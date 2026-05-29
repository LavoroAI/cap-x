from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr
import robodm
import tyro


@dataclass
class VisualizeArgs:
    """Visualise one or more RoboDM ``.vla`` trajectories in the Rerun viewer."""

    path: str
    """A ``.vla`` file, or a directory containing ``traj_*.vla`` files."""

    num: int | None = None
    """When ``path`` is a directory, cap the number of trajectories to load."""

    pattern: str = "traj_*.vla"
    """Glob applied inside a directory."""

    app_id: str = "capx-vla"
    """Rerun application id."""


# Module-level so RecordingStream wrappers aren't GC'd between files —
# their __del__ flushes and disconnects, which kills in-flight data.
_ACTIVE_STREAMS: list[rr.RecordingStream] = []

_DEFAULT_SAMPLE_RATE_HZ = 20.0
_TIMELINE = "time"


def _load_stats(stats_path: Path) -> tuple[float | None, dict | None]:
    """Return (sample_rate_hz, full blob) from the sidecar, or (None, None)."""
    if not stats_path.is_file():
        return None, None
    try:
        blob = json.loads(stats_path.read_text())
    except Exception as e:
        print(f"[visualize_data] could not read {stats_path}: {e}")
        return None, None
    if not isinstance(blob, dict):
        return None, blob
    rate = blob.get("sample_rate_hz")
    if rate is None:
        env_meta = blob.get("env_metadata") or {}
        rate = env_meta.get("sample_rate_hz")
        cf = env_meta.get("control_freq_hz")
        sub = env_meta.get("subsample_rate")
        if rate is None and cf and sub:
            rate = float(cf) / float(sub)
    return (float(rate) if rate is not None else None), blob


def _log_step(rec: rr.RecordingStream, key: str, arr: np.ndarray, t: int) -> None:
    """Log one feature at one timestep using a per-step archetype call."""
    if t >= len(arr):
        return
    v = arr[t]
    # (T, H, W, 3) uint8 → RGB image
    if arr.ndim == 4 and arr.shape[-1] == 3 and arr.dtype == np.uint8:
        rec.log(key, rr.Image(v))
        return
    # (T, H, W) → depth (float) or segmentation (int)
    if arr.ndim == 3:
        if np.issubdtype(arr.dtype, np.floating):
            rec.log(key, rr.DepthImage(np.asarray(v, dtype=np.float32)))
        else:
            rec.log(key, rr.SegmentationImage(np.asarray(v, dtype=np.uint16)))
        return
    # (T, N) vector → one scalar per component
    if arr.ndim == 2:
        for i in range(arr.shape[1]):
            rec.log(f"{key}/{i}", rr.Scalars(float(v[i])))
        return
    # (T,) scalar
    if arr.ndim == 1:
        rec.log(key, rr.Scalars(float(v)))


def _visualize_one(path: Path, app_id: str, *, spawn: bool) -> None:
    """Open one ``.vla`` file as a fresh recording under the shared app_id."""
    print(f"[visualize_data] loading {path}")

    datamanager = robodm.Trajectory(path=str(path), mode="r")
    data: dict[str, np.ndarray] = datamanager.load()
    if not data:
        print(f"[visualize_data] {path} is empty; skipping")
        return

    # One RecordingStream per file under the same application_id. Held in a
    # module-level list so the Python wrapper isn't GC'd at function return.
    rec = rr.RecordingStream(application_id=app_id, recording_id=path.stem)
    _ACTIVE_STREAMS.append(rec)
    if spawn:
        rec.spawn()
    else:
        try:
            rec.connect_grpc()
        except Exception:
            pass

    stats_path = path.with_name(path.stem + "_stats.json")
    sample_rate_hz, stats_blob = _load_stats(stats_path)
    if stats_blob is not None:
        rec.log(
            "meta/stats",
            rr.TextDocument(json.dumps(stats_blob, indent=2)),
            static=True,
        )
    if sample_rate_hz is None or sample_rate_hz <= 0:
        sample_rate_hz = _DEFAULT_SAMPLE_RATE_HZ
    rec.log(
        "meta/sample_rate_hz",
        rr.TextDocument(f"{sample_rate_hz:.3f} Hz"),
        static=True,
    )

    # The recorded ``timestep`` column is redundant once we drive the timeline
    # from sample_rate_hz — drop it so it isn't logged as a scalar series.
    data.pop("timestep", None)

    # Bucket features: numeric arrays get the per-step loop; string/object
    # arrays are summarised as a static TextDocument up-front.
    arrays: dict[str, np.ndarray] = {}
    n_steps = 0
    for key, values in data.items():
        arr = np.asarray(values)
        if arr.dtype == object or np.issubdtype(arr.dtype, np.str_):
            uniq = sorted({str(x) for x in arr.tolist() if x is not None and str(x) != ""})
            if uniq:
                text = "\n---\n".join(uniq) if len(uniq) > 1 else uniq[0]
                entity = "instruction" if key == "language_instruction" else f"text/{key}"
                rec.log(entity, rr.TextDocument(text), static=True)
            continue
        if not np.issubdtype(arr.dtype, np.number):
            print(f"[visualize_data] skipping {key}: unsupported dtype {arr.dtype}")
            continue
        arrays[key] = arr
        n_steps = max(n_steps, int(arr.shape[0]))

    # Per-step loop: set the timeline from the sample rate, then log each
    # feature with its single-step archetype. No send_columns anywhere.
    for t in range(n_steps):
        rec.set_time(_TIMELINE, duration=t / sample_rate_hz)
        for key, arr in arrays.items():
            try:
                _log_step(rec, key, arr, t)
            except Exception as e:
                print(f"[visualize_data] failed to log {key} at t={t}: {e}")

    close = getattr(datamanager, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _resolve_files(path: str, pattern: str, num: int | None) -> list[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        files = sorted(p.glob(pattern))
        return files[:num] if num is not None else files
    raise FileNotFoundError(f"No such file or directory: {path}")


def main(args: VisualizeArgs) -> None:
    files = _resolve_files(args.path, args.pattern, args.num)
    if not files:
        print(f"[visualize_data] no trajectories matched {args.path}/{args.pattern}")
        return

    print(f"[visualize_data] visualizing {len(files)} trajectory(ies)")
    for idx, p in enumerate(files):
        _visualize_one(p, args.app_id, spawn=(idx == 0))


if __name__ == "__main__":
    main(tyro.cli(VisualizeArgs))
