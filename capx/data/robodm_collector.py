"""Per-worker collector that writes RoboDM .vla files for VLA training.

One folder per task_id, one .vla file per successful demonstration:

    output_root/<task_id>/traj_NNNN.vla
    output_root/<task_id>/traj_NNNN_stats.json

Disk I/O is drained off the simulator thread via an internal queue +
background writer thread so the sim step never blocks on libx264
encoding. Atomic per-task index reservation makes this safe under
parallel workers writing to the same task directory.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any

import robodm

from capx.data.schema import Step, step_to_dict

logger = logging.getLogger(__name__)

_WRITER_SENTINEL: tuple[None, None] = (None, None)


def _reserve_next_traj_path(task_dir: Path) -> Path:
    """Atomically reserve the next free `traj_NNNN.vla` path in task_dir.

    Uses O_CREAT|O_EXCL to claim an empty placeholder file so two parallel
    workers cannot grab the same index. robodm overwrites the placeholder
    on first write.
    """
    task_dir.mkdir(parents=True, exist_ok=True)
    while True:
        used: set[int] = set()
        for p in task_dir.glob("traj_*.vla"):
            stem = p.stem.removeprefix("traj_")
            if stem.isdigit():
                used.add(int(stem))
        next_idx = (max(used) + 1) if used else 0
        candidate = task_dir / f"traj_{next_idx:04d}.vla"
        try:
            fd = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.close(fd)
            return candidate
        except FileExistsError:
            continue


class _EpisodeHandle:
    """Bookkeeping for one in-flight episode."""

    __slots__ = (
        "path", "task_id", "language_instruction", "seed", "env_metadata",
        "traj", "q", "thread", "start_time", "num_steps", "dropped_steps",
        "closed",
    )

    def __init__(
        self,
        path: Path,
        task_id: str,
        language_instruction: str,
        seed: int,
        env_metadata: dict[str, Any],
        traj: robodm.Trajectory,
        q: "queue.Queue[tuple[dict | None, float | None]]",
        thread: threading.Thread,
        start_time: float,
    ) -> None:
        self.path = path
        self.task_id = task_id
        self.language_instruction = language_instruction
        self.seed = seed
        self.env_metadata = env_metadata
        self.traj = traj
        self.q = q
        self.thread = thread
        self.start_time = start_time
        self.num_steps = 0
        self.dropped_steps = 0
        self.closed = False


class RoboDMCollector:
    """Collector that writes one .vla per successful demonstration."""

    def __init__(
        self,
        output_root: Path | str,
        *,
        worker_id: int = 0,
        video_codec: str = "libx264",
        codec_options: dict[str, Any] | None = None,
        keep_failures: bool = False,
        log_system_stats: bool = False,
        queue_size: int = 1000,
    ) -> None:
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.worker_id = worker_id
        self.video_codec = video_codec
        self.codec_options = codec_options or {}
        self.keep_failures = keep_failures
        self.log_system_stats = log_system_stats
        self.queue_size = queue_size
        self._handles: dict[str, _EpisodeHandle] = {}

    # -------------------------------------------------------------------- lifecycle

    def start_episode(
        self,
        *,
        task_id: str,
        language_instruction: str,
        seed: int,
        env_metadata: dict[str, Any],
    ) -> str:
        """Reserve the next traj path, open a fresh Trajectory, spawn writer."""
        task_dir = self.output_root / task_id
        path = _reserve_next_traj_path(task_dir)

        try:
            traj = robodm.Trajectory(
                str(path),
                mode="w",
                video_codec=self.video_codec,
                codec_options=self.codec_options or None,
                time_unit="s",
                enforce_monotonic=True,
            )
        except Exception:
            # Couldn't open writer; clean up placeholder and re-raise
            path.unlink(missing_ok=True)
            raise

        q: "queue.Queue[tuple[dict | None, float | None]]" = queue.Queue(maxsize=self.queue_size)
        start_time = time.time()

        handle_id = str(path)
        thread = threading.Thread(
            target=self._writer_loop,
            args=(handle_id, traj, q),
            daemon=True,
            name=f"robodm-writer-w{self.worker_id}-{task_id}",
        )

        h = _EpisodeHandle(
            path=path,
            task_id=task_id,
            language_instruction=language_instruction,
            seed=seed,
            env_metadata=env_metadata,
            traj=traj,
            q=q,
            thread=thread,
            start_time=start_time,
        )
        self._handles[handle_id] = h
        thread.start()
        return handle_id

    def append_step(self, handle_id: str, step: Step) -> None:
        """Non-blocking enqueue. Drops the step (with a warning) if full."""
        h = self._handles.get(handle_id)
        if h is None or h.closed:
            return
        flat = step_to_dict(step)
        try:
            h.q.put_nowait((flat, float(step.timestep)))
            h.num_steps += 1
        except queue.Full:
            h.dropped_steps += 1
            if h.dropped_steps == 1 or h.dropped_steps % 50 == 0:
                logger.warning(
                    "[robodm] writer queue full for %s; dropped %d step(s)",
                    h.task_id, h.dropped_steps,
                )

    def finish_episode(
        self,
        handle_id: str,
        *,
        success: bool,
        terminal_reward: float,
        info: dict[str, Any] | None = None,
    ) -> bool:
        """Drain writer, close trajectory, write _stats.json. Returns True iff kept."""
        h = self._handles.pop(handle_id, None)
        if h is None:
            return False
        if h.closed:
            return False
        h.closed = True

        # Signal writer to drain + close and wait for it.
        try:
            h.q.put(_WRITER_SENTINEL, timeout=10.0)
        except queue.Full:
            logger.warning("[robodm] queue full while sending close sentinel")
        h.thread.join(timeout=120.0)
        if h.thread.is_alive():
            logger.error("[robodm] writer thread did not exit for %s", h.path)

        end_time = time.time()
        stats_path = h.path.with_name(h.path.stem + "_stats.json")

        if success or self.keep_failures:
            stats = {
                "file_name": str(h.path),
                "task_id": h.task_id,
                "language_instruction": h.language_instruction,
                "seed": h.seed,
                "success": bool(success),
                "terminal_reward": float(terminal_reward),
                "num_steps": h.num_steps,
                "dropped_steps": h.dropped_steps,
                "start_time": h.start_time,
                "end_time": end_time,
                "total_time": end_time - h.start_time,
                # Lifted from env_metadata so downstream tools (e.g. the rerun
                # visualizer) don't have to dig for the cadence.
                "sample_rate_hz": h.env_metadata.get("sample_rate_hz"),
                "env_metadata": h.env_metadata,
                "agent_info": info or {},
            }
            if self.log_system_stats:
                stats["system_stats"] = _collect_system_stats()
            try:
                stats_path.write_text(json.dumps(stats, indent=2, default=_json_default))
            except Exception as e:
                logger.error("[robodm] failed to write %s: %s", stats_path, e)
            return True
        else:
            h.path.unlink(missing_ok=True)
            stats_path.unlink(missing_ok=True)
            return False

    def committed_count(self, task_id: str) -> int:
        """Number of .vla files currently on disk for this task."""
        return len(list((self.output_root / task_id).glob("traj_*.vla")))

    def close(self) -> None:
        """Idempotent: any in-flight episode is finished with success=False."""
        for handle_id in list(self._handles.keys()):
            try:
                self.finish_episode(
                    handle_id, success=False, terminal_reward=0.0,
                    info={"aborted": True},
                )
            except Exception as e:
                logger.error("[robodm] error closing %s: %s", handle_id, e)

    # ----------------------------------------------------------------- background

    def _writer_loop(
        self,
        handle_id: str,
        traj: robodm.Trajectory,
        q: "queue.Queue[tuple[dict | None, float | None]]",
    ) -> None:
        while True:
            try:
                item = q.get(timeout=300.0)
            except queue.Empty:
                logger.error("[robodm] writer for %s timed out; closing", handle_id)
                break
            if item == _WRITER_SENTINEL:
                break
            flat, ts = item
            try:
                traj.add_by_dict(flat, timestamp=ts, time_unit="s")
            except Exception as e:
                logger.error("[robodm] add_by_dict failed at ts=%s: %s", ts, e)
        try:
            traj.close()
        except Exception as e:
            logger.error("[robodm] traj.close failed for %s: %s", handle_id, e)


def _json_default(o: Any) -> Any:
    """Make numpy / Path objects JSON-serialisable for the stats sidecar."""
    try:
        import numpy as np
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
    except Exception:
        pass
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _collect_system_stats() -> dict[str, Any]:
    """Best-effort CPU/RAM/GPU stats. Skips silently if psutil/nvidia-smi absent."""
    stats: dict[str, Any] = {}
    try:
        import psutil
        stats["avg_cpu_usage"] = float(psutil.cpu_percent(interval=0.1))
        stats["avg_ram_mem"] = float(psutil.virtual_memory().percent)
    except Exception:
        pass
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"], timeout=2.0,
        ).decode()
        gpus_util, gpus_mem = [], []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                gpus_util.append(float(parts[0]))
                gpus_mem.append(int(parts[1]) * 1024 * 1024)
        if gpus_util:
            stats["avg_gpus_usage"] = gpus_util
            stats["avg_gpus_mem"] = gpus_mem
    except Exception:
        pass
    return stats
