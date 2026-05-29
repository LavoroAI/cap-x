"""Quota-driven curriculum loop for VLA data collection.

Drives :func:`capx.envs.runner._run_headless_trials` once per task and counts
successful ``.vla`` files on disk until each task hits its quota (or its
``max_trials`` safety cap, whichever comes first). The eval entry point
(``launch.py``) is *not* involved here — collection is fully owned by
``collect_data.py``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------- spec


@dataclass
class CurriculumTask:
    task_id: str
    language_instruction: str
    quota_successes: int
    max_trials: int
    prefer_oracle: bool = True
    seed_offset: int = 0
    # Optional dotted-path overrides applied to ``env_factory["cfg"]["low_level"]``
    # before instantiation (used by LIBERO to switch suite_name / task_id per task).
    env_factory_overrides: dict[str, Any] | None = None


@dataclass
class CurriculumSpec:
    tasks: list[CurriculumTask]
    output_root: Path
    num_workers: int = 8
    keep_failures: bool = False
    log_system_stats: bool = False
    video_codec: str = "libx264"
    codec_options: dict[str, Any] = field(default_factory=dict)
    save_depth: bool = False
    record_hz: float | None = None  # Target recording frequency; None ⇒ env default.


@dataclass
class CurriculumSummary:
    per_task: dict[str, dict[str, Any]] = field(default_factory=dict)

    def note_task_done(self, task_id: str, *, kept: int, attempted: int) -> None:
        self.per_task[task_id] = {"kept": kept, "attempted": attempted}

    def as_dict(self) -> dict[str, Any]:
        return {"per_task": self.per_task}


# ---------------------------------------------------------------------------- helpers


def _load_collect_block(config_path: str) -> dict[str, Any]:
    """Parse the YAML and return its top-level ``collect:`` block (or empty dict)."""
    with open(config_path) as f:
        data = yaml.safe_load(f) or {}
    return data.get("collect", {}) or {}


def _derive_task_id(config_path: str) -> str:
    """Fallback task_id derived from the YAML filename stem."""
    return Path(config_path).stem.removesuffix("_collect")


# ---------------------------------------------------------------------------- API


def collect_curriculum(
    spec: CurriculumSpec,
    *,
    env_factory: dict[str, Any],
    base_config: dict[str, Any],
    launch_args: Any,
    start_time: float,
) -> CurriculumSummary:
    """Run every task in ``spec.tasks`` up to its per-task success quota.

    One :func:`_run_headless_trials` call per task. The collector lives in each
    worker process; the curriculum counts ``.vla`` files on disk before/after
    each call to determine how many demos were committed.

    API servers are assumed to already be running (started by the caller, e.g.
    ``collect_data.main``). This function does not touch them.
    """
    from capx.envs.runner import _run_headless_trials

    out = Path(spec.output_root)
    out.mkdir(parents=True, exist_ok=True)
    summary = CurriculumSummary()

    for task in spec.tasks:
        task_dir = out / task.task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        def _committed() -> int:
            return len(list(task_dir.glob("traj_*.vla")))

        already = _committed()
        if already >= task.quota_successes:
            logger.info(
                "[curriculum] task=%s already has %d/%d demos — skipping",
                task.task_id, already, task.quota_successes,
            )
            summary.note_task_done(task.task_id, kept=already, attempted=0)
            continue

        logger.info(
            "[curriculum] task=%s quota=%d max_trials=%d (have=%d)",
            task.task_id, task.quota_successes, task.max_trials, already,
        )

        # Per-task copies so we don't mutate caller's state.
        task_env_factory = _apply_env_factory_overrides(
            env_factory, task.env_factory_overrides
        )
        task_config = {
            **base_config,
            "task_id": task.task_id,
            "language_instruction": task.language_instruction,
            "total_trials": task.max_trials,
            "num_workers": spec.num_workers or base_config.get("num_workers", 1),
            "use_oracle_code": (
                task.prefer_oracle
                if base_config.get("use_oracle_code") is None
                else base_config["use_oracle_code"]
            ),
            "collect_save_depth": bool(spec.save_depth),
        }
        collector_kwargs = {
            "output_root": Path(spec.output_root),
            "video_codec": spec.video_codec,
            "codec_options": dict(spec.codec_options or {}),
            "keep_failures": spec.keep_failures,
            "log_system_stats": spec.log_system_stats,
        }
        if spec.record_hz is not None:
            # Forwarded through collector_kwargs purely as a delivery channel;
            # the runner pops it before instantiating the collector.
            collector_kwargs["record_hz"] = float(spec.record_hz)

        try:
            _run_headless_trials(
                launch_args,
                task_env_factory,
                task_config,
                start_time,
                collector_kwargs=collector_kwargs,
            )
        except Exception as e:
            logger.exception("[curriculum] task=%s run failed: %s", task.task_id, e)

        kept = _committed()
        logger.info(
            "[curriculum] task=%s done kept=%d/%d (max_trials=%d)",
            task.task_id, kept, task.quota_successes, task.max_trials,
        )
        summary.note_task_done(task.task_id, kept=kept, attempted=task.max_trials)

    summary_path = out / "curriculum_summary.json"
    try:
        summary_path.write_text(json.dumps(summary.as_dict(), indent=2))
    except Exception as e:
        logger.error("[curriculum] failed to write %s: %s", summary_path, e)

    return summary


def _apply_env_factory_overrides(
    env_factory: dict[str, Any], overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    """Deep-copy ``env_factory`` and shallow-merge ``overrides`` into ``cfg.low_level``.

    Used by LIBERO to swap ``suite_name`` / ``task_id`` per curriculum task without
    re-loading the YAML.
    """
    if not overrides:
        return env_factory
    import copy
    new_factory = copy.deepcopy(env_factory)
    low = new_factory.setdefault("cfg", {}).setdefault("low_level", {})
    if isinstance(low, dict):
        low.update(overrides)
    else:
        logger.warning(
            "[curriculum] cfg.low_level is not a dict (%r); env_factory_overrides ignored",
            type(low).__name__,
        )
    return new_factory


# ---------------------------------------------------------------------------- spec builders


def _spec_from_args(
    args: Any, config: dict[str, Any], collect_block: dict[str, Any],
) -> CurriculumSpec:
    """Build a one-task spec from CLI args + the loaded eval config + collect: block.

    CLI overrides (``args.target_successes`` etc.) win over YAML ``collect:`` values.
    """
    task_id = collect_block.get("task_id") or _derive_task_id(args.config_path)

    quota = (
        getattr(args, "target_successes", None)
        if getattr(args, "target_successes", None) is not None
        else collect_block.get("target_successes")
    )
    if quota is None:
        quota = config.get("total_trials") or 1

    max_trials = (
        getattr(args, "max_trials", None)
        if getattr(args, "max_trials", None) is not None
        else collect_block.get("max_trials")
    )
    if max_trials is None:
        max_trials = config.get("total_trials") or max(int(quota) * 3, int(quota) + 1)

    keep_failures = (
        getattr(args, "keep_failures", None)
        if getattr(args, "keep_failures", None) is not None
        else bool(collect_block.get("keep_failures", False))
    )

    out_root = (
        getattr(args, "demo_output_root", None)
        or collect_block.get("output_root")
        or "./demos"
    )

    record_hz = (
        getattr(args, "record_hz", None)
        if getattr(args, "record_hz", None) is not None
        else collect_block.get("record_hz")
    )

    task = CurriculumTask(
        task_id=task_id,
        language_instruction=collect_block.get("language_instruction") or "",
        quota_successes=int(quota),
        max_trials=int(max_trials),
        prefer_oracle=bool(config.get("use_oracle_code", False)),
    )

    return CurriculumSpec(
        tasks=[task],
        output_root=Path(out_root),
        num_workers=int(config.get("num_workers", 8) or 8),
        keep_failures=bool(keep_failures),
        log_system_stats=bool(collect_block.get("log_system_stats", False)),
        video_codec=collect_block.get("video_codec", "libx264"),
        codec_options=dict(collect_block.get("codec_options") or {}),
        save_depth=bool(collect_block.get("save_depth", False)),
        record_hz=float(record_hz) if record_hz is not None else None,
    )


# ----------------------------------------------------------------- LIBERO helpers


def build_libero_curriculum(
    *,
    args: Any,
    base_config: dict[str, Any],
    collect_block: dict[str, Any],
    suite_names: list[str] | None = None,
) -> CurriculumSpec:
    """Enumerate LIBERO tasks via ``libero.benchmark.get_benchmark_dict()`` and
    return a multi-task CurriculumSpec.

    Each :class:`CurriculumTask` carries ``env_factory_overrides`` so
    :func:`collect_curriculum` can swap ``suite_name`` / ``task_id`` on the
    shared template env factory per task.

    ``suite_names`` defaults to all suites in ``base_config['libero_suites']``
    or, failing that, to the suite already configured in ``cfg.low_level``.
    """
    from libero.libero import benchmark  # type: ignore[import-not-found]

    if suite_names is None:
        suite_names = (
            collect_block.get("libero_suites")
            or base_config.get("libero_suites")
            or []
        )
    if not suite_names:
        # Fall back to whatever the template config declared.
        ll = base_config.get("env", {}).get("cfg", {}).get("low_level", {})
        if isinstance(ll, dict) and ll.get("suite_name"):
            suite_names = [ll["suite_name"]]
        else:
            raise ValueError(
                "build_libero_curriculum: no suite_names provided; set "
                "`collect.libero_suites: [libero_object, ...]` in the YAML or "
                "pass --multi-task libero with an explicit list."
            )

    quota_per_task = int(
        getattr(args, "target_successes", None)
        or collect_block.get("target_successes")
        or 10
    )
    max_trials_per_task = int(
        getattr(args, "max_trials", None)
        or collect_block.get("max_trials")
        or quota_per_task * 3
    )
    keep_failures = bool(
        getattr(args, "keep_failures", None)
        if getattr(args, "keep_failures", None) is not None
        else collect_block.get("keep_failures", False)
    )

    benchmark_dict = benchmark.get_benchmark_dict()
    tasks: list[CurriculumTask] = []
    for suite_name in suite_names:
        if suite_name not in benchmark_dict:
            logger.warning("[curriculum] unknown libero suite: %s", suite_name)
            continue
        task_suite = benchmark_dict[suite_name]()
        for libero_task_id in range(task_suite.n_tasks):
            try:
                task_lang = task_suite.get_task(libero_task_id).language
            except Exception:
                task_lang = ""
            tasks.append(
                CurriculumTask(
                    task_id=f"franka_libero_{suite_name}_{libero_task_id}",
                    language_instruction=task_lang,
                    quota_successes=quota_per_task,
                    max_trials=max_trials_per_task,
                    prefer_oracle=bool(base_config.get("use_oracle_code", False)),
                    env_factory_overrides={
                        "suite_name": suite_name,
                        "task_id": libero_task_id,
                    },
                )
            )

    out_root = (
        getattr(args, "demo_output_root", None)
        or collect_block.get("output_root")
        or "./demos"
    )

    record_hz = (
        getattr(args, "record_hz", None)
        if getattr(args, "record_hz", None) is not None
        else collect_block.get("record_hz")
    )

    return CurriculumSpec(
        tasks=tasks,
        output_root=Path(out_root),
        num_workers=int(base_config.get("num_workers", 8) or 8),
        keep_failures=keep_failures,
        log_system_stats=bool(collect_block.get("log_system_stats", False)),
        video_codec=collect_block.get("video_codec", "libx264"),
        codec_options=dict(collect_block.get("codec_options") or {}),
        save_depth=bool(collect_block.get("save_depth", False)),
        record_hz=float(record_hz) if record_hz is not None else None,
    )
