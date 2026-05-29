"""Autonomous VLA data-collection entry point.

CaP-X has two top-level CLIs:

  - ``capx.envs.launch``        — eval / benchmark; never touches the collector.
  - ``capx.envs.collect_data``  — data collection; intercepts every sim step
                                   and writes RoboDM ``.vla`` trajectory files
                                   under ``demos/<task_id>/``.

Usage::

    uv run capx/envs/collect_data.py \\
        --config-path env_configs/cube_stack/franka_robosuite_cube_stack_collect.yaml \\
        --use-oracle-code

Multi-task LIBERO sweep (enumerates every task in the named suite via
``libero.benchmark.get_benchmark_dict()``)::

    uv run capx/envs/collect_data.py \\
        --config-path env_configs/libero/franka_libero_collect_template.yaml \\
        --multi-task libero --use-oracle-code

Execution flow::

    main()
      ├─ _load_config(args)                       (eval-shared YAML loader)
      ├─ _load_collect_block(args.config_path)    (curriculum-only YAML helper)
      ├─ _spec_from_args / build_libero_curriculum
      ├─ _start_api_servers(api_servers)
      └─ collect_curriculum(spec, ...)            (per-task → _run_headless_trials with collector)
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

import tyro

from capx.envs.launch import LaunchArgs
from capx.utils.launch_utils import _load_config

os.environ.setdefault("MUJOCO_GL", "egl")


# ---------------------------------------------------------------------------
# CLI argument dataclass
# ---------------------------------------------------------------------------


@dataclass
class CollectArgs(LaunchArgs):
    """Adds collection-specific flags on top of the eval ``LaunchArgs``."""

    demo_output_root: str | None = None
    """Root directory for ``.vla`` files (one subdir per task_id). Overrides YAML ``collect.output_root``."""

    target_successes: int | None = None
    """Per-task quota of successful ``.vla`` files. Overrides YAML ``collect.target_successes``."""

    max_trials: int | None = None
    """Safety upper bound on attempts per task. Overrides YAML ``collect.max_trials``."""

    keep_failures: bool | None = None
    """If True, retain ``.vla`` files for failed trials. Overrides YAML ``collect.keep_failures``."""

    record_hz: float | None = None
    """Target recording frequency in Hz (e.g. 20.0). Overrides YAML ``collect.record_hz``.
    ``None`` falls back to YAML; if still unset, simulator records every control step."""

    multi_task: str | None = None
    """Multi-task curriculum mode. ``libero`` enumerates LIBERO suites; ``None`` means single-task."""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(args: CollectArgs) -> None:
    """Load config, build the curriculum, then run it to per-task quota."""
    from capx.data.curriculum import (
        _load_collect_block,
        _spec_from_args,
        build_libero_curriculum,
        collect_curriculum,
    )
    from capx.envs.runner import _start_api_servers, _stop_api_servers

    start_time = time.time()
    env_factory, config, api_servers = _load_config(args)
    collect_block = _load_collect_block(args.config_path)

    if args.multi_task == "libero":
        spec = build_libero_curriculum(
            args=args, base_config=config, collect_block=collect_block,
        )
    elif args.multi_task is None:
        spec = _spec_from_args(args, config, collect_block)
    else:
        raise ValueError(
            f"--multi-task={args.multi_task!r} is not supported. "
            "Use 'libero' or omit the flag for single-task collection."
        )

    if not spec.tasks:
        print("[collect_data] No tasks in curriculum; exiting.")
        return

    print(
        f"[collect_data] Starting curriculum with {len(spec.tasks)} task(s), "
        f"output_root={spec.output_root}"
    )

    server_procs = _start_api_servers(api_servers)
    try:
        summary = collect_curriculum(
            spec,
            env_factory=env_factory,
            base_config=config,
            launch_args=args,
            start_time=start_time,
        )
        print(f"[collect_data] curriculum summary: {summary.as_dict()}")
    finally:
        try:
            _stop_api_servers(server_procs)
        except KeyboardInterrupt:
            sys.exit(1)


if __name__ == "__main__":
    main(tyro.cli(CollectArgs))
