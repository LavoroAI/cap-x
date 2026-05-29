# Autonomous VLA Data Collection via CaP-Agent0

## Context

CaP-Agent0 today is a *code-generation evaluation* framework: it queries an LLM,
the LLM writes Python that calls high-level perception+control APIs, that code
drives a simulator (RoboSuite / LIBERO / OmniGibson), and at the end the trial
is graded as success/failure. The artifacts it saves are LLM-facing (code,
prompts, MP4 videos for debugging) — none of them constitute a VLA training
demonstration.

The proposal: **turn CaP-Agent0 into an autonomous data engine.** Run the
existing curriculum of tasks repeatedly, intercept the low-level
`(observation, action)` trace each time the code-policy steps the simulator,
keep only the trials that satisfy `task_completed`, and write them to disk as
**RoboDM `.vla` trajectories** — one folder per task, one `.vla` file per
demonstration, compressed in real-time as the simulator runs.

The goal of this milestone is a **sim-only validation**: prove that the
existing curriculum can produce VLA-shaped data autonomously, at
worker-parallel scale, with successful-only filtering, without disturbing the
existing eval workflow.

---

## What exists today (entry points to reuse)

| Concern | Where | Notes |
| --- | --- | --- |
| Top-level launch | `capx/envs/launch.py:main` (`LaunchArgs` dataclass at L37) | `tyro` CLI; what `uv run capx/envs/launch.py …` invokes |
| Headless orchestration | `capx/envs/runner.py:_run_headless_trials`, `_start_api_servers` | Spawns parallel workers, brings up SAM3 / GraspNet / PyRoKi |
| Per-trial loop | `capx/envs/trial.py:_run_single_trial` (L629) | Reset → init code gen → step code blocks → multi-turn decide → save |
| Trial dir naming | `capx/envs/trial.py:_trial_video_dir` (L113); `_save_trial_artifacts` (L376 of `launch_utils.py`) | Already encodes `taskcompleted_{0,1}` |
| Frame buffer (main + wrist) | `RobosuiteBaseEnv._frame_buffer`, `_wrist_frame_buffer`, `enable_video_capture`, `_record_frame`, `get_video_frames` (`simulators/robosuite_base.py`) | Subsampled at rate 5 (4 for LIBERO / R1Pro) |
| Low-level control chokepoint | `RobosuiteBaseEnv.move_to_joints_blocking` / `move_to_joints_non_blocking` / `_step_once` / `_set_gripper` (L114–208) | All actions to the sim pass through these three call sites |
| Success signal | `task_completed()` per env; surfaced as `info_step["task_completed"]` in `_run_single_trial`; already gates skill extraction (L928 of `trial.py`) |
| Task curriculum | 170 YAMLs in `env_configs/**` across 8 RoboSuite families + LIBERO + R1Pro/BEHAVIOR |
| Batch curriculum runners | `capx/envs/scripts/run_batch.py`, `run_libero_batch.py` | Iterate over YAML configs, hand each to `launch.main` |
| Env registry | `capx/envs/simulators/__init__.py:register_env` | Used by `_target_` YAML resolution |
| Skill library | `capx/skills/` — auto-promotes functions seen in ≥2 successes | Pattern to mirror for "promote episodes on success" |

---

## What is missing (this milestone builds it)

1. A **per-sim-step recorder hook** on the low-level env, snapshotting
   `(action, observation)` at MuJoCo-step granularity — not LLM-turn
   granularity.
2. A **`Step` schema** (dataclasses) that describes one timestep in a way
   that flattens cleanly to RoboDM's nested-feature dict.
3. A **`RoboDMCollector`** that wraps `robodm.Trajectory` with an async
   writer thread (mirroring the reference recorder in the user-provided
   example), so the simulator step never blocks on codec encoding.
4. A **curriculum-driven outer loop** that keeps running each task until a
   per-task quota of *successful* `.vla` files lands in its task directory,
   instead of a fixed `trials: N`.
5. A **finalize step** that emits `curriculum_summary.json` aggregating
   per-task counts, average episode length, and pointer to the
   per-trajectory `_stats.json` sidecars RoboDM already writes.

---

## Dataset format: RoboDM `.vla`

We adopt the [RoboDM](https://github.com/RoboDM) data standard via the
`robodm` Python package. Each demonstration is a single `.vla` file written
by `robodm.Trajectory(path, mode="w", video_codec="libx264", raw_codec="rawvideo_pyarrow")`.
Per-feature codec choice is automatic:

- **RGB streams** (`observation/rgb_main`, `observation/rgb_wrist`) → libx264
  (compressed video, ~10–100× smaller than raw arrays).
- **Everything else** (proprio, actions, scalars, strings) → `rawvideo_pyarrow`
  (uncompressed Arrow, fast read).

Each step appends one record via `traj.add_by_dict(state, timestamp=t, time_unit="s")`.
The dict can be nested (`{"observation": {"rgb_main": ...}}`) or pre-flattened
(`{"observation/rgb_main": ...}`) — RoboDM accepts both.

### Disk layout

```
demos/                                           # the curriculum output root
├── franka_robosuite_cube_stack/                 # one folder per task_id
│   ├── traj_0000.vla
│   ├── traj_0000_stats.json
│   ├── traj_0001.vla
│   ├── traj_0001_stats.json
│   └── ...
├── franka_libero_spatial_2/
│   ├── traj_0000.vla
│   └── ...
├── franka_robosuite_two_arm_handover/
│   └── ...
└── curriculum_summary.json
```

The `_stats.json` sidecar is produced by the recorder itself (see the
reference `Recorder._save_stats()`): start/end time, num steps, optional
CPU/RAM/GPU usage, per-step rewards, final reward. We extend it with
CaP-Agent0–specific fields (success flag, LLM model, oracle/code,
num_regenerations, sandbox_rc, config_path).

### Why `.vla` (vs HDF5 / LeRobot / RLDS)

- **Real-time optimised.** The reference recorder pattern decouples disk
  I/O from the producer via a request queue + background drainer. The
  simulator step never stalls on libx264 encoding. Critical for the
  real-robot generalisation later.
- **One file per demo.** Trivial to shard, ship, dedupe, version with git
  LFS / S3, or stream from an object store. No "one giant HDF5" foot-gun.
- **Per-task folders.** Curriculum-coverage queries become `ls | wc -l`.
- **Lossy where it matters, lossless elsewhere.** RGB gets video-coded
  (cheap), proprio stays exact (Arrow).
- **Schema is just the dict keys.** No fragile RLDS feature spec, no
  LeRobot-version pinning.

---

## The CaP-X `Step` schema

`capx/data/schema.py` (new file) defines what one timestep looks like.
Mirrors the convention of the reference example: a top-level `Step`
dataclass containing nested `Observation` / `Action` dataclasses. The
`step_to_dict` helper (also a direct port from the reference) flattens
nested dataclasses into `parent/child` keys before handing them to
`robodm.add_by_dict`.

```python
# capx/data/schema.py  (new)
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Observation:
    rgb_main:        np.ndarray | None = None  # (H, W, 3) uint8, agentview / robot0_robotview
    rgb_wrist:       np.ndarray | None = None  # (H, W, 3) uint8, robot0_eye_in_hand
    joint_pos:       np.ndarray | None = None  # (7,)  float32 radians
    ee_pose:         np.ndarray | None = None  # (7,)  float32 wxyz_xyz in robot-base frame
    gripper_fraction: float | None    = None   # 0.0 closed → 1.0 open
    # Optional / opt-in:
    depth_main:      np.ndarray | None = None  # (H, W) float32 metric depth
    segmentation:    np.ndarray | None = None  # (H, W) int32 instance ids


@dataclass
class Action:
    # The "canonical" action: what's commanded into the controller.
    joint_target:    np.ndarray | None = None  # (7,) float32 absolute joint targets (rad)
    gripper_command: float | None      = None  # 0.0–1.0 fraction (capx convention)
    # The "native" action: what the underlying env actually saw at step().
    # Same as joint_target for RoboSuite; (8,) delta-scaled for LIBERO.
    native:          np.ndarray | None = None


@dataclass
class Step:
    timestep:              float                # sim-time seconds, not wall-clock
    observation:           Observation
    action:                Action
    language_instruction:  str | None = None    # constant per episode; recorded every step
                                                # so consumers don't need separate metadata
    reward:                float | None = None  # per-step reward (sim reward function)
    info:                  dict[str, Any] = field(default_factory=dict)


def step_to_dict(step: Step) -> dict[str, Any]:
    """Flatten a Step into the hierarchical-key dict robodm consumes.

    Direct port of the helper in the user-provided reference Recorder
    example. Nested dataclasses → `parent/child` keys. `None` values are
    dropped (so an episode without a wrist camera writes no rgb_wrist column).
    """
    out: dict[str, Any] = {}
    items = step.__dict__.items() if hasattr(step, "__dataclass_fields__") else step.items()
    for k, v in items:
        if hasattr(v, "__dataclass_fields__") or isinstance(v, dict):
            for k2, v2 in step_to_dict(v).items():
                if v2 is not None:
                    out[f"{k}/{k2}"] = v2
        elif v is not None:
            out[k] = v
    return out
```

Resulting RoboDM feature names (when both cameras and the LIBERO native
action are present):

```
observation/rgb_main           → libx264
observation/rgb_wrist          → libx264
observation/joint_pos          → rawvideo_pyarrow
observation/ee_pose            → rawvideo_pyarrow
observation/gripper_fraction   → rawvideo_pyarrow
action/joint_target            → rawvideo_pyarrow
action/gripper_command         → rawvideo_pyarrow
action/native                  → rawvideo_pyarrow
language_instruction           → rawvideo_pyarrow (string)
reward                         → rawvideo_pyarrow
timestep                       → (timestamp axis)
```

---

## `RoboDMCollector` design

`capx/data/robodm_collector.py` (new file) wraps `robodm.Trajectory` and
hides the async-writer plumbing behind a small lifecycle API the
trial loop calls. It is **not** a `Protocol` — it's a single concrete class.
The reference RIO `Recorder` is the template (queue + drainer thread,
per-traj `_stats.json`, atomic save). We don't pull in `rio_hw` as a
dependency; we mirror just the pattern.

```python
# capx/data/robodm_collector.py  (new)
from __future__ import annotations
import json, os, queue, threading, time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import robodm

from capx.data.schema import Step, step_to_dict


class RoboDMCollector:
    """Per-worker collector that writes one `.vla` file per successful demo
    into a per-task directory.  Drains writes off the simulator thread via
    an internal queue + background writer thread (mirrors the reference
    Recorder pattern). One folder per task_id, one file per demonstration.

        demos/<task_id>/traj_<NNNN>.vla
        demos/<task_id>/traj_<NNNN>_stats.json
    """

    def __init__(
        self,
        output_root: Path,                       # e.g. demos/
        *,
        worker_id: int = 0,
        video_codec: str = "libx264",
        codec_options: dict[str, Any] | None = None,
        raw_codec: str = "rawvideo_pyarrow",
        keep_failures: bool = False,
        log_system_stats: bool = False,
        queue_size: int = 1000,                   # ~50 s @ 20 Hz
    ): ...

    # ── Lifecycle ───────────────────────────────────────────────────────
    def start_episode(self, *, task_id: str, language_instruction: str,
                      seed: int, env_metadata: dict[str, Any]) -> str:
        """Atomically reserve the next traj index in demos/<task_id>/,
        open a fresh robodm.Trajectory, spin up a writer thread, return
        an opaque handle."""

    def append_step(self, handle: str, step: Step) -> None:
        """Non-blocking enqueue of `step_to_dict(step)` for the background
        writer.  Drops the step (with a warning) if the queue is full —
        better to lose one frame than stall the simulator."""

    def finish_episode(self, handle: str, *, success: bool,
                       terminal_reward: float, info: dict[str, Any]) -> bool:
        """Drain the queue, close the robodm Trajectory, write the
        `_stats.json` sidecar. If `success=False` and `keep_failures=False`,
        delete both files. Returns True iff the .vla was kept on disk."""

    def committed_count(self, task_id: str) -> int:
        """Returns len(list(demos/<task_id>/traj_*.vla)) — used by the
        curriculum loop's quota check."""

    def close(self) -> None:
        """Idempotent: any in-flight episode is finished with success=False
        and dropped. Worker shutdown calls this in a try/finally."""
```

### Atomic per-task index reservation (parallel-worker safe)

The user's directive — *folder per task, file per demonstration,
`traj_NNNN.vla`* — must work under parallel workers writing to the same
task directory. The reference recorder picks `max(existing_indices)` at
init time only, which would let two simultaneously-launching workers both
grab index 0.

We side-step it by reserving an empty placeholder atomically (`O_CREAT |
O_EXCL`) *before* handing the path to `robodm.Trajectory`:

```python
def _reserve_next_traj_path(task_dir: Path) -> Path:
    task_dir.mkdir(parents=True, exist_ok=True)
    while True:
        used = {int(p.stem.removeprefix("traj_"))
                for p in task_dir.glob("traj_*.vla")
                if p.stem.removeprefix("traj_").isdigit()}
        next_idx = (max(used) + 1) if used else 0
        candidate = task_dir / f"traj_{next_idx:04d}.vla"
        try:
            fd = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.close(fd)
            return candidate            # we own this index
        except FileExistsError:
            continue                    # another worker took it; retry
```

robodm overwrites the empty placeholder on first write. No worker-subdir
scaffolding, no post-collection flatten step — the layout the user wants
is the layout on disk while collection is running.

### Async writer (real-time-friendly)

Mirroring the reference Recorder's `pubreq` loop:

```python
def _writer_loop(self, handle: str, traj: robodm.Trajectory, q: queue.Queue):
    while True:
        item = q.get()
        if item is None:                # sentinel → close
            traj.close()
            return
        flat_dict, ts = item
        try:
            traj.add_by_dict(flat_dict, timestamp=ts, time_unit="s")
        except Exception as e:
            logger.error(f"robodm write failed @ ts={ts}: {e}")
        finally:
            q.task_done()
```

Producer side (`append_step`) simply does `q.put_nowait((flat_dict,
step.timestep))`. If the queue is full (default 1000 entries ≈ 50 s @ 20 Hz),
we log and drop — better than stalling the simulator. Sim-only runs will
never see this in practice; the bound matters for real-robot collection.

### `_stats.json` sidecar

We write a JSON sibling per trajectory (file naming matches the reference:
`traj_NNNN.vla` ↔ `traj_NNNN_stats.json`). Schema:

```json
{
  "file_name":           "demos/franka_robosuite_cube_stack/traj_0042.vla",
  "task_id":             "franka_robosuite_cube_stack",
  "language_instruction":"Stack the red cube on top of the green cube.",
  "seed":                42,
  "success":             true,
  "terminal_reward":     1.0,
  "num_steps":           187,
  "start_time":          1716745201.123,
  "end_time":            1716745214.892,
  "total_time":          13.769,
  "env_metadata": {
    "robot":           "franka_panda",
    "sim_backend":     "robosuite",
    "control_freq_hz": 20,
    "subsample_rate":  5,
    "action_space":    { "shape": [8], "layout": "joint_targets_plus_gripper_frac" },
    "cameras":         { "main": {...}, "wrist": {...} },
    "image_size":      [512, 512]
  },
  "agent_info": {
    "llm_model":         "google/gemini-3.1-pro-preview",
    "used_oracle_code":  false,
    "sandbox_rc":        0,
    "num_code_blocks":   3,
    "num_regenerations": 1,
    "config_path":       "env_configs/cube_stack/franka_robosuite_cube_stack_collect.yaml"
  },
  "system_stats": {           // optional, only if log_system_stats=True
    "avg_cpu_usage":      42.1,
    "avg_ram_mem":        38.7,
    "avg_gpus_usage":     [71.4],
    "avg_gpus_mem":       [3214000000]
  }
}
```

The `env_metadata` and `agent_info` blocks are CaP-X-specific extensions on
top of the reference recorder's bookkeeping fields.

---

## Where the collector plugs in

### 1. Step-level recorder hook on the low-level env

**Caveat from the code audit:** `robosuite_env.step(...)` is called from
multiple sites — not just `RobosuiteBaseEnv._do_robosuite_step`. Subclasses
override the pattern and call `robosuite_env.step(...)` directly:

| File | Direct `robosuite_env.step` call sites |
| --- | --- |
| `robosuite_base.py` | 1 (inside `_do_robosuite_step`) |
| `robosuite_nut_assembly.py` | 4 (lines 197, 199, 241, 243) |
| `robosuite_two_arm_lift.py` | 4 (lines 240, 281, 331, 370) |
| `robosuite_handover.py` | 8 (across `move_to_joints_blocking{,_arm1,_both}`, `_step_once`) |
| `libero.py` | calls its own LIBERO env step internally |

Hooking only at `_do_robosuite_step` would silently drop steps from
nut_assembly, two_arm_lift, and handover. So the design uses a **small
refactor to a single post-step helper** rather than relying on a single
existing chokepoint:

```python
# in capx/envs/simulators/robosuite_base.py  (mirrored in libero.py)

class RobosuiteBaseEnv(BaseEnv):
    ...
    def attach_step_recorder(
        self,
        callback: Callable[[Step], None] | None,
        *,
        every_n_sim_steps: int | None = None,   # default = self._subsample_rate
    ) -> None:
        """Register a callback invoked once per simulator step (post-step).

        The callback receives a `capx.data.schema.Step` already populated
        with the action *target*, the latest proprio, and the most-recently
        rendered frames (main + wrist if enabled).
        """
        self._step_recorder_cb = callback
        self._recorder_stride = every_n_sim_steps or self._subsample_rate

    def _post_step(self, action: np.ndarray) -> None:
        """Centralised post-step bookkeeping. Replaces ad-hoc `_record_frame`
        + viser-update + step-counter calls in subclasses."""
        if self._record_frames and self._sim_step_count % self._subsample_rate == 0:
            self._record_frame()
        if (
            self._step_recorder_cb is not None
            and self._sim_step_count % self._recorder_stride == 0
        ):
            self._emit_step_record(action)
        if hasattr(self, "viser_server") and self._sim_step_count % self._subsample_rate == 0:
            self._update_viser_server()
        self._sim_step_count += 1
```

The refactor itself is mechanical:

- In `_do_robosuite_step` / `_step_once` / `move_to_joints_*` (across
  `robosuite_base.py`, `robosuite_nut_assembly.py`, `robosuite_two_arm_lift.py`,
  `robosuite_handover.py`, `libero.py`), replace the inline pattern
  ```python
  if self._record_frames and ...: self._record_frame()
  if hasattr(self, "viser_server") and ...: self._update_viser_server()
  self._sim_step_count += 1
  ```
  with a single `self._post_step(action)` call after each `robosuite_env.step`.
- Behaviour is identical when no recorder is attached. Verification step #7
  (perf regression) confirms.

`_emit_step_record(action)` builds a `Step` (the dataclass from
`capx.data.schema`) by reading what the env already computes —
`_current_joints`, `_gripper_fraction`, `gripper_link_wxyz_xyz` — and
grabs the most recent main/wrist frames from the existing `_frame_buffer`
/ `_wrist_frame_buffer`. **No extra rendering**: we reuse the frames that
`_record_frame` is already producing on the same subsample cadence.

Per-simulator adapter classes provide `_emit_step_record` and
`get_dataset_metadata`:

- `RobosuiteBaseEnv` — single-arm, action shape `(8,)`.
- `RobosuiteTwoArmLiftEnv`, `RobosuiteHandoverEnv` — two-arm, action shape
  `(14,)`, different `_ACTION_SLICE`.
- `FrankaRobosuiteNutAssembly` — same base, but its overridden movement
  routines must be touched by the same refactor.
- `FrankaLiberoEnv` (`libero.py`) — wraps LIBERO's `OffScreenRenderEnv`. Hook
  applies identically, but the call-site chain is different — see §1b.
- `R1ProBehaviourLowLevel` (`r1pro_b1k.py`) — holonomic-base + bimanual;
  defer to milestone 2.
- `FrankaRealLowLevel` (`franka_real.py`) — hook signature compatible;
  wiring deferred to the real-robot milestone.

### 1b. LIBERO hook — explicit walkthrough

LIBERO is registered in `capx/envs/simulators/__init__.py` via
`franka_libero_{suite}_{task_id}_low_level` (130+ entries spanning
`libero_10`, `libero_90`, `libero_object`, `libero_spatial`, `libero_goal`).
The implementation in `capx/envs/simulators/libero.py:FrankaLiberoEnv` is
*independent* of `RobosuiteBaseEnv` — it does **not** inherit from it,
inherits from `BaseEnv` directly, and routes actions through its own
`self.handle.step(action)` rather than `robosuite_env.step(...)`.

Key differences that shape the recorder adapter:

| Aspect | RoboSuite (`RobosuiteBaseEnv`) | LIBERO (`FrankaLiberoEnv`) |
| --- | --- | --- |
| Underlying sim | RoboSuite MuJoCo | LIBERO's `OffScreenRenderEnv` (still MuJoCo) |
| Step entry | `self.robosuite_env.step(sliced)` | `self.handle.step(action)` (libero.py L245, L283) |
| Native action layout | abs joint target + gripper, `(8,)` | **joint deltas × control_freq + gripper, `(8,)`** — *not* absolute |
| Gripper convention | `action[-2:] = 1 − fraction×2`, two channels | `action[-1] = 1 − fraction×2`, one channel |
| Control freq | sim default (~50 Hz post-controller) | `control_freq=20` (configurable in `__init__`) |
| Cameras | `robot0_robotview` + `robot0_eye_in_hand` | `agentview` + `robot0_eye_in_hand` |
| Move primitives | `move_to_joints_{blocking,non_blocking}`, `_step_once` | only `move_to_joints_blocking`, `_step_once` |
| Frame buffer | `_frame_buffer`, rate 5 | `_frame_buffer`, **rate 4** |
| Language label | from YAML / `obs["full_prompt"]` | **`self.handle.task_language`** (per-task string returned in `info["task_prompt"]` at reset) |
| Init randomisation | `UniformRandomSampler` driven by env seed | per-task **discrete init-state list**; `state_idx = (seed-1) % len(init_states)` (libero.py L160–167) |
| Success check | `robosuite_env._check_success()` | `self.handle.env.check_success()` |

The refactor is the same shape — introduce `FrankaLiberoEnv._post_step(action)`
and route both call sites (`_step_once` and `move_to_joints_blocking`) through
it. There are exactly two `self.handle.step(action)` invocations in
`libero.py` (L245, L283), so the diff is small:

```python
# libero.py — current (move_to_joints_blocking, L244–263):
self._current_obs, self._current_reward, self._current_done, self._current_info = (
    self.handle.step(action)
)
self._sim_step_count += 1
self.gripper_link_wxyz_xyz = np.concatenate([...])
if self.viser_debug and ...: ...
if self._record_frames and ...: self._record_frame()

# after refactor:
self._current_obs, self._current_reward, self._current_done, self._current_info = (
    self.handle.step(action)
)
self.gripper_link_wxyz_xyz = np.concatenate([...])
self._post_step(action)   # increments _sim_step_count, viser, _record_frame,
                          # and fires _step_recorder_cb (when collector is attached)
```

`FrankaLiberoEnv._emit_step_record(action)` constructs a `Step` whose
`Action` carries both the **normalised** target (absolute joints + gripper
fraction — same shape RoboSuite reports) and the **native** LIBERO action
(`[Δj1..Δj7 × control_freq, grip_cmd]`) so VLA recipes that expect either
format can train from the same dataset.

`FrankaLiberoEnv.get_dataset_metadata()` returns the env_metadata dict
that goes into `_stats.json`:

```python
{
    "robot":           "franka_panda",
    "sim_backend":     "libero",
    "suite":           self.handle.suite_name,
    "task_id":         self.handle.task_id,
    "task_language":   self.handle.task_language,
    "control_freq_hz": self._control_freq,         # 20
    "subsample_rate":  self._subsample_rate,       # 4
    "action_space": {
        "native":     {"shape": (8,), "layout": "joint_deltas_scaled_plus_gripper"},
        "normalised": {"shape": (8,), "layout": "joint_targets_plus_gripper_frac"},
    },
    "cameras": {
        "main":  {"name": "agentview",          "intrinsics": K_main,  "extrinsics": ...},
        "wrist": {"name": "robot0_eye_in_hand", "intrinsics": K_wrist, "extrinsics": ...},
    },
    "image_size":      (self._render_height, self._render_width),
    "max_steps":       self.max_steps,
    "init_state_idx":  (seed - 1) % len(self.handle.init_states),
}
```

The LIBERO batch runner `capx/envs/scripts/run_libero_batch.py` already
enumerates `(suite, task_id, task_name, num_trials)` tuples from
`libero.benchmark.get_benchmark_dict()`. The curriculum module reuses that
enumeration to build a `CurriculumTask` per LIBERO task with zero
hand-typed YAML.

### 2. Trial-level lifecycle in `_run_single_trial`

Three edits in `capx/envs/trial.py`:

```python
def _run_single_trial(env, trial, args, config, multi_turn_prompt, ...):
    obs, _ = env.reset(options={"trial": trial}, seed=trial)
    ...
    if config["record_video"] and hasattr(env, "enable_video_capture"):
        env.enable_video_capture(True, clear=True, wrist_camera=use_wrist)

+   # ── DATA COLLECTION HOOK ─────────────────────────────────────────
+   collector: RoboDMCollector | None = config.get("_collector")
+   episode_handle = None
+   if collector is not None:
+       language = (config.get("language_instruction")
+                   or env.get_language_instruction()    # libero: handle.task_language
+                   or obs["full_prompt_text_fallback"])
+       episode_handle = collector.start_episode(
+           task_id              = config["task_id"],
+           language_instruction = language,
+           seed                 = trial,
+           env_metadata         = env.get_dataset_metadata(),
+       )
+       env.attach_step_recorder(
+           lambda step: collector.append_step(episode_handle, step)
+       )
    ...
    # (existing 5-stage loop runs unchanged: visual feedback → code gen →
    #  step blocks → multi-turn decisions → save artifacts)
    ...
+   if collector is not None and episode_handle is not None:
+       env.attach_step_recorder(None)
+       kept = collector.finish_episode(
+           episode_handle,
+           success         = info_step.get("task_completed", False),
+           terminal_reward = reward,
+           info = {
+               "sandbox_rc":         info_step["sandbox_rc"],
+               "num_code_blocks":    num_code_blocks,
+               "num_regenerations":  num_regenerations,
+               "num_finishes":       num_finishes,
+               "llm_model":          args.model,
+               "used_oracle_code":   config["use_oracle_code"],
+               "config_path":        args.config_path,
+           },
+       )
+       print(f"[robodm] trial {trial}: "
+             f"{'kept' if kept else 'dropped'} "
+             f"(success={info_step.get('task_completed', False)})")
```

The existing trial save path (`_save_trial_artifacts`, `_save_trial_video`)
is **untouched**. Eval debug artifacts continue to land under `outputs/...`;
`.vla` demos go to `demos/...`. Turning data collection on/off is risk-free
for downstream eval consumers.

### 3. Worker lifecycle in `_run_headless_trials`

`capx/envs/runner.py` constructs one `RoboDMCollector` per worker process
(via `run_parallel_with_setup`'s `setup_fn`) and tears it down on exit. The
collector is injected into the per-worker `config` dict so
`_run_single_trial` picks it up.

A worker's collector is constructed with `output_root=demos/` (not the
per-task directory). `start_episode(task_id=...)` is what computes
`demos/<task_id>/` lazily, calling `mkdir(parents=True, exist_ok=True)`
and reserving the next `traj_NNNN.vla` index atomically. This lets a
single worker write into multiple task directories across the lifetime of
one `launch.main` invocation if needed (the curriculum loop does this when
several small tasks share a launch).

### 4. Curriculum-driven outer loop

New module `capx/data/curriculum.py`:

```python
@dataclass
class CurriculumTask:
    config_path:          str                  # path into env_configs/
    task_id:              str                  # stable id, becomes the demos/ subdir name
    language_instruction: str                  # constant per episode
    quota_successes:      int                  # commit until this many `.vla` files exist
    max_trials:           int                  # safety cap
    prefer_oracle:        bool = True          # use *_privileged.yaml variant when available
    seed_offset:          int  = 0             # avoid seed re-use across tasks

@dataclass
class CurriculumSpec:
    tasks:                  list[CurriculumTask]
    output_root:            Path               # demos/
    num_workers:            int  = 8
    seed_base:              int  = 0
    batch_size_per_attempt: int  = 16
    keep_failures:          bool = False
    log_system_stats:       bool = False
    video_codec:            str  = "libx264"
    codec_options:          dict[str, Any] = field(default_factory=dict)
```

Counting committed episodes is now a filesystem query
(`len(list((output_root / task_id).glob("traj_*.vla")))`), which is the
same number the user sees when they `ls` the directory. Stops when the
quota is reached or `max_trials` is exhausted — whichever comes first.

### 5. New YAML key (loaded by `_load_config`)

```yaml
# env_configs/cube_stack/franka_robosuite_cube_stack_collect.yaml
env:
  _target_: capx.envs.tasks.franka.franka_pick_place.FrankaPickPlaceCodeEnv
  cfg:
    low_level: franka_robosuite_cubes_low_level
    apis: [FrankaControlApi]

collect:
  output_root:          ./demos
  task_id:              franka_robosuite_cube_stack
  language_instruction: "Stack the red cube on top of the green cube."
  target_successes:     100
  max_trials:           300
  keep_failures:        false
  video_codec:          libx264
  codec_options:        {crf: 23, preset: veryfast}
  log_system_stats:     false

trials:      300            # safety upper bound, overridden by max_trials
num_workers: 12
```

### 6. New `LaunchArgs` flags

Minimal additions to `capx/envs/launch.py:LaunchArgs`:

```python
collect_demos:     bool        = False        # turn on the RoboDMCollector path
demo_output_root:  str | None  = None         # overrides YAML
target_successes:  int | None  = None         # overrides YAML
keep_failures:     bool | None = None         # overrides YAML
```

When `--collect-demos` is set, `launch.main` defers to
`capx.data.curriculum.collect_curriculum` instead of `_run_headless_trials`
directly.

---

## End-to-end data flow

```
collect_curriculum(spec)                          [capx/data/curriculum.py]
└── for task in spec.tasks:
    └── launch.main(LaunchArgs(...))              [capx/envs/launch.py]
        ├── _start_api_servers()                  [SAM3 / GraspNet / PyRoKi]
        └── _run_headless_trials(...)             [capx/envs/runner.py]
            └── run_parallel_with_setup(worker_fn=
                  setup: collector = RoboDMCollector(output_root, worker_id=wid)
                  per-trial:
                    _run_single_trial(env, trial, ...)   [capx/envs/trial.py]
                    │   obs, _ = env.reset(seed=trial)
                    │   handle = collector.start_episode(task_id, lang, seed, env.get_dataset_metadata())
                    │       └── atomically reserve demos/<task_id>/traj_<NNNN>.vla
                    │       └── open robodm.Trajectory(path, mode="w", video_codec="libx264", ...)
                    │       └── spawn background writer thread; create queue
                    │   env.attach_step_recorder(cb=lambda step: collector.append_step(handle, step))
                    │   ┌──────────────────────────────────────────────────────────┐
                    │   │ Existing code-as-policy turn loop:                        │
                    │   │   LLM → code block → env.step(code)                       │
                    │   │     └── APIs (`goto_pose`, `close_gripper`, …)            │
                    │   │           └── env.move_to_joints_blocking(...)            │
                    │   │                 └── _do_robosuite_step(action)            │
                    │   │                       ├── robosuite.step / handle.step    │
                    │   │                       └── env._post_step(action)          │
                    │   │                             ├── _record_frame (main+wrist)│
                    │   │                             └── _step_recorder_cb(step)   │
                    │   │                                   └── collector.append_step│
                    │   │                                         └── queue.put_nowait
                    │   │                                                ↓          │
                    │   │       [background writer thread]   ── traj.add_by_dict ── │
                    │   │       (libx264 encodes RGB, Arrow                          │
                    │   │        writes proprio/action streams)                      │
                    │   └──────────────────────────────────────────────────────────┘
                    │   env.attach_step_recorder(None)
                    │   kept = collector.finish_episode(
                    │             handle, success=task_completed,
                    │             terminal_reward=reward, info={...})
                    │       └── drain queue (sentinel None)
                    │       └── traj.close()
                    │       └── write demos/<task_id>/traj_<NNNN>_stats.json
                    │       └── if (not success and not keep_failures): unlink both files
                    │   (existing) _save_trial_artifacts(...)  ← unchanged
                    └── cleanup: collector.close() per worker
        └── return; orchestrator counts demos/<task_id>/traj_*.vla
            until quota met or max_trials exhausted
└── write demos/curriculum_summary.json
    { task_id: { attempted, succeeded, kept_vla_count,
                 mean_episode_len, total_disk_bytes, ... } }
```

---

## Pseudo-code: the collection loop

Three nested loops. From outermost (curriculum) to innermost (per-step
recording). Written to make integration points explicit; not the final
code.

### A. Outer: `collect_curriculum(spec)` — per-task quota loop

```
function collect_curriculum(spec: CurriculumSpec) -> CurriculumSummary:
    out = spec.output_root
    mkdir(out)
    summary = CurriculumSummary()

    for task in spec.tasks:
        task_dir = out / task.task_id
        seed     = spec.seed_base + task.seed_offset
        attempts = 0

        function vla_count() -> int:
            return len(list(task_dir.glob("traj_*.vla")))

        while vla_count() < task.quota_successes and attempts < task.max_trials:

            remaining = task.max_trials - attempts
            needed    = task.quota_successes - vla_count()
            batch     = min(spec.batch_size_per_attempt,
                            remaining,
                            max(needed * 2, 1))      # over-provision 2× so a
                                                     # bad success rate still
                                                     # makes progress per round

            # Build LaunchArgs as if the user had typed the CLI by hand.
            args = LaunchArgs(
                config_path     = task.config_path,
                total_trials    = batch,
                num_workers     = spec.num_workers,
                output_dir      = str(out / "_eval_artifacts" / task.task_id
                                          / f"attempt_{attempts:04d}"),
                record_video    = false,             # debug MP4 not needed
                use_oracle_code = task.prefer_oracle and has_oracle(task.config_path),
                collect_demos   = true,              # NEW flag
            )
            args.demo_output_root  = str(spec.output_root)
            args.target_successes  = task.quota_successes
            args.keep_failures     = spec.keep_failures
            args.seed_base         = seed

            # Per-worker RoboDMCollector is instantiated inside
            # _run_headless_trials' setup_fn (see §3), pointed at
            # spec.output_root with the task_id provided by the YAML.
            launch.main(args)

            attempts += batch
            seed     += batch

            summary.note_attempt(task.task_id, attempted=batch,
                                 kept_so_far=vla_count())

        summary.note_task_done(task.task_id,
                               kept=vla_count(),
                               attempted=attempts)

    write_json(out / "curriculum_summary.json", summary.as_dict())
    return summary
```

Key properties:

- **Quota, not trial-count.** Stops when *kept `.vla` files* reach
  `quota_successes`, not when a fixed `trials` count is exhausted.
- **Bounded budget.** `max_trials` caps the work; partial results land in
  the summary so the user can see where coverage fell short.
- **Re-uses everything.** `launch.main` is called as-is. API servers,
  parallel workers, multi-turn LLM, ensembles, skill library — all
  unchanged.
- **Filesystem is the ground truth.** No in-memory accounting — counting
  `traj_*.vla` files matches what the user sees on disk and what
  downstream loaders will iterate.
- **Seed monotonicity.** `seed` strictly increases across attempts, so
  RoboSuite (`UniformRandomSampler`) and LIBERO
  (`init_states[(seed-1) % N]`) both see new initial conditions every
  trial.

### B. Middle: per-worker trial loop (lives inside `_run_single_trial`)

Collection-relevant lines only — existing LLM / multi-turn /
save-artifacts logic is elided.

```
function _run_single_trial(env, trial, args, config, multi_turn_prompt):

    # ── Existing reset ──────────────────────────────────────────────
    obs, _ = env.reset(options={"trial": trial}, seed=trial)

    # ── NEW: open a demo episode (RoboDMCollector hands back a handle)
    collector: RoboDMCollector | None = config.get("_collector")
    episode_handle = None
    if collector is not None:
        language = (config.get("language_instruction")
                    or env.get_language_instruction()
                    or obs["full_prompt_text_fallback"])
        episode_handle = collector.start_episode(
            task_id              = config["task_id"],
            language_instruction = language,
            seed                 = trial,
            env_metadata         = env.get_dataset_metadata(),
        )
        env.attach_step_recorder(
            lambda step: collector.append_step(episode_handle, step)
        )

    # ── Existing turn loop (LLM → code → env.step → multi-turn) ─────
    try:
        ... existing code unchanged ...
    finally:
        # ── NEW: close the demo episode regardless of how we exited ──
        if collector is not None and episode_handle is not None:
            env.attach_step_recorder(None)
            kept = collector.finish_episode(
                episode_handle,
                success         = info_step.get("task_completed", False),
                terminal_reward = reward,
                info = {
                    "sandbox_rc":        info_step["sandbox_rc"],
                    "num_code_blocks":   num_code_blocks,
                    "num_regenerations": num_regenerations,
                    "num_finishes":      num_finishes,
                    "llm_model":         args.model,
                    "used_oracle_code":  config["use_oracle_code"],
                    "config_path":       args.config_path,
                },
            )
            log(f"[robodm] trial {trial}: {'kept' if kept else 'dropped'} "
                f"(success={info_step.get('task_completed', False)})")

    # ── Existing save path (untouched) ──────────────────────────────
    _save_trial_artifacts(...)
    _save_turn_and_combined_videos(...)
    return TrialSummary(...)
```

The `finally` block guarantees `finish_episode` runs even if the LLM call
raises, the timeout fires, or the code-as-policy explodes mid-step. The
collector handles the success-vs-drop decision: on failure with
`keep_failures=False`, the just-finalized `.vla` and `_stats.json` files
are unlinked so the directory only contains successful demos.

### C. Inner: per-sim-step recorder (lives inside the env)

Smallest loop; runs at MuJoCo step cadence. Identical shape for
`RobosuiteBaseEnv` and `FrankaLiberoEnv`; only the action layout and
proprio source differ.

```
# Generic shape (each env subclass provides _emit_step_record):

function _post_step(env, action):
    # 1. Existing video frame capture, on subsample cadence.
    if env._record_frames and env._sim_step_count % env._subsample_rate == 0:
        env._record_frame()

    # 2. NEW: data-collection callback, on (possibly different) cadence.
    if env._step_recorder_cb is not None \
       and env._sim_step_count % env._recorder_stride == 0:
        env._emit_step_record(action)        # builds Step, fires cb

    # 3. Existing viser update.
    if has_viser(env) and env._sim_step_count % env._subsample_rate == 0:
        env._update_viser_server()

    env._sim_step_count += 1

# RoboSuite-specific Step builder (single-arm; 2-arm subclasses override):

function RobosuiteBaseEnv._emit_step_record(env, action):
    step = Step(
        timestep = env._sim_step_count / env._control_freq_hz(),
        observation = Observation(
            rgb_main         = env._frame_buffer[-1].copy(),
            rgb_wrist        = (env._wrist_frame_buffer[-1].copy()
                                if env._record_wrist_camera else None),
            joint_pos        = env._current_joints.copy(),
            ee_pose          = env.gripper_link_wxyz_xyz.copy(),    # (7,) wxyz+xyz
            gripper_fraction = env._gripper_fraction,
        ),
        action = Action(
            joint_target    = action[:7].copy(),
            gripper_command = env._gripper_fraction,
            native          = action.copy(),                        # same as joint_target+grip for robosuite
        ),
        language_instruction = env._current_language_instruction,   # set by collector at start_episode
        reward = None,
    )
    env._step_recorder_cb(step)

# LIBERO-specific Step builder (delta-scaled native action; same shape):

function FrankaLiberoEnv._emit_step_record(env, action):
    # `action` here is the native LIBERO action: [Δj1..Δj7 × control_freq, grip_cmd].
    step = Step(
        timestep = env._sim_step_count / env._control_freq,         # 20 Hz
        observation = Observation(
            rgb_main         = env._frame_buffer[-1].copy(),        # agentview
            rgb_wrist        = (env._wrist_frame_buffer[-1].copy()
                                if env._record_wrist_camera else None),
            joint_pos        = qpos_read(env.handle.env.sim,
                                         env._panda_joint_qpos_addrs),
            ee_pose          = env.gripper_link_wxyz_xyz.copy(),
            gripper_fraction = env._gripper_fraction,
        ),
        action = Action(
            joint_target    = env._current_joints.copy(),           # what move_to_joints_blocking is targeting
            gripper_command = env._gripper_fraction,
            native          = action.copy(),                        # raw env action
        ),
        language_instruction = env._current_language_instruction,
        reward = env._current_reward,
    )
    env._step_recorder_cb(step)
```

The recorder does **no extra rendering**: it reuses the most recent frame
that `_record_frame` already pushed to `_frame_buffer`. That's why
`attach_step_recorder` defaults `every_n_sim_steps` to the env's
`_subsample_rate` — the two cadences match, so `_frame_buffer[-1]` is
always the frame for *this* step.

`env._current_language_instruction` is a single-string attribute the
collector sets on the env at `start_episode` time, so the env doesn't need
to know about the collector beyond storing the string. The string ends up
copied into every `Step` and is recorded as a per-step feature in the
`.vla` file (RoboDM dedup it internally; the storage overhead is
negligible).

---

## File-level change list

| File | Change | Size |
| --- | --- | --- |
| `capx/data/__init__.py` | new package | small |
| `capx/data/schema.py` | `Observation`, `Action`, `Step` dataclasses + `step_to_dict` (port of the reference helper) | ~80 LOC |
| `capx/data/robodm_collector.py` | `RoboDMCollector` (atomic index reservation, async writer thread, `_stats.json` sidecar) | ~250 LOC |
| `capx/data/curriculum.py` | `CurriculumSpec`, `CurriculumTask`, `collect_curriculum` | ~200 LOC |
| `capx/envs/simulators/robosuite_base.py` | `attach_step_recorder`, `_emit_step_record`, new `_post_step(action)` helper, `get_dataset_metadata`, `get_language_instruction`; hoist `_record_frame` + viser-update + step-count into `_post_step` | +90 LOC |
| `capx/envs/simulators/robosuite_nut_assembly.py` | Replace 4 ad-hoc post-step blocks with `_post_step(action)` | +/- 20 LOC |
| `capx/envs/simulators/robosuite_two_arm_lift.py` | Replace 4 post-step blocks with `_post_step(action)`; two-arm `_emit_step_record` (14-D action) | +40 LOC |
| `capx/envs/simulators/robosuite_handover.py` | Replace 8 post-step blocks with `_post_step(action)`; bimanual `_emit_step_record` | +40 LOC |
| `capx/envs/simulators/libero.py` | mirror of base; LIBERO-specific `_emit_step_record` and `get_dataset_metadata`; `get_language_instruction()` returns `self.handle.task_language` | +70 LOC |
| `capx/envs/trial.py` | insert `start_episode` / `attach_step_recorder` / `finish_episode` around existing trial loop (≈25 lines, no rewrites) | +30 LOC |
| `capx/envs/runner.py` | per-worker `RoboDMCollector` setup_fn / teardown | +25 LOC |
| `capx/envs/launch.py` | new `LaunchArgs` flags, dispatch to `collect_curriculum` when `--collect-demos` is set | +20 LOC |
| `capx/utils/launch_utils.py:_load_config` | parse new `collect:` YAML block | +20 LOC |
| `env_configs/**/*_collect.yaml` | one collection variant per RoboSuite family (8 files) + a LIBERO template (`franka_libero_collect.yaml.j2`) auto-expanded by the curriculum builder | small |
| `pyproject.toml` | add `robodm` dependency under a new `[project.optional-dependencies] vla-collect` extra (kept optional so the eval-only install path doesn't pull libx264 deps) | 1 line |

The eval-path code (`_save_trial_artifacts`, `_save_trial_video`,
`_save_turn_and_combined_videos`, all multi-turn / ensemble logic) is
**not modified** — milestone 1 is strictly additive.

---

## Verification plan

End-to-end checks, smallest-first:

1. **Smoke: one trial, one `.vla`.** Run one trial of
   `franka_robosuite_cube_stack` with `--collect-demos` and a quota of 1.
   Assert:
   - `demos/franka_robosuite_cube_stack/traj_0000.vla` exists and is
     non-empty.
   - `demos/franka_robosuite_cube_stack/traj_0000_stats.json` exists,
     `success == true`, `num_steps > 0`, `language_instruction` matches the
     YAML.
   - existing eval artifacts (`outputs/.../trial_*/code.py`, etc.) still
     appear unchanged.

2. **Roundtrip via robodm.** Open the produced `.vla` file in read mode
   (`robodm.Trajectory(path, mode="r")`), iterate every record. Assert:
   - the timestamp stream is monotonic and starts near 0.
   - `observation/rgb_main` decodes to `(H, W, 3) uint8` with the expected
     resolution from `env_metadata`.
   - `action/joint_target.shape == (7,)`, `action/gripper_command` is a
     scalar in `[0, 1]`.
   - `language_instruction` is the same string on every step.

3. **Action-shape invariants (RoboSuite).** For cube_stack, cube_lift,
   nut_assembly, spill_wipe, two_arm_lift, two_arm_handover: run 1 trial
   each with `quota_successes=1`. For each produced `.vla`, assert the
   action shape matches `env_metadata.action_space.shape`. Catches per-env
   adapter bugs (in particular the two-arm 14-D switch).

4. **LIBERO parity.** Repeat #3 for one task per LIBERO suite
   (`libero_10/0`, `libero_object/0`, `libero_spatial/2`, `libero_goal/1`,
   `libero_90/35`). Assert:
   - `action/native.shape == (8,)` and `action/joint_target.shape == (7,)`.
   - `observation/rgb_main` comes from `agentview` (per
     `env_metadata.cameras.main.name`), not `robot0_robotview`.
   - `language_instruction == self.handle.task_language` and is non-empty.
   - Re-running seed=N yields the same first observation (deterministic
     init-state lookup).

5. **Parallel-worker atomicity.** Run 32 trials of `cube_stack` with
   `num_workers=8`, `quota_successes=32`, oracle code (success rate ~100%).
   After completion, assert:
   - `demos/franka_robosuite_cube_stack/` contains exactly 32 `.vla` files
     and 32 `_stats.json` files.
   - The traj indices `0000..0031` are contiguous (no gaps from collisions).
   - All `_stats.json` files load with non-overlapping `start_time` ⇒
     `end_time` intervals *per worker* (sanity-check timing) and unique
     `seed` values.

6. **Failure isolation.** Re-run the same task with a deliberately broken
   LLM endpoint (`server_url` → dead socket) and `keep_failures=false`.
   Assert:
   - every trial raises in the existing `_run_single_trial`;
   - `finish_episode` is still called for partially-stepped episodes
     (the `finally` block);
   - the task directory is **empty** at the end — no partial `.vla` leaks.

7. **Multi-task curriculum dry-run.** `quota_successes=3` on three
   RoboSuite tasks and two LIBERO tasks at once with oracle code where
   available. Assert: each task directory has exactly 3 `.vla` files,
   `curriculum_summary.json` has correct kept-counts, no inter-task
   filename collisions.

8. **VLA loader contract.** Point a downstream LeRobot- or robodm-native
   loader at one of the produced task directories. Verify the
   trajectories load and the action / observation tensor shapes match
   `env_metadata`. This is the contract test for "the .vla files are
   actually usable for finetuning."

9. **Performance regression.** Time 10 trials of cube_stack:
   (a) with `--collect-demos`, libx264, async writer;
   (b) with `--collect-demos` swapped for a no-op recorder (callback that
       discards the Step);
   (c) without `--collect-demos`.
   Expect (a) within ~10% of (b) of (c) — the async writer should hide
   codec cost. If (a) ≫ (b), the queue is back-pressuring; consider
   raising `queue_size` or downgrading the codec for sim runs.

---

## Out of scope for this milestone

- Real-robot collection (`franka_real` backend). The hook lives in
  `RobosuiteBaseEnv`; mirroring it onto `FrankaRealLowLevel` is mechanical
  but needs hardware time. The RoboDMCollector itself is already
  real-time-friendly (async writer), so the data-side change for real
  robot is essentially zero.
- BEHAVIOR / R1Pro holonomic bimanual collection. Different action
  topology; worth its own design pass.
- Finetuning a VLA on the collected data. The `.vla` files hand off to
  whichever loader the chosen VLA stack uses; downstream training is a
  separate workstream.
- Negative / partial-success data, contrastive sampling, DAgger-style
  re-labeling. All compatible with the design (`keep_failures=True`
  retains failed episodes; per-step `reward` and `info` are already
  recorded) but disabled for milestone 1.
- Cross-process / cross-host distributed collection via the RIO
  middleware (`RecorderServer` / `RecorderClient` pattern from the
  reference example). Single-host parallel workers via the per-worker
  in-process `RoboDMCollector` are sufficient for milestone 1.
