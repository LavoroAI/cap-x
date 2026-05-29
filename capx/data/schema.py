"""Per-timestep schema for VLA data collection.

A trajectory is a sequence of `Step` records. `step_to_dict` flattens a
Step into the hierarchical-key dict shape that RoboDM consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Observation:
    # camera_name -> modality -> array. Modalities: "rgb" (H,W,3) uint8,
    # "depth" (H,W) float32 metric, "segmentation" (H,W) int32 instance ids.
    images: dict[str, dict[str, np.ndarray]] | None = None
    joint_pos: np.ndarray | None = None        # (7,) float32, radians
    ee_pose: np.ndarray | None = None          # (7,) float32, wxyz_xyz in robot base frame
    gripper_fraction: float | None = None      # 0.0 closed → 1.0 open


@dataclass
class Action:
    joint_target: np.ndarray | None = None     # (7,) float32 absolute joint targets (rad)
    gripper_command: float | None = None       # 0.0–1.0 fraction
    native: np.ndarray | None = None           # env-native action layout


@dataclass
class Step:
    timestep: float
    observation: Observation
    action: Action
    language_instruction: str | None = None
    reward: float | None = None
    info: dict[str, Any] = field(default_factory=dict)


def step_to_dict(obj: Any) -> dict[str, Any]:
    """Flatten a Step (or nested dataclass/dict) into a hierarchical-key dict.

    Nested dataclasses and dicts produce `parent/child` keys. None values are
    dropped so that an episode missing a wrist camera writes no rgb_wrist
    column.
    """
    out: dict[str, Any] = {}
    if hasattr(obj, "__dataclass_fields__"):
        items = obj.__dict__.items()
    elif isinstance(obj, dict):
        items = obj.items()
    else:
        return out

    for k, v in items:
        if v is None:
            continue
        if hasattr(v, "__dataclass_fields__") or isinstance(v, dict):
            for k2, v2 in step_to_dict(v).items():
                out[f"{k}/{k2}"] = v2
        else:
            out[k] = v
    return out
