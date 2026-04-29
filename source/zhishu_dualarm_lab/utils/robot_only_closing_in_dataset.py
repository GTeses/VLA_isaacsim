from __future__ import annotations

"""LeIsaac-style HDF5 helpers for the clean robot-only closing-in pipeline.

这条链路的原则是：
- 控制逻辑完全留在 `run_robot_only_closing_in_v2.py`
- 这里不参与 IK / reset / 目标采样
- 这里只负责把已经稳定的 teacher rollout 写成统一的 HDF5 结构
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


TASK_NAME = "LeIsaac-Zhishu-RobotOnly-ClosingIn-v0"
ACTION_DIM = 14
STATE_DIM = 70
ROOT_GROUP = "data"
IMAGE_NAMES = ("external_image", "left_wrist_image", "right_wrist_image")
ACTION_NAMES = [*(f"left_joint{i}" for i in range(1, 8)), *(f"right_joint{i}" for i in range(1, 8))]
DEFAULT_PROMPT = "bring both arms in toward a shared tabletop disk from two sides"


@dataclass(frozen=True)
class RobotOnlyClosingInEpisodeSpec:
    """Per-episode metadata for the clean closing-in task."""

    episode_index: int
    prompt: str
    gap_m: float
    left_target: np.ndarray
    right_target: np.ndarray
    center_target: np.ndarray
    disk_center: np.ndarray


def ensure_hdf5_path(path: Path) -> Path:
    return path if path.suffix == ".hdf5" else path.with_suffix(".hdf5")


def create_or_open_dataset(path: Path, *, resume: bool = False) -> tuple[h5py.File, h5py.Group]:
    """Create or append a LeIsaac-style HDF5 file rooted at `/data`."""

    resolved = ensure_hdf5_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume else "w"
    file_handle = h5py.File(resolved, mode)
    data_group = file_handle.require_group(ROOT_GROUP)
    if not resume:
        data_group.attrs["env_name"] = TASK_NAME
        data_group.attrs["type"] = "zhishu_robot_only_closing_in_v0"
        data_group.attrs["total"] = 0
    return file_handle, data_group


def next_episode_index(data_group: h5py.Group) -> int:
    return len(data_group.keys())


def list_episode_names(data_group: h5py.Group) -> list[str]:
    return sorted(data_group.keys(), key=lambda name: int(name.split("_")[-1]))


def parse_policy_state(state: np.ndarray) -> dict[str, np.ndarray]:
    """Parse the fixed 70D policy state contract used by the robot_only env."""

    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape[0] != STATE_DIM:
        raise ValueError(f"Expected a {STATE_DIM}D policy state, got {state.shape}")
    return {
        "joint_pos": state[0:14],
        "joint_vel": state[14:28],
        "last_action": state[28:42],
        "left_tcp_pose": state[42:49],
        "right_tcp_pose": state[49:56],
        "object_pose": state[56:63],
        "target_pose": state[63:70],
    }


def compute_success(state: np.ndarray, spec: RobotOnlyClosingInEpisodeSpec, threshold: float) -> tuple[bool, float, float]:
    """Return success and left/right TCP distances to the current structured targets."""

    parsed = parse_policy_state(state)
    left_tcp = parsed["left_tcp_pose"][:3]
    right_tcp = parsed["right_tcp_pose"][:3]
    left_dist = float(np.linalg.norm(left_tcp - spec.left_target))
    right_dist = float(np.linalg.norm(right_tcp - spec.right_target))
    success = left_dist < threshold and right_dist < threshold
    return success, left_dist, right_dist


def _write_recursive(group: h5py.Group, key: str, value: Any) -> None:
    if isinstance(value, dict):
        subgroup = group.require_group(key)
        for sub_key, sub_value in value.items():
            _write_recursive(subgroup, sub_key, sub_value)
        return
    if isinstance(value, str):
        group.create_dataset(key, data=np.asarray(value, dtype=h5py.string_dtype("utf-8")))
        return
    data = np.asarray(value)
    group.create_dataset(key, data=data, compression="lzf" if data.ndim > 0 else None)


def write_episode(
    data_group: h5py.Group,
    *,
    spec: RobotOnlyClosingInEpisodeSpec,
    payload: dict[str, Any],
    success: bool,
    num_samples: int,
) -> str:
    """Append one episode to the HDF5 dataset and return its group name."""

    episode_name = f"demo_{next_episode_index(data_group)}"
    episode_group = data_group.create_group(episode_name)
    episode_group.attrs["task_name"] = TASK_NAME
    episode_group.attrs["episode_index"] = spec.episode_index
    episode_group.attrs["prompt"] = spec.prompt
    episode_group.attrs["gap_m"] = spec.gap_m
    episode_group.attrs["success"] = bool(success)
    episode_group.attrs["num_samples"] = int(num_samples)
    for key, value in payload.items():
        _write_recursive(episode_group, key, value)
    data_group.attrs["total"] = int(data_group.attrs.get("total", 0)) + int(num_samples)
    return episode_name
