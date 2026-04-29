from __future__ import annotations

"""Helpers for the first Zhishu closing-in simulation dataset.

This module keeps the HDF5 layout, state slicing, and episode-spec sampling in
one place so the collector, replay script, and converter all agree on the same
schema.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


TASK_NAME = "LeIsaac-Zhishu-ClosingIn-v0"
ACTION_DIM = 14
STATE_DIM = 70
ROOT_GROUP = "data"
IMAGE_NAMES = ("external_image", "left_wrist_image", "right_wrist_image")
ACTION_NAMES = [*(f"left_joint{i}" for i in range(1, 8)), *(f"right_joint{i}" for i in range(1, 8))]

_CENTER_PROMPTS = (
    "bring both arms closer to the center target",
    "move both arms toward the center goal region",
    "close both arms in around the middle target",
)
_SYMMETRIC_PROMPTS = (
    "move both arms toward mirrored targets around the center",
    "bring the left and right arms to symmetric target points",
    "close both arms toward two balanced target points",
)


@dataclass(frozen=True)
class ClosingInEpisodeSpec:
    """Per-episode task spec stored together with the rollout."""

    episode_index: int
    target_mode: str
    prompt: str
    speed_scale: float
    hold_steps: int
    jitter_scale: float
    success_threshold: float
    center_target: np.ndarray
    left_target: np.ndarray
    right_target: np.ndarray


def ensure_hdf5_path(path: Path) -> Path:
    return path if path.suffix == ".hdf5" else path.with_suffix(".hdf5")


def create_or_open_dataset(path: Path, *, resume: bool = False) -> tuple[h5py.File, h5py.Group]:
    """Create a LeIsaac-style HDF5 root with a `/data` group."""

    resolved = ensure_hdf5_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume else "w"
    file_handle = h5py.File(resolved, mode)
    data_group = file_handle.require_group(ROOT_GROUP)
    if not resume:
        data_group.attrs["env_name"] = TASK_NAME
        data_group.attrs["type"] = "zhishu_closing_in_v0"
        data_group.attrs["total"] = 0
    return file_handle, data_group


def next_episode_index(data_group: h5py.Group) -> int:
    return len(data_group.keys())


def list_episode_names(data_group: h5py.Group) -> list[str]:
    return sorted(data_group.keys(), key=lambda name: int(name.split("_")[-1]))


def parse_policy_state(state: np.ndarray) -> dict[str, np.ndarray]:
    """Extract structured terms from the fixed 70D state contract."""

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


def sample_episode_spec(rng: np.random.Generator, episode_index: int, base_target: np.ndarray) -> ClosingInEpisodeSpec:
    """Sample one closing-in task variant with prompt and motion diversity.

    The first successful GUI trials showed that target diversity was too small:
    trajectories stayed semantically valid, but the goal stayed in almost the
    same tabletop patch. For training we want "same task, broader workspace",
    so the center target now covers most of the reachable tabletop middle area
    while still staying away from the far edges.
    """

    target_mode = str(rng.choice(["center", "symmetric"], p=[0.45, 0.55]))
    center_target = np.asarray(base_target, dtype=np.float32).copy()
    center_target[0] += float(rng.uniform(-0.12, 0.12))
    # 中文说明：
    # 这里之前让 center_target 的 y 覆盖了过宽的 [-0.28, 0.28]。
    # 对 symmetric 任务来说，这会导致“中心点本身已经偏左很多”，
    # 即使再减一个 lateral_offset，右臂目标仍可能落在身体左侧。
    # 这样看起来就像“两个手臂都在往左边靠”，但根因其实是任务采样本身。
    #
    # closing-in 的目标不是考察跨身体大范围横摆，而是考察双臂在桌面中部
    # 工作区内向中线/对称点聚拢。因此这里把中心目标约束在更窄的中线带内。
    center_target[1] = float(rng.uniform(-0.08, 0.08))
    center_target[2] += float(rng.uniform(-0.015, 0.035))

    if target_mode == "center":
        left_target = center_target.copy()
        right_target = center_target.copy()
        prompt = str(rng.choice(_CENTER_PROMPTS))
    else:
        # 中文说明：
        # symmetric 模式必须保证左目标在 y>0，右目标在 y<0，
        # 否则右臂会被任务本身要求穿过身体中线去左侧，视觉上就会像“右臂
        # 紧贴身体也要往左边凑”。这里显式保证左右目标分居中线两侧。
        lateral_offset = float(rng.uniform(0.12, 0.20))
        left_target = center_target.copy()
        right_target = center_target.copy()
        left_target[1] += lateral_offset
        right_target[1] -= lateral_offset
        left_target[1] = float(max(left_target[1], 0.10))
        right_target[1] = float(min(right_target[1], -0.10))
        prompt = str(rng.choice(_SYMMETRIC_PROMPTS))

    return ClosingInEpisodeSpec(
        episode_index=episode_index,
        target_mode=target_mode,
        prompt=prompt,
        speed_scale=float(rng.uniform(0.55, 1.10)),
        hold_steps=int(rng.integers(4, 14)),
        # Jitter stays intentionally small. We want diversity from workspace,
        # start pose, and timing first, not from noisy task-space dithering.
        jitter_scale=float(rng.uniform(0.0, 0.012)),
        success_threshold=float(rng.uniform(0.040, 0.065)),
        center_target=center_target.astype(np.float32),
        left_target=left_target.astype(np.float32),
        right_target=right_target.astype(np.float32),
    )


def compute_success(state: np.ndarray, spec: ClosingInEpisodeSpec) -> tuple[bool, float, float]:
    """Return success and the left/right TCP distances to their current targets."""

    parsed = parse_policy_state(state)
    left_tcp = parsed["left_tcp_pose"][:3]
    right_tcp = parsed["right_tcp_pose"][:3]
    left_dist = float(np.linalg.norm(left_tcp - spec.left_target))
    right_dist = float(np.linalg.norm(right_tcp - spec.right_target))
    success = left_dist < spec.success_threshold and right_dist < spec.success_threshold
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
    spec: ClosingInEpisodeSpec,
    payload: dict[str, Any],
    success: bool,
    num_samples: int,
) -> str:
    """Append a new episode to the dataset and return its group name."""

    episode_name = f"demo_{next_episode_index(data_group)}"
    episode_group = data_group.create_group(episode_name)
    episode_group.attrs["task_name"] = TASK_NAME
    episode_group.attrs["episode_index"] = spec.episode_index
    episode_group.attrs["target_mode"] = spec.target_mode
    episode_group.attrs["prompt"] = spec.prompt
    episode_group.attrs["speed_scale"] = spec.speed_scale
    episode_group.attrs["hold_steps"] = spec.hold_steps
    episode_group.attrs["jitter_scale"] = spec.jitter_scale
    episode_group.attrs["success_threshold"] = spec.success_threshold
    episode_group.attrs["success"] = bool(success)
    episode_group.attrs["num_samples"] = int(num_samples)
    for key, value in payload.items():
        _write_recursive(episode_group, key, value)
    data_group.attrs["total"] = int(data_group.attrs.get("total", 0)) + int(num_samples)
    return episode_name
