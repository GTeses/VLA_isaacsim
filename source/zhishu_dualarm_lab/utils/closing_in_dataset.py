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
    # Do not inherit the scene's default target-zone lateral bias here.
    # For closing-in data we want targets to cover the tabletop middle band on
    # both sides of the robot so we can distinguish posture failures from
    # "the target just happened to sit on the left" artifacts.
    center_target[1] = float(rng.uniform(-0.22, 0.22))
    center_target[2] += float(rng.uniform(-0.015, 0.035))

    if target_mode == "center":
        left_target = center_target.copy()
        right_target = center_target.copy()
        prompt = str(rng.choice(_CENTER_PROMPTS))
    else:
        lateral_offset = float(rng.uniform(0.08, 0.18))
        left_target = center_target.copy()
        right_target = center_target.copy()
        left_target[1] += lateral_offset
        right_target[1] -= lateral_offset
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
