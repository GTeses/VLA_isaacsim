#!/usr/bin/env python3
"""Convert canonical Zhishu rollout folders into a LeRobot dataset.

Expected raw layout:

raw_dir/
  episode_0000/
    metadata.json                # optional, can contain {"task": "...", "fps": 10}
    prompt.txt                   # optional fallback for task text
    observation_state.npy        # required, shape [T, state_dim]
    action.npy                   # required, shape [T, 14]
    external_images/000000.png   # required, T frames
    left_wrist_images/000000.png # required, T frames
    right_wrist_images/000000.png# required, T frames

This script defines the canonical LeRobot feature names used by the Zhishu
openpi scaffold:

- `observation.images.external`
- `observation.images.left_wrist`
- `observation.images.right_wrist`
- `observation.state`
- `action`
- `task`
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

try:
    from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "LeRobot is required for this script. Install it in the environment that will run the conversion."
    ) from exc


ACTION_NAMES = [*(f"left_joint{i}" for i in range(1, 8)), *(f"right_joint{i}" for i in range(1, 8))]


@dataclasses.dataclass(frozen=True)
class Args:
    raw_dir: Path
    repo_id: str
    fps: int = 10
    use_videos: bool = False
    image_writer_threads: int = 8
    image_writer_processes: int = 4
    push_to_hub: bool = False
    private: bool = False
    license: str = "apache-2.0"
    robot_type: str = "zhishu_dualarm_nohand"
    mode: str = "image"


def _load_metadata(ep_dir: Path) -> dict:
    metadata_path = ep_dir / "metadata.json"
    if metadata_path.is_file():
        return json.loads(metadata_path.read_text())
    return {}


def _load_task(ep_dir: Path, metadata: dict, episode_index: int) -> str:
    if "task" in metadata:
        return str(metadata["task"])
    prompt_path = ep_dir / "prompt.txt"
    if prompt_path.is_file():
        return prompt_path.read_text().strip()
    return f"zhishu episode {episode_index}"


def _sorted_image_files(image_dir: Path) -> list[Path]:
    return sorted([*image_dir.glob("*.png"), *image_dir.glob("*.jpg"), *image_dir.glob("*.jpeg")])


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _infer_state_dim(episode_dirs: list[Path]) -> int:
    state = np.load(episode_dirs[0] / "observation_state.npy")
    if state.ndim != 2:
        raise ValueError(f"Expected observation_state.npy to have shape [T, state_dim], got {state.shape}")
    return int(state.shape[1])


def _infer_image_shape(episode_dirs: list[Path]) -> tuple[int, int, int]:
    sample_dir = episode_dirs[0] / "external_images"
    files = _sorted_image_files(sample_dir)
    if not files:
        raise ValueError(f"No images found under {sample_dir}")
    return tuple(_load_rgb(files[0]).shape)


def _create_dataset(args: Args, state_dim: int, image_shape: tuple[int, int, int]) -> LeRobotDataset:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": [[f"state_{i}" for i in range(state_dim)]],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(ACTION_NAMES),),
            "names": [ACTION_NAMES],
        },
        "observation.images.external": {
            "dtype": args.mode,
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        },
        "observation.images.left_wrist": {
            "dtype": args.mode,
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        },
        "observation.images.right_wrist": {
            "dtype": args.mode,
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        },
    }

    output_dir = LEROBOT_HOME / args.repo_id
    if output_dir.exists():
        shutil.rmtree(output_dir)

    return LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        robot_type=args.robot_type,
        features=features,
        use_videos=args.use_videos,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
    )


def _validate_episode(ep_dir: Path, state_dim: int) -> tuple[np.ndarray, np.ndarray, list[Path], list[Path], list[Path]]:
    state = np.load(ep_dir / "observation_state.npy").astype(np.float32)

    action_path = ep_dir / "action.npy"
    if not action_path.is_file():
        action_path = ep_dir / "actions.npy"
    action = np.load(action_path).astype(np.float32)

    if state.ndim != 2 or state.shape[1] != state_dim:
        raise ValueError(f"{ep_dir}: expected state shape [T, {state_dim}], got {state.shape}")
    if action.ndim != 2 or action.shape[1] != len(ACTION_NAMES):
        raise ValueError(f"{ep_dir}: expected action shape [T, 14], got {action.shape}")

    external = _sorted_image_files(ep_dir / "external_images")
    left_wrist = _sorted_image_files(ep_dir / "left_wrist_images")
    right_wrist = _sorted_image_files(ep_dir / "right_wrist_images")

    num_frames = state.shape[0]
    for name, files in (
        ("external_images", external),
        ("left_wrist_images", left_wrist),
        ("right_wrist_images", right_wrist),
    ):
        if len(files) != num_frames:
            raise ValueError(f"{ep_dir}: {name} has {len(files)} frames, expected {num_frames}")
    if action.shape[0] != num_frames:
        raise ValueError(f"{ep_dir}: action has {action.shape[0]} frames, expected {num_frames}")

    return state, action, external, left_wrist, right_wrist


def main(args: Args) -> None:
    episode_dirs = sorted([p for p in args.raw_dir.iterdir() if p.is_dir()])
    if not episode_dirs:
        raise ValueError(f"No episode directories found under {args.raw_dir}")

    state_dim = _infer_state_dim(episode_dirs)
    image_shape = _infer_image_shape(episode_dirs)
    dataset = _create_dataset(args, state_dim, image_shape)

    for ep_idx, ep_dir in enumerate(episode_dirs):
        metadata = _load_metadata(ep_dir)
        task = _load_task(ep_dir, metadata, ep_idx)
        state, action, external, left_wrist, right_wrist = _validate_episode(ep_dir, state_dim)

        for frame_idx in range(state.shape[0]):
            dataset.add_frame(
                {
                    "observation.state": state[frame_idx],
                    "action": action[frame_idx],
                    "observation.images.external": _load_rgb(external[frame_idx]),
                    "observation.images.left_wrist": _load_rgb(left_wrist[frame_idx]),
                    "observation.images.right_wrist": _load_rgb(right_wrist[frame_idx]),
                }
            )

        dataset.save_episode(task=task)

    if args.push_to_hub:
        dataset.push_to_hub(
            tags=["zhishu", "dualarm", "nohand", "openpi", "lerobot"],
            private=args.private,
            push_videos=args.use_videos,
            license=args.license,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", type=Path, required=True)
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--fps", type=int, default=Args.fps)
    parser.add_argument("--use_videos", action="store_true")
    parser.add_argument("--image_writer_threads", type=int, default=Args.image_writer_threads)
    parser.add_argument("--image_writer_processes", type=int, default=Args.image_writer_processes)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--license", default=Args.license)
    parser.add_argument("--robot_type", default=Args.robot_type)
    parser.add_argument("--mode", default=Args.mode)
    ns = parser.parse_args()
    main(
        Args(
            raw_dir=ns.raw_dir,
            repo_id=ns.repo_id,
            fps=ns.fps,
            use_videos=ns.use_videos,
            image_writer_threads=ns.image_writer_threads,
            image_writer_processes=ns.image_writer_processes,
            push_to_hub=ns.push_to_hub,
            private=ns.private,
            license=ns.license,
            robot_type=ns.robot_type,
            mode=ns.mode,
        )
    )
