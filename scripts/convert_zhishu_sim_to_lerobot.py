#!/usr/bin/env python3
"""Convert LeIsaac-Zhishu-ClosingIn-v0 HDF5 episodes into a LeRobot dataset."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

try:
    from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit("LeRobot is required for this converter. Install it before running this script.") from exc

from zhishu_dualarm_lab.utils.closing_in_dataset import ACTION_NAMES, ROOT_GROUP


def _create_dataset(repo_id: str, *, fps: int, image_shape: tuple[int, int, int], state_dim: int) -> LeRobotDataset:
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
            "dtype": "image",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        },
        "observation.images.left_wrist": {
            "dtype": "image",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        },
        "observation.images.right_wrist": {
            "dtype": "image",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        },
    }

    output_dir = LEROBOT_HOME / repo_id
    if output_dir.exists():
        shutil.rmtree(output_dir)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type="zhishu_dualarm_nohand",
        features=features,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_file", type=Path, required=True, help="Closing-in HDF5 file.")
    parser.add_argument("--repo_id", required=True, help="Target LeRobot dataset repo id.")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--successful_only", action="store_true", help="Skip failed episodes.")
    parser.add_argument("--skip_initial_frames", type=int, default=0)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--license", default="apache-2.0")
    args = parser.parse_args()

    dataset_path = args.dataset_file if args.dataset_file.suffix == ".hdf5" else args.dataset_file.with_suffix(".hdf5")
    with h5py.File(dataset_path, "r") as handle:
        data_group = handle[ROOT_GROUP]
        episode_names = sorted(data_group.keys(), key=lambda name: int(name.split("_")[-1]))
        if not episode_names:
            raise ValueError(f"No episodes found in {dataset_path}")

        sample_episode = data_group[episode_names[0]]
        state = np.asarray(sample_episode["observation"]["state"], dtype=np.float32)
        images = np.asarray(sample_episode["observation"]["external_image"], dtype=np.uint8)
        state_dim = int(state.shape[1])
        image_shape = tuple(images.shape[1:])
        dataset = _create_dataset(args.repo_id, fps=args.fps, image_shape=image_shape, state_dim=state_dim)

        saved_count = 0
        for episode_name in episode_names:
            episode_group = data_group[episode_name]
            success = bool(episode_group.attrs.get("success", False))
            if args.successful_only and not success:
                continue

            state = np.asarray(episode_group["observation"]["state"], dtype=np.float32)
            actions = np.asarray(episode_group["actions"], dtype=np.float32)
            external = np.asarray(episode_group["observation"]["external_image"], dtype=np.uint8)
            left_wrist = np.asarray(episode_group["observation"]["left_wrist_image"], dtype=np.uint8)
            right_wrist = np.asarray(episode_group["observation"]["right_wrist_image"], dtype=np.uint8)
            task = str(episode_group.attrs["prompt"])

            start_idx = min(args.skip_initial_frames, state.shape[0] - 1)
            for frame_idx in range(start_idx, state.shape[0]):
                dataset.add_frame(
                    {
                        "observation.state": state[frame_idx],
                        "action": actions[frame_idx],
                        "observation.images.external": external[frame_idx],
                        "observation.images.left_wrist": left_wrist[frame_idx],
                        "observation.images.right_wrist": right_wrist[frame_idx],
                    }
                )

            dataset.save_episode(task=task)
            saved_count += 1
            print(f"[INFO] saved {episode_name} -> {args.repo_id} success={success}")

        if args.push_to_hub:
            dataset.push_to_hub(
                tags=["zhishu", "closing-in", "dualarm", "nohand", "openpi", "lerobot"],
                private=args.private,
                license=args.license,
            )

    print(f"[INFO] converted {saved_count} episodes into {args.repo_id}")


if __name__ == "__main__":
    main()
