#!/usr/bin/env python3
"""Replay LeIsaac-Zhishu-ClosingIn-v0 HDF5 episodes in Isaac."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from isaaclab.app import AppLauncher

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


class RateLimiter:
    def __init__(self, hz: int):
        self.sleep_duration = 1.0 / float(hz)
        self.last_time = time.time()
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env) -> None:
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()
        self.last_time = next_wakeup_time


def _restore_episode_state(env, episode_group: h5py.Group) -> None:
    env_ids = torch.tensor([0], device=env.device, dtype=torch.long)

    joint_pos = env._robot.data.joint_pos.clone()
    joint_vel = torch.zeros_like(env._robot.data.joint_vel)
    arm_joint_pos = torch.as_tensor(
        np.asarray(episode_group["initial"]["arm_joint_pos"], dtype=np.float32),
        device=env.device,
    )
    joint_pos[0, env._arm_joint_ids] = arm_joint_pos
    env._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
    env._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    env._joint_targets[:] = joint_pos
    env._last_action[:] = 0.0

    object_root_state = torch.as_tensor(
        np.asarray(episode_group["initial"]["object_root_state"], dtype=np.float32)[None, :],
        device=env.device,
    )
    env._object.write_root_pose_to_sim(object_root_state[:, :7], env_ids=env_ids)
    env._object.write_root_velocity_to_sim(object_root_state[:, 7:], env_ids=env_ids)

    target_root_state = torch.as_tensor(
        np.asarray(episode_group["initial"]["target_root_state"], dtype=np.float32)[None, :],
        device=env.device,
    )
    env._target_zone.write_root_pose_to_sim(target_root_state[:, :7], env_ids=env_ids)
    env._target_zone.write_root_velocity_to_sim(target_root_state[:, 7:], env_ids=env_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_file", type=Path, required=True, help="Closing-in HDF5 file to replay.")
    parser.add_argument("--select_episodes", type=int, nargs="*", default=[], help="Episode indices to replay.")
    parser.add_argument("--step_hz", type=int, default=10, help="Replay stepping rate.")
    parser.add_argument("--pause_between_episodes", type=float, default=0.5)
    parser.add_argument("--num_envs", type=int, default=1)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if hasattr(args, "enable_cameras"):
        args.enable_cameras = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        from zhishu_dualarm_lab.tasks.dualarm_tabletop.env import ZhishuDualArmTabletopEnv
        from zhishu_dualarm_lab.tasks.dualarm_tabletop.env_cfg import ZhishuDualArmTabletopEnvCfg
        from zhishu_dualarm_lab.utils.closing_in_dataset import ROOT_GROUP, TASK_NAME, list_episode_names

        cfg = ZhishuDualArmTabletopEnvCfg()
        cfg.scene.num_envs = args.num_envs
        env = ZhishuDualArmTabletopEnv(cfg=cfg, render_mode="human" if not args.headless else None)
        rate_limiter = RateLimiter(args.step_hz)

        dataset_path = args.dataset_file if args.dataset_file.suffix == ".hdf5" else args.dataset_file.with_suffix(".hdf5")
        with h5py.File(dataset_path, "r") as handle:
            data_group = handle[ROOT_GROUP]
            episode_names = list_episode_names(data_group)
            indices = args.select_episodes if args.select_episodes else list(range(len(episode_names)))

            for replay_count, episode_index in enumerate(indices, start=1):
                if episode_index >= len(episode_names):
                    raise IndexError(f"Episode index {episode_index} is out of range for {len(episode_names)} episodes.")

                episode_name = episode_names[episode_index]
                episode_group = data_group[episode_name]
                if episode_group.attrs.get("task_name", "") != TASK_NAME:
                    raise ValueError(f"{episode_name} is not a {TASK_NAME} episode.")

                prompt = episode_group.attrs["prompt"]
                target_mode = episode_group.attrs["target_mode"]
                success = bool(episode_group.attrs["success"])
                actions = np.asarray(episode_group["actions"], dtype=np.float32)

                env.reset()
                _restore_episode_state(env, episode_group)
                print(
                    f"[INFO] replay {replay_count}: {episode_name} mode={target_mode} "
                    f"success={success} steps={actions.shape[0]} prompt={prompt!r}"
                )

                for step_idx in range(actions.shape[0]):
                    action = torch.as_tensor(actions[step_idx : step_idx + 1], device=env.device)
                    env.step(action)
                    rate_limiter.sleep(env)

                if args.pause_between_episodes > 0:
                    time.sleep(args.pause_between_episodes)

        env.close()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
