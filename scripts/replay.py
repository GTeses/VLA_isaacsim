#!/usr/bin/env python3
"""Replay clean robot-only closing-in HDF5 episodes in Isaac."""

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

TARGET_MARKER_SIZE = (0.035, 0.035, 0.035)
DISK_DIAMETER_M = 0.178
DISK_THICKNESS_M = 0.010
DISK_RADIUS_M = 0.5 * DISK_DIAMETER_M


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


def _normalize_path(path: Path) -> Path:
    return path if path.suffix == ".hdf5" else path.with_suffix(".hdf5")


def _build_target_markers():
    from isaaclab.markers import VisualizationMarkers
    from isaaclab.markers.config import CUBOID_MARKER_CFG

    marker_cfg = CUBOID_MARKER_CFG.replace(prim_path="/Visuals/ZhishuRobotOnlyClosingInReplayTargets")
    marker_cfg.markers["cuboid"].size = TARGET_MARKER_SIZE
    return VisualizationMarkers(marker_cfg)


def _build_disk_marker():
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/ZhishuRobotOnlyClosingInReplayDisk",
        markers={
            "disk": sim_utils.CylinderCfg(
                radius=DISK_RADIUS_M,
                height=DISK_THICKNESS_M,
                axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.05, 0.10, 0.45),
                    roughness=0.45,
                    metallic=0.05,
                ),
            )
        },
    )
    return VisualizationMarkers(marker_cfg)


def _list_episode_names(data_group: h5py.Group) -> list[str]:
    return sorted(data_group.keys(), key=lambda name: int(name.split("_")[-1]))


def _restore_robot_state_from_policy_state(env, state: np.ndarray) -> None:
    """Replay one recorded 70D state by directly writing the robot joint state."""

    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape[0] != 70:
        raise ValueError(f"Expected 70D policy state, got {state.shape}")

    env_ids = torch.tensor([0], device=env.device, dtype=torch.long)
    joint_pos = env._robot.data.joint_pos.clone()
    joint_vel = torch.zeros_like(env._robot.data.joint_vel)
    joint_pos[0, env._arm_joint_ids] = torch.as_tensor(state[0:14], dtype=torch.float32, device=env.device)
    joint_vel[0, env._arm_joint_ids] = torch.as_tensor(state[14:28], dtype=torch.float32, device=env.device)
    env._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
    env._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    env._joint_targets[:] = joint_pos
    env._last_action[0] = torch.as_tensor(state[28:42], dtype=torch.float32, device=env.device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_file", type=Path, required=True, help="Clean closing-in HDF5 file to replay.")
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

    env = None
    try:
        from zhishu_dualarm_lab.tasks.robot_only.env import ZhishuDualArmRobotOnlyEnv
        from zhishu_dualarm_lab.tasks.robot_only.env_cfg import ZhishuDualArmRobotOnlyEnvCfg
        from zhishu_dualarm_lab.utils.robot_only_closing_in_dataset import ROOT_GROUP, TASK_NAME

        cfg = ZhishuDualArmRobotOnlyEnvCfg()
        cfg.scene.num_envs = args.num_envs
        env = ZhishuDualArmRobotOnlyEnv(cfg=cfg, render_mode="human" if not args.headless else None)
        rate_limiter = RateLimiter(args.step_hz)
        markers = _build_target_markers()
        disk_marker = _build_disk_marker()
        marker_quats = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
            device=env.device,
        )
        disk_marker_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=env.device)

        dataset_path = _normalize_path(args.dataset_file)
        with h5py.File(dataset_path, "r") as handle:
            data_group = handle[ROOT_GROUP]
            episode_names = _list_episode_names(data_group)
            indices = args.select_episodes if args.select_episodes else list(range(len(episode_names)))

            for replay_count, episode_index in enumerate(indices, start=1):
                if episode_index >= len(episode_names):
                    raise IndexError(f"Episode index {episode_index} is out of range for {len(episode_names)} episodes.")

                episode_name = episode_names[episode_index]
                episode_group = data_group[episode_name]
                if episode_group.attrs.get("task_name", "") != TASK_NAME:
                    raise ValueError(f"{episode_name} is not a {TASK_NAME} episode.")

                prompt = str(episode_group.attrs.get("prompt", ""))
                success = bool(episode_group.attrs.get("success", False))
                states = np.asarray(episode_group["observation"]["state"], dtype=np.float32)
                left_target = np.asarray(episode_group["task"]["left_target"], dtype=np.float32)
                right_target = np.asarray(episode_group["task"]["right_target"], dtype=np.float32)
                disk_center = np.asarray(episode_group["task"]["disk_center"], dtype=np.float32)

                env.reset()
                markers.visualize(
                    translations=torch.as_tensor(np.stack([left_target, right_target], axis=0), dtype=torch.float32, device=env.device),
                    orientations=marker_quats,
                )
                disk_marker.visualize(
                    translations=torch.as_tensor(disk_center[None, :], dtype=torch.float32, device=env.device),
                    orientations=disk_marker_quat,
                )
                print(
                    f"[INFO] replay {replay_count}: {episode_name} "
                    f"success={success} steps={states.shape[0]} prompt={prompt!r}"
                )

                for step_idx in range(states.shape[0]):
                    _restore_robot_state_from_policy_state(env, states[step_idx])
                    env._sync_camera_mounts()
                    env.sim.render()
                    rate_limiter.sleep(env)

                if args.pause_between_episodes > 0:
                    time.sleep(args.pause_between_episodes)

        env.close()
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        simulation_app.close()


if __name__ == "__main__":
    main()
