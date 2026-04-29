from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import CUBOID_MARKER_CFG
from isaaclab.utils.math import combine_frame_transforms, sample_uniform

from zhishu_dualarm_lab.utils.action_adapter import JointActionAdapter, JointActionAdapterCfg
from zhishu_dualarm_lab.utils.obs_builder import ObservationBuilder, ObservationBuilderCfg

from .constants import (
    ACTION_DIM,
    ARM_ACTION_DELTA_SCALE,
    ARM_JOINT_NAMES,
    CAMERA_BODY_SIZE,
    HEAD_CAMERA_LINK_NAME,
    HEAD_CAMERA_OFFSET_POS,
    HEAD_CAMERA_OFFSET_ROT,
    JOINT_VELOCITY_SCALE,
    LEFT_TCP_LINK_NAME,
    LEFT_TCP_NAME,
    LEFT_WRIST_CAMERA_OFFSET_POS,
    LEFT_WRIST_CAMERA_OFFSET_ROT,
    POLICY_PROMPT,
    RESET_JOINT_NOISE,
    RIGHT_TCP_LINK_NAME,
    RIGHT_TCP_NAME,
    RIGHT_WRIST_CAMERA_OFFSET_POS,
    RIGHT_WRIST_CAMERA_OFFSET_ROT,
    WAIST_CAMERA_LINK_NAME,
    WAIST_CAMERA_OFFSET_POS,
    WAIST_CAMERA_OFFSET_ROT,
)
from .env_cfg import ZhishuDualArmRobotOnlyEnvCfg


class ZhishuDualArmRobotOnlyEnv(DirectRLEnv):
    """Robot-only dual-arm environment with the same action/obs contract."""

    cfg: ZhishuDualArmRobotOnlyEnvCfg
    _TABLETOP_START_JOINT4_RAD = float(np.deg2rad(96.0))

    def __init__(self, cfg: ZhishuDualArmRobotOnlyEnvCfg, render_mode: str | None = None, **kwargs):
        self._latest_policy_input: dict | None = None
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        self.action_dim = ACTION_DIM
        self._arm_joint_ids = self._robot.find_joints(list(ARM_JOINT_NAMES))[0]
        self._joint_lower_limits = self._robot.data.soft_joint_pos_limits[0, self._arm_joint_ids, 0].clone()
        self._joint_upper_limits = self._robot.data.soft_joint_pos_limits[0, self._arm_joint_ids, 1].clone()
        self._default_joint_pos = self._robot.data.default_joint_pos.clone()
        self._left_joint4_id = self._robot.find_joints(["left_joint4"])[0][0]
        self._right_joint4_id = self._robot.find_joints(["right_joint4"])[0][0]
        self._default_joint_pos[:, self._left_joint4_id] = self._TABLETOP_START_JOINT4_RAD
        self._default_joint_pos[:, self._right_joint4_id] = self._TABLETOP_START_JOINT4_RAD
        self._joint_targets = self._default_joint_pos.clone()
        self._last_action = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self._raw_action = torch.zeros_like(self._last_action)

        self._left_tcp_idx = self._tcp_frames.data.target_frame_names.index(LEFT_TCP_NAME)
        self._right_tcp_idx = self._tcp_frames.data.target_frame_names.index(RIGHT_TCP_NAME)
        self._head_camera_body_id = self._robot.find_bodies([HEAD_CAMERA_LINK_NAME])[0][0]
        self._waist_camera_body_id = self._robot.find_bodies([WAIST_CAMERA_LINK_NAME])[0][0]
        self._left_camera_body_id = self._robot.find_bodies([LEFT_TCP_LINK_NAME])[0][0]
        self._right_camera_body_id = self._robot.find_bodies([RIGHT_TCP_LINK_NAME])[0][0]
        marker_cfg = CUBOID_MARKER_CFG.replace(prim_path="/Visuals/ZhishuRobotOnlyCameraBodies")
        marker_cfg.markers["cuboid"].size = CAMERA_BODY_SIZE
        self._camera_body_markers = VisualizationMarkers(marker_cfg)

        self._action_adapter = JointActionAdapter(JointActionAdapterCfg(delta_scale=ARM_ACTION_DELTA_SCALE))
        self._obs_builder = ObservationBuilder(ObservationBuilderCfg(prompt=POLICY_PROMPT))

    def _setup_scene(self):
        self._robot: Articulation = self.scene["robot"]
        self._external_camera = self.scene["external_camera"]
        self._waist_camera = self.scene["waist_camera"]
        self._left_wrist_camera = self.scene["left_wrist_camera"]
        self._right_wrist_camera = self.scene["right_wrist_camera"]
        self._tcp_frames = self.scene["tcp_frames"]

    def _camera_world_pose(
        self,
        *,
        parent_body_id: int,
        offset_pos: tuple[float, float, float],
        offset_rot: tuple[float, float, float, float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        parent_pos = self._robot.data.body_link_pos_w[:, parent_body_id]
        parent_quat = self._robot.data.body_link_quat_w[:, parent_body_id]
        offset_pos_tensor = torch.tensor(offset_pos, dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)
        offset_quat_tensor = torch.tensor(offset_rot, dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)
        return combine_frame_transforms(parent_pos, parent_quat, offset_pos_tensor, offset_quat_tensor)

    def _sync_camera_mounts(self) -> None:
        head_pos, _ = self._camera_world_pose(
            parent_body_id=self._head_camera_body_id, offset_pos=HEAD_CAMERA_OFFSET_POS, offset_rot=HEAD_CAMERA_OFFSET_ROT
        )
        # Keep the head camera level in world coordinates while still
        # translating it with the head link.
        head_quat = torch.tensor(HEAD_CAMERA_OFFSET_ROT, dtype=torch.float32, device=self.device).repeat(
            self.num_envs, 1
        )
        waist_pos, waist_quat = self._camera_world_pose(
            parent_body_id=self._waist_camera_body_id, offset_pos=WAIST_CAMERA_OFFSET_POS, offset_rot=WAIST_CAMERA_OFFSET_ROT
        )
        left_pos, left_quat = self._camera_world_pose(
            parent_body_id=self._left_camera_body_id,
            offset_pos=LEFT_WRIST_CAMERA_OFFSET_POS,
            offset_rot=LEFT_WRIST_CAMERA_OFFSET_ROT,
        )
        right_pos, right_quat = self._camera_world_pose(
            parent_body_id=self._right_camera_body_id,
            offset_pos=RIGHT_WRIST_CAMERA_OFFSET_POS,
            offset_rot=RIGHT_WRIST_CAMERA_OFFSET_ROT,
        )
        self._external_camera.set_world_poses(positions=head_pos, orientations=head_quat, convention="world")
        self._left_wrist_camera.set_world_poses(positions=left_pos, orientations=left_quat, convention="world")
        self._right_wrist_camera.set_world_poses(positions=right_pos, orientations=right_quat, convention="world")
        self._waist_camera.set_world_poses(positions=waist_pos, orientations=waist_quat, convention="world")
        self._camera_body_markers.visualize(translations=head_pos, orientations=head_quat)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._raw_action = actions.clone().clamp(-1.0, 1.0)
        current_arm_pos = self._robot.data.joint_pos[:, self._arm_joint_ids]
        arm_targets = self._action_adapter.compute_joint_targets(
            raw_action=self._raw_action,
            current_joint_pos=current_arm_pos,
            lower_limits=self._joint_lower_limits,
            upper_limits=self._joint_upper_limits,
        )
        self._joint_targets[:] = self._default_joint_pos
        self._joint_targets[:, self._arm_joint_ids] = arm_targets
        self._last_action[:] = self._raw_action

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self._joint_targets)

    def _get_observations(self) -> dict:
        self._sync_camera_mounts()
        if self.sim.has_rtx_sensors():
            self.sim.render()

        left_tcp_pos = self._tcp_frames.data.target_pos_w[:, self._left_tcp_idx]
        left_tcp_quat = self._tcp_frames.data.target_quat_w[:, self._left_tcp_idx]
        right_tcp_pos = self._tcp_frames.data.target_pos_w[:, self._right_tcp_idx]
        right_tcp_quat = self._tcp_frames.data.target_quat_w[:, self._right_tcp_idx]

        zeros_pos = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        zeros_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)
        obs_dict = self._obs_builder.build(
            external_camera=self._external_camera,
            left_wrist_camera=self._left_wrist_camera,
            right_wrist_camera=self._right_wrist_camera,
            joint_pos=self._robot.data.joint_pos[:, self._arm_joint_ids],
            joint_vel=self._robot.data.joint_vel[:, self._arm_joint_ids] * JOINT_VELOCITY_SCALE,
            last_action=self._last_action,
            left_tcp_pos=left_tcp_pos,
            left_tcp_quat=left_tcp_quat,
            right_tcp_pos=right_tcp_pos,
            right_tcp_quat=right_tcp_quat,
            object_pos=zeros_pos,
            object_quat=zeros_quat,
            target_pos=zeros_pos,
            target_quat=zeros_quat,
        )
        self._latest_policy_input = obs_dict
        return {"policy": obs_dict}

    def _get_rewards(self) -> torch.Tensor:
        return -0.01 * torch.sum(self._last_action**2, dim=-1)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids)

        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()
        joint_pos[:, self._arm_joint_ids] += sample_uniform(
            -RESET_JOINT_NOISE, RESET_JOINT_NOISE, (len(env_ids), len(self._arm_joint_ids)), self.device
        )
        joint_pos[:, self._left_joint4_id] = self._TABLETOP_START_JOINT4_RAD
        joint_pos[:, self._right_joint4_id] = self._TABLETOP_START_JOINT4_RAD
        joint_pos[:, self._arm_joint_ids] = torch.clamp(
            joint_pos[:, self._arm_joint_ids], self._joint_lower_limits, self._joint_upper_limits
        )
        joint_vel.zero_()
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._joint_targets[env_ids] = joint_pos
        self._last_action[env_ids] = 0.0

    def get_policy_input(self) -> dict:
        if self._latest_policy_input is None:
            _ = self._get_observations()
        if self._latest_policy_input is None:
            raise RuntimeError("Policy input requested before robot-only observations were initialized.")
        return {
            "prompt": self._latest_policy_input["prompt"][0],
            "observation/external_image": self._latest_policy_input["observation/external_image"][0].cpu().numpy(),
            "observation/left_wrist_image": self._latest_policy_input["observation/left_wrist_image"][0].cpu().numpy(),
            "observation/right_wrist_image": self._latest_policy_input["observation/right_wrist_image"][0].cpu().numpy(),
            "observation/state": self._latest_policy_input["observation/state"][0].cpu().numpy(),
        }
