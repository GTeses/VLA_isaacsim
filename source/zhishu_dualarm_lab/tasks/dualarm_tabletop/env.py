from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import CUBOID_MARKER_CFG
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.utils.math import sample_uniform

from zhishu_dualarm_lab.utils.action_adapter import (
    JointActionAdapter,
    JointActionAdapterCfg,
    PolicyActionChunkAdapter,
)
from zhishu_dualarm_lab.utils.obs_builder import ObservationBuilder, ObservationBuilderCfg

from .constants import (
    ACTION_DIM,
    ACTION_PENALTY_SCALE,
    ARM_ACTION_DELTA_SCALE,
    ARM_JOINT_NAMES,
    HEAD_CAMERA_LINK_NAME,
    HEAD_CAMERA_OFFSET_POS,
    HEAD_CAMERA_OFFSET_ROT,
    CAMERA_BODY_SIZE,
    OBJECT_TARGET_PROGRESS_SCALE,
    OBJECT_TARGET_SHAPING_SCALE,
    JOINT_VELOCITY_SCALE,
    LEFT_TCP_LINK_NAME,
    LEFT_TCP_NAME,
    LEFT_WRIST_CAMERA_OFFSET_POS,
    LEFT_WRIST_CAMERA_OFFSET_ROT,
    POLICY_PROMPT,
    REACH_SUCCESS_BONUS,
    RESET_JOINT_NOISE,
    RIGHT_TCP_LINK_NAME,
    RIGHT_TCP_NAME,
    RIGHT_WRIST_CAMERA_OFFSET_POS,
    RIGHT_WRIST_CAMERA_OFFSET_ROT,
    TARGET_SUCCESS_BONUS,
    TARGET_REACHED_THRESHOLD,
    TCP_GATHER_SHAPING_SCALE,
    TCP_MIDPOINT_OBJECT_THRESHOLD,
    TCP_OBJECT_REACHED_THRESHOLD,
    TCP_OBJECT_SHAPING_SCALE,
    WAIST_CAMERA_LINK_NAME,
    WAIST_CAMERA_OFFSET_POS,
    WAIST_CAMERA_OFFSET_ROT,
)
from .env_cfg import ZhishuDualArmTabletopEnvCfg


class ZhishuDualArmTabletopEnv(DirectRLEnv):
    """Minimal dual-arm tabletop prototype environment.

    Actions:
    - [num_envs, 14] normalized joint deltas in [-1, 1]

    Observations:
    - dict of RGB images and low-dimensional state terms
    - image format is HWC uint8
    """

    cfg: ZhishuDualArmTabletopEnvCfg

    def __init__(self, cfg: ZhishuDualArmTabletopEnvCfg, render_mode: str | None = None, **kwargs):
        self._latest_policy_input: dict | None = None
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        self.action_dim = ACTION_DIM
        # Keep a compact list of arm-only joint ids. The robot asset may
        # contain head/waist/wheels, but first-version control only exposes
        # the two 7-DoF arms.
        self._arm_joint_ids = self._robot.find_joints(list(ARM_JOINT_NAMES))[0]
        self._joint_lower_limits = self._robot.data.soft_joint_pos_limits[0, self._arm_joint_ids, 0].clone()
        self._joint_upper_limits = self._robot.data.soft_joint_pos_limits[0, self._arm_joint_ids, 1].clone()
        self._default_joint_pos = self._robot.data.default_joint_pos.clone()
        # Full-joint target tensor is preserved because the articulation
        # still contains non-arm joints that should remain at their default values.
        self._joint_targets = self._default_joint_pos.clone()
        self._last_action = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self._raw_action = torch.zeros_like(self._last_action)
        # Placeholder buffer for future websocket / chunked policy outputs.
        self._action_plan_buffer = torch.zeros((0, self.action_dim), device=self.device)
        self._prev_object_target_dist = torch.zeros(self.num_envs, device=self.device)
        self.replan_steps = 1

        self._left_tcp_idx = self._tcp_frames.data.target_frame_names.index(LEFT_TCP_NAME)
        self._right_tcp_idx = self._tcp_frames.data.target_frame_names.index(RIGHT_TCP_NAME)
        self._head_camera_body_id = self._robot.find_bodies([HEAD_CAMERA_LINK_NAME])[0][0]
        self._waist_camera_body_id = self._robot.find_bodies([WAIST_CAMERA_LINK_NAME])[0][0]
        self._left_camera_body_id = self._robot.find_bodies([LEFT_TCP_LINK_NAME])[0][0]
        self._right_camera_body_id = self._robot.find_bodies([RIGHT_TCP_LINK_NAME])[0][0]
        marker_cfg = CUBOID_MARKER_CFG.replace(prim_path="/Visuals/ZhishuCameraBodies")
        marker_cfg.markers["cuboid"].size = CAMERA_BODY_SIZE
        self._camera_body_markers = VisualizationMarkers(marker_cfg)

        self._action_adapter = JointActionAdapter(JointActionAdapterCfg(delta_scale=ARM_ACTION_DELTA_SCALE))
        self._policy_action_adapter = PolicyActionChunkAdapter(action_dim=self.action_dim)
        self._obs_builder = ObservationBuilder(ObservationBuilderCfg(prompt=POLICY_PROMPT))

    def _setup_scene(self):
        # All scene handles are cached once here so later env methods can
        # stay small and read like task logic instead of scene lookups.
        self._robot: Articulation = self.scene["robot"]
        self._table: RigidObject = self.scene["table"]
        self._object: RigidObject = self.scene["object"]
        self._target_zone: RigidObject = self.scene["target_zone"]
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
        head_pos, head_quat = self._camera_world_pose(
            parent_body_id=self._head_camera_body_id,
            offset_pos=HEAD_CAMERA_OFFSET_POS,
            offset_rot=HEAD_CAMERA_OFFSET_ROT,
        )
        waist_pos, waist_quat = self._camera_world_pose(
            parent_body_id=self._waist_camera_body_id,
            offset_pos=WAIST_CAMERA_OFFSET_POS,
            offset_rot=WAIST_CAMERA_OFFSET_ROT,
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
        marker_pos = torch.cat([head_pos, waist_pos, left_pos, right_pos], dim=0)
        marker_quat = torch.cat([head_quat, waist_quat, left_quat, right_quat], dim=0)
        self._camera_body_markers.visualize(translations=marker_pos, orientations=marker_quat)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Convert normalized external actions into arm joint position targets.
        # Non-arm joints are intentionally left at their default values.
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
        # The imported robot articulation does not propagate moving body poses
        # back to child USD prims, so wrist/head cameras are driven through
        # explicit kinematic camera-body assets that follow the live body state.
        self._sync_camera_mounts()
        if self.sim.has_rtx_sensors():
            self.sim.render()

        # FrameTransformer provides the TCP frames as virtual target frames,
        # which lets us change TCP offsets later without rewriting the rest
        # of the observation or task code.
        left_tcp_pos = self._tcp_frames.data.target_pos_w[:, self._left_tcp_idx]
        left_tcp_quat = self._tcp_frames.data.target_quat_w[:, self._left_tcp_idx]
        right_tcp_pos = self._tcp_frames.data.target_pos_w[:, self._right_tcp_idx]
        right_tcp_quat = self._tcp_frames.data.target_quat_w[:, self._right_tcp_idx]

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
            object_pos=self._object.data.root_pos_w,
            object_quat=self._object.data.root_quat_w,
            target_pos=self._target_zone.data.root_pos_w,
            target_quat=self._target_zone.data.root_quat_w,
        )
        self._latest_policy_input = obs_dict
        return {"policy": obs_dict}

    def _compute_task_metrics(self) -> dict[str, torch.Tensor]:
        """Collect the small set of XY metrics used by reward and done logic.

        Stage two treats this environment as a no-hand tabletop platform:
        both TCPs should approach the cube, close in around it, and nudge it
        toward the target zone. Using XY distances keeps the tabletop task
        robust to small height noise while preserving the fixed 70D state
        contract exposed to policies.
        """
        object_xy = self._object.data.root_pos_w[:, :2]
        target_xy = self._target_zone.data.root_pos_w[:, :2]
        left_tcp_xy = self._tcp_frames.data.target_pos_w[:, self._left_tcp_idx, :2]
        right_tcp_xy = self._tcp_frames.data.target_pos_w[:, self._right_tcp_idx, :2]
        tcp_midpoint_xy = 0.5 * (left_tcp_xy + right_tcp_xy)
        left_object_dist = torch.linalg.norm(left_tcp_xy - object_xy, dim=-1)
        right_object_dist = torch.linalg.norm(right_tcp_xy - object_xy, dim=-1)
        tcp_midpoint_object_dist = torch.linalg.norm(tcp_midpoint_xy - object_xy, dim=-1)
        object_target_dist = torch.linalg.norm(object_xy - target_xy, dim=-1)
        reach_success = (left_object_dist < TCP_OBJECT_REACHED_THRESHOLD) & (
            right_object_dist < TCP_OBJECT_REACHED_THRESHOLD
        )
        gather_success = tcp_midpoint_object_dist < TCP_MIDPOINT_OBJECT_THRESHOLD
        object_close_to_target = object_target_dist < TARGET_REACHED_THRESHOLD
        return {
            "left_object_dist": left_object_dist,
            "right_object_dist": right_object_dist,
            "tcp_midpoint_object_dist": tcp_midpoint_object_dist,
            "object_target_dist": object_target_dist,
            "reach_success": reach_success,
            "gather_success": gather_success,
            "object_close_to_target": object_close_to_target,
        }

    def _get_rewards(self) -> torch.Tensor:
        metrics = self._compute_task_metrics()
        # Reaching reward: both arms should move close to the cube.
        reaching_reward = 0.5 * (
            torch.exp(-TCP_OBJECT_SHAPING_SCALE * metrics["left_object_dist"])
            + torch.exp(-TCP_OBJECT_SHAPING_SCALE * metrics["right_object_dist"])
        )
        # Gather reward: the midpoint between both TCPs should close in on the cube,
        # which biases the behavior toward a dual-arm surround / nudge posture.
        gather_reward = torch.exp(-TCP_GATHER_SHAPING_SCALE * metrics["tcp_midpoint_object_dist"])
        # Target shaping keeps the pushing objective alive even before the cube enters
        # the target zone, and progress reward pays only for actual motion toward it.
        object_target_reward = torch.exp(-OBJECT_TARGET_SHAPING_SCALE * metrics["object_target_dist"])
        object_target_progress = (self._prev_object_target_dist - metrics["object_target_dist"]).clamp(-0.05, 0.05)
        self._prev_object_target_dist[:] = metrics["object_target_dist"]
        reach_bonus = (metrics["reach_success"] & metrics["gather_success"]).float() * REACH_SUCCESS_BONUS
        target_bonus = metrics["object_close_to_target"].float() * TARGET_SUCCESS_BONUS
        action_penalty = torch.sum(self._last_action**2, dim=-1)
        return (
            reaching_reward
            + gather_reward
            + object_target_reward
            + OBJECT_TARGET_PROGRESS_SCALE * object_target_progress
            + reach_bonus
            + target_bonus
            - ACTION_PENALTY_SCALE * action_penalty
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        metrics = self._compute_task_metrics()
        # Stage two explicitly supports both "reach / close in" and "push to target"
        # outcomes, so either success condition can end the episode early.
        task_success = metrics["object_close_to_target"] | (
            metrics["reach_success"] & metrics["gather_success"]
        )
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return task_success, time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids)

        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

        # Reset arm joints near the default pose with light noise so the
        # environment does not always start from a numerically identical state.
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()
        joint_pos[:, self._arm_joint_ids] += sample_uniform(
            -RESET_JOINT_NOISE,
            RESET_JOINT_NOISE,
            (len(env_ids), len(self._arm_joint_ids)),
            self.device,
        )
        joint_pos[:, self._arm_joint_ids] = torch.clamp(
            joint_pos[:, self._arm_joint_ids],
            self._joint_lower_limits,
            self._joint_upper_limits,
        )
        joint_vel.zero_()
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._joint_targets[env_ids] = joint_pos

        # Reset the tabletop object in a small randomized patch on the table.
        object_state = self._object.data.default_root_state[env_ids].clone()
        object_state[:, :3] += self.scene.env_origins[env_ids]
        object_state[:, 0] += sample_uniform(-0.05, 0.05, (len(env_ids), 1), self.device).squeeze(-1)
        object_state[:, 1] += sample_uniform(-0.08, 0.08, (len(env_ids), 1), self.device).squeeze(-1)
        object_state[:, 7:] = 0.0
        self._object.write_root_pose_to_sim(object_state[:, :7], env_ids=env_ids)
        self._object.write_root_velocity_to_sim(object_state[:, 7:], env_ids=env_ids)

        target_state = self._target_zone.data.default_root_state[env_ids].clone()
        target_state[:, :3] += self.scene.env_origins[env_ids]
        target_state[:, 7:] = 0.0
        self._target_zone.write_root_pose_to_sim(target_state[:, :7], env_ids=env_ids)
        self._target_zone.write_root_velocity_to_sim(target_state[:, 7:], env_ids=env_ids)

        table_state = self._table.data.default_root_state[env_ids].clone()
        table_state[:, :3] += self.scene.env_origins[env_ids]
        table_state[:, 7:] = 0.0
        self._table.write_root_pose_to_sim(table_state[:, :7], env_ids=env_ids)
        self._table.write_root_velocity_to_sim(table_state[:, 7:], env_ids=env_ids)

        self._last_action[env_ids] = 0.0
        object_xy = object_state[:, :2]
        target_xy = target_state[:, :2]
        self._prev_object_target_dist[env_ids] = torch.linalg.norm(object_xy - target_xy, dim=-1)

    def get_policy_input(self) -> dict:
        """Return the minimal stable policy input contract for websocket inference.

        The first-version schema is intentionally small and fixed:
        - prompt: str
        - observation/external_image: HWC uint8 RGB
        - observation/left_wrist_image: HWC uint8 RGB
        - observation/right_wrist_image: HWC uint8 RGB
        - observation/state: float32 vector

        The state order is fixed by ObservationBuilder:
        [joint_pos, joint_vel, last_action, left_tcp_pose, right_tcp_pose, object_pose, target_pose]
        """
        if self._latest_policy_input is None:
            return {}
        return {
            "prompt": self._latest_policy_input["prompt"][0],
            "observation/external_image": self._latest_policy_input["observation/external_image"][0].cpu().numpy(),
            "observation/left_wrist_image": self._latest_policy_input["observation/left_wrist_image"][0].cpu().numpy(),
            "observation/right_wrist_image": self._latest_policy_input["observation/right_wrist_image"][0].cpu().numpy(),
            "observation/state": self._latest_policy_input["observation/state"][0].cpu().numpy().astype(np.float32),
        }

    def apply_policy_output(self, action_chunk_or_vec) -> torch.Tensor:
        """Cache an action chunk or vector for future stepping.

        Accepts:
        - [action_dim]
        - [chunk_len, action_dim]

        Right now this only caches the incoming tensor. The future websocket
        client can call this method and then decide whether to step the env
        with the first action immediately or consume a short action chunk over
        several control steps.
        """
        normalized_chunk = self._policy_action_adapter.normalize_action_chunk(action_chunk_or_vec).copy()
        action_tensor = torch.as_tensor(normalized_chunk, device=self.device, dtype=torch.float32)
        if self.replan_steps > 0:
            action_tensor = action_tensor[: self.replan_steps]
        self._action_plan_buffer = action_tensor.clone()
        return self._action_plan_buffer[0]

    @property
    def action_plan_buffer(self) -> torch.Tensor:
        return self._action_plan_buffer

    @property
    def action_plan_length(self) -> int:
        return int(self._action_plan_buffer.shape[0])

    def consume_action_plan_step(self) -> torch.Tensor:
        """Pop the next action from the local plan buffer."""
        if self.action_plan_length == 0:
            raise RuntimeError("Action plan buffer is empty. Call apply_policy_output() before consuming actions.")
        next_action = self._action_plan_buffer[0].clone()
        self._action_plan_buffer = self._action_plan_buffer[1:].clone()
        return next_action.unsqueeze(0)
