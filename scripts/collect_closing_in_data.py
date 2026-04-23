#!/usr/bin/env python3
"""Collect a small-scale HDF5 dataset for LeIsaac-Zhishu-ClosingIn-v0.

This collector is intentionally narrow:
- only the dual-arm no-hand closing-in task
- only scripted/heuristic actions
- only small HDF5 recording for semantic inspection and replay

Each episode samples either:
- one shared center target, or
- two symmetric targets around a shared center

The resulting HDF5 stores 14D joint-delta actions, 3 RGB camera streams,
the fixed 70D policy state, prompt text, episode metadata, and success labels.
"""

from __future__ import annotations

import argparse
import traceback
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

# The original URDF is only globally mirrored. The two arms are not a
# "copy/paste with one sign flipped" pair:
# - left_joint1 limit = [-pi, 1.0], right_joint1 = [-1.0, pi]
# - left_joint2 limit = [-0.523, 1.57], right_joint2 = [-1.57, 0.523]
# - several joint origins/rpy are mirrored but not numerically identical
# Safe-start therefore uses independent tabletop templates for each arm instead
# of forcing the right arm to inherit a left-arm-shaped posture prior.
LEFT_SAFE_TABLETOP_HOME_POSES = (
    np.array([-0.95, 0.00, 0.28, 0.00, 0.05, 0.00, 0.55], dtype=np.float32),
    np.array([-1.15, 0.05, 0.40, -0.05, 0.08, 0.00, 0.80], dtype=np.float32),
    np.array([-0.75, -0.05, 0.18, 0.02, -0.05, 0.02, 0.35], dtype=np.float32),
    np.array([-1.05, 0.00, 0.48, -0.10, 0.12, -0.05, 0.95], dtype=np.float32),
    np.array([-0.88, 0.03, 0.32, 0.04, 0.00, 0.06, 0.68], dtype=np.float32),
)

RIGHT_SAFE_TABLETOP_HOME_POSES = (
    np.array([1.42, -0.12, -0.08, 1.12, -0.30, 0.16, 0.00], dtype=np.float32),
    np.array([1.56, -0.04, -0.16, 1.26, -0.18, 0.10, 0.02], dtype=np.float32),
    np.array([1.30, -0.18, -0.04, 1.02, -0.42, 0.18, -0.02], dtype=np.float32),
    np.array([1.70, 0.02, -0.22, 1.34, -0.10, 0.04, 0.04], dtype=np.float32),
    np.array([1.48, -0.08, -0.12, 1.18, -0.24, 0.14, -0.04], dtype=np.float32),
)

# Left-arm posture anchoring can stay close to the earlier probe result because
# the left side already looks broadly natural. On the right side, however, the
# visually important "shoulder + elbow" appearance depends much more on
# right_joint2/right_joint4 than on the earlier probe-only set. Keep the right
# arm anchored on the joints that actually determine whether it looks lifted and
# tabletop-ready, not just on the joints that happened to move the TCP most.
LEFT_PROXIMAL_LOCAL_IDS = (0, 2, 6)
RIGHT_PROXIMAL_LOCAL_IDS = (7, 8, 10, 11)

LEFT_SAFE_CHAIN_LINK_NAMES = ("left_link3", "left_link5")
RIGHT_SAFE_CHAIN_LINK_NAMES = ("right_link3", "right_link5")


def _clip_xyz_step(err: torch.Tensor, max_norm: float) -> torch.Tensor:
    norms = torch.linalg.norm(err, dim=-1, keepdim=True).clamp_min(1e-6)
    scale = torch.clamp(max_norm / norms, max=1.0)
    return err * scale


class ClosingInIKPolicy:
    """Task-space teacher policy built on Isaac Lab's Differential IK controller.

    The collector's long-term job is to generate semantically consistent data,
    not to hand-design a joint-sign map. Using differential IK keeps the teacher
    aligned with the robot's actual Jacobian and avoids the brittle
    "err_x/err_y/err_z -> guessed joint delta" behavior.
    """

    def __init__(
        self,
        *,
        env,
        left_body_name: str,
        right_body_name: str,
        diff_ik_controller_cfg_cls,
        diff_ik_controller_cls,
        delta_scale: float,
    ):
        self.env = env
        self.delta_scale = float(delta_scale)
        self.left_joint_ids = env._arm_joint_ids[:7]
        self.right_joint_ids = env._arm_joint_ids[7:]
        self.left_tcp_frame_idx = env._left_tcp_idx
        self.right_tcp_frame_idx = env._right_tcp_idx
        self.left_body_idx = env._robot.find_bodies([left_body_name])[0][0]
        self.right_body_idx = env._robot.find_bodies([right_body_name])[0][0]
        if env._robot.is_fixed_base:
            self.left_jacobi_body_idx = self.left_body_idx - 1
            self.right_jacobi_body_idx = self.right_body_idx - 1
            self.left_jacobi_joint_ids = self.left_joint_ids
            self.right_jacobi_joint_ids = self.right_joint_ids
        else:
            self.left_jacobi_body_idx = self.left_body_idx
            self.right_jacobi_body_idx = self.right_body_idx
            self.left_jacobi_joint_ids = [i + 6 for i in self.left_joint_ids]
            self.right_jacobi_joint_ids = [i + 6 for i in self.right_joint_ids]

        controller_cfg = diff_ik_controller_cfg_cls(
            command_type="position",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 0.08},
        )
        self.left_controller = diff_ik_controller_cls(controller_cfg, num_envs=env.num_envs, device=env.device)
        self.right_controller = diff_ik_controller_cls(controller_cfg, num_envs=env.num_envs, device=env.device)

    def infer(self, spec, step_idx: int) -> np.ndarray:
        del step_idx
        left_action, _ = self._solve_arm(
            controller=self.left_controller,
            tcp_frame_idx=self.left_tcp_frame_idx,
            jacobi_body_idx=self.left_jacobi_body_idx,
            joint_ids=self.left_joint_ids,
            jacobi_joint_ids=self.left_jacobi_joint_ids,
            target_world=torch.as_tensor(spec.left_target, dtype=torch.float32, device=self.env.device)[None, :],
            speed_scale=spec.speed_scale,
            jitter_scale=spec.jitter_scale,
        )
        right_action, _ = self._solve_arm(
            controller=self.right_controller,
            tcp_frame_idx=self.right_tcp_frame_idx,
            jacobi_body_idx=self.right_jacobi_body_idx,
            joint_ids=self.right_joint_ids,
            jacobi_joint_ids=self.right_jacobi_joint_ids,
            target_world=torch.as_tensor(spec.right_target, dtype=torch.float32, device=self.env.device)[None, :],
            speed_scale=spec.speed_scale,
            jitter_scale=spec.jitter_scale,
        )
        return torch.cat([left_action, right_action], dim=-1).detach().cpu().numpy()

    def solve_joint_targets(self, *, left_target: np.ndarray, right_target: np.ndarray, speed_scale: float) -> torch.Tensor:
        """Return one full 14D joint-position target for a tabletop-safe pre-positioning step."""

        _, left_joint_pos_des = self._solve_arm(
            controller=self.left_controller,
            tcp_frame_idx=self.left_tcp_frame_idx,
            jacobi_body_idx=self.left_jacobi_body_idx,
            joint_ids=self.left_joint_ids,
            jacobi_joint_ids=self.left_jacobi_joint_ids,
            target_world=torch.as_tensor(left_target, dtype=torch.float32, device=self.env.device)[None, :],
            speed_scale=speed_scale,
            jitter_scale=0.0,
        )
        _, right_joint_pos_des = self._solve_arm(
            controller=self.right_controller,
            tcp_frame_idx=self.right_tcp_frame_idx,
            jacobi_body_idx=self.right_jacobi_body_idx,
            joint_ids=self.right_joint_ids,
            jacobi_joint_ids=self.right_jacobi_joint_ids,
            target_world=torch.as_tensor(right_target, dtype=torch.float32, device=self.env.device)[None, :],
            speed_scale=speed_scale,
            jitter_scale=0.0,
        )
        joint_pos = self.env._robot.data.joint_pos.clone()
        joint_pos[:, self.left_joint_ids] = left_joint_pos_des
        joint_pos[:, self.right_joint_ids] = right_joint_pos_des
        return joint_pos

    def _solve_arm(
        self,
        *,
        controller,
        tcp_frame_idx: int,
        jacobi_body_idx: int,
        joint_ids,
        jacobi_joint_ids,
        target_world: torch.Tensor,
        speed_scale: float,
        jitter_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current_pos = self.env._tcp_frames.data.target_pos_w[:, tcp_frame_idx]
        current_quat = self.env._tcp_frames.data.target_quat_w[:, tcp_frame_idx]
        task_error = target_world - current_pos
        max_step = 0.020 + 0.018 * float(speed_scale)
        desired_pos = current_pos + _clip_xyz_step(task_error, max_norm=max_step)
        if jitter_scale > 0.0:
            # Jitter stays in task space and is kept deliberately tiny so it
            # adds data diversity without overpowering the IK command.
            desired_pos[:, 0] += float(jitter_scale) * 0.003
            desired_pos[:, 1] -= float(jitter_scale) * 0.002
        controller.set_command(desired_pos, ee_quat=current_quat)
        jacobian = self.env._robot.root_physx_view.get_jacobians()[:, jacobi_body_idx, :, jacobi_joint_ids]
        joint_pos = self.env._robot.data.joint_pos[:, joint_ids]
        joint_pos_des = controller.compute(current_pos, current_quat, jacobian, joint_pos)
        normalized_delta = (joint_pos_des - joint_pos) / self.delta_scale
        return normalized_delta.clamp(-1.0, 1.0), joint_pos_des


def _disable_default_task_success(env) -> None:
    """Collector-specific override: do not auto-reset on the env's cube task."""

    def _collector_dones(self):
        false = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return false, time_out

    env._get_dones = types.MethodType(_collector_dones, env)


def _get_safe_chain_body_ids(env) -> dict[str, tuple[int, ...]]:
    """Resolve and cache the arm-link ids used for tabletop-safe start checks."""

    if not hasattr(env, "_closing_in_safe_chain_body_ids"):
        left_ids = env._robot.find_bodies(list(LEFT_SAFE_CHAIN_LINK_NAMES))[0]
        right_ids = env._robot.find_bodies(list(RIGHT_SAFE_CHAIN_LINK_NAMES))[0]
        env._closing_in_safe_chain_body_ids = {
            "left": tuple(int(i) for i in left_ids),
            "right": tuple(int(i) for i in right_ids),
        }
    return env._closing_in_safe_chain_body_ids


def _is_safe_tcp_start(env, *, table_top_z: float, table_front_x: float, table_back_x: float) -> bool:
    left_tcp = env._tcp_frames.data.target_pos_w[0, env._left_tcp_idx]
    right_tcp = env._tcp_frames.data.target_pos_w[0, env._right_tcp_idx]
    safe_chain_body_ids = _get_safe_chain_body_ids(env)
    left_chain_ids = safe_chain_body_ids["left"]
    right_chain_ids = safe_chain_body_ids["right"]
    left_chain_pos = env._robot.data.body_link_pos_w[0, list(left_chain_ids)]
    right_chain_pos = env._robot.data.body_link_pos_w[0, list(right_chain_ids)]
    min_clearance_z = table_top_z + 0.06
    min_chain_z = table_top_z + 0.02
    # "Safe start" should mean "clearly above the tabletop and not hidden deep
    # behind the robot", not "already fully inside the table workspace". The
    # closing-in rollout itself will move the arms forward. Requiring both TCPs
    # to already cross the front edge rejected valid tabletop-above starts.
    min_forward_x = table_front_x - 0.15
    max_forward_x = table_back_x - 0.10
    return bool(
        (left_tcp[2] > min_clearance_z)
        and (right_tcp[2] > min_clearance_z)
        and torch.all(left_chain_pos[:, 2] > min_chain_z)
        and torch.all(right_chain_pos[:, 2] > min_chain_z)
        and (left_tcp[0] > min_forward_x)
        and (right_tcp[0] > min_forward_x)
        and (left_tcp[0] < max_forward_x)
        and (right_tcp[0] < max_forward_x)
    )


def _sample_safe_home_joint_pos(
    env,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Sample a tabletop-safe start pose from a small library of natural seeds.

    A single home seed made the IK settle into nearly identical shoulder/elbow
    branches. We instead choose between several curated natural tabletop seeds,
    then add bounded noise around that seed.
    """

    joint_pos = env._robot.data.joint_pos.clone()
    left_seed_index = int(rng.integers(0, len(LEFT_SAFE_TABLETOP_HOME_POSES)))
    right_seed_index = int(rng.integers(0, len(RIGHT_SAFE_TABLETOP_HOME_POSES)))
    left_base = torch.as_tensor(LEFT_SAFE_TABLETOP_HOME_POSES[left_seed_index], dtype=torch.float32, device=env.device)
    right_base = torch.as_tensor(RIGHT_SAFE_TABLETOP_HOME_POSES[right_seed_index], dtype=torch.float32, device=env.device)
    # Use the live Isaac probe result rather than assumed URDF semantics:
    # around reset, the main visible posture levers are left {0, 2, 6} and
    # right {7, 9, 11}. Sample around those directly so safe-start diversity
    # shows up in the shoulder/upper-arm chain instead of only in joints that
    # barely move the TCP.
    left_noise = np.array(
        [
            rng.uniform(-0.18, 0.18),
            rng.uniform(-0.04, 0.04),
            rng.uniform(-0.22, 0.22),
            rng.uniform(-0.06, 0.06),
            rng.uniform(-0.06, 0.06),
            rng.uniform(-0.06, 0.06),
            rng.uniform(-0.32, 0.32),
        ],
        dtype=np.float32,
    )
    # Right-arm noise stays narrower around the tabletop templates because the
    # URDF's asymmetric joint1/joint2 limits make it much easier to fall into a
    # low tucked-under-table branch if we perturb too aggressively.
    right_noise = np.array(
        [
            rng.uniform(-0.12, 0.12),
            rng.uniform(-0.12, 0.12),
            rng.uniform(-0.08, 0.08),
            rng.uniform(-0.06, 0.06),
            rng.uniform(-0.15, 0.15),
            rng.uniform(-0.04, 0.04),
            rng.uniform(-0.04, 0.04),
        ],
        dtype=np.float32,
    )
    noise = torch.as_tensor(np.concatenate([left_noise, right_noise]), dtype=torch.float32, device=env.device)
    base = torch.cat([left_base, right_base], dim=0)
    sampled = torch.clamp(base + noise, env._joint_lower_limits, env._joint_upper_limits)
    joint_pos[0, env._arm_joint_ids] = sampled
    return joint_pos, sampled.clone(), left_seed_index, right_seed_index


def _reinject_proximal_posture(
    current_joint_pos: torch.Tensor,
    sampled_seed: torch.Tensor,
    *,
    left_blend: float,
    right_blend: float,
) -> torch.Tensor:
    """Blend shoulder/elbow joints back toward the sampled natural seed.

    Position-only IK is free to reuse the same redundant branch over and over.
    A stronger proximal reinjection keeps the shoulder and upper-arm posture tied
    to the chosen natural seed while leaving enough freedom for TCP motion.
    """

    blended = current_joint_pos.clone()
    for joint_idx in LEFT_PROXIMAL_LOCAL_IDS:
        blended[:, joint_idx] = (
            (1.0 - left_blend) * blended[:, joint_idx] + left_blend * sampled_seed[joint_idx]
        )
    for joint_idx in RIGHT_PROXIMAL_LOCAL_IDS:
        blended[:, joint_idx] = (
            (1.0 - right_blend) * blended[:, joint_idx] + right_blend * sampled_seed[joint_idx]
        )
    return blended


def _lift_right_arm_out_of_table(
    joint_pos_des: torch.Tensor,
    env,
) -> torch.Tensor:
    """Bias the right arm toward a higher, more forward tabletop branch.

    The current articulation consistently falls into a low right-arm branch
    during safe-start. Use a small joint-space bias on the empirically active
    right-arm joints before writing targets to sim.
    """

    adjusted = joint_pos_des.clone()
    right = adjusted[:, env._arm_joint_ids[7:]]
    # Explicitly lift the visually important shoulder/elbow chain.
    right[:, 0] = torch.clamp(right[:, 0] + 0.18, env._joint_lower_limits[7], env._joint_upper_limits[7])
    right[:, 1] = torch.clamp(right[:, 1] + 0.20, env._joint_lower_limits[8], env._joint_upper_limits[8])
    right[:, 3] = torch.clamp(right[:, 3] + 0.22, env._joint_lower_limits[10], env._joint_upper_limits[10])
    right[:, 4] = torch.clamp(right[:, 4] - 0.14, env._joint_lower_limits[11], env._joint_upper_limits[11])
    adjusted[:, env._arm_joint_ids[7:]] = right
    return adjusted


def _drive_right_arm_to_tabletop_template(
    env,
    current_joint_pos: torch.Tensor,
    sampled_seed: torch.Tensor,
    *,
    steps: int = 12,
) -> torch.Tensor:
    """Hard-reset the right arm toward a tabletop-safe joint template.

    The right arm repeatedly falls into a low IK branch even when its target TCP
    is above the table. When that happens, stop asking IK to discover the
    tabletop posture on its own; directly interpolate the right-arm joints back
    toward the sampled tabletop seed, with a small extra lift bias on the main
    active joints.
    """

    target = current_joint_pos.clone()
    right_target = sampled_seed[7:].clone()
    # Do not trust the noisy sampled seed for the right arm once we already
    # know it collapsed into the low branch. Snap it toward a more explicit
    # tabletop posture: higher shoulder yaw, a much less tucked shoulder-2, a
    # clearly bent main elbow joint, and a less tucked wrist chain.
    tabletop_template = torch.tensor(
        [1.60, -0.02, -0.10, 1.38, -0.16, 0.20, 0.00],
        dtype=torch.float32,
        device=env.device,
    )
    right_target = 0.25 * right_target + 0.75 * tabletop_template
    right_target[0] = torch.clamp(right_target[0], env._joint_lower_limits[7], env._joint_upper_limits[7])
    right_target[1] = torch.clamp(right_target[1], env._joint_lower_limits[8], env._joint_upper_limits[8])
    right_target[2] = torch.clamp(right_target[2], env._joint_lower_limits[9], env._joint_upper_limits[9])
    right_target[3] = torch.clamp(right_target[3], env._joint_lower_limits[10], env._joint_upper_limits[10])
    right_target[4] = torch.clamp(right_target[4], env._joint_lower_limits[11], env._joint_upper_limits[11])
    right_target[5] = torch.clamp(right_target[5], env._joint_lower_limits[12], env._joint_upper_limits[12])
    right_target[6] = torch.clamp(right_target[6], env._joint_lower_limits[13], env._joint_upper_limits[13])
    target[:, env._arm_joint_ids[7:]] = right_target

    env_ids = torch.tensor([0], device=env.device, dtype=torch.long)
    zero_action = torch.zeros((env.num_envs, env.action_dim), device=env.device)
    start = current_joint_pos.clone()
    for alpha in torch.linspace(0.15, 1.0, steps, device=env.device):
        blended = start.clone()
        blended[:, env._arm_joint_ids[7:]] = (
            (1.0 - alpha) * start[:, env._arm_joint_ids[7:]]
            + alpha * target[:, env._arm_joint_ids[7:]]
        )
        blended[:, env._arm_joint_ids[7:]] = torch.clamp(
            blended[:, env._arm_joint_ids[7:]],
            env._joint_lower_limits[7:],
            env._joint_upper_limits[7:],
        )
        env._robot.set_joint_position_target(blended, env_ids=env_ids)
        env._joint_targets[:] = blended
        env.step(zero_action)
    return env._robot.data.joint_pos.clone()


def _sample_start_tcp_targets(
    spec,
    rng: np.random.Generator,
    *,
    table_top_z: float,
    table_front_x: float,
    table_back_x: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample tabletop-safe initial TCP goals that sit behind/outside the task target.

    A single fixed home pose makes the rollout too local: the IK teacher can
    satisfy most of the remaining task error with wrist/distal joints. By
    sampling a task-conditioned start waypoint behind and outside the final
    target, the robot has to move its shoulder/elbow chain to enter the
    closing-in region.
    """

    left_start = spec.left_target.astype(np.float32).copy()
    right_start = spec.right_target.astype(np.float32).copy()

    rear_offset = float(rng.uniform(0.12, 0.20))
    left_start[0] -= rear_offset
    right_start[0] -= rear_offset

    if spec.target_mode == "center":
        outward = float(rng.uniform(0.14, 0.24))
        left_start[1] += outward
        right_start[1] -= outward
    else:
        extra_outward = float(rng.uniform(0.05, 0.12))
        left_start[1] += extra_outward
        right_start[1] -= extra_outward

    lift = float(rng.uniform(0.03, 0.08))
    right_extra_lift = float(rng.uniform(0.08, 0.14))
    left_start[2] = max(left_start[2] + lift, table_top_z + 0.12)
    right_start[2] = max(right_start[2] + lift + right_extra_lift, table_top_z + 0.20)

    min_x = table_front_x + 0.04
    max_x = table_back_x - 0.14
    left_start[0] = float(np.clip(left_start[0], min_x, max_x))
    right_start[0] = float(np.clip(right_start[0], min_x, max_x))

    left_start[1] = float(np.clip(left_start[1], -0.42, 0.42))
    right_start[1] = float(np.clip(right_start[1], -0.42, 0.42))
    return left_start, right_start


def _set_manual_scene_state(
    env,
    policy: ClosingInIKPolicy,
    spec,
    rng: np.random.Generator,
    *,
    table_top_z: float,
    table_front_x: float,
    table_back_x: float,
) -> None:
    """Drive the robot into a tabletop-safe, task-conditioned closing-in start pose."""

    env_ids = torch.tensor([0], device=env.device, dtype=torch.long)

    target_state = env._target_zone.data.default_root_state.clone()
    target_state[0, :3] += env.scene.env_origins[0]
    target_state[0, :3] = torch.as_tensor(spec.center_target, device=env.device)
    target_state[0, 7:] = 0.0
    env._target_zone.write_root_pose_to_sim(target_state[:, :7], env_ids=env_ids)
    env._target_zone.write_root_velocity_to_sim(target_state[:, 7:], env_ids=env_ids)

    # Keep the cube away from the center so this dataset stays focused on
    # closing-in motion rather than object interaction.
    object_state = env._object.data.default_root_state.clone()
    object_state[0, :3] += env.scene.env_origins[0]
    object_state[0, 0] += 0.12
    object_state[0, 1] += -0.18
    object_state[0, 7:] = 0.0
    env._object.write_root_pose_to_sim(object_state[:, :7], env_ids=env_ids)
    env._object.write_root_velocity_to_sim(object_state[:, 7:], env_ids=env_ids)

    env._last_action[:] = 0.0
    env._robot.reset(env_ids)
    zero_action = torch.zeros((env.num_envs, env.action_dim), device=env.device)
    joint_pos, sampled_seed, left_seed_index, right_seed_index = _sample_safe_home_joint_pos(env, rng)
    joint_vel = torch.zeros_like(env._robot.data.joint_vel)
    env._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
    env._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    env._joint_targets[:] = joint_pos
    env._last_action[:] = 0.0
    env._robot.reset(env_ids)
    env.step(zero_action)

    left_start_target, right_start_target = _sample_start_tcp_targets(
        spec,
        rng,
        table_top_z=table_top_z,
        table_front_x=table_front_x,
        table_back_x=table_back_x,
    )
    # Use a more aggressive pre-positioning pass than the rollout teacher.
    # The goal here is not fine control; it is to drive the robot into a
    # task-conditioned tabletop start pose where shoulder and elbow joints have
    # already participated, instead of leaving the whole burden to distal joints.
    for _ in range(20):
        joint_pos_des = policy.solve_joint_targets(
            left_target=left_start_target,
            right_target=right_start_target,
            speed_scale=2.4,
        )
        joint_pos_des[:, env._arm_joint_ids] = _reinject_proximal_posture(
            joint_pos_des[:, env._arm_joint_ids],
            sampled_seed,
            left_blend=0.62,
            right_blend=0.12,
        )
        joint_pos_des = _lift_right_arm_out_of_table(joint_pos_des, env)
        joint_pos_des[:, env._arm_joint_ids] = torch.clamp(
            joint_pos_des[:, env._arm_joint_ids],
            env._joint_lower_limits,
            env._joint_upper_limits,
        )
        env._robot.set_joint_position_target(joint_pos_des, env_ids=env_ids)
        env._joint_targets[:] = joint_pos_des
        env.step(zero_action)

    if not _is_safe_tcp_start(
        env,
        table_top_z=table_top_z,
        table_front_x=table_front_x,
        table_back_x=table_back_x,
    ):
        # If the right arm is still trapped in a low branch, stop relying on
        # IK alone. First pull the right arm back toward a curated tabletop
        # template in joint space, then do a short TCP refinement pass.
        joint_pos_now = _drive_right_arm_to_tabletop_template(env, env._robot.data.joint_pos.clone(), sampled_seed)

        # Then give the right arm a short TCP recovery pass with only mild
        # posture reinjection so it can stay near the tabletop seed branch.
        recovery_right_target = right_start_target.copy()
        recovery_right_target[0] = float(np.clip(recovery_right_target[0] + 0.10, table_front_x + 0.10, table_back_x - 0.10))
        recovery_right_target[2] = max(float(recovery_right_target[2] + 0.18), table_top_z + 0.32)
        for _ in range(12):
            joint_pos_des = policy.solve_joint_targets(
                left_target=left_start_target,
                right_target=recovery_right_target,
                speed_scale=2.8,
            )
            joint_pos_des[:, env._arm_joint_ids] = _reinject_proximal_posture(
                joint_pos_des[:, env._arm_joint_ids],
                sampled_seed,
                left_blend=0.55,
                right_blend=0.22,
            )
            joint_pos_des = _lift_right_arm_out_of_table(joint_pos_des, env)
            joint_pos_des[:, env._arm_joint_ids[7:]] = 0.55 * joint_pos_des[:, env._arm_joint_ids[7:]] + 0.45 * joint_pos_now[:, env._arm_joint_ids[7:]]
            joint_pos_des[:, env._arm_joint_ids] = torch.clamp(
                joint_pos_des[:, env._arm_joint_ids],
                env._joint_lower_limits,
                env._joint_upper_limits,
            )
            env._robot.set_joint_position_target(joint_pos_des, env_ids=env_ids)
            env._joint_targets[:] = joint_pos_des
            env.step(zero_action)

    left_tcp = env._tcp_frames.data.target_pos_w[0, env._left_tcp_idx].detach().cpu().numpy()
    right_tcp = env._tcp_frames.data.target_pos_w[0, env._right_tcp_idx].detach().cpu().numpy()
    safe_chain_body_ids = _get_safe_chain_body_ids(env)
    left_chain_ids = safe_chain_body_ids["left"]
    right_chain_ids = safe_chain_body_ids["right"]
    left_chain_z = env._robot.data.body_link_pos_w[0, list(left_chain_ids), 2].detach().cpu().numpy()
    right_chain_z = env._robot.data.body_link_pos_w[0, list(right_chain_ids), 2].detach().cpu().numpy()
    arm_joint_pos = env._robot.data.joint_pos[0, env._arm_joint_ids].detach().cpu().numpy()
    print(
        "[INFO] safe-start check "
        f"left_seed_index={left_seed_index} "
        f"right_seed_index={right_seed_index} "
        f"left_start_target={np.round(left_start_target, 4).tolist()} "
        f"right_start_target={np.round(right_start_target, 4).tolist()} "
        f"left_tcp={np.round(left_tcp, 4).tolist()} "
        f"right_tcp={np.round(right_tcp, 4).tolist()} "
        f"left_chain_z={np.round(left_chain_z, 4).tolist()} "
        f"right_chain_z={np.round(right_chain_z, 4).tolist()} "
        f"table_top_z={table_top_z:.4f} "
        f"arm_joint_pos={np.round(arm_joint_pos, 4).tolist()}"
    )
    if _is_safe_tcp_start(
        env,
        table_top_z=table_top_z,
        table_front_x=table_front_x,
        table_back_x=table_back_x,
    ):
        return

    raise RuntimeError(
        "Failed to drive both TCPs into a tabletop-safe closing-in home region. "
        "Adjust the safe start waypoints or IK warmup steps."
    )


def _warmup_env(env, warmup_steps: int) -> None:
    zero_action = torch.zeros((env.num_envs, env.action_dim), device=env.device)
    for _ in range(warmup_steps):
        env.step(zero_action)


def _episode_payload(summary, *, spec, success, current_hold, fps):
    return {
        "task": {
            "name": summary["task_name"],
            "prompt": spec.prompt,
            "center_target": spec.center_target.astype(np.float32),
            "left_target": spec.left_target.astype(np.float32),
            "right_target": spec.right_target.astype(np.float32),
        },
        "observation": {
            "state": np.stack(summary["states"], axis=0).astype(np.float32),
            "external_image": np.stack(summary["external_images"], axis=0).astype(np.uint8),
            "left_wrist_image": np.stack(summary["left_wrist_images"], axis=0).astype(np.uint8),
            "right_wrist_image": np.stack(summary["right_wrist_images"], axis=0).astype(np.uint8),
        },
        "actions": np.stack(summary["actions"], axis=0).astype(np.float32),
        "reward": np.asarray(summary["rewards"], dtype=np.float32),
        "done": np.asarray(summary["dones"], dtype=np.bool_),
        "success": np.asarray([success], dtype=np.bool_),
        "timestamp": (np.arange(len(summary["actions"]), dtype=np.float32) / float(fps)),
        "metrics": {
            "left_target_dist": np.asarray(summary["left_distances"], dtype=np.float32),
            "right_target_dist": np.asarray(summary["right_distances"], dtype=np.float32),
            "hold_count": np.asarray([current_hold], dtype=np.int32),
        },
        "initial": {
            "arm_joint_pos": summary["initial_arm_joint_pos"].astype(np.float32),
            "object_root_state": summary["initial_object_root_state"].astype(np.float32),
            "target_root_state": summary["initial_target_root_state"].astype(np.float32),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_file", type=Path, required=True, help="Output HDF5 file.")
    parser.add_argument("--num_episodes", type=int, default=16, help="Number of episodes to record.")
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=None,
        help="Maximum episode attempts before stopping. Defaults to 3x num_episodes.",
    )
    parser.add_argument("--max_steps", type=int, default=80, help="Maximum recorded steps per episode.")
    parser.add_argument("--warmup_steps", type=int, default=5, help="Warmup steps before recording frames.")
    parser.add_argument("--fps", type=int, default=10, help="Nominal dataset FPS used for timestamps.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for episode sampling.")
    parser.add_argument("--resume", action="store_true", help="Append episodes to an existing HDF5 file.")
    parser.add_argument("--num_envs", type=int, default=1, help="First version expects a single environment.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if hasattr(args, "enable_cameras"):
        args.enable_cameras = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        import h5py  # noqa: F401

        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

        from zhishu_dualarm_lab.tasks.dualarm_tabletop.constants import (
            ARM_ACTION_DELTA_SCALE,
            LEFT_TCP_LINK_NAME,
            RIGHT_TCP_LINK_NAME,
            TABLE_POS,
            TABLE_SIZE,
        )
        from zhishu_dualarm_lab.tasks.dualarm_tabletop.env import ZhishuDualArmTabletopEnv
        from zhishu_dualarm_lab.tasks.dualarm_tabletop.env_cfg import ZhishuDualArmTabletopEnvCfg
        from zhishu_dualarm_lab.utils.closing_in_dataset import (
            TASK_NAME,
            compute_success,
            create_or_open_dataset,
            parse_policy_state,
            sample_episode_spec,
            write_episode,
        )

        rng = np.random.default_rng(args.seed)

        cfg = ZhishuDualArmTabletopEnvCfg()
        cfg.scene.num_envs = args.num_envs
        cfg.episode_length_s = max(cfg.episode_length_s, args.max_steps * cfg.sim.dt * cfg.decimation + 2.0)
        env = ZhishuDualArmTabletopEnv(cfg=cfg, render_mode="human" if not args.headless else None)
        _disable_default_task_success(env)
        policy = ClosingInIKPolicy(
            env=env,
            left_body_name=LEFT_TCP_LINK_NAME,
            right_body_name=RIGHT_TCP_LINK_NAME,
            diff_ik_controller_cfg_cls=DifferentialIKControllerCfg,
            diff_ik_controller_cls=DifferentialIKController,
            delta_scale=ARM_ACTION_DELTA_SCALE,
        )

        file_handle, data_group = create_or_open_dataset(args.dataset_file, resume=args.resume)
        print(
            f"[INFO] collecting {TASK_NAME} to {args.dataset_file} "
            f"episodes={args.num_episodes} max_steps={args.max_steps}"
        )

        recorded_episodes = 0
        attempt_index = 0
        max_attempts = int(args.max_attempts) if args.max_attempts is not None else max(3 * args.num_episodes, 1)
        while recorded_episodes < args.num_episodes and attempt_index < max_attempts:
            try:
                episode_index = recorded_episodes
                print(
                    f"[INFO] attempt={attempt_index} episode={episode_index} "
                    "reset begin"
                )
                env.reset()
                print(f"[INFO] attempt={attempt_index} episode={episode_index} reset done")
                table_top_z = float(TABLE_POS[2] + 0.5 * TABLE_SIZE[2])
                table_front_x = float(TABLE_POS[0] - 0.5 * TABLE_SIZE[0])
                table_back_x = float(TABLE_POS[0] + 0.5 * TABLE_SIZE[0])
                base_target = env._target_zone.data.default_root_state[0, :3].detach().cpu().numpy()
                # Closing-in targets should live above the table, not inherit an
                # arbitrary reset-time TCP height that might be below the tabletop.
                base_target[0] = float(np.clip(base_target[0], table_front_x + 0.18, table_back_x - 0.18))
                base_target[2] = table_top_z + 0.18
                spec = sample_episode_spec(rng, episode_index, base_target=base_target)
                print(
                    f"[INFO] attempt={attempt_index} episode={episode_index} spec "
                    f"mode={spec.target_mode} "
                    f"prompt={spec.prompt!r} "
                    f"center_target={np.round(spec.center_target, 4).tolist()} "
                    f"left_target={np.round(spec.left_target, 4).tolist()} "
                    f"right_target={np.round(spec.right_target, 4).tolist()} "
                    f"speed={spec.speed_scale:.2f} hold={spec.hold_steps} jitter={spec.jitter_scale:.3f}"
                )
                print(f"[INFO] attempt={attempt_index} episode={episode_index} safe-start begin")
                _set_manual_scene_state(
                    env,
                    policy,
                    spec,
                    rng,
                    table_top_z=table_top_z,
                    table_front_x=table_front_x,
                    table_back_x=table_back_x,
                )
                print("[INFO] safe-start finished, entering warmup")
                _warmup_env(env, args.warmup_steps)
                print("[INFO] warmup finished, entering episode rollout")

                summary = {
                    "task_name": TASK_NAME,
                    "states": [],
                    "external_images": [],
                    "left_wrist_images": [],
                    "right_wrist_images": [],
                    "actions": [],
                    "rewards": [],
                    "dones": [],
                    "left_distances": [],
                    "right_distances": [],
                    "initial_arm_joint_pos": env._robot.data.joint_pos[0, env._arm_joint_ids].detach().cpu().numpy(),
                    "initial_object_root_state": env._object.data.root_state_w[0].detach().cpu().numpy(),
                    "initial_target_root_state": env._target_zone.data.root_state_w[0].detach().cpu().numpy(),
                }

                success = False
                hold_counter = 0
                start_time = time.monotonic()
                for step_idx in range(args.max_steps):
                    if step_idx == 0:
                        print("[INFO] rollout step=0 fetching policy input")
                    policy_input = env.get_policy_input()
                    if not policy_input:
                        raise RuntimeError("Policy input is empty during closing-in collection.")

                    state = np.asarray(policy_input["observation/state"], dtype=np.float32).copy()
                    summary["states"].append(state)
                    summary["external_images"].append(
                        np.asarray(policy_input["observation/external_image"], dtype=np.uint8)
                    )
                    summary["left_wrist_images"].append(
                        np.asarray(policy_input["observation/left_wrist_image"], dtype=np.uint8)
                    )
                    summary["right_wrist_images"].append(
                        np.asarray(policy_input["observation/right_wrist_image"], dtype=np.uint8)
                    )

                    if step_idx == 0:
                        print("[INFO] rollout step=0 solving teacher action")
                    action_chunk = policy.infer(spec, step_idx)
                    first_action = env.apply_policy_output(action_chunk)
                    summary["actions"].append(first_action.detach().cpu().numpy().reshape(-1))

                    if step_idx == 0:
                        print("[INFO] rollout step=0 stepping env with buffered action")
                    obs, _, _, _, _ = env.step(env.consume_action_plan_step())
                    del obs

                    if step_idx == 0:
                        print("[INFO] rollout step=0 fetching post-step policy input")
                    next_policy_input = env.get_policy_input()
                    if not next_policy_input:
                        raise RuntimeError("Policy input is empty after stepping the environment.")
                    next_state = np.asarray(next_policy_input["observation/state"], dtype=np.float32)
                    task_success, left_dist, right_dist = compute_success(next_state, spec)
                    parsed = parse_policy_state(next_state)
                    mean_dist = 0.5 * (left_dist + right_dist)
                    motion_penalty = 0.01 * float(np.linalg.norm(parsed["joint_vel"]))
                    reward = float(np.exp(-12.0 * mean_dist) - motion_penalty)
                    if task_success:
                        hold_counter += 1
                    else:
                        hold_counter = 0
                    success = hold_counter >= spec.hold_steps

                    summary["left_distances"].append(left_dist)
                    summary["right_distances"].append(right_dist)
                    summary["rewards"].append(reward)
                    summary["dones"].append(success)

                    if success:
                        break

                payload = _episode_payload(
                    summary,
                    spec=spec,
                    success=success,
                    current_hold=hold_counter,
                    fps=args.fps,
                )
                episode_name = write_episode(
                    data_group,
                    spec=spec,
                    payload=payload,
                    success=success,
                    num_samples=len(summary["actions"]),
                )
                file_handle.flush()
                print(
                    f"[INFO] {episode_name} mode={spec.target_mode} prompt={spec.prompt!r} "
                    f"steps={len(summary['actions'])} success={success} "
                    f"speed={spec.speed_scale:.2f} hold={spec.hold_steps} jitter={spec.jitter_scale:.3f} "
                    f"duration_s={time.monotonic() - start_time:.2f}"
                )
                recorded_episodes += 1
            except BaseException as exc:
                print(
                    f"[ERROR] attempt={attempt_index} episode={recorded_episodes} "
                    f"failed with {type(exc).__name__}: {exc}"
                )
                traceback.print_exc()
            finally:
                attempt_index += 1

        print(
            f"[INFO] closing-in collection finished "
            f"recorded={recorded_episodes}/{args.num_episodes} attempts={attempt_index}/{max_attempts}"
        )
    finally:
        if "file_handle" in locals():
            file_handle.close()
        if "env" in locals():
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
