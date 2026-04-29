#!/usr/bin/env python3
"""Drive the Zhishu dual-arm robot toward a structured closing-in target.

This v2 variant keeps the original task logic, but fixes one confirmed control
bug from the first script:
- the robot_only env exposes arm joints in an interleaved order
- therefore left/right arms cannot be split with [:7] and [7:]
- IK must gather left/right joint ids by name, then scatter actions back into
  the env action order
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

TARGET_MARKER_SIZE = (0.035, 0.035, 0.035)
LEFT_ARM_JOINT_NAMES = tuple(f"left_joint{i}" for i in range(1, 8))
RIGHT_ARM_JOINT_NAMES = tuple(f"right_joint{i}" for i in range(1, 8))
DISK_DIAMETER_M = 0.178
DISK_THICKNESS_M = 0.010
DISK_RADIUS_M = 0.5 * DISK_DIAMETER_M

# 最小侵入版手心朝向保持：
# - 主 IK 仍然负责 7 轴位置求解
# - 但在动作输出最后，对 joint5 做“软拉回”
# - joint5 更接近沿小臂轴的滚转，用它比 joint7 更适合调手心朝向
# - 如果你要对着 GUI 调整手心向上角度，优先改下面这两个角度
# - SOFTNESS 越大，joint5 越倾向于跟随 IK；越小，越倾向于保持目标角
#
# 单位：度
LEFT_HAND_JOINT5_BIAS_DEG = -1150.0
RIGHT_HAND_JOINT5_BIAS_DEG = 1150.0
JOINT5_BIAS_SOFTNESS = 0.01


def _clip_xyz_step(err: torch.Tensor, max_norm: float) -> torch.Tensor:
    norms = torch.linalg.norm(err, dim=-1, keepdim=True).clamp_min(1e-6)
    scale = torch.clamp(max_norm / norms, max=1.0)
    return err * scale


def _tcp_pos(env) -> tuple[np.ndarray, np.ndarray]:
    left = env._tcp_frames.data.target_pos_w[0, env._left_tcp_idx].detach().cpu().numpy().astype(np.float32)
    right = env._tcp_frames.data.target_pos_w[0, env._right_tcp_idx].detach().cpu().numpy().astype(np.float32)
    return left, right


def _settle(env, steps: int) -> None:
    zero_action = torch.zeros((env.num_envs, env.action_dim), device=env.device)
    for _ in range(max(0, steps)):
        env.step(zero_action)


def _build_target_markers():
    from isaaclab.markers import VisualizationMarkers
    from isaaclab.markers.config import CUBOID_MARKER_CFG

    marker_cfg = CUBOID_MARKER_CFG.replace(prim_path="/Visuals/ZhishuRobotOnlyClosingInTargets")
    marker_cfg.markers["cuboid"].size = TARGET_MARKER_SIZE
    return VisualizationMarkers(marker_cfg)


def _build_disk_marker():
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/ZhishuRobotOnlyClosingInDisk",
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


def _disk_center_from_target_body(center: np.ndarray, forward_axis_xy: np.ndarray) -> np.ndarray:
    """Compute the horizontal disk center.

    约定：
    - 共同目标体由两个红色目标点及其 marker 尺寸隐式定义
    - 前侧面指向机器人正前方
    - 圆盘水平放置，因此圆柱轴沿世界 Z
    - 圆盘侧面与共同目标体前侧面相切
    """

    target_body_half_depth = 0.5 * float(TARGET_MARKER_SIZE[0])
    disk_forward_offset = target_body_half_depth + DISK_RADIUS_M
    disk_center = center.copy()
    disk_center[0] += float(forward_axis_xy[0] * disk_forward_offset)
    disk_center[1] += float(forward_axis_xy[1] * disk_forward_offset)
    return disk_center.astype(np.float32)


def _sample_closing_in_targets(
    *,
    env,
    rng: np.random.Generator,
    left_start: np.ndarray,
    right_start: np.ndarray,
    left_shoulder_body_id: int,
    right_shoulder_body_id: int,
    waist_body_id: int,
    head_body_id: int,
    target_gap_m: float,
    target_gap_jitter_m: float,
    min_move: float,
    max_attempts: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Sample a structured pair of targets for a shared target body."""

    left_shoulder = env._robot.data.body_link_pos_w[0, left_shoulder_body_id].detach().cpu().numpy().astype(np.float32)
    right_shoulder = env._robot.data.body_link_pos_w[0, right_shoulder_body_id].detach().cpu().numpy().astype(np.float32)
    waist_pos = env._robot.data.body_link_pos_w[0, waist_body_id].detach().cpu().numpy().astype(np.float32)
    head_pos = env._robot.data.body_link_pos_w[0, head_body_id].detach().cpu().numpy().astype(np.float32)

    shoulder_center = 0.5 * (left_shoulder + right_shoulder)
    shoulder_vec_xy = (left_shoulder - right_shoulder)[:2].astype(np.float64)
    shoulder_width = float(np.linalg.norm(shoulder_vec_xy))
    if shoulder_width < 1e-6:
        shoulder_axis_xy = np.array([0.0, 1.0], dtype=np.float64)
    else:
        shoulder_axis_xy = shoulder_vec_xy / shoulder_width

    forward_xy_a = np.array([-shoulder_axis_xy[1], shoulder_axis_xy[0]], dtype=np.float64)
    forward_xy_b = -forward_xy_a
    tcp_center_xy = 0.5 * (left_start[:2] + right_start[:2]).astype(np.float64)
    shoulder_center_xy = shoulder_center[:2].astype(np.float64)
    if np.dot(tcp_center_xy - shoulder_center_xy, forward_xy_a) >= np.dot(
        tcp_center_xy - shoulder_center_xy, forward_xy_b
    ):
        forward_axis_xy = forward_xy_a
    else:
        forward_axis_xy = forward_xy_b

    start_center = 0.5 * (left_start + right_start)
    start_forward_dist = float(np.dot(start_center[:2] - shoulder_center_xy, forward_axis_xy))
    # 共同目标体的后平面（靠近身体一侧）向身体外侧再推 5cm，
    # 避免目标整体刷新得过近，导致双臂几乎贴着身体起手。
    forward_min = max(0.10, start_forward_dist - 0.03 + 0.08)
    forward_max = max(forward_min + 0.08, start_forward_dist + 0.12)

    gap = float(target_gap_m + rng.uniform(-target_gap_jitter_m, target_gap_jitter_m))
    gap = max(0.08, gap)
    half_gap = 0.5 * gap

    shoulder_half_width = 0.5 * shoulder_width
    # 共同目标体整体更收在身前中间区域，两侧各额外向内收 5cm。
    lateral_margin = 0.02
    max_center_lateral = max(0.0, shoulder_half_width - half_gap - lateral_margin)

    # 共同目标体整体下移：
    # - 下平面相对原设置下移 3cm
    # - 上平面相对原设置下移 10cm
    z_low = float(waist_pos[2] + 0.12)
    z_high = float(min(shoulder_center[2] - 0.23, head_pos[2] - 0.36))
    if z_high <= z_low:
        z_mid = 0.5 * (z_low + z_high)
        z_low = z_mid - 0.01
        z_high = z_mid + 0.01

    best_left = left_start.copy()
    best_right = right_start.copy()
    best_center = start_center.copy()
    best_score = -1.0

    for _ in range(max(1, max_attempts)):
        forward_dist = rng.uniform(forward_min, forward_max)
        lateral_center = rng.uniform(-max_center_lateral, max_center_lateral) if max_center_lateral > 1e-6 else 0.0
        center_xy = shoulder_center_xy + forward_axis_xy * forward_dist + shoulder_axis_xy * lateral_center
        center_z = rng.uniform(z_low, z_high)
        center = np.array([center_xy[0], center_xy[1], center_z], dtype=np.float32)

        offset = np.array([shoulder_axis_xy[0] * half_gap, shoulder_axis_xy[1] * half_gap, 0.0], dtype=np.float32)
        left_target = center + offset
        right_target = center - offset

        left_lateral = float(np.dot(left_target[:2] - shoulder_center_xy, shoulder_axis_xy))
        right_lateral = float(np.dot(right_target[:2] - shoulder_center_xy, shoulder_axis_xy))
        if left_lateral > shoulder_half_width - lateral_margin:
            continue
        if right_lateral < -(shoulder_half_width - lateral_margin):
            continue

        left_move = float(np.linalg.norm(left_target - left_start))
        right_move = float(np.linalg.norm(right_target - right_start))
        score = min(left_move, right_move)
        if left_move >= min_move and right_move >= min_move:
            disk_center = _disk_center_from_target_body(center, forward_axis_xy)
            return (
                left_target.astype(np.float32),
                right_target.astype(np.float32),
                center.astype(np.float32),
                disk_center,
                gap,
            )
        if score > best_score:
            best_score = score
            best_left = left_target.astype(np.float32)
            best_right = right_target.astype(np.float32)
            best_center = center.astype(np.float32)

    best_disk_center = _disk_center_from_target_body(best_center, forward_axis_xy)
    return best_left, best_right, best_center, best_disk_center, gap


class DualArmPositionIK:
    """Small wrapper around Isaac Lab Differential IK for two 7-DoF arms."""

    def __init__(self, *, env, left_body_name: str, right_body_name: str, delta_scale: float, ik_lambda: float):
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

        self.env = env
        self.delta_scale = float(delta_scale)
        self.env_arm_joint_ids = list(env._arm_joint_ids)
        self.left_joint_ids = list(env._robot.find_joints(list(LEFT_ARM_JOINT_NAMES))[0])
        self.right_joint_ids = list(env._robot.find_joints(list(RIGHT_ARM_JOINT_NAMES))[0])
        env_action_index_by_joint_id = {joint_id: idx for idx, joint_id in enumerate(self.env_arm_joint_ids)}
        self.left_action_indices = [env_action_index_by_joint_id[joint_id] for joint_id in self.left_joint_ids]
        self.right_action_indices = [env_action_index_by_joint_id[joint_id] for joint_id in self.right_joint_ids]
        self.left_tcp_frame_idx = env._left_tcp_idx
        self.right_tcp_frame_idx = env._right_tcp_idx
        self.left_body_idx = env._robot.find_bodies([left_body_name])[0][0]
        self.right_body_idx = env._robot.find_bodies([right_body_name])[0][0]
        self.left_joint5_id = self.left_joint_ids[4]
        self.right_joint5_id = self.right_joint_ids[4]
        self.left_joint5_action_idx = self.left_action_indices[4]
        self.right_joint5_action_idx = self.right_action_indices[4]
        self.left_joint5_target_rad = math.radians(LEFT_HAND_JOINT5_BIAS_DEG)
        self.right_joint5_target_rad = math.radians(RIGHT_HAND_JOINT5_BIAS_DEG)
        self.joint5_bias_softness = float(np.clip(JOINT5_BIAS_SOFTNESS, 0.0, 1.0))

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

        controller_cfg = DifferentialIKControllerCfg(
            command_type="position",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": float(ik_lambda)},
        )
        self.left_controller = DifferentialIKController(controller_cfg, num_envs=env.num_envs, device=env.device)
        self.right_controller = DifferentialIKController(controller_cfg, num_envs=env.num_envs, device=env.device)

    def _solve_arm(
        self,
        *,
        controller,
        tcp_frame_idx: int,
        jacobi_body_idx: int,
        joint_ids: list[int],
        jacobi_joint_ids: list[int],
        target_world: torch.Tensor,
        max_task_step: float,
    ) -> torch.Tensor:
        current_pos = self.env._tcp_frames.data.target_pos_w[:, tcp_frame_idx]
        current_quat = self.env._tcp_frames.data.target_quat_w[:, tcp_frame_idx]
        desired_pos = current_pos + _clip_xyz_step(target_world - current_pos, max_norm=max_task_step)
        controller.set_command(desired_pos, ee_quat=current_quat)

        jacobian = self.env._robot.root_physx_view.get_jacobians()[:, jacobi_body_idx, :, jacobi_joint_ids]
        joint_pos = self.env._robot.data.joint_pos[:, joint_ids]
        joint_pos_des = controller.compute(current_pos, current_quat, jacobian, joint_pos)
        action = (joint_pos_des - joint_pos) / self.delta_scale
        return action.clamp(-1.0, 1.0)

    def infer(self, *, left_target: torch.Tensor, right_target: torch.Tensor, max_task_step: float) -> torch.Tensor:
        left_action = self._solve_arm(
            controller=self.left_controller,
            tcp_frame_idx=self.left_tcp_frame_idx,
            jacobi_body_idx=self.left_jacobi_body_idx,
            joint_ids=self.left_joint_ids,
            jacobi_joint_ids=self.left_jacobi_joint_ids,
            target_world=left_target,
            max_task_step=max_task_step,
        )
        right_action = self._solve_arm(
            controller=self.right_controller,
            tcp_frame_idx=self.right_tcp_frame_idx,
            jacobi_body_idx=self.right_jacobi_body_idx,
            joint_ids=self.right_joint_ids,
            jacobi_joint_ids=self.right_jacobi_joint_ids,
            target_world=right_target,
            max_task_step=max_task_step,
        )

        action = torch.zeros((self.env.num_envs, self.env.action_dim), dtype=torch.float32, device=self.env.device)
        action[:, self.left_action_indices] = left_action
        action[:, self.right_action_indices] = right_action

        # 最小侵入地给 joint5 加一个“软拉回”偏置：
        # joint5 仍然参与 IK，但最终动作会在“IK 输出”和“朝目标角收敛”之间做线性混合。
        # 当前 env 的动作语义是：
        #   next_target = current_joint_pos + action * delta_scale
        # 因此这里先算出一个“如果只想把 joint5 拉回目标角，需要给多大动作”，
        # 再和 IK 的原始输出按 softness 做插值。
        left_joint5_pos = self.env._robot.data.joint_pos[:, self.left_joint5_id]
        right_joint5_pos = self.env._robot.data.joint_pos[:, self.right_joint5_id]
        left_joint5_bias_action = ((self.left_joint5_target_rad - left_joint5_pos) / self.delta_scale).clamp(-1.0, 1.0)
        right_joint5_bias_action = ((self.right_joint5_target_rad - right_joint5_pos) / self.delta_scale).clamp(-1.0, 1.0)
        action[:, self.left_joint5_action_idx] = (
            self.joint5_bias_softness * action[:, self.left_joint5_action_idx]
            + (1.0 - self.joint5_bias_softness) * left_joint5_bias_action
        ).clamp(-1.0, 1.0)
        action[:, self.right_joint5_action_idx] = (
            self.joint5_bias_softness * action[:, self.right_joint5_action_idx]
            + (1.0 - self.joint5_bias_softness) * right_joint5_bias_action
        ).clamp(-1.0, 1.0)
        return action


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_rounds", type=int, default=10, help="Number of structured closing-in target pairs to execute.")
    parser.add_argument("--max_steps_per_round", type=int, default=180, help="Control steps before a round times out.")
    parser.add_argument("--settle_steps", type=int, default=12, help="Zero-action warmup steps after each reset.")
    parser.add_argument("--success_threshold", type=float, default=0.035, help="TCP distance threshold in meters.")
    parser.add_argument("--hold_steps", type=int, default=6, help="Consecutive in-threshold steps required for success.")
    parser.add_argument("--max_task_step", type=float, default=0.045, help="Max Cartesian IK command step in meters.")
    parser.add_argument("--ik_lambda", type=float, default=0.08, help="DLS IK damping lambda.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_envs", type=int, default=1, help="This debugging script is intended for one env.")
    parser.add_argument("--target_gap_m", type=float, default=0.20, help="Nominal distance between left/right target points.")
    parser.add_argument("--target_gap_jitter_m", type=float, default=0.01, help="Small symmetric gap perturbation around the nominal distance.")
    parser.add_argument("--min_separation_from_start", type=float, default=0.08, help="Minimum move distance for each TCP from its current start.")
    parser.add_argument("--target_sample_attempts", type=int, default=64)
    parser.add_argument("--log_every", type=int, default=10)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs != 1:
        raise ValueError("run_robot_only_closing_in_v2.py currently supports --num_envs 1 only.")
    if hasattr(args, "enable_cameras"):
        args.enable_cameras = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    env = None
    try:
        print("[INFO] importing robot-only task modules...", flush=True)
        from zhishu_dualarm_lab.tasks.robot_only.constants import (
            ARM_ACTION_DELTA_SCALE,
            HEAD_CAMERA_LINK_NAME,
            LEFT_TCP_LINK_NAME,
            RIGHT_TCP_LINK_NAME,
            WAIST_CAMERA_LINK_NAME,
        )
        from zhishu_dualarm_lab.tasks.robot_only.env import ZhishuDualArmRobotOnlyEnv
        from zhishu_dualarm_lab.tasks.robot_only.env_cfg import ZhishuDualArmRobotOnlyEnvCfg

        print("[INFO] creating robot-only environment...", flush=True)
        rng = np.random.default_rng(args.seed)
        torch.manual_seed(args.seed)
        cfg = ZhishuDualArmRobotOnlyEnvCfg()
        cfg.scene.num_envs = args.num_envs
        env = ZhishuDualArmRobotOnlyEnv(cfg=cfg, render_mode="human" if not args.headless else None)
        print("[INFO] resetting robot-only environment...", flush=True)
        env.reset()

        left_shoulder_body_id = env._robot.find_bodies(["left_link1"])[0][0]
        right_shoulder_body_id = env._robot.find_bodies(["right_link1"])[0][0]
        head_body_id = env._robot.find_bodies([HEAD_CAMERA_LINK_NAME])[0][0]
        waist_body_id = env._robot.find_bodies([WAIST_CAMERA_LINK_NAME])[0][0]

        print("[INFO] initializing controllers and target markers...", flush=True)
        ik = DualArmPositionIK(
            env=env,
            left_body_name=LEFT_TCP_LINK_NAME,
            right_body_name=RIGHT_TCP_LINK_NAME,
            delta_scale=ARM_ACTION_DELTA_SCALE,
            ik_lambda=args.ik_lambda,
        )
        markers = _build_target_markers()
        disk_marker = _build_disk_marker()
        marker_quats = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
            device=env.device,
        )
        disk_marker_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=env.device)

        joint_names = env._robot.joint_names
        env_action_joint_names = [joint_names[joint_id] for joint_id in ik.env_arm_joint_ids]
        left_action_joint_names = [joint_names[joint_id] for joint_id in ik.left_joint_ids]
        right_action_joint_names = [joint_names[joint_id] for joint_id in ik.right_joint_ids]
        print(
            "[INFO] v2 joint mapping "
            f"env_action_order={env_action_joint_names} "
            f"left_arm_order={left_action_joint_names} "
            f"right_arm_order={right_action_joint_names}",
            flush=True,
        )

        print(
            "[INFO] clean closing-in task started "
            f"usd={cfg.scene.robot.spawn.usd_path} "
            f"target_gap={args.target_gap_m:.3f}m±{args.target_gap_jitter_m:.3f}m"
        )
        print(
            "[INFO] joint5 soft-bias targets "
            f"left_joint5={LEFT_HAND_JOINT5_BIAS_DEG:.1f}deg "
            f"right_joint5={RIGHT_HAND_JOINT5_BIAS_DEG:.1f}deg "
            f"softness={JOINT5_BIAS_SOFTNESS:.2f}",
            flush=True,
        )

        for round_idx in range(args.num_rounds):
            env.reset()
            _settle(env, args.settle_steps)
            left_start, right_start = _tcp_pos(env)
            left_target_np, right_target_np, center_np, disk_center_np, gap = _sample_closing_in_targets(
                env=env,
                rng=rng,
                left_start=left_start,
                right_start=right_start,
                left_shoulder_body_id=left_shoulder_body_id,
                right_shoulder_body_id=right_shoulder_body_id,
                waist_body_id=waist_body_id,
                head_body_id=head_body_id,
                target_gap_m=args.target_gap_m,
                target_gap_jitter_m=args.target_gap_jitter_m,
                min_move=args.min_separation_from_start,
                max_attempts=args.target_sample_attempts,
            )
            left_target = torch.as_tensor(left_target_np, dtype=torch.float32, device=env.device).view(1, 3)
            right_target = torch.as_tensor(right_target_np, dtype=torch.float32, device=env.device).view(1, 3)
            disk_center = torch.as_tensor(disk_center_np, dtype=torch.float32, device=env.device).view(1, 3)
            markers.visualize(translations=torch.cat([left_target, right_target], dim=0), orientations=marker_quats)
            disk_marker.visualize(translations=disk_center, orientations=disk_marker_quat)

            print(
                f"[INFO] round={round_idx} "
                f"left_start={np.round(left_start, 4).tolist()} "
                f"right_start={np.round(right_start, 4).tolist()} "
                f"center_target={np.round(center_np, 4).tolist()} "
                f"disk_center={np.round(disk_center_np, 4).tolist()} "
                f"gap={gap:.4f} "
                f"left_target={np.round(left_target_np, 4).tolist()} "
                f"right_target={np.round(right_target_np, 4).tolist()}"
            )

            left_hold = 0
            right_hold = 0
            left_done = False
            right_done = False
            for step_idx in range(args.max_steps_per_round):
                action = ik.infer(
                    left_target=left_target,
                    right_target=right_target,
                    max_task_step=args.max_task_step,
                )
                if left_done:
                    action[:, ik.left_action_indices] = 0.0
                if right_done:
                    action[:, ik.right_action_indices] = 0.0
                env.step(action)
                left_now, right_now = _tcp_pos(env)
                left_dist = float(np.linalg.norm(left_now - left_target_np))
                right_dist = float(np.linalg.norm(right_now - right_target_np))

                if not left_done:
                    left_hold = left_hold + 1 if left_dist <= args.success_threshold else 0
                    left_done = left_hold >= args.hold_steps
                if not right_done:
                    right_hold = right_hold + 1 if right_dist <= args.success_threshold else 0
                    right_done = right_hold >= args.hold_steps

                if args.log_every > 0 and (step_idx % args.log_every == 0 or (left_done and right_done)):
                    print(
                        f"[INFO] round={round_idx} step={step_idx} "
                        f"left_dist={left_dist:.4f} right_dist={right_dist:.4f} "
                        f"left_done={left_done} right_done={right_done}"
                    )
                if left_done and right_done:
                    print(f"[INFO] round={round_idx} success.")
                    break
            else:
                print(f"[WARN] round={round_idx} timeout: left_done={left_done} right_done={right_done}.")

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
