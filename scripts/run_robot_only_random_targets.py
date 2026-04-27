#!/usr/bin/env python3
"""Move the Zhishu dual-arm robot to two independent random TCP targets.

This script is intentionally robot-only: it loads the current project USD asset
and drives left_link7/right_link7 TCP frames to two sampled free-space points.
The current USD is expected to have symmetric left/right arm kinematics, so the
controller does not contain side-specific compensation.
"""

from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

TARGET_MARKER_SIZE = (0.035, 0.035, 0.035)


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in rpy]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis / norm
    c = math.cos(float(angle))
    s = math.sin(float(angle))
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def _transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _rpy_matrix(rpy)
    transform[:3, 3] = xyz.astype(np.float64)
    return transform


def _quat_wxyz_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in quat]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


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


def _parse_7_values(spec: str, *, arg_name: str) -> np.ndarray:
    values = np.asarray([float(v.strip()) for v in spec.split(",") if v.strip()], dtype=np.float32)
    if values.shape != (7,):
        raise ValueError(f"{arg_name} expects 7 comma-separated values, got {len(values)}: {spec}")
    return values


def _sample_targets(
    rng: np.random.Generator,
    left_start: np.ndarray,
    right_start: np.ndarray,
    *,
    x_min: float,
    x_max: float,
    left_y_min: float,
    left_y_max: float,
    right_y_min: float,
    right_y_max: float,
    z_min: float,
    z_max: float,
    min_separation_from_start: float,
    max_attempts: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample one target in each arm's workspace half-plane."""

    def sample_one(start: np.ndarray, y_min: float, y_max: float) -> np.ndarray:
        best = start.copy()
        best_dist = -1.0
        for _ in range(max(1, max_attempts)):
            candidate = np.array(
                [
                    rng.uniform(x_min, x_max),
                    rng.uniform(y_min, y_max),
                    rng.uniform(z_min, z_max),
                ],
                dtype=np.float32,
            )
            dist = float(np.linalg.norm(candidate - start))
            if dist >= min_separation_from_start:
                return candidate
            if dist > best_dist:
                best_dist = dist
                best = candidate
        return best.astype(np.float32)

    return (
        sample_one(left_start, left_y_min, left_y_max),
        sample_one(right_start, right_y_min, right_y_max),
    )


class UrdfLinkPositionFk:
    """Minimal URDF FK for link positions without mutating the simulator."""

    def __init__(self, urdf_path: Path, *, base_link: str = "base_link"):
        self.base_link = base_link
        root = ET.parse(urdf_path).getroot()
        self._joint_by_child: dict[str, dict] = {}
        for joint in root.findall("joint"):
            origin = joint.find("origin")
            axis = joint.find("axis")
            self._joint_by_child[joint.find("child").attrib["link"]] = {
                "name": joint.attrib["name"],
                "type": joint.attrib.get("type", "fixed"),
                "parent": joint.find("parent").attrib["link"],
                "xyz": np.fromstring(origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" "),
                "rpy": np.fromstring(origin.attrib.get("rpy", "0 0 0") if origin is not None else "0 0 0", sep=" "),
                "axis": np.fromstring(axis.attrib.get("xyz", "0 0 0") if axis is not None else "0 0 0", sep=" "),
            }
        self._chain_cache: dict[str, list[dict]] = {}

    def _chain_to(self, link_name: str) -> list[dict]:
        cached = self._chain_cache.get(link_name)
        if cached is not None:
            return cached
        chain: list[dict] = []
        current = link_name
        while current != self.base_link:
            joint = self._joint_by_child[current]
            chain.append(joint)
            current = joint["parent"]
        chain.reverse()
        self._chain_cache[link_name] = chain
        return chain

    def link_position(self, link_name: str, joint_positions: dict[str, float]) -> np.ndarray:
        transform = np.eye(4, dtype=np.float64)
        for joint in self._chain_to(link_name):
            transform = transform @ _transform(joint["xyz"], joint["rpy"])
            q = float(joint_positions.get(joint["name"], 0.0))
            if joint["type"] in {"revolute", "continuous"}:
                motion = np.eye(4, dtype=np.float64)
                motion[:3, :3] = _axis_angle_matrix(joint["axis"], q)
                transform = transform @ motion
            elif joint["type"] == "prismatic":
                motion = np.eye(4, dtype=np.float64)
                motion[:3, 3] = joint["axis"] * q
                transform = transform @ motion
        return transform[:3, 3].astype(np.float32)


def _joint_position_map(env) -> dict[str, float]:
    joint_pos = env._robot.data.joint_pos[0].detach().cpu().numpy()
    return {name: float(joint_pos[idx]) for idx, name in enumerate(env._robot.data.joint_names)}


def _sample_reachable_targets_from_joint_fk(
    *,
    env,
    fk: UrdfLinkPositionFk,
    rng: np.random.Generator,
    left_link_name: str,
    right_link_name: str,
    joint_noise_abs: np.ndarray,
    sample_attempts: int,
    min_move: float,
    max_move: float,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """Sample reachable TCP targets with URDF FK, without moving the sim state."""

    arm_start = env._robot.data.joint_pos[:, env._arm_joint_ids].clone()
    arm_lower = env._joint_lower_limits.view(1, 14)
    arm_upper = env._joint_upper_limits.view(1, 14)
    left_start, right_start = _tcp_pos(env)
    joint_pos_map = _joint_position_map(env)
    left_fk_start = fk.link_position(left_link_name, joint_pos_map)
    right_fk_start = fk.link_position(right_link_name, joint_pos_map)
    root_quat = env._robot.data.root_link_quat_w[0].detach().cpu().numpy()
    root_rot = _quat_wxyz_matrix(root_quat)

    noise_abs = torch.as_tensor(
        np.concatenate([joint_noise_abs, joint_noise_abs]),
        dtype=torch.float32,
        device=env.device,
    ).view(1, 14)
    best_score = float("inf")
    best_left_target = left_start.copy()
    best_right_target = right_start.copy()
    best_arm_goal = arm_start.clone()

    for _ in range(max(1, sample_attempts)):
        noise = (2.0 * torch.rand((1, 14), dtype=torch.float32, device=env.device) - 1.0) * noise_abs
        arm_goal = torch.clamp(arm_start + noise, arm_lower, arm_upper)
        goal_map = dict(joint_pos_map)
        for joint_id, q in zip(env._arm_joint_ids, arm_goal[0].detach().cpu().numpy(), strict=True):
            goal_map[env._robot.data.joint_names[joint_id]] = float(q)
        left_fk_goal = fk.link_position(left_link_name, goal_map)
        right_fk_goal = fk.link_position(right_link_name, goal_map)
        left_target = left_start + (root_rot @ (left_fk_goal - left_fk_start)).astype(np.float32)
        right_target = right_start + (root_rot @ (right_fk_goal - right_fk_start)).astype(np.float32)
        left_move = float(np.linalg.norm(left_target - left_start))
        right_move = float(np.linalg.norm(right_target - right_start))
        valid = min_move <= left_move <= max_move and min_move <= right_move <= max_move
        score = abs(left_move - 0.5 * (min_move + max_move)) + abs(right_move - 0.5 * (min_move + max_move))
        if valid:
            best_left_target = left_target
            best_right_target = right_target
            best_arm_goal = arm_goal.clone()
            break
        if score < best_score:
            best_score = score
            best_left_target = left_target
            best_right_target = right_target
            best_arm_goal = arm_goal.clone()

    return best_left_target.astype(np.float32), best_right_target.astype(np.float32), best_arm_goal


def _build_target_markers():
    from isaaclab.markers import VisualizationMarkers
    from isaaclab.markers.config import CUBOID_MARKER_CFG

    marker_cfg = CUBOID_MARKER_CFG.replace(prim_path="/Visuals/ZhishuRobotOnlyRandomTargets")
    marker_cfg.markers["cuboid"].size = TARGET_MARKER_SIZE
    return VisualizationMarkers(marker_cfg)


class DualArmPositionIK:
    """Small wrapper around Isaac Lab Differential IK for two 7-DoF arms."""

    def __init__(self, *, env, left_body_name: str, right_body_name: str, delta_scale: float, ik_lambda: float):
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

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
        joint_ids,
        jacobi_joint_ids,
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

    def infer(
        self,
        *,
        left_target: torch.Tensor,
        right_target: torch.Tensor,
        max_task_step: float,
        left_active: bool,
        right_active: bool,
    ) -> torch.Tensor:
        left_action = torch.zeros((self.env.num_envs, 7), device=self.env.device)
        right_action = torch.zeros((self.env.num_envs, 7), device=self.env.device)
        if left_active:
            left_action = self._solve_arm(
                controller=self.left_controller,
                tcp_frame_idx=self.left_tcp_frame_idx,
                jacobi_body_idx=self.left_jacobi_body_idx,
                joint_ids=self.left_joint_ids,
                jacobi_joint_ids=self.left_jacobi_joint_ids,
                target_world=left_target,
                max_task_step=max_task_step,
            )
        if right_active:
            right_action = self._solve_arm(
                controller=self.right_controller,
                tcp_frame_idx=self.right_tcp_frame_idx,
                jacobi_body_idx=self.right_jacobi_body_idx,
                joint_ids=self.right_joint_ids,
                jacobi_joint_ids=self.right_jacobi_joint_ids,
                target_world=right_target,
                max_task_step=max_task_step,
            )
        return torch.cat([left_action, right_action], dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_rounds", type=int, default=10, help="Number of random target pairs to execute.")
    parser.add_argument("--max_steps_per_round", type=int, default=180, help="Control steps before a round times out.")
    parser.add_argument("--settle_steps", type=int, default=12, help="Zero-action warmup steps after each reset.")
    parser.add_argument("--success_threshold", type=float, default=0.035, help="TCP distance threshold in meters.")
    parser.add_argument("--hold_steps", type=int, default=6, help="Consecutive in-threshold steps required for success.")
    parser.add_argument("--max_task_step", type=float, default=0.045, help="Max Cartesian IK command step in meters.")
    parser.add_argument("--ik_lambda", type=float, default=0.08, help="DLS IK damping lambda.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_envs", type=int, default=1, help="This debugging script is intended for one env.")
    parser.add_argument(
        "--target_mode",
        choices=["reachable_fk", "box"],
        default="reachable_fk",
        help="reachable_fk samples random joint goals first, so target points are guaranteed reachable.",
    )
    parser.add_argument(
        "--controller",
        choices=["joint", "ik"],
        default="joint",
        help="joint tracks the sampled FK joint goal; ik uses Cartesian Differential IK.",
    )
    parser.add_argument(
        "--joint_goal_noise",
        type=str,
        default="0.45,0.35,0.45,0.35,0.45,0.30,0.45",
        help="Per-arm joint sampling amplitude in radians for target_mode=reachable_fk.",
    )
    parser.add_argument("--fk_probe_settle_steps", type=int, default=3, help=argparse.SUPPRESS)
    parser.add_argument("--fk_sample_attempts", type=int, default=64)
    parser.add_argument("--target_min_move", type=float, default=0.06)
    parser.add_argument("--target_max_move", type=float, default=0.24)
    parser.add_argument("--x_min", type=float, default=0.28)
    parser.add_argument("--x_max", type=float, default=0.78)
    parser.add_argument("--left_y_min", type=float, default=0.10)
    parser.add_argument("--left_y_max", type=float, default=0.48)
    parser.add_argument("--right_y_min", type=float, default=-0.48)
    parser.add_argument("--right_y_max", type=float, default=-0.10)
    parser.add_argument("--z_min", type=float, default=0.62)
    parser.add_argument("--z_max", type=float, default=1.05)
    parser.add_argument("--min_separation_from_start", type=float, default=0.08)
    parser.add_argument("--target_sample_attempts", type=int, default=64)
    parser.add_argument("--log_every", type=int, default=10)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs != 1:
        raise ValueError("run_robot_only_random_targets.py currently supports --num_envs 1 only.")
    if hasattr(args, "enable_cameras"):
        args.enable_cameras = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    env = None
    try:
        print("[INFO] importing robot-only task modules...", flush=True)
        from zhishu_dualarm_lab.tasks.robot_only.constants import (
            ARM_ACTION_DELTA_SCALE,
            LEFT_TCP_LINK_NAME,
            RIGHT_TCP_LINK_NAME,
        )
        from zhishu_dualarm_lab.tasks.robot_only.env import ZhishuDualArmRobotOnlyEnv
        from zhishu_dualarm_lab.tasks.robot_only.env_cfg import ZhishuDualArmRobotOnlyEnvCfg
        from zhishu_dualarm_lab.utils.local_paths import resolve_robot_urdf_path

        print("[INFO] creating robot-only environment...", flush=True)
        rng = np.random.default_rng(args.seed)
        torch.manual_seed(args.seed)
        joint_goal_noise = _parse_7_values(args.joint_goal_noise, arg_name="--joint_goal_noise")
        fk = UrdfLinkPositionFk(resolve_robot_urdf_path())
        cfg = ZhishuDualArmRobotOnlyEnvCfg()
        cfg.scene.num_envs = args.num_envs
        env = ZhishuDualArmRobotOnlyEnv(cfg=cfg, render_mode="human" if not args.headless else None)
        print("[INFO] resetting robot-only environment...", flush=True)
        env.reset()

        print("[INFO] initializing controllers and target markers...", flush=True)
        ik = DualArmPositionIK(
            env=env,
            left_body_name=LEFT_TCP_LINK_NAME,
            right_body_name=RIGHT_TCP_LINK_NAME,
            delta_scale=ARM_ACTION_DELTA_SCALE,
            ik_lambda=args.ik_lambda,
        )
        markers = _build_target_markers()
        marker_quats = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
            device=env.device,
        )

        print(
            "[INFO] random target task started "
            f"usd={cfg.scene.robot.spawn.usd_path} "
            f"workspace=x[{args.x_min:.2f},{args.x_max:.2f}] "
            f"left_y[{args.left_y_min:.2f},{args.left_y_max:.2f}] "
            f"right_y[{args.right_y_min:.2f},{args.right_y_max:.2f}] "
            f"z[{args.z_min:.2f},{args.z_max:.2f}] "
            f"target_mode={args.target_mode} controller={args.controller}"
        )

        for round_idx in range(args.num_rounds):
            env.reset()
            _settle(env, args.settle_steps)
            left_start, right_start = _tcp_pos(env)
            arm_goal = None
            if args.target_mode == "reachable_fk":
                left_target_np, right_target_np, arm_goal = _sample_reachable_targets_from_joint_fk(
                    env=env,
                    fk=fk,
                    rng=rng,
                    left_link_name=LEFT_TCP_LINK_NAME,
                    right_link_name=RIGHT_TCP_LINK_NAME,
                    joint_noise_abs=joint_goal_noise,
                    sample_attempts=args.fk_sample_attempts,
                    min_move=args.target_min_move,
                    max_move=args.target_max_move,
                )
            else:
                left_target_np, right_target_np = _sample_targets(
                    rng,
                    left_start,
                    right_start,
                    x_min=args.x_min,
                    x_max=args.x_max,
                    left_y_min=args.left_y_min,
                    left_y_max=args.left_y_max,
                    right_y_min=args.right_y_min,
                    right_y_max=args.right_y_max,
                    z_min=args.z_min,
                    z_max=args.z_max,
                    min_separation_from_start=args.min_separation_from_start,
                    max_attempts=args.target_sample_attempts,
                )
            left_target = torch.as_tensor(left_target_np, dtype=torch.float32, device=env.device).view(1, 3)
            right_target = torch.as_tensor(right_target_np, dtype=torch.float32, device=env.device).view(1, 3)
            markers.visualize(translations=torch.cat([left_target, right_target], dim=0), orientations=marker_quats)

            print(
                f"[INFO] round={round_idx} "
                f"left_start={np.round(left_start, 4).tolist()} "
                f"right_start={np.round(right_start, 4).tolist()} "
                f"left_target={np.round(left_target_np, 4).tolist()} "
                f"right_target={np.round(right_target_np, 4).tolist()}"
            )
            if arm_goal is not None:
                print(f"[INFO] round={round_idx} sampled_arm_goal={np.round(arm_goal[0].detach().cpu().numpy(), 4).tolist()}")

            left_hold = 0
            right_hold = 0
            left_done = False
            right_done = False
            for step_idx in range(args.max_steps_per_round):
                if args.controller == "joint" and arm_goal is not None:
                    current_arm_pos = env._robot.data.joint_pos[:, env._arm_joint_ids]
                    action = ((arm_goal - current_arm_pos) / ARM_ACTION_DELTA_SCALE).clamp(-1.0, 1.0)
                    if left_done:
                        action[:, :7] = 0.0
                    if right_done:
                        action[:, 7:] = 0.0
                else:
                    action = ik.infer(
                        left_target=left_target,
                        right_target=right_target,
                        max_task_step=args.max_task_step,
                        left_active=not left_done,
                        right_active=not right_done,
                    )
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
    except SystemExit as exc:
        print(f"[ERROR] SystemExit before/during random-target task: code={exc.code}", flush=True)
        raise
    except BaseException as exc:
        print(f"[ERROR] random-target task failed: {type(exc).__name__}: {exc}", flush=True)
        raise
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        simulation_app.close()


if __name__ == "__main__":
    main()
