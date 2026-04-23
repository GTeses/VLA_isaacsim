#!/usr/bin/env python3
"""Probe per-joint TCP response signs for the Zhishu dual-arm robot.

This is a diagnostic tool for validating the real motion direction of each
joint against the TCP position change it causes. It is useful when a
hand-written joint-sign mapping looks suspicious.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from isaaclab.app import AppLauncher

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def _step_zero(env, steps: int) -> None:
    zero_action = torch.zeros((env.num_envs, env.action_dim), device=env.device)
    for _ in range(steps):
        env.step(zero_action)


def _tcp_pos(env, arm: str) -> torch.Tensor:
    frame_idx = env._left_tcp_idx if arm == "left" else env._right_tcp_idx
    return env._tcp_frames.data.target_pos_w[:, frame_idx].clone()


def _joint_ids_for_arm(env, arm: str):
    return env._arm_joint_ids[:7] if arm == "left" else env._arm_joint_ids[7:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["left", "right"], default="left")
    parser.add_argument("--joint_delta", type=float, default=0.08, help="Normalized action magnitude in [-1, 1].")
    parser.add_argument("--settle_steps", type=int, default=6, help="Number of steps to apply each probe action.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if hasattr(args, "enable_cameras"):
        args.enable_cameras = False

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        from zhishu_dualarm_lab.tasks.dualarm_tabletop.env import ZhishuDualArmTabletopEnv
        from zhishu_dualarm_lab.tasks.dualarm_tabletop.env_cfg import ZhishuDualArmTabletopEnvCfg

        cfg = ZhishuDualArmTabletopEnvCfg()
        env = ZhishuDualArmTabletopEnv(cfg=cfg, render_mode="human" if not args.headless else None)
        joint_ids = _joint_ids_for_arm(env, args.arm)

        print(f"[INFO] probing {args.arm} arm joint-to-TCP response")
        for local_joint_idx, joint_id in enumerate(joint_ids.tolist()):
            for sign in (+1.0, -1.0):
                env.reset()
                _step_zero(env, 2)
                tcp_before = _tcp_pos(env, args.arm)
                action = torch.zeros((env.num_envs, env.action_dim), device=env.device)
                action_index = local_joint_idx if args.arm == "left" else local_joint_idx + 7
                action[:, action_index] = sign * args.joint_delta
                for _ in range(args.settle_steps):
                    env.step(action)
                tcp_after = _tcp_pos(env, args.arm)
                delta = (tcp_after - tcp_before)[0].detach().cpu().numpy()
                print(
                    f"[INFO] joint={env._robot.data.joint_names[joint_id]} sign={sign:+.0f} "
                    f"delta_tcp_w=({delta[0]:+.4f}, {delta[1]:+.4f}, {delta[2]:+.4f})"
                )

        env.close()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
