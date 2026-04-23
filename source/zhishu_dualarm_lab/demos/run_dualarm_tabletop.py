from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def describe_observation(obs: dict) -> None:
    print("[INFO] observation keys:")
    for key, value in obs.items():
        if hasattr(value, "shape"):
            print(f"  - {key}: shape={tuple(value.shape)}, dtype={getattr(value, 'dtype', type(value))}")
        elif isinstance(value, (list, tuple)):
            print(f"  - {key}: len={len(value)}")
        else:
            print(f"  - {key}: type={type(value)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Zhishu dual-arm tabletop environment demo.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments. First version expects 1.")
    parser.add_argument("--max_steps", type=int, default=2000, help="Number of environment steps to run.")
    parser.add_argument(
        "--motion-scale",
        type=float,
        default=1.0,
        help="Scale factor for the handwritten demo action profile.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    if hasattr(args_cli, "enable_cameras"):
        args_cli.enable_cameras = True

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import torch

    from zhishu_dualarm_lab.tasks.dualarm_tabletop.env import ZhishuDualArmTabletopEnv
    from zhishu_dualarm_lab.tasks.dualarm_tabletop.env_cfg import ZhishuDualArmTabletopEnvCfg

    cfg = ZhishuDualArmTabletopEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    env = ZhishuDualArmTabletopEnv(cfg=cfg, render_mode="human")

    obs, _ = env.reset()
    describe_observation(obs["policy"])
    print(f"[INFO] action shape: ({env.num_envs}, {env.cfg.action_space.shape[0]})")

    for step in range(args_cli.max_steps):
        phase = step * env.step_dt
        action = torch.zeros((env.num_envs, env.action_dim), device=env.device)
        motion_scale = args_cli.motion_scale
        # Drive several shoulder/elbow joints so the demo is visually obvious.
        action[:, 0] = 0.85 * motion_scale * math.sin(phase * 0.8)
        action[:, 1] = 0.65 * motion_scale * math.sin(phase * 0.6 + 0.5)
        action[:, 3] = 0.55 * motion_scale * math.sin(phase * 1.0 + 0.2)
        action[:, 5] = 0.45 * motion_scale * math.sin(phase * 1.1 + 1.0)
        action[:, 7] = -0.85 * motion_scale * math.sin(phase * 0.8)
        action[:, 8] = -0.65 * motion_scale * math.sin(phase * 0.6 + 0.5)
        action[:, 10] = -0.55 * motion_scale * math.sin(phase * 1.0 + 0.2)
        action[:, 12] = -0.45 * motion_scale * math.sin(phase * 1.1 + 1.0)
        action.clamp_(-1.0, 1.0)

        obs, reward, terminated, truncated, _ = env.step(action)
        if step % 100 == 0:
            print(
                f"[INFO] step={step} reward_mean={reward.mean().item():.4f} "
                f"terminated={terminated.any().item()} truncated={truncated.any().item()}"
            )
        if terminated.any() or truncated.any():
            print(f"[INFO] environment requested reset at step {step}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
