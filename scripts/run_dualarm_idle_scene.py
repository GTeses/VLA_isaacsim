#!/usr/bin/env python3
"""Launch the Zhishu dual-arm tabletop scene and hold it idle.

This entrypoint is meant for visual inspection:
- load the current tabletop scene
- keep the robot in the reset pose
- do not inject any demo, policy, or scripted motion
- do not auto-exit after a fixed number of steps
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments. First version expects 1.")
    parser.add_argument(
        "--log_every",
        type=int,
        default=300,
        help="Print one heartbeat line every N idle steps.",
    )
    parser.add_argument(
        "--start_paused",
        action="store_true",
        help="Load the scene, then pause the Isaac Sim timeline until you press Play in the UI.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if hasattr(args, "enable_cameras"):
        args.enable_cameras = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        import torch

        from zhishu_dualarm_lab.tasks.dualarm_tabletop.env import ZhishuDualArmTabletopEnv
        from zhishu_dualarm_lab.tasks.dualarm_tabletop.env_cfg import ZhishuDualArmTabletopEnvCfg

        cfg = ZhishuDualArmTabletopEnvCfg()
        cfg.scene.num_envs = args.num_envs
        env = ZhishuDualArmTabletopEnv(cfg=cfg, render_mode="human" if not args.headless else None)
        env.reset()
        timeline = None
        if args.start_paused:
            import omni.timeline

            timeline = omni.timeline.get_timeline_interface()
            timeline.pause()
            print("[INFO] scene loaded and timeline paused; open Physics Inspector, then press Play in Isaac Sim.")

        zero_action = torch.zeros((env.num_envs, env.action_dim), device=env.device)
        step_idx = 0
        print("[INFO] idle scene ready; stepping zero actions until you close the app or Ctrl-C.")
        while simulation_app.is_running():
            if timeline is not None and not timeline.is_playing():
                simulation_app.update()
                continue
            env.step(zero_action)
            if args.log_every > 0 and step_idx % args.log_every == 0:
                print(f"[INFO] idle step={step_idx}")
            step_idx += 1
    finally:
        if "env" in locals():
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
