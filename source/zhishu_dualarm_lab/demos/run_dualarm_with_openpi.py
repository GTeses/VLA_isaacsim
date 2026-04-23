from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def _print_policy_input_summary(policy_input: dict) -> None:
    print(f"[INFO] policy input keys: {sorted(policy_input.keys())}")
    for image_key in (
        "observation/external_image",
        "observation/left_wrist_image",
        "observation/right_wrist_image",
    ):
        image = policy_input[image_key]
        print(f"[INFO] {image_key}: shape={image.shape} dtype={image.dtype}")
    state = policy_input["observation/state"]
    print(f"[INFO] observation/state: shape={state.shape} dtype={state.dtype}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Zhishu dual-arm environment with an openpi-style policy client.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments. Current bridge expects 1.")
    parser.add_argument("--policy_mode", choices=["fake", "websocket"], default="fake", help="Policy client mode.")
    parser.add_argument("--policy_host", type=str, default="127.0.0.1", help="Websocket policy server host.")
    parser.add_argument("--policy_port", type=int, default=8000, help="Websocket policy server port.")
    parser.add_argument(
        "--policy_input_schema",
        choices=["native", "zhishu14", "libero"],
        default="zhishu14",
        help="Schema adapter used before sending observations to the policy server. Use zhishu14 for the long-term native contract.",
    )
    parser.add_argument(
        "--policy_output_contract",
        choices=["native14", "zhishu14", "libero7"],
        default="zhishu14",
        help="How to interpret the action field returned by the policy server. Use zhishu14 for the long-term native contract.",
    )
    parser.add_argument("--replan_steps", type=int, default=8, help="How many actions to keep from each returned chunk.")
    parser.add_argument("--max_steps", type=int, default=2000, help="Maximum number of environment steps.")
    parser.add_argument("--debug_print_every", type=int, default=50, help="Print debug summaries every N steps.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    if hasattr(args_cli, "enable_cameras"):
        args_cli.enable_cameras = True

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import torch

    from zhishu_dualarm_lab.tasks.dualarm_tabletop.env import ZhishuDualArmTabletopEnv
    from zhishu_dualarm_lab.tasks.dualarm_tabletop.env_cfg import ZhishuDualArmTabletopEnvCfg
    from zhishu_dualarm_lab.utils.policy_client import (
        FakePolicyClient,
        OpenPiWebsocketClient,
        PolicyClientConfig,
        PolicyConnectionError,
        PolicyResponseError,
        PolicyTimeoutError,
    )

    cfg = ZhishuDualArmTabletopEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    env = ZhishuDualArmTabletopEnv(cfg=cfg, render_mode="human" if not args_cli.headless else None)
    env.replan_steps = args_cli.replan_steps

    client_cfg = PolicyClientConfig(
        host=args_cli.policy_host,
        port=args_cli.policy_port,
        timeout_s=10.0,
        action_dim=env.action_dim,
        input_schema=args_cli.policy_input_schema,
        output_contract=args_cli.policy_output_contract,
    )
    if args_cli.policy_mode == "fake":
        policy_client = FakePolicyClient(client_cfg, chunk_length=max(args_cli.replan_steps, 8))
    else:
        policy_client = OpenPiWebsocketClient(client_cfg)

    print(
        f"[INFO] policy client mode={args_cli.policy_mode} host={args_cli.policy_host} port={args_cli.policy_port} "
        f"replan_steps={args_cli.replan_steps} input_schema={args_cli.policy_input_schema} "
        f"output_contract={args_cli.policy_output_contract}"
    )

    try:
        policy_client.connect()
        if args_cli.policy_mode == "websocket":
            print(f"[INFO] server metadata keys: {sorted(policy_client.server_metadata.keys())}")

        obs, _ = env.reset()
        del obs

        for step in range(args_cli.max_steps):
            if env.action_plan_length == 0:
                policy_input = env.get_policy_input()
                if not policy_input:
                    raise RuntimeError("Policy input is empty. This usually means env observations were not initialized.")
                if step % args_cli.debug_print_every == 0:
                    _print_policy_input_summary(policy_input)
                action_chunk = policy_client.infer(policy_input)
                first_action = env.apply_policy_output(action_chunk)
                if step % args_cli.debug_print_every == 0:
                    print(f"[INFO] received action chunk shape: {tuple(action_chunk.shape)}")
                    print(f"[INFO] staged action buffer length: {env.action_plan_length}")
                    print(f"[INFO] first staged action shape: {tuple(first_action.shape)}")

            action = env.consume_action_plan_step()
            if step % args_cli.debug_print_every == 0:
                print(
                    f"[INFO] consumed action shape: {tuple(action.shape)} "
                    f"buffer_remaining={env.action_plan_length}"
                )

            obs, reward, terminated, truncated, _ = env.step(action)
            del obs
            if step % args_cli.debug_print_every == 0:
                print(
                    f"[INFO] step={step} reward_mean={reward.mean().item():.4f} "
                    f"terminated={terminated.any().item()} truncated={truncated.any().item()}"
                )

    except PolicyConnectionError as exc:
        raise RuntimeError(f"websocket connection failure: {exc}") from exc
    except PolicyTimeoutError as exc:
        raise RuntimeError(f"policy timeout: {exc}") from exc
    except PolicyResponseError as exc:
        raise RuntimeError(f"policy response error: {exc}") from exc
    finally:
        policy_client.close()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
