#!/usr/bin/env python3
"""Collect Zhishu dual-arm no-hand simulation rollouts into the raw layout used by the LeRobot converter.

The collector records one folder per episode:

output_dir/
  episode_0000/
    metadata.json
    prompt.txt
    observation_state.npy
    action.npy
    reward.npy
    terminated.npy
    truncated.npy
    timestamp.npy
    external_images/000000.png
    left_wrist_images/000000.png
    right_wrist_images/000000.png

By default the collector uses a lightweight scripted joint-space policy so the
first training batch can come entirely from Isaac simulation without waiting
for teleop or real-robot data.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher
from PIL import Image

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


@dataclass(frozen=True)
class EpisodeSummary:
    episode_index: int
    num_steps: int
    total_reward: float
    terminated: bool
    truncated: bool
    task: str
    policy_mode: str


def _save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(path)


class ScriptedZhishuPolicy:
    """Small joint-space heuristic for reach / gather / push data collection.

    This is not intended to be a high-quality controller. Its job is to create
    a first batch of simulation trajectories that are at least task-directed:
    both arms move toward the cube, close in around it, and then bias toward
    pushing the cube toward the target zone.
    """

    def __init__(self, action_dim: int = 14):
        self.action_dim = action_dim

    def infer(self, policy_input: dict) -> np.ndarray:
        state = np.asarray(policy_input["observation/state"], dtype=np.float32)
        left_tcp = state[42:49][:3]
        right_tcp = state[49:56][:3]
        object_pos = state[56:63][:3]
        target_pos = state[63:70][:3]

        left_rel = object_pos - left_tcp
        right_rel = object_pos - right_tcp
        push_dir = target_pos - object_pos

        action = np.zeros((self.action_dim,), dtype=np.float32)
        action[:7] = self._arm_action(left_rel, push_dir, is_left=True)
        action[7:] = self._arm_action(right_rel, push_dir, is_left=False)
        return action[None, :]

    def _arm_action(self, tcp_to_object: np.ndarray, object_to_target: np.ndarray, *, is_left: bool) -> np.ndarray:
        x_err, y_err, z_err = tcp_to_object
        push_x, push_y, _ = object_to_target
        side = 1.0 if is_left else -1.0

        # Shoulder and elbow terms bias both arms to move forward and slightly
        # inward toward the cube. Once close, the object->target direction adds
        # a push bias so the resulting rollouts are not pure hovering data.
        action = np.array(
            [
                side * (1.6 * y_err + 0.6 * push_y),
                -1.1 * x_err - 0.2 * z_err - 0.5 * push_x,
                -0.3 * side * y_err,
                1.4 * x_err + 0.35 * push_x,
                0.0,
                -0.8 * z_err,
                0.15 * side * push_y,
            ],
            dtype=np.float32,
        )
        return np.clip(action, -1.0, 1.0)


def _make_policy(mode: str, env_action_dim: int, args, policy_client_module):
    if mode == "scripted":
        return ScriptedZhishuPolicy(action_dim=env_action_dim), None

    client_cfg = policy_client_module.PolicyClientConfig(
        host=args.policy_host,
        port=args.policy_port,
        timeout_s=10.0,
        action_dim=env_action_dim,
        input_schema=args.policy_input_schema,
        output_contract=args.policy_output_contract,
    )
    if mode == "fake":
        return policy_client_module.FakePolicyClient(client_cfg, chunk_length=max(args.replan_steps, 8)), client_cfg
    return policy_client_module.OpenPiWebsocketClient(client_cfg), client_cfg


def _record_episode(
    env,
    *,
    episode_index: int,
    episode_dir: Path,
    max_steps: int,
    fps: int,
    policy_mode: str,
    replan_steps: int,
    policy,
) -> EpisodeSummary:
    obs, _ = env.reset()
    del obs

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    terminated_flags: list[bool] = []
    truncated_flags: list[bool] = []
    timestamps: list[float] = []

    episode_dir.mkdir(parents=True, exist_ok=True)
    (episode_dir / "external_images").mkdir(exist_ok=True)
    (episode_dir / "left_wrist_images").mkdir(exist_ok=True)
    (episode_dir / "right_wrist_images").mkdir(exist_ok=True)

    prompt = ""
    total_reward = 0.0
    start_time = time.monotonic()

    for step in range(max_steps):
        policy_input = env.get_policy_input()
        if not policy_input:
            raise RuntimeError("Policy input is empty during collection.")

        if not prompt:
            prompt = str(policy_input["prompt"])

        states.append(np.asarray(policy_input["observation/state"], dtype=np.float32).copy())
        timestamps.append(step / float(fps))

        _save_rgb(episode_dir / "external_images" / f"{step:06d}.png", policy_input["observation/external_image"])
        _save_rgb(episode_dir / "left_wrist_images" / f"{step:06d}.png", policy_input["observation/left_wrist_image"])
        _save_rgb(episode_dir / "right_wrist_images" / f"{step:06d}.png", policy_input["observation/right_wrist_image"])

        action_chunk = policy.infer(policy_input)
        first_action = env.apply_policy_output(action_chunk)
        actions.append(first_action.detach().cpu().numpy().reshape(-1).astype(np.float32))

        reward = None
        terminated = None
        truncated = None
        for _ in range(replan_steps):
            action = env.consume_action_plan_step()
            obs, reward, terminated, truncated, _ = env.step(action)
            del obs
            if bool(terminated.any().item()) or bool(truncated.any().item()):
                break

        if reward is None or terminated is None or truncated is None:
            raise RuntimeError("Episode step loop did not produce reward / done signals.")

        reward_value = float(reward.mean().item())
        terminated_value = bool(terminated.any().item())
        truncated_value = bool(truncated.any().item())
        rewards.append(reward_value)
        terminated_flags.append(terminated_value)
        truncated_flags.append(truncated_value)
        total_reward += reward_value

        if terminated_value or truncated_value:
            break

    np.save(episode_dir / "observation_state.npy", np.stack(states, axis=0).astype(np.float32))
    np.save(episode_dir / "action.npy", np.stack(actions, axis=0).astype(np.float32))
    np.save(episode_dir / "reward.npy", np.asarray(rewards, dtype=np.float32))
    np.save(episode_dir / "terminated.npy", np.asarray(terminated_flags, dtype=np.bool_))
    np.save(episode_dir / "truncated.npy", np.asarray(truncated_flags, dtype=np.bool_))
    np.save(episode_dir / "timestamp.npy", np.asarray(timestamps, dtype=np.float32))
    (episode_dir / "prompt.txt").write_text(f"{prompt}\n", encoding="utf-8")

    summary = EpisodeSummary(
        episode_index=episode_index,
        num_steps=len(actions),
        total_reward=total_reward,
        terminated=terminated_flags[-1] if terminated_flags else False,
        truncated=truncated_flags[-1] if truncated_flags else False,
        task=prompt,
        policy_mode=policy_mode,
    )
    metadata = {
        **asdict(summary),
        "fps": fps,
        "record_duration_s": time.monotonic() - start_time,
    }
    (episode_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True, help="Root directory for raw episode folders.")
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=80)
    parser.add_argument("--replan_steps", type=int, default=1)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--policy_mode",
        choices=["scripted", "fake", "websocket"],
        default="scripted",
        help="Action source for raw rollout generation.",
    )
    parser.add_argument("--policy_host", type=str, default="127.0.0.1")
    parser.add_argument("--policy_port", type=int, default=8000)
    parser.add_argument(
        "--policy_input_schema",
        choices=["native", "zhishu14", "libero"],
        default="zhishu14",
        help="Use zhishu14 for the long-term native contract. libero is smoke-test-only.",
    )
    parser.add_argument(
        "--policy_output_contract",
        choices=["native14", "zhishu14", "libero7"],
        default="zhishu14",
        help="Use zhishu14 for the long-term native contract. libero7 is smoke-test-only.",
    )
    parser.add_argument("--num_envs", type=int, default=1)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if hasattr(args, "enable_cameras"):
        args.enable_cameras = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        from zhishu_dualarm_lab.tasks.dualarm_tabletop.env import ZhishuDualArmTabletopEnv
        from zhishu_dualarm_lab.tasks.dualarm_tabletop.env_cfg import ZhishuDualArmTabletopEnvCfg
        from zhishu_dualarm_lab.utils import policy_client as policy_client_module

        cfg = ZhishuDualArmTabletopEnvCfg()
        cfg.scene.num_envs = args.num_envs
        env = ZhishuDualArmTabletopEnv(cfg=cfg, render_mode="human" if not args.headless else None)
        env.replan_steps = args.replan_steps

        policy, _ = _make_policy(args.policy_mode, env.action_dim, args, policy_client_module)
        if hasattr(policy, "connect"):
            policy.connect()

        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[INFO] collecting raw Zhishu sim data output_dir={args.output_dir} policy_mode={args.policy_mode} "
            f"episodes={args.num_episodes} max_steps={args.max_steps} replan_steps={args.replan_steps}"
        )

        for episode_index in range(args.num_episodes):
            episode_dir = args.output_dir / f"episode_{episode_index:04d}"
            summary = _record_episode(
                env,
                episode_index=episode_index,
                episode_dir=episode_dir,
                max_steps=args.max_steps,
                fps=args.fps,
                policy_mode=args.policy_mode,
                replan_steps=args.replan_steps,
                policy=policy,
            )
            print(
                f"[INFO] episode={summary.episode_index} steps={summary.num_steps} "
                f"reward_sum={summary.total_reward:.4f} terminated={summary.terminated} truncated={summary.truncated}"
            )

    finally:
        try:
            if "policy" in locals() and hasattr(policy, "close"):
                policy.close()
        finally:
            if "env" in locals():
                env.close()
            simulation_app.close()


if __name__ == "__main__":
    main()
