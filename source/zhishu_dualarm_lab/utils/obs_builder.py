from __future__ import annotations

from dataclasses import dataclass

import torch


def _rgb_hwc(camera_sensor) -> torch.Tensor:
    """Return RGB images in HWC uint8 format: [N, H, W, 3]."""
    rgb = camera_sensor.data.output["rgb"][..., :3]
    if rgb.dtype != torch.uint8:
        rgb = rgb.to(torch.uint8)
    return rgb.contiguous()


def _pose_tensor(pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """Stack pose as [x, y, z, qw, qx, qy, qz]."""
    return torch.cat([pos, quat], dim=-1)


@dataclass
class ObservationBuilderCfg:
    prompt: str


class ObservationBuilder:
    """Build the environment observation dict.

    The returned images are fixed to HWC format.

    This builder exists so the policy-facing observation schema stays
    concentrated in one place. The environment can evolve internally
    without forcing downstream policy glue code to chase changes across
    env.py.
    """

    def __init__(self, cfg: ObservationBuilderCfg):
        self.cfg = cfg

    def build(
        self,
        *,
        external_camera,
        left_wrist_camera,
        right_wrist_camera,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        last_action: torch.Tensor,
        left_tcp_pos: torch.Tensor,
        left_tcp_quat: torch.Tensor,
        right_tcp_pos: torch.Tensor,
        right_tcp_quat: torch.Tensor,
        object_pos: torch.Tensor,
        object_quat: torch.Tensor,
        target_pos: torch.Tensor,
        target_quat: torch.Tensor,
    ) -> dict[str, torch.Tensor | list[str]]:
        # Pose tensors use a single fixed layout:
        # [x, y, z, qw, qx, qy, qz]
        left_tcp_pose = _pose_tensor(left_tcp_pos, left_tcp_quat)
        right_tcp_pose = _pose_tensor(right_tcp_pos, right_tcp_quat)
        object_pose = _pose_tensor(object_pos, object_quat)
        target_pose = _pose_tensor(target_pos, target_quat)
        # "observation/state" is a compact low-dimensional state vector
        # for debugging and future policy-server integration.
        state = torch.cat(
            [
                joint_pos,
                joint_vel,
                last_action,
                left_tcp_pose,
                right_tcp_pose,
                object_pose,
                target_pose,
            ],
            dim=-1,
        )
        num_envs = joint_pos.shape[0]
        return {
            # prompt is kept as a per-env list of strings so future VLA /
            # websocket policies can consume it without extra reshaping.
            "prompt": [self.cfg.prompt for _ in range(num_envs)],
            "observation/external_image": _rgb_hwc(external_camera),
            "observation/left_wrist_image": _rgb_hwc(left_wrist_camera),
            "observation/right_wrist_image": _rgb_hwc(right_wrist_camera),
            "observation/joint_pos": joint_pos,
            "observation/joint_vel": joint_vel,
            "observation/last_action": last_action,
            "observation/left_tcp_pose": left_tcp_pose,
            "observation/right_tcp_pose": right_tcp_pose,
            "observation/object_pose": object_pose,
            "observation/target_pose": target_pose,
            "observation/state": state,
        }
