from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class JointActionAdapterCfg:
    """Joint-space action mapping config.

    Input shape:
    - raw action: [num_envs, action_dim], normalized to [-1, 1]

    Output shape:
    - joint targets: [num_envs, num_arm_joints]
    """

    delta_scale: float
    use_delta_targets: bool = True


class JointActionAdapter:
    """Map external action vectors into arm joint targets."""

    def __init__(self, cfg: JointActionAdapterCfg):
        self.cfg = cfg

    def compute_joint_targets(
        self,
        raw_action: torch.Tensor,
        current_joint_pos: torch.Tensor,
        lower_limits: torch.Tensor,
        upper_limits: torch.Tensor,
    ) -> torch.Tensor:
        # Current first-version contract:
        #   action in [-1, 1]
        #   -> scaled per-joint delta
        #   -> clipped to the articulation joint limits
        #
        # If the future policy server changes its output convention,
        # we want the translation logic to stay here instead of spreading
        # action slicing / scaling rules across env.py.
        clipped_action = raw_action.clamp(-1.0, 1.0)
        if self.cfg.use_delta_targets:
            joint_targets = current_joint_pos + clipped_action * self.cfg.delta_scale
        else:
            center = 0.5 * (lower_limits + upper_limits)
            half_range = 0.5 * (upper_limits - lower_limits)
            joint_targets = center + clipped_action * half_range
        return torch.clamp(joint_targets, lower_limits, upper_limits)


class PolicyActionChunkAdapter:
    """Validate and normalize policy outputs into a [T, action_dim] action chunk."""

    def __init__(self, action_dim: int):
        self.action_dim = action_dim

    def normalize_action_chunk(self, action_chunk_or_vec) -> np.ndarray:
        action_array = np.asarray(action_chunk_or_vec, dtype=np.float32)
        if action_array.ndim == 1:
            if action_array.shape[0] != self.action_dim:
                raise ValueError(
                    f"Expected single-step action shape ({self.action_dim},), got {tuple(action_array.shape)}"
                )
            return action_array[None, :]
        if action_array.ndim == 2:
            if action_array.shape[1] != self.action_dim:
                raise ValueError(
                    f"Expected action chunk shape (T, {self.action_dim}), got {tuple(action_array.shape)}"
                )
            return action_array
        raise ValueError(
            f"Expected policy action output to have rank 1 or 2, got rank {action_array.ndim} with shape "
            f"{tuple(action_array.shape)}"
        )

    def normalize_libero_action_chunk(self, action_chunk_or_vec) -> np.ndarray:
        """Map a LIBERO-style 7D action or [T, 7] chunk into the current 14D dual-arm contract.

        This is a placeholder compatibility bridge for end-to-end transport testing only.
        It is not a semantics-correct controller for the Zhishu robot.

        Mapping rule:
        - LIBERO 7D action is duplicated onto both arms
        - output shape becomes [T, 14]
        """
        action_array = np.asarray(action_chunk_or_vec, dtype=np.float32)
        if action_array.ndim == 1:
            if action_array.shape[0] != 7:
                raise ValueError(f"Expected LIBERO action shape (7,), got {tuple(action_array.shape)}")
            action_array = action_array[None, :]
        elif action_array.ndim == 2:
            if action_array.shape[1] != 7:
                raise ValueError(f"Expected LIBERO action chunk shape (T, 7), got {tuple(action_array.shape)}")
        else:
            raise ValueError(
                f"Expected LIBERO policy action output rank 1 or 2, got rank {action_array.ndim} "
                f"with shape {tuple(action_array.shape)}"
            )

        dual_arm = np.concatenate([action_array, action_array], axis=-1)
        return self.normalize_action_chunk(dual_arm)
