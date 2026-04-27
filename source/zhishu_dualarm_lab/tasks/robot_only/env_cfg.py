from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .constants import ACTION_SPACE, DECIMATION, EPISODE_LENGTH_S, OBSERVATION_SPACE, SIM_DT
from .scene_cfg import ZhishuDualArmRobotOnlySceneCfg


@configclass
class ZhishuDualArmRobotOnlyEnvCfg(DirectRLEnvCfg):
    """DirectRLEnv config for a robot-only dual-arm scene."""

    episode_length_s = EPISODE_LENGTH_S
    decimation = DECIMATION
    observation_space = OBSERVATION_SPACE
    action_space = ACTION_SPACE
    state_space = 0
    rerender_on_reset = True

    sim: SimulationCfg = SimulationCfg(
        dt=SIM_DT,
        render_interval=DECIMATION,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    scene: ZhishuDualArmRobotOnlySceneCfg = ZhishuDualArmRobotOnlySceneCfg(
        num_envs=1,
        env_spacing=4.0,
        replicate_physics=False,
        clone_in_fabric=False,
    )

    def __post_init__(self) -> None:
        self.viewer.eye = (2.2, 2.0, 1.8)
        self.viewer.lookat = (0.5, 0.0, 0.8)
        self.sim.render.antialiasing_mode = "OFF"
