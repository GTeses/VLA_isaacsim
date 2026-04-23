from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from zhishu_dualarm_lab.tasks.dualarm_tabletop.constants import (
    ARM_JOINT_NAMES,
    LOCKED_JOINT_NAMES,
    ROBOT_INIT_JOINT_POS,
    ROBOT_INIT_POS,
    ROBOT_INIT_ROT,
    ROBOT_USD_PATH,
)


def _assert_robot_usd_exists() -> None:
    if not Path(ROBOT_USD_PATH).is_file():
        raise FileNotFoundError(
            f"Robot USD not found: {ROBOT_USD_PATH}\n"
            "Run scripts/import_zhishu_robot_usd.py once before starting the environment."
        )


def build_zhishu_robot_cfg(prim_path: str = "{ENV_REGEX_NS}/Robot") -> ArticulationCfg:
    """Create the articulation config for the imported Zhishu robot USD."""
    _assert_robot_usd_exists()
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ROBOT_USD_PATH),
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=ROBOT_INIT_POS,
            rot=ROBOT_INIT_ROT,
            joint_pos=ROBOT_INIT_JOINT_POS,
        ),
        actuators={
            "dual_arms": ImplicitActuatorCfg(
                joint_names_expr=list(ARM_JOINT_NAMES),
                effort_limit_sim=200.0,
                velocity_limit_sim=5.0,
                stiffness=200.0,
                damping=20.0,
            ),
            "locked_joints": ImplicitActuatorCfg(
                joint_names_expr=list(LOCKED_JOINT_NAMES),
                effort_limit_sim=500.0,
                velocity_limit_sim=1.0,
                stiffness=5_000.0,
                damping=500.0,
            ),
        },
    )


