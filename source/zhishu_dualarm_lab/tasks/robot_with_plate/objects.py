from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

from .constants import PLATE_INIT_POS, PLATE_INIT_ROT, PLATE_USD_PATH


def _assert_plate_usd_exists() -> None:
    if not Path(PLATE_USD_PATH).is_file():
        raise FileNotFoundError(f"Plate USD not found: {PLATE_USD_PATH}")


def build_plate_cfg() -> RigidObjectCfg:
    _assert_plate_usd_exists()
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Plate",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(PLATE_USD_PATH),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                max_depenetration_velocity=1.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=PLATE_INIT_POS,
            rot=PLATE_INIT_ROT,
        ),
    )
