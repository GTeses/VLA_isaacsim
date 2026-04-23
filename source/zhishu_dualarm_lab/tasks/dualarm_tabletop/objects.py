from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

from .constants import (
    CAMERA_BODY_SIZE,
    GROUND_Z,
    HEAD_CAMERA_BODY_COLOR,
    HEAD_CAMERA_BODY_NAME,
    LEFT_WRIST_CAMERA_BODY_COLOR,
    LEFT_WRIST_CAMERA_BODY_NAME,
    OBJECT_COLOR,
    OBJECT_SIZE,
    OBJECT_START_POS,
    RIGHT_WRIST_CAMERA_BODY_COLOR,
    RIGHT_WRIST_CAMERA_BODY_NAME,
    TABLE_COLOR,
    TABLE_POS,
    TABLE_SIZE,
    TARGET_ZONE_COLOR,
    TARGET_ZONE_POS,
    TARGET_ZONE_SIZE,
    WAIST_CAMERA_BODY_COLOR,
    WAIST_CAMERA_BODY_NAME,
)


def build_ground_cfg() -> AssetBaseCfg:
    return AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())


def build_light_cfg() -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75)),
    )


def build_table_cfg() -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                max_depenetration_velocity=1.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=TABLE_COLOR),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=TABLE_POS),
    )


def build_object_cfg() -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=OBJECT_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=3.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=OBJECT_COLOR),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=OBJECT_START_POS),
    )


def build_target_zone_cfg() -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetZone",
        spawn=sim_utils.CuboidCfg(
            size=TARGET_ZONE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                max_depenetration_velocity=1.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=TARGET_ZONE_COLOR),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=TARGET_ZONE_POS),
    )


def _build_camera_body_cfg(name: str, color: tuple[float, float, float]) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=CAMERA_BODY_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                max_depenetration_velocity=1.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )


def build_head_camera_body_cfg() -> RigidObjectCfg:
    return _build_camera_body_cfg(HEAD_CAMERA_BODY_NAME, HEAD_CAMERA_BODY_COLOR)


def build_waist_camera_body_cfg() -> RigidObjectCfg:
    return _build_camera_body_cfg(WAIST_CAMERA_BODY_NAME, WAIST_CAMERA_BODY_COLOR)


def build_left_wrist_camera_body_cfg() -> RigidObjectCfg:
    return _build_camera_body_cfg(LEFT_WRIST_CAMERA_BODY_NAME, LEFT_WRIST_CAMERA_BODY_COLOR)


def build_right_wrist_camera_body_cfg() -> RigidObjectCfg:
    return _build_camera_body_cfg(RIGHT_WRIST_CAMERA_BODY_NAME, RIGHT_WRIST_CAMERA_BODY_COLOR)
