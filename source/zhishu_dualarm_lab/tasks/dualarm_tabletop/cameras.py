from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg

from .constants import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    EXTERNAL_CAMERA_NAME,
    HEAD_CAMERA_LINK_NAME,
    LEFT_WRIST_CAMERA_NAME,
    LEFT_TCP_LINK_NAME,
    RIGHT_TCP_LINK_NAME,
    RIGHT_WRIST_CAMERA_NAME,
    WAIST_CAMERA_LINK_NAME,
    WAIST_CAMERA_NAME,
)


def build_external_camera_cfg() -> CameraCfg:
    return CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{HEAD_CAMERA_LINK_NAME}/{EXTERNAL_CAMERA_NAME}",
        update_period=0.0,
        update_latest_camera_pose=True,
        height=CAMERA_HEIGHT,
        width=CAMERA_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
    )


def build_waist_camera_cfg() -> CameraCfg:
    return CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{WAIST_CAMERA_LINK_NAME}/{WAIST_CAMERA_NAME}",
        update_period=0.0,
        update_latest_camera_pose=True,
        height=CAMERA_HEIGHT,
        width=CAMERA_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
    )


def build_left_wrist_camera_cfg() -> CameraCfg:
    return CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{LEFT_TCP_LINK_NAME}/{LEFT_WRIST_CAMERA_NAME}",
        update_period=0.0,
        update_latest_camera_pose=True,
        height=CAMERA_HEIGHT,
        width=CAMERA_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.02, 100.0),
        ),
        # The live wrist camera mount is updated every frame in env.py.
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
    )


def build_right_wrist_camera_cfg() -> CameraCfg:
    return CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{RIGHT_TCP_LINK_NAME}/{RIGHT_WRIST_CAMERA_NAME}",
        update_period=0.0,
        update_latest_camera_pose=True,
        height=CAMERA_HEIGHT,
        width=CAMERA_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.02, 100.0),
        ),
        # The live wrist camera mount is updated every frame in env.py.
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
    )
