from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg

from .constants import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    EXTERNAL_CAMERA_NAME,
    HEAD_CAMERA_LINK_NAME,
    HEAD_CAMERA_OFFSET_POS,
    HEAD_CAMERA_OFFSET_ROT,
    LEFT_TCP_LINK_NAME,
    LEFT_WRIST_CAMERA_NAME,
    LEFT_WRIST_CAMERA_OFFSET_POS,
    LEFT_WRIST_CAMERA_OFFSET_ROT,
    RIGHT_TCP_LINK_NAME,
    RIGHT_WRIST_CAMERA_NAME,
    RIGHT_WRIST_CAMERA_OFFSET_POS,
    RIGHT_WRIST_CAMERA_OFFSET_ROT,
    WAIST_CAMERA_LINK_NAME,
    WAIST_CAMERA_NAME,
    WAIST_CAMERA_OFFSET_POS,
    WAIST_CAMERA_OFFSET_ROT,
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
        offset=CameraCfg.OffsetCfg(pos=HEAD_CAMERA_OFFSET_POS, rot=HEAD_CAMERA_OFFSET_ROT, convention="world"),
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
        offset=CameraCfg.OffsetCfg(pos=WAIST_CAMERA_OFFSET_POS, rot=WAIST_CAMERA_OFFSET_ROT, convention="world"),
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
        offset=CameraCfg.OffsetCfg(
            pos=LEFT_WRIST_CAMERA_OFFSET_POS,
            rot=LEFT_WRIST_CAMERA_OFFSET_ROT,
            convention="world",
        ),
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
        offset=CameraCfg.OffsetCfg(
            pos=RIGHT_WRIST_CAMERA_OFFSET_POS,
            rot=RIGHT_WRIST_CAMERA_OFFSET_ROT,
            convention="world",
        ),
    )
