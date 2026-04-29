from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg

from .constants import (
    D405_FAR_CLIP,
    D405_NEAR_CLIP,
    D405_RGB_FOCAL_LENGTH,
    D405_RGB_HORIZONTAL_APERTURE,
    EXTERNAL_CAMERA_HEIGHT,
    EXTERNAL_CAMERA_WIDTH,
    EXTERNAL_CAMERA_NAME,
    HEAD_CAMERA_LINK_NAME,
    LEFT_WRIST_CAMERA_NAME,
    RIGHT_WRIST_CAMERA_NAME,
    WAIST_CAMERA_LINK_NAME,
    WAIST_CAMERA_NAME,
    WAIST_CAMERA_HEIGHT,
    WAIST_CAMERA_WIDTH,
    WRIST_CAMERA_HEIGHT,
    WRIST_CAMERA_WIDTH,
)


def build_external_camera_cfg() -> CameraCfg:
    return CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{HEAD_CAMERA_LINK_NAME}/{EXTERNAL_CAMERA_NAME}",
        update_period=0.0,
        update_latest_camera_pose=True,
        height=EXTERNAL_CAMERA_HEIGHT,
        width=EXTERNAL_CAMERA_WIDTH,
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
        height=WAIST_CAMERA_HEIGHT,
        width=WAIST_CAMERA_WIDTH,
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
        # Wrist cameras are driven by explicit world poses in env.py, so keep
        # them outside the robot prim hierarchy to avoid inheriting stale link
        # transforms while also being repositioned each frame.
        prim_path=f"{{ENV_REGEX_NS}}/{LEFT_WRIST_CAMERA_NAME}",
        update_period=0.0,
        update_latest_camera_pose=True,
        height=WRIST_CAMERA_HEIGHT,
        width=WRIST_CAMERA_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=D405_RGB_FOCAL_LENGTH,
            focus_distance=400.0,
            horizontal_aperture=D405_RGB_HORIZONTAL_APERTURE,
            clipping_range=(D405_NEAR_CLIP, D405_FAR_CLIP),
        ),
        # The live wrist camera mount is updated every frame in env.py.
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
    )


def build_right_wrist_camera_cfg() -> CameraCfg:
    return CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{RIGHT_WRIST_CAMERA_NAME}",
        update_period=0.0,
        update_latest_camera_pose=True,
        height=WRIST_CAMERA_HEIGHT,
        width=WRIST_CAMERA_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=D405_RGB_FOCAL_LENGTH,
            focus_distance=400.0,
            horizontal_aperture=D405_RGB_HORIZONTAL_APERTURE,
            clipping_range=(D405_NEAR_CLIP, D405_FAR_CLIP),
        ),
        # The live wrist camera mount is updated every frame in env.py.
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
    )
