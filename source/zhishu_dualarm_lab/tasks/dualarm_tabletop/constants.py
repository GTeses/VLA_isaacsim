from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
from zhishu_dualarm_lab.utils.local_paths import resolve_robot_urdf_path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
PACKAGE_ROOT = Path(__file__).resolve().parents[2]

PROJECT_NAME = "zhishu_dualarm_lab"
ENV_ID = "Zhishu-DualArm-Tabletop-Direct-v0"
POLICY_PROMPT = "Move both arms toward the cube and nudge it into the target zone without grasping."

ROBOT_URDF_PATH = resolve_robot_urdf_path()
ROBOT_USD_PATH = PACKAGE_ROOT / "assets/robots/zhishu_robot/usd/zhishu_robot.usd"

LEFT_ARM_JOINT_NAMES = tuple(f"left_joint{i}" for i in range(1, 8))
RIGHT_ARM_JOINT_NAMES = tuple(f"right_joint{i}" for i in range(1, 8))
ARM_JOINT_NAMES = LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES
LOCKED_JOINT_NAMES = (
    "waist_up_down_join",
    "waist_y_joint",
    "head_y_joint",
    "head_P_joint",
    "left_wheel_joint",
    "right_wheel_joint",
)
#123
LEFT_TCP_LINK_NAME = "left_link7"
RIGHT_TCP_LINK_NAME = "right_link7"
HEAD_CAMERA_LINK_NAME = "head_P_link"
WAIST_CAMERA_LINK_NAME = "waist_y_link"
LEFT_TCP_NAME = "left_tcp"
RIGHT_TCP_NAME = "right_tcp"
LEFT_TCP_OFFSET_POS = (0.0, 0.0, 0.0)
LEFT_TCP_OFFSET_ROT = (1.0, 0.0, 0.0, 0.0)
RIGHT_TCP_OFFSET_POS = (0.0, 0.0, 0.0)
RIGHT_TCP_OFFSET_ROT = (1.0, 0.0, 0.0, 0.0)

EXTERNAL_CAMERA_NAME = "external_camera"
WAIST_CAMERA_NAME = "waist_camera"
LEFT_WRIST_CAMERA_NAME = "left_wrist_camera"
RIGHT_WRIST_CAMERA_NAME = "right_wrist_camera"
HEAD_CAMERA_BODY_NAME = "head_camera_body"
WAIST_CAMERA_BODY_NAME = "waist_camera_body"
LEFT_WRIST_CAMERA_BODY_NAME = "left_wrist_camera_body"
RIGHT_WRIST_CAMERA_BODY_NAME = "right_wrist_camera_body"
EXTERNAL_CAMERA_HEIGHT = 256
EXTERNAL_CAMERA_WIDTH = 256
WAIST_CAMERA_HEIGHT = 256
WAIST_CAMERA_WIDTH = 256
WRIST_CAMERA_HEIGHT = 400
WRIST_CAMERA_WIDTH = 640
CAMERA_RGB_CHANNELS = 3
CAMERA_BODY_SIZE = (0.02, 0.05, 0.05)
HEAD_CAMERA_BODY_COLOR = (0.95, 0.80, 0.20)
WAIST_CAMERA_BODY_COLOR = (0.75, 0.75, 0.75)
LEFT_WRIST_CAMERA_BODY_COLOR = (0.25, 0.65, 0.95)
RIGHT_WRIST_CAMERA_BODY_COLOR = (0.95, 0.45, 0.25)

# The current URDF ends at link7 on both arms: there is no separate fixed
# hand adapter / dexterous hand link in this asset yet. Place the cameras
# back near the wrist-side connection area so the mounting logic matches the
# intended wrist location once the hand model is attached.
LEFT_WRIST_CAMERA_OFFSET_POS = (-0.045, 0.05, -0.05)
RIGHT_WRIST_CAMERA_OFFSET_POS = (-0.045, 0.05, 0.05)
LEFT_WRIST_CAMERA_OFFSET_ROT = (0.5, -0.5, 0.5, 0.5)
RIGHT_WRIST_CAMERA_OFFSET_ROT = (0.0, -0.7071, 0.0, -0.7071)

# Approximate RGB intrinsics for a wrist-mounted RealSense D405.
# This task still exposes RGB only, but the image size, FOV, and
# close-range clipping are moved closer to D405 behavior.
D405_RGB_HORIZONTAL_APERTURE = 20.955
D405_RGB_FOCAL_LENGTH = 11.3
D405_NEAR_CLIP = 0.07
D405_FAR_CLIP = 2.0

# Keep the policy-facing observation key name "external_image", but remount
# that camera onto the head near the mouth area instead of using a fixed
# world camera.
HEAD_CAMERA_OFFSET_POS = (-0.12, 0.0, 0.11)
HEAD_CAMERA_OFFSET_ROT = (0.0, -0.3007058, 0.0, 0.95371695)

# Keep an additional waist camera in the scene for future use even though it
# is not part of the current policy observation contract.
WAIST_CAMERA_OFFSET_POS = (-0.04, 0.0, 0.18)
WAIST_CAMERA_OFFSET_ROT = (1.0, 0.0, 0.0, 0.0)

ROBOT_INIT_POS = (0.0, 0.0, 0.0)
ROBOT_INIT_ROT = (1.0, 0.0, 0.0, 0.0)
ROBOT_INIT_JOINT_POS = {joint_name: 0.0 for joint_name in ARM_JOINT_NAMES + LOCKED_JOINT_NAMES}

GROUND_Z = 0.0
TABLE_SIZE = (0.90, 1.20, 0.05)
TABLE_POS = (0.78, 0.0, 0.73)
TABLE_COLOR = (0.28, 0.22, 0.16)
OBJECT_SIZE = (0.05, 0.05, 0.05)
OBJECT_START_POS = (0.65, -0.10, 0.78)
OBJECT_COLOR = (0.85, 0.20, 0.20)
TARGET_ZONE_SIZE = (0.16, 0.16, 0.01)
TARGET_ZONE_POS = (0.65, 0.22, 0.760)
TARGET_ZONE_COLOR = (0.15, 0.65, 0.25)
TARGET_REACHED_THRESHOLD = 0.08
TCP_OBJECT_REACHED_THRESHOLD = 0.16
TCP_MIDPOINT_OBJECT_THRESHOLD = 0.12
OBJECT_TARGET_PROGRESS_SCALE = 8.0
OBJECT_TARGET_SHAPING_SCALE = 4.0
TCP_OBJECT_SHAPING_SCALE = 6.0
TCP_GATHER_SHAPING_SCALE = 8.0
ACTION_PENALTY_SCALE = 0.01
REACH_SUCCESS_BONUS = 1.0
TARGET_SUCCESS_BONUS = 2.5

SIM_DT = 1.0 / 120.0
DECIMATION = 2
EPISODE_LENGTH_S = 10.0
ACTION_DIM = 14
ARM_ACTION_DELTA_SCALE = 0.3
JOINT_VELOCITY_SCALE = 0.1
RESET_JOINT_NOISE = 0.05

OBSERVATION_SPACE = gym.spaces.Dict(
    {
        "observation/external_image": gym.spaces.Box(
            low=0,
            high=255,
            shape=(EXTERNAL_CAMERA_HEIGHT, EXTERNAL_CAMERA_WIDTH, CAMERA_RGB_CHANNELS),
            dtype=np.uint8,
        ),
        "observation/left_wrist_image": gym.spaces.Box(
            low=0,
            high=255,
            shape=(WRIST_CAMERA_HEIGHT, WRIST_CAMERA_WIDTH, CAMERA_RGB_CHANNELS),
            dtype=np.uint8,
        ),
        "observation/right_wrist_image": gym.spaces.Box(
            low=0,
            high=255,
            shape=(WRIST_CAMERA_HEIGHT, WRIST_CAMERA_WIDTH, CAMERA_RGB_CHANNELS),
            dtype=np.uint8,
        ),
        "observation/joint_pos": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(ACTION_DIM,), dtype=np.float32),
        "observation/joint_vel": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(ACTION_DIM,), dtype=np.float32),
        "observation/last_action": gym.spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32),
        "observation/left_tcp_pose": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32),
        "observation/right_tcp_pose": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32),
        "observation/object_pose": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32),
        "observation/target_pose": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32),
        "observation/state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(70,), dtype=np.float32),
    }
)
ACTION_SPACE = gym.spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)
