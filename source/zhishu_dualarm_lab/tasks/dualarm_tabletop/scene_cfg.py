from __future__ import annotations

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass

from zhishu_dualarm_lab.assets.robots.zhishu_robot import build_zhishu_robot_cfg
from zhishu_dualarm_lab.utils.tcp_frames import make_tcp_frame

from .cameras import (
    build_external_camera_cfg,
    build_left_wrist_camera_cfg,
    build_right_wrist_camera_cfg,
    build_waist_camera_cfg,
)
from .constants import (
    LEFT_TCP_LINK_NAME,
    LEFT_TCP_NAME,
    LEFT_TCP_OFFSET_POS,
    LEFT_TCP_OFFSET_ROT,
    RIGHT_TCP_LINK_NAME,
    RIGHT_TCP_NAME,
    RIGHT_TCP_OFFSET_POS,
    RIGHT_TCP_OFFSET_ROT,
)
from .objects import build_ground_cfg, build_light_cfg, build_object_cfg, build_table_cfg, build_target_zone_cfg


@configclass
class ZhishuDualArmSceneCfg(InteractiveSceneCfg):
    """Scene with robot, tabletop objects, cameras, and TCP virtual frames."""

    ground = build_ground_cfg()
    dome_light = build_light_cfg()
    robot = build_zhishu_robot_cfg()
    table = build_table_cfg()
    object = build_object_cfg()
    target_zone = build_target_zone_cfg()
    external_camera = build_external_camera_cfg()
    waist_camera = build_waist_camera_cfg()
    left_wrist_camera = build_left_wrist_camera_cfg()
    right_wrist_camera = build_right_wrist_camera_cfg()
    tcp_frames = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        debug_vis=False,
        target_frames=[
            make_tcp_frame(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{LEFT_TCP_LINK_NAME}",
                name=LEFT_TCP_NAME,
                pos=LEFT_TCP_OFFSET_POS,
                rot=LEFT_TCP_OFFSET_ROT,
            ),
            make_tcp_frame(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{RIGHT_TCP_LINK_NAME}",
                name=RIGHT_TCP_NAME,
                pos=RIGHT_TCP_OFFSET_POS,
                rot=RIGHT_TCP_OFFSET_ROT,
            ),
        ],
    )
