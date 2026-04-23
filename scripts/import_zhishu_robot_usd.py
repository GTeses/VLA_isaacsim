from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Import the Zhishu robot URDF into a project-local USD asset.")
    parser.add_argument("--fix-base", action="store_true", default=True, help="Fix the robot base in the generated USD.")
    parser.add_argument("--merge-joints", action="store_true", default=False, help="Merge fixed joints during import.")
    parser.add_argument("--force", action="store_true", help="Force re-conversion even if the USD already exists.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    from zhishu_dualarm_lab.tasks.dualarm_tabletop.constants import ROBOT_URDF_PATH, ROBOT_USD_PATH

    # Allow URDF package:// mesh resolution for the zhishu_robot_description package.
    urdf_package_root = ROBOT_URDF_PATH.parents[2]
    existing_ros_package_path = os.environ.get("ROS_PACKAGE_PATH", "")
    os.environ["ROS_PACKAGE_PATH"] = (
        f"{urdf_package_root}:{existing_ros_package_path}" if existing_ros_package_path else str(urdf_package_root)
    )

    ROBOT_USD_PATH.parent.mkdir(parents=True, exist_ok=True)
    converter_cfg = UrdfConverterCfg(
        asset_path=str(ROBOT_URDF_PATH),
        usd_dir=str(ROBOT_USD_PATH.parent),
        usd_file_name=ROBOT_USD_PATH.name,
        fix_base=args_cli.fix_base,
        merge_fixed_joints=args_cli.merge_joints,
        force_usd_conversion=args_cli.force,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=400.0, damping=40.0),
            target_type="position",
        ),
    )
    converter = UrdfConverter(converter_cfg)
    print(f"[INFO] URDF: {ROBOT_URDF_PATH}")
    print(f"[INFO] USD : {converter.usd_path}")
    simulation_app.close()


if __name__ == "__main__":
    main()
