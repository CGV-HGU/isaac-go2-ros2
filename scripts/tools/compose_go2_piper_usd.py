# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Composite USD generator for Unitree Go2 + AgileX PiPER Mobile Manipulator.
Runs inside Isaac Sim (Isaac Sim 6.2.0 / 5.x / 4.5).

Usage:
    ./isaaclab.sh -p scripts/tools/compose_go2_piper_usd.py
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Compose Go2 + AgileX PiPER USD.")
parser.add_argument(
    "--output_path",
    type=str,
    default="robot_arms/piper/descriptions/go2_piper.usd",
    help="Output USD file path.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Gf, Usd, UsdGeom, UsdPhysics
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


def main():
    print("=" * 70)
    print(" [USD Composer] Assembling Unitree Go2 + AgileX PiPER in Isaac Sim")
    print("=" * 70)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    output_usd = os.path.join(repo_root, args_cli.output_path)
    os.makedirs(os.path.dirname(output_usd), exist_ok=True)

    # 1. Create a fresh stage
    stage = Usd.Stage.CreateNew(output_usd)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # Root Robot Prim
    robot_prim_path = "/World/Robot"
    robot_xform = UsdGeom.Xform.Define(stage, robot_prim_path)
    stage.SetDefaultPrim(robot_xform.GetPrim())

    # 2. Reference Unitree Go2 USD
    go2_usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd"
    print(f"[INFO] Referencing Go2 base from: {go2_usd_path}")
    go2_prim = stage.DefinePrim(f"{robot_prim_path}/Go2")
    go2_prim.GetReferences().AddReference(go2_usd_path)

    # 3. Reference AgileX PiPER USD on top deck
    piper_usd_path = os.path.join(repo_root, "robot_arms/piper/piper_isaac_sim/USD/piper_v1.usd")
    if os.path.exists(piper_usd_path):
        print(f"[INFO] Referencing AgileX PiPER arm from: {piper_usd_path}")
        piper_prim = stage.DefinePrim(f"{robot_prim_path}/Go2/base/arm_mount")
        piper_prim.GetReferences().AddReference(piper_usd_path)

        # Set mount transform: 10cm forward, 12.5cm up on trunk
        arm_xform = UsdGeom.Xformable(piper_prim)
        arm_xform.AddTranslateOp().Set(Gf.Vec3d(0.10, 0.0, 0.125))
        arm_xform.AddRotateXYZOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    else:
        print(f"[WARN] PiPER USD not found at {piper_usd_path}. Please initialize submodule first.")

    # 4. Save composite stage
    stage.GetRootLayer().Save()
    print(f"[SUCCESS] Composite Go2 + PiPER USD generated at: {output_usd}")

    simulation_app.close()


if __name__ == "__main__":
    main()
