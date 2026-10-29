# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Composite USD generator for Unitree AS2 + AgileX PiPER Mobile Manipulator.
Runs inside Isaac Sim (Isaac Sim 6.2.0 / 5.x / 4.5).

Usage:
    ./isaaclab.sh -p scripts/tools/compose_as2_piper_usd.py
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Compose AS2 + AgileX PiPER USD.")
parser.add_argument(
    "--output_path",
    type=str,
    default="robot_arms/piper/descriptions/as2_piper.usd",
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
    print(" [USD Composer] Assembling Unitree AS2 + AgileX PiPER in Isaac Sim 6.2.0")
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

    # 2. Reference Unitree AS2 Base (can be overridden with UNITREE_AS2_USD_PATH)
    as2_usd_path = os.getenv("UNITREE_AS2_USD_PATH", f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd")
    print(f"[INFO] Referencing AS2 base from: {as2_usd_path}")
    as2_prim = stage.DefinePrim(f"{robot_prim_path}/AS2")
    as2_prim.GetReferences().AddReference(as2_usd_path)

    # 3. Reference AgileX PiPER arm on top deck
    piper_usd_path = os.path.join(repo_root, "robot_arms/piper/piper_isaac_sim/USD/piper_v1.usd")
    if os.path.exists(piper_usd_path):
        print(f"[INFO] Referencing AgileX PiPER arm from: {piper_usd_path}")
        piper_prim = stage.DefinePrim(f"{robot_prim_path}/AS2/base/arm_mount")
        piper_prim.GetReferences().AddReference(piper_usd_path)

        # Set mount transform: 10cm forward, 13cm up on AS2 top deck
        arm_xform = UsdGeom.Xformable(piper_prim)
        arm_xform.AddTranslateOp().Set(Gf.Vec3d(0.10, 0.0, 0.13))
        arm_xform.AddRotateXYZOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    else:
        print(f"[WARN] PiPER USD not found at {piper_usd_path}. Please initialize submodule first.")

    # 4. Save composite stage
    stage.GetRootLayer().Save()
    print(f"[SUCCESS] Composite AS2 + PiPER USD generated at: {output_usd}")

    simulation_app.close()


if __name__ == "__main__":
    main()
