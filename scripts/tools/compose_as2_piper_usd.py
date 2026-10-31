# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Composite USD generator for Unitree AS2 + AgileX PiPER Mobile Manipulator.
Runs inside Isaac Sim (Isaac Sim 6.2.0 / 5.x / 4.5).

Note:
    By default, ``AS2_PIPER_CFG`` converts the unified 20-DOF ``as2_piper.urdf``
    directly at runtime. Use this script only if you want to generate a standalone
    single-root USD stage welding an AS2 USD and ``piper_v1.usd`` via ``UsdPhysics.FixedJoint``,
    and set ``AS2_PIPER_USD_PATH`` to use it.

Usage:
    ./isaaclab.sh -p scripts/tools/compose_as2_piper_usd.py
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Compose AS2 + AgileX PiPER USD.")
parser.add_argument(
    "--output_path",
    type=str,
    default="robot_arms/piper/descriptions/as2_piper_composed.usd",
    help="Output USD file path.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


def main():
    print("=" * 70)
    print(" [USD Composer] Assembling Unitree AS2 + AgileX PiPER in Isaac Sim")
    print("=" * 70)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    piper_usd_path = os.path.join(repo_root, "robot_arms/piper/piper_isaac_sim/USD/piper_v1.usd")
    if not os.path.isfile(piper_usd_path):
        print(
            f"[ERROR] PiPER USD not found at '{piper_usd_path}'. "
            "Run 'bash robot_arms/piper/setup_piper.sh' first."
        )
        simulation_app.close()
        sys.exit(1)

    output_usd = (
        args_cli.output_path
        if os.path.isabs(args_cli.output_path)
        else os.path.join(repo_root, args_cli.output_path)
    )
    os.makedirs(os.path.dirname(output_usd), exist_ok=True)

    # Create or overwrite output stage cleanly
    layer = Sdf.Layer.FindOrOpen(output_usd)
    if layer is not None:
        layer.Clear()
        stage = Usd.Stage.Open(layer)
    else:
        stage = Usd.Stage.CreateNew(output_usd)

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # Root Robot Prim matching '{ENV_REGEX_NS}/Robot/base' convention
    robot_prim_path = "/Robot"
    robot_prim = stage.DefinePrim(robot_prim_path, "Xform")
    stage.SetDefaultPrim(robot_prim)

    as2_usd_path = os.getenv("UNITREE_AS2_USD_PATH", f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd")
    print(f"[INFO] Referencing AS2 base directly onto {robot_prim_path} from: {as2_usd_path}")
    robot_prim.GetReferences().AddReference(as2_usd_path)

    print(f"[INFO] Referencing AgileX PiPER arm from: {piper_usd_path}")
    piper_prim_path = f"{robot_prim_path}/PiPER"
    piper_prim = stage.DefinePrim(piper_prim_path, "Xform")
    piper_prim.GetReferences().AddReference(piper_usd_path)

    # Strip nested ArticulationRootAPI on PiPER subtree so PhysX sees a single articulation root
    for prim in Usd.PrimRange(piper_prim):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)

    arm_xform = UsdGeom.Xformable(piper_prim)
    arm_xform.ClearXformOpOrder()
    arm_xform.AddTranslateOp().Set(Gf.Vec3d(0.10, 0.0, 0.13))
    arm_xform.AddRotateXYZOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))

    # Weld AS2 '/Robot/base' to PiPER '/Robot/PiPER/base_link' via UsdPhysics.FixedJoint
    arm_root_body = f"{piper_prim_path}/base_link"
    if not stage.GetPrimAtPath(arm_root_body).IsValid():
        arm_root_body = piper_prim_path

    mount_joint = UsdPhysics.FixedJoint.Define(stage, f"{robot_prim_path}/as2_to_piper_mount")
    mount_joint.CreateBody0Rel().SetTargets([Sdf.Path(f"{robot_prim_path}/base")])
    mount_joint.CreateBody1Rel().SetTargets([Sdf.Path(arm_root_body)])
    mount_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.10, 0.0, 0.13))
    mount_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    mount_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    mount_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    stage.GetRootLayer().Save()
    print(f"[SUCCESS] Single-articulation AS2 + PiPER USD saved at: {output_usd}")
    print(f"[INFO] Export AS2_PIPER_USD_PATH='{output_usd}' to use this USD instead of URDF.")

    simulation_app.close()


if __name__ == "__main__":
    main()
