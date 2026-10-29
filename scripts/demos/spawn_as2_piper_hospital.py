# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Demonstration and Interactive Simulation: Unitree AS2 + AgileX PiPER in Hospital Environment.
Features opening and closing of an interactive articulated door (revolute hinge + handle).

Compatible with NVIDIA Isaac Sim 6.2.0 and Isaac Lab.

Usage:
    ./spawn_as2_piper_hospital.sh
    ./isaaclab.sh -p scripts/demos/spawn_as2_piper_hospital.py
    ./isaaclab.sh -p scripts/demos/spawn_as2_piper_hospital.py --auto_door
    ./isaaclab.sh -p scripts/demos/spawn_as2_piper_hospital.py --hospital_usd <path_to_custom_hospital.usd>
"""

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

# 1. CLI Arguments
parser = argparse.ArgumentParser(
    description="Spawn Unitree AS2 + AgileX PiPER with interactive opening/closing door in Hospital scene."
)
parser.add_argument(
    "--hospital_usd",
    type=str,
    default=os.getenv("HOSPITAL_USD_PATH", None),
    help="Custom path or Nucleus path for Hospital USD environment.",
)
parser.add_argument(
    "--auto_door",
    action="store_true",
    default=True,
    help="Run automated door opening and closing manipulation sequence.",
)
parser.add_argument(
    "--enable_ros2",
    action="store_true",
    default=False,
    help="Enable omni.isaac.ros2_bridge extension.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 2. Launch Omniverse Simulation App
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs inside Omniverse Python environment."""

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets.robots.as2_piper import AS2_PIPER_CFG
from isaaclab_assets.robots.hospital_door import HOSPITAL_DOOR_CFG


def main():
    print("=" * 75)
    print(" [Isaac Sim 6.2.0] Unitree AS2 + AgileX PiPER Hospital Mobile Manipulation")
    print(" Interactive Articulated Door (Open & Close Simulation)")
    print(" Computer Graphics and Vision Lab (CGV)")
    print("=" * 75)

    # 1. Optional ROS 2 Bridge
    if args_cli.enable_ros2:
        try:
            from omni.isaac.core.utils.extensions import enable_extension
            enable_extension("omni.isaac.ros2_bridge")
            print("[INFO] omni.isaac.ros2_bridge enabled successfully.")
        except Exception as e:
            print(f"[WARN] Could not enable ROS 2 Bridge: {e}")

    # 2. Simulation Context Setup
    sim_cfg = sim_utils.SimulationCfg(device="cuda:0" if torch.cuda.is_available() else "cpu")
    sim = sim_utils.SimulationContext(sim_cfg)

    # Cinematic Camera View overlooking AS2, PiPER arm, and Hospital Door
    sim.set_camera_view(eye=[2.6, -1.8, 1.6], target=[0.7, 0.0, 0.8])

    # 3. Environment & Lighting Setup
    hospital_path = args_cli.hospital_usd
    if hospital_path is None:
        hospital_path = f"{ISAAC_NUCLEUS_DIR}/Environments/Hospital/hospital.usd"

    print(f"[INFO] Resolving Hospital Environment: {hospital_path}")
    try:
        env_cfg = sim_utils.UsdFileCfg(usd_path=hospital_path)
        env_cfg.func("/World/HospitalEnvironment", env_cfg)
        print("[INFO] Hospital USD environment successfully referenced.")
    except Exception as e:
        print(f"[WARN] Remote Nucleus hospital.usd unavailable ({e}). Using local high-fidelity hospital floor stage.")
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    # Studio Dome Lighting for clear visual tracking
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.95, 0.95, 1.0))
    light_cfg.func("/World/HospitalLighting", light_cfg)

    # 4. Spawn Articulated Hospital Door (Revolute Hinge + Spring Latch Handle)
    print("[INFO] Spawning Articulated Hospital Door at doorway (x=1.1, y=0.0)...")
    door_cfg = HOSPITAL_DOOR_CFG.replace(prim_path="/World/Door")
    # Position door frame right in front of the robot's approach path
    door_cfg.init_state.pos = (1.1, -0.45, 0.0)
    door_cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    door = Articulation(door_cfg)

    # 5. Spawn Unitree AS2 + AgileX PiPER Mobile Manipulator
    print("[INFO] Spawning Unitree AS2 Quadruped + AgileX PiPER 6-DOF Arm...")
    robot_cfg = AS2_PIPER_CFG.replace(prim_path="/World/Robot")
    robot_cfg.init_state.pos = (0.0, 0.0, 0.45)
    robot = Articulation(robot_cfg)

    # 6. Initialize Physics State
    sim.reset()
    robot.reset()
    door.reset()

    print("[INFO] Multi-body Articulation initialized. Door & Robot ready.")
    print("=" * 75)
    print(" Door Controls:")
    print("   - Automatic Mode: AS2 positions arm -> Gripper reaches handle -> Pushes door open -> Door closes.")
    print("=" * 75)

    step_count = 0
    sim_dt = sim.get_physics_dt()

    # Base initial joint positions
    default_robot_pos = robot.data.default_joint_pos.clone()
    default_door_pos = door.data.default_joint_pos.clone()

    cycle_period = 14.0  # seconds per full open/close interaction cycle

    while simulation_app.is_running():
        sim.step()
        robot.update(sim_dt)
        door.update(sim_dt)

        t = step_count * sim_dt
        cycle_t = t % cycle_period

        robot_targets = default_robot_pos.clone()
        door_targets = default_door_pos.clone()

        # Find Joint Indices
        # PiPER Arm Joints
        arm_j1_idx = [i for i, name in enumerate(robot.data.joint_names) if "joint1" in name]
        arm_j2_idx = [i for i, name in enumerate(robot.data.joint_names) if "joint2" in name]
        arm_j3_idx = [i for i, name in enumerate(robot.data.joint_names) if "joint3" in name]
        arm_j4_idx = [i for i, name in enumerate(robot.data.joint_names) if "joint4" in name]
        arm_j5_idx = [i for i, name in enumerate(robot.data.joint_names) if "joint5" in name]
        arm_j6_idx = [i for i, name in enumerate(robot.data.joint_names) if "joint6" in name]
        gripper_idx = [i for i, name in enumerate(robot.data.joint_names) if "joint7" in name or "joint8" in name]

        # Door Joints
        hinge_idx = [i for i, name in enumerate(door.data.joint_names) if "hinge" in name]
        handle_idx = [i for i, name in enumerate(door.data.joint_names) if "handle" in name]

        if args_cli.auto_door:
            # Phase 1 (0 ~ 2.5s): Neutral stand & arm preparation
            if cycle_t < 2.5:
                blend = min(1.0, cycle_t / 2.0)
                if arm_j2_idx:
                    robot_targets[:, arm_j2_idx[0]] = 0.4 * blend
                if arm_j3_idx:
                    robot_targets[:, arm_j3_idx[0]] = -0.6 * blend

            # Phase 2 (2.5 ~ 6.0s): Arm reaches toward door handle
            elif cycle_t < 6.0:
                p2 = (cycle_t - 2.5) / 3.5
                if arm_j1_idx:
                    robot_targets[:, arm_j1_idx[0]] = 0.15 * p2  # yaw toward latch
                if arm_j2_idx:
                    robot_targets[:, arm_j2_idx[0]] = 0.4 + 0.5 * p2  # extend forward
                if arm_j3_idx:
                    robot_targets[:, arm_j3_idx[0]] = -0.6 - 0.4 * p2
                if arm_j5_idx:
                    robot_targets[:, arm_j5_idx[0]] = -0.3 * p2  # wrist tilt
                for g_idx in gripper_idx:
                    robot_targets[:, g_idx] = 0.025 * p2  # open gripper ready for handle

            # Phase 3 (6.0 ~ 10.0s): Press handle & push door open (0 -> ~1.4 rad / 80 deg)
            elif cycle_t < 10.0:
                p3 = (cycle_t - 6.0) / 4.0
                open_angle = 1.35 * (1.0 - math.cos(p3 * math.pi * 0.5))  # smooth open

                # Depress door handle
                if handle_idx:
                    door_targets[:, handle_idx[0]] = 0.4 * math.sin(p3 * math.pi)

                # Drive door hinge open
                if hinge_idx:
                    door_targets[:, hinge_idx[0]] = open_angle

                # Arm follows the door swing trajectory
                if arm_j1_idx:
                    robot_targets[:, arm_j1_idx[0]] = 0.15 - 0.35 * p3
                if arm_j2_idx:
                    robot_targets[:, arm_j2_idx[0]] = 0.9 - 0.2 * p3
                if arm_j3_idx:
                    robot_targets[:, arm_j3_idx[0]] = -1.0 + 0.3 * p3

            # Phase 4 (10.0 ~ 14.0s): Door slowly closes with damping, arm resets to ready pose
            else:
                p4 = (cycle_t - 10.0) / 4.0
                close_factor = 0.5 * (1.0 + math.cos(p4 * math.pi))  # smooth close from 1.0 to 0.0

                if hinge_idx:
                    door_targets[:, hinge_idx[0]] = 1.35 * close_factor
                if handle_idx:
                    door_targets[:, handle_idx[0]] = 0.0

                # Retract arm smoothly
                if arm_j2_idx:
                    robot_targets[:, arm_j2_idx[0]] = 0.7 * close_factor
                if arm_j3_idx:
                    robot_targets[:, arm_j3_idx[0]] = -0.7 * close_factor
                for g_idx in gripper_idx:
                    robot_targets[:, g_idx] = 0.01 * close_factor

        # Write actuation targets to simulator
        robot.set_joint_position_target(robot_targets)
        door.set_joint_position_target(door_targets)

        robot.write_data_to_sim()
        door.write_data_to_sim()

        step_count += 1
        if step_count % 300 == 0:
            hinge_val = door.data.joint_pos[0, hinge_idx[0]].item() if hinge_idx else 0.0
            print(
                f"[t={t:.1f}s | Cycle {cycle_t:.1f}s] Door Hinge Angle: {math.degrees(hinge_val):.1f}° | FPS: {1.0/sim_dt:.0f}"
            )

    simulation_app.close()


if __name__ == "__main__":
    main()
