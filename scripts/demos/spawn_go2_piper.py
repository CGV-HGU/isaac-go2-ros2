# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Demonstration and Spawner script for Unitree Go2 + AgileX PiPER Mobile Manipulator.
Compatible with NVIDIA Isaac Sim 6.2.0 and Isaac Lab.

Usage:
    ./isaaclab.sh -p scripts/demos/spawn_go2_piper.py
    ./isaaclab.sh -p scripts/demos/spawn_go2_piper.py --num_envs 4
    ./isaaclab.sh -p scripts/demos/spawn_go2_piper.py --standalone_arm
"""

import argparse
import math
import sys

from isaaclab.app import AppLauncher

# 1. Argument Parser
parser = argparse.ArgumentParser(description="Spawn Unitree Go2 with AgileX PiPER Arm in Isaac Sim.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of robot instances to spawn.")
parser.add_argument("--standalone_arm", action="store_true", default=False, help="Spawn only standalone PiPER arm.")
parser.add_argument("--enable_ros2", action="store_true", default=False, help="Enable omni.isaac.ros2_bridge.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 2. Launch Isaac Sim / Omniverse App
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows after Omniverse app launch."""

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab_assets.robots.go2_piper import GO2_PIPER_CFG, PIPER_ARM_CFG


def main():
    """Main simulation loop for Go2 + AgileX PiPER mobile manipulator."""
    print("=" * 70)
    print(" [Isaac Sim 6.2.0 / Isaac Lab] Go2 + AgileX PiPER Mobile Manipulator")
    print(" Computer Graphics and Vision Lab (CGV)")
    print("=" * 70)

    # Enable ROS 2 Bridge if requested
    if args_cli.enable_ros2:
        try:
            from omni.isaac.core.utils.extensions import enable_extension
            enable_extension("omni.isaac.ros2_bridge")
            print("[INFO] omni.isaac.ros2_bridge enabled successfully.")
        except Exception as e:
            print(f"[WARN] Failed to enable ROS 2 Bridge: {e}")

    # Build Scene Configuration
    sim_cfg = sim_utils.SimulationCfg(device="cuda:0" if torch.cuda.is_available() else "cpu")
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.5, 2.5, 1.8], target=[0.0, 0.0, 0.4])

    # Ground plane and lighting
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.85, 0.85, 0.85))
    light_cfg.func("/World/defaultDomeLight", light_cfg)

    # Spawn Robot Articulation
    if args_cli.standalone_arm:
        print("[INFO] Spawning Standalone AgileX PiPER Robotic Arm...")
        robot_cfg = PIPER_ARM_CFG.replace(prim_path="/World/Robot")
    else:
        print("[INFO] Spawning Unitree Go2 + AgileX PiPER Mobile Manipulator...")
        robot_cfg = GO2_PIPER_CFG.replace(prim_path="/World/Robot")

    robot = Articulation(robot_cfg)

    # Reset simulation
    sim.reset()
    robot.reset()
    print("[INFO] Simulation initialized. Starting demonstration motion loop...")

    step_count = 0
    sim_dt = sim.get_physics_dt()

    # Default joint targets
    default_joint_pos = robot.data.default_joint_pos.clone()

    while simulation_app.is_running():
        sim.step()
        robot.update(sim_dt)

        t = step_count * sim_dt
        targets = default_joint_pos.clone()

        # Arm dynamic trajectory demonstration (smooth sine wave)
        # joint1: base yaw (-0.5 to 0.5 rad)
        # joint2: shoulder pitch
        # joint3: elbow pitch
        # joint7/8: gripper open/close
        if not args_cli.standalone_arm and robot.data.joint_names:
            arm_yaw_idx = [i for i, name in enumerate(robot.data.joint_names) if "joint1" in name]
            arm_pitch_idx = [i for i, name in enumerate(robot.data.joint_names) if "joint2" in name]
            gripper_idx = [i for i, name in enumerate(robot.data.joint_names) if "joint7" in name or "joint8" in name]

            if arm_yaw_idx:
                targets[:, arm_yaw_idx[0]] = 0.6 * math.sin(t * 1.5)
            if arm_pitch_idx:
                targets[:, arm_pitch_idx[0]] = 0.6 + 0.3 * math.sin(t * 1.0)
            for g_idx in gripper_idx:
                targets[:, g_idx] = 0.015 * (1.0 + math.sin(t * 2.0))

        robot.set_joint_position_target(targets)
        robot.write_data_to_sim()

        step_count += 1
        if step_count % 500 == 0:
            print(f"[Step {step_count}] Running simulation at t={t:.2f}s | Target FPS: {1.0/sim_dt:.0f}")

    # Shutdown
    simulation_app.close()


if __name__ == "__main__":
    main()
