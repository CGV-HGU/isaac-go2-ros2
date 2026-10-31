# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Demonstration and Spawner script for Unitree AS2 + AgileX PiPER Mobile Manipulator.
Compatible with NVIDIA Isaac Sim 6.2.0 / 5.x and Isaac Lab.

Usage:
    ./isaaclab.sh -p scripts/demos/spawn_as2_piper.py
    ./isaaclab.sh -p scripts/demos/spawn_as2_piper.py --num_envs 4
    ./isaaclab.sh -p scripts/demos/spawn_as2_piper.py --standalone_arm
    ./isaaclab.sh -p scripts/demos/spawn_as2_piper.py --standalone_as2
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Spawn Unitree AS2 with AgileX PiPER Arm in Isaac Sim.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of robot instances to spawn.")
mode_group = parser.add_mutually_exclusive_group()
mode_group.add_argument("--standalone_arm", action="store_true", default=False, help="Spawn only standalone PiPER arm.")
mode_group.add_argument(
    "--standalone_as2", action="store_true", default=False, help="Spawn only standalone AS2 quadruped."
)
parser.add_argument(
    "--enable_sensors",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Attach AS2 RGB camera, metric Depth camera, and 16-ch 3D LiDAR sensors.",
)
parser.add_argument(
    "--enable_ros2",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Enable Isaac Sim ROS 2 Bridge (RGB, Depth, CameraInfo, 3D LiDAR, TF, Clock).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.standalone_arm and (args_cli.enable_sensors or args_cli.enable_ros2):
    parser.error("--standalone_arm cannot be combined with --enable_sensors or --enable_ros2 (no AS2 'base' trunk).")
if args_cli.enable_sensors or args_cli.enable_ros2:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows after Omniverse app launch."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.as2_piper import AS2_PIPER_CFG, PIPER_ARM_CFG, UNITREE_AS2_CFG
from isaaclab_assets.robots.as2_sensors import (
    attach_as2_sensors_to_scene_cfg,
    setup_as2_ros2_sensors_graph,
    summarize_as2_sensor_readings,
)


def _selected_robot_cfg():
    if args_cli.standalone_arm:
        print("[INFO] Spawning Standalone AgileX PiPER Robotic Arm (fixed root)...")
        return PIPER_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    if args_cli.standalone_as2:
        print("[INFO] Spawning Standalone Unitree AS2 Quadruped...")
        return UNITREE_AS2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    print("[INFO] Spawning Unitree AS2 + AgileX PiPER Mobile Manipulator...")
    return AS2_PIPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class DemoSceneCfg(InteractiveSceneCfg):
    """Multi-environment scene configuration honoring ``--num_envs``."""

    ground = sim_utils.AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = sim_utils.AssetBaseCfg(
        prim_path="/World/defaultDomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
    )
    robot = _selected_robot_cfg()


def main():
    """Main simulation loop for Unitree AS2 + AgileX PiPER mobile manipulator."""
    print("=" * 70)
    print(f" [Isaac Sim / Isaac Lab] AS2 + PiPER Demo (num_envs={args_cli.num_envs})")
    print(" Computer Graphics and Vision Lab (CGV)")
    print("=" * 70)

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device if args_cli.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.8, 2.8, 2.0], target=[0.0, 0.0, 0.45])

    scene_cfg = DemoSceneCfg(num_envs=max(1, args_cli.num_envs), env_spacing=2.2)
    if args_cli.enable_sensors:
        attach_as2_sensors_to_scene_cfg(
            scene_cfg, enable_rgbd=True, enable_lidar=True, ground_prim_path="/World/defaultGroundPlane"
        )
        print("[INFO] Attached AS2 RGB camera, metric Depth camera, and 16-ch 3D LiDAR.")
    scene = InteractiveScene(scene_cfg)
    robot: Articulation = scene["robot"]

    sim.reset()
    scene.reset()
    robot.set_joint_position_target(robot.data.default_joint_pos.clone())
    scene.write_data_to_sim()

    if args_cli.enable_ros2:
        existing_rp = None
        if args_cli.enable_sensors and "rgbd_camera" in scene.keys():
            rp_paths = getattr(scene["rgbd_camera"], "render_product_paths", None)
            if rp_paths:
                existing_rp = str(rp_paths[0])
        if scene.num_envs > 1:
            print("[INFO] --enable_ros2 publishes sensor streams for env_0 (/World/envs/env_0/Robot/base).")
        ros2_info = setup_as2_ros2_sensors_graph(
            sim.stage,
            simulation_app,
            robot_base_path="/World/envs/env_0/Robot/base",
            existing_camera_rp_path=existing_rp,
        )
        print(f"[INFO] ROS 2 Sensor Bridge active: {', '.join(ros2_info['topics'])}")

    print("[INFO] Simulation initialized. Starting dynamic demonstration loop...")

    step_count = 0
    sim_dt = sim.get_physics_dt()
    default_joint_pos = robot.data.default_joint_pos.clone()

    # Cache joint indices once outside the physics loop
    arm_yaw_idx, _ = robot.find_joints(["joint1"])
    arm_pitch_idx, _ = robot.find_joints(["joint2"])
    arm_elbow_idx, _ = robot.find_joints(["joint3"])
    grip_j7_idx, _ = robot.find_joints(["joint7"])
    grip_j8_idx, _ = robot.find_joints(["joint8"])

    while simulation_app.is_running():
        sim.step()
        scene.update(sim_dt)

        t = step_count * sim_dt
        targets = default_joint_pos.clone()

        if not args_cli.standalone_as2:
            if arm_yaw_idx:
                targets[:, arm_yaw_idx[0]] = 0.6 * math.sin(t * 1.5)
            if arm_pitch_idx:
                targets[:, arm_pitch_idx[0]] = 0.6 + 0.3 * math.sin(t * 1.0)
            if arm_elbow_idx:
                targets[:, arm_elbow_idx[0]] = -1.2 + 0.4 * math.cos(t * 1.0)
            # Symmetric jaw motion: joint7 in [0, +0.035], joint8 in [-0.035, 0]
            grip_open = 0.015 * (1.0 + math.sin(t * 2.0))
            if grip_j7_idx:
                targets[:, grip_j7_idx[0]] = grip_open
            if grip_j8_idx:
                targets[:, grip_j8_idx[0]] = -grip_open

        robot.set_joint_position_target(targets)
        scene.write_data_to_sim()

        step_count += 1
        if step_count % 500 == 0:
            sensor_msg = ""
            if args_cli.enable_sensors and not args_cli.standalone_arm:
                stats = summarize_as2_sensor_readings(scene)
                if stats:
                    sensor_msg = " | " + ", ".join(f"{k}={v:.2f}" for k, v in stats.items())
            print(f"[Step {step_count}] Simulation running at t={t:.2f}s | Envs: {scene.num_envs}{sensor_msg}")

    simulation_app.close()


if __name__ == "__main__":
    main()
