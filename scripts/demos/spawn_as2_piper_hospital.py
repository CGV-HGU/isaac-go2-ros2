# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Demonstration and Interactive Simulation: Unitree AS2 + AgileX PiPER in Hospital Environment.
Supports manual PiPER 6-DOF arm + symmetric gripper teleop AND proximity-gated automatic door
opening (Key '1') across multiple hospital door mechanisms (lever_push, lever_pull, sliding, push_bar).

Usage:
    ./spawn_as2_piper_hospital.sh
    ./isaaclab.sh -p scripts/demos/spawn_as2_piper_hospital.py --door_type lever_push
    ./isaaclab.sh -p scripts/demos/spawn_as2_piper_hospital.py --door_type sliding --auto_door
    ./isaaclab.sh -p scripts/demos/spawn_as2_piper_hospital.py --spawn_far  # test proximity rejection
"""

import argparse
import math
import os

from isaaclab.app import AppLauncher

DOOR_TYPE_CHOICES = ("lever_push", "lever_pull", "sliding", "push_bar")

parser = argparse.ArgumentParser(
    description="Spawn Unitree AS2 + AgileX PiPER with multi-type interactive door in Hospital scene."
)
parser.add_argument(
    "--hospital_usd",
    type=str,
    default=os.getenv("HOSPITAL_USD_PATH", None),
    help="Custom path or Nucleus path for Hospital USD environment.",
)
parser.add_argument(
    "--door_type",
    type=str,
    choices=DOOR_TYPE_CHOICES,
    default="lever_push",
    help="Hospital door mechanism type (lever_push, lever_pull, sliding, push_bar).",
)
parser.add_argument(
    "--auto_door",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Automatically trigger door opening sequence without waiting for Key '1'.",
)
parser.add_argument(
    "--spawn_far",
    action="store_true",
    default=False,
    help="Spawn robot 1.6m away from door to test proximity rejection and WASD planar approach.",
)
parser.add_argument(
    "--enable_sensors",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Attach AS2 RGB camera, metric Depth camera, and 3D LiDAR sensors.",
)
parser.add_argument(
    "--enable_ros2",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Enable Isaac Sim ROS 2 Bridge (RGB, Depth, CameraInfo, RTX 3D LiDAR, TF, Clock).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.enable_sensors or args_cli.enable_ros2:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs inside Omniverse Python environment."""

import carb
import carb.input
import omni.appwindow
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets.robots.as2_piper import AS2_PIPER_CFG
from isaaclab_assets.robots.as2_sensors import (
    create_as2_rgbd_camera_cfg,
    setup_as2_ros2_sensors_graph,
    spawn_as2_rtx_lidar_prim,
)
from isaaclab_assets.robots.door_manipulation import (
    DoorAutoOpener,
    TeleopControlState,
    compute_door_handle_world_pos,
)
from isaaclab_assets.robots.hospital_door import get_hospital_door_cfg


def main():
    print("=" * 75)
    print(" [Isaac Sim 6.2.0 / 5.x] Unitree AS2 + AgileX PiPER Hospital Manipulation")
    print(f" Door Mechanism : {args_cli.door_type} | Auto Loop: {args_cli.auto_door}")
    print(" Computer Graphics and Vision Lab (CGV)")
    print("=" * 75)

    sim_device = args_cli.device if args_cli.device else ("cuda:0" if torch.cuda.is_available() else "cpu")
    sim_cfg = sim_utils.SimulationCfg(device=sim_device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[3.1, -2.0, 1.45], target=[1.65, -0.55, 0.75])

    # Resolve Hospital USD environment safely
    explicit_hospital = args_cli.hospital_usd is not None
    hospital_path = args_cli.hospital_usd
    if hospital_path is None and ISAAC_NUCLEUS_DIR is not None:
        hospital_path = f"{ISAAC_NUCLEUS_DIR}/Environments/Hospital/hospital.usd"

    if hospital_path is not None:
        print(f"[INFO] Resolving Hospital Environment: {hospital_path}")
        try:
            env_cfg = sim_utils.UsdFileCfg(usd_path=hospital_path)
            env_cfg.func("/World/HospitalEnvironment", env_cfg)
            print("[INFO] Hospital USD environment successfully referenced.")
        except Exception as exc:
            if explicit_hospital:
                raise FileNotFoundError(f"Explicit --hospital_usd '{hospital_path}' failed to load: {exc}") from exc
            print(f"[WARN] Remote Nucleus hospital.usd unavailable ({exc}). Using local ground plane.")
            ground_cfg = sim_utils.GroundPlaneCfg()
            ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    else:
        print("[WARN] ISAAC_NUCLEUS_DIR is None; spawning local ground plane.")
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.95, 0.95, 1.0))
    light_cfg.func("/World/HospitalLighting", light_cfg)

    # Spawn Articulated Door at (1.10, -0.45, 0.0) — closed panel spans x in [1.10, 2.00], y in [-0.47, -0.43]
    door_cfg = get_hospital_door_cfg(args_cli.door_type).replace(prim_path="/World/Door")
    door_cfg.init_state.pos = (1.10, -0.45, 0.0)
    door_cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    door = Articulation(door_cfg)

    # Spawn AS2 + PiPER cleanly in front of the -Y face of the door, facing +Y (yaw = +90 deg)
    # At y=-0.95 (or -2.10 if --spawn_far), trunk front is at y=-0.725 (>=25.5cm clear of closed door panel!)
    robot_cfg = AS2_PIPER_CFG.replace(prim_path="/World/Robot")
    start_y = -2.10 if args_cli.spawn_far else -0.95
    robot_cfg.init_state.pos = (1.84, start_y, 0.45)
    robot_cfg.init_state.rot = (0.7071068, 0.0, 0.0, 0.7071068)
    robot = Articulation(robot_cfg)

    rgbd_camera: Camera | None = None
    if args_cli.enable_sensors:
        rgbd_camera = Camera(create_as2_rgbd_camera_cfg(prim_path="/World/Robot/base/front_cam"))
        print("[INFO] Attached AS2 RGB + Depth camera at /World/Robot/base/front_cam.")

    sim.reset()
    robot.reset()
    door.reset()
    if rgbd_camera is not None:
        rgbd_camera.reset()

    if args_cli.enable_sensors and not args_cli.enable_ros2:
        lidar_prim = spawn_as2_rtx_lidar_prim(
            sim.stage, simulation_app, robot_base_path="/World/Robot/base"
        )
        print(f"[INFO] Attached AS2 RTX 3D LiDAR at {lidar_prim}.")

    if args_cli.enable_ros2:
        existing_rp = None
        if rgbd_camera is not None:
            rp_paths = getattr(rgbd_camera, "render_product_paths", None)
            if rp_paths:
                existing_rp = str(rp_paths[0])
        ros2_info = setup_as2_ros2_sensors_graph(
            sim.stage,
            simulation_app,
            robot_base_path="/World/Robot/base",
            existing_camera_rp_path=existing_rp,
        )
        print(f"[INFO] Hospital ROS 2 Sensor Bridge active: {', '.join(ros2_info['topics'])}")

    # Synchronize initial joint position targets with default posture so arm holds tucked pose cleanly
    robot.set_joint_position_target(robot.data.default_joint_pos.clone())
    door.set_joint_position_target(door.data.default_joint_pos.clone())
    robot.write_data_to_sim()
    door.write_data_to_sim()

    # Headless-safe keyboard subscription
    teleop_state = TeleopControlState()
    input_interface = None
    keyboard = None
    keyboard_sub = None
    if not args_cli.headless:
        app_window = omni.appwindow.get_default_app_window()
        if app_window is not None:
            keyboard = app_window.get_keyboard()
            input_interface = carb.input.acquire_input_interface()

            def on_keyboard_event(event, *args, **kwargs):
                is_press = event.type == carb.input.KeyboardEventType.KEY_PRESS
                is_release = event.type == carb.input.KeyboardEventType.KEY_RELEASE
                key_name = event.input.name if hasattr(event.input, "name") else str(event.input)
                teleop_state.update_from_key_event(key_name, is_press, is_release)
                return True

            keyboard_sub = input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)

    print("=" * 75)
    print(" Controls:")
    print("   - [1] Key       : Proximity-gated auto door open (0.28m~0.90m, <=60 deg)")
    print("   - J/L, I/K, U/O : Manual PiPER arm shoulder/elbow control")
    print("   - F/H, T/G, R/Y : Manual PiPER wrist roll/pitch/yaw control")
    print("   - Z / X         : Symmetric gripper Open / Close")
    print("   - C             : Abort auto-door & reset arm to tucked pose")
    print("=" * 75)

    sim_dt = sim.get_physics_dt()
    step_count = 0
    cooldown_s = 0.0

    arm_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "joint8"]
    arm_joint_ids, _ = robot.find_joints(arm_joint_names, preserve_order=True)
    hinge_ids, _ = door.find_joints(["door_hinge"])
    handle_ids, _ = door.find_joints(["door_handle_joint"])

    door_opener = DoorAutoOpener(door_type=args_cli.door_type)
    robot_targets = robot.data.default_joint_pos.clone()
    door_targets = door.data.default_joint_pos.clone()

    while simulation_app.is_running():
        sim.step()
        robot.update(sim_dt)
        door.update(sim_dt)
        if rgbd_camera is not None:
            rgbd_camera.update(sim_dt)

        t = step_count * sim_dt
        base_pos_w = robot.data.root_pos_w[0].tolist()
        base_quat_w = robot.data.root_quat_w[0].tolist()
        door_pos_w = door.data.root_pos_w[0].tolist()
        door_quat_w = door.data.root_quat_w[0].tolist()
        hinge_val = door.data.joint_pos[0, hinge_ids[0]].item() if hinge_ids else 0.0
        handle_pos_w = compute_door_handle_world_pos(
            door_pos_w, door_quat_w, hinge_val=hinge_val, door_type=args_cli.door_type
        )

        if teleop_state.reset_arm_requested and door_opener.active:
            print(door_opener.abort())

        zone_msg = door_opener.poll_zone_notification(base_pos_w, base_quat_w, handle_pos_w)
        if zone_msg is not None:
            print(zone_msg)

        if args_cli.auto_door and not door_opener.active:
            cooldown_s -= sim_dt
            if cooldown_s <= 0.0:
                # Reset door closed before each automated demonstration cycle
                door_targets[:] = 0.0
                ok, msg = door_opener.try_trigger(base_pos_w, base_quat_w, handle_pos_w, teleop_state.arm_targets)
                print(msg)
                cooldown_s = 3.0 if ok else 2.0

        if teleop_state.trigger_auto_door_requested:
            teleop_state.trigger_auto_door_requested = False
            _, msg = door_opener.try_trigger(base_pos_w, base_quat_w, handle_pos_w, teleop_state.arm_targets)
            print(msg)

        if door_opener.active:
            arm_q8, _, (hinge_t, handle_t) = door_opener.step(
                sim_dt, base_pos_w, base_quat_w, handle_pos_w, teleop_state.arm_targets
            )
            teleop_state.arm_targets = arm_q8
            teleop_state.gripper_opening = max(0.0, arm_q8[6])
            if hinge_ids:
                door_targets[:, hinge_ids[0]] = hinge_t
            if handle_ids:
                door_targets[:, handle_ids[0]] = handle_t
        else:
            arm_q8 = teleop_state.step_manual_arm(sim_dt)

        if len(arm_joint_ids) == 8:
            robot_targets[:, arm_joint_ids] = torch.tensor(
                arm_q8, device=robot.device, dtype=torch.float32
            ).unsqueeze(0)

        robot.set_joint_position_target(robot_targets)
        door.set_joint_position_target(door_targets)
        robot.write_data_to_sim()
        door.write_data_to_sim()

        step_count += 1
        if step_count % 300 == 0:
            unit = "m" if args_cli.door_type == "sliding" else "deg"
            disp_val = hinge_val if args_cli.door_type == "sliding" else math.degrees(hinge_val)
            depth_info = ""
            if rgbd_camera is not None:
                depth_tensor = rgbd_camera.data.output.get("distance_to_image_plane")
                if depth_tensor is not None and depth_tensor.numel() > 0:
                    cy, cx = depth_tensor.shape[1] // 2, depth_tensor.shape[2] // 2
                    center_d = float(depth_tensor[0, cy, cx].item())
                    if math.isfinite(center_d):
                        depth_info = f" | CenterDepth: {center_d:.2f}m"
            print(
                f"[t={t:.1f}s] Door ({args_cli.door_type}): {disp_val:.2f} {unit} | "
                f"AutoActive: {door_opener.active}{depth_info}"
            )

    if input_interface is not None and keyboard_sub is not None:
        input_interface.unsubscribe_to_keyboard_events(keyboard, keyboard_sub)

    simulation_app.close()


if __name__ == "__main__":
    main()
