# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play an RSL-RL locomotion checkpoint for Unitree AS2 / AS2+PiPER with simultaneous arm teleop,
proximity-gated multi-type door auto-opening (Key '1'), and Isaac Sim 5.x/6.x ROS 2 Bridge."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

DOOR_TYPE_CHOICES = ("lever_push", "lever_pull", "sliding", "push_bar")

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL and simultaneous PiPER teleop.")
parser.add_argument("--video", action="store_true", default=False, help="Record evaluation video.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--enable_sensors",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Attach AS2 RGB camera, metric Depth camera, and 16-ch 3D LiDAR sensors to the scene.",
)
parser.add_argument(
    "--enable_ros2",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Enable ROS 2 RGB/Depth/CameraInfo/3D-LiDAR/TF/Clock OmniGraph bridge.",
)
parser.add_argument(
    "--spawn_door",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Spawn an interactive articulated door ~2.0m ahead for WASD approach + Key '1' auto-open.",
)
parser.add_argument(
    "--door_type",
    type=str,
    choices=DOOR_TYPE_CHOICES,
    default="lever_push",
    help="Door mechanism type when --spawn_door is enabled.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# enable RTX camera pipeline whenever video recording, onboard RGB-D/LiDAR sensors, or ROS 2 bridge is requested
if args_cli.video or args_cli.enable_sensors or args_cli.enable_ros2:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import os
import time

import carb
import carb.input
import gymnasium as gym
import omni.appwindow
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.assets import Articulation
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_assets.robots.as2_sensors import (
    attach_as2_sensors_to_scene_cfg,
    setup_as2_ros2_sensors_graph,
    summarize_as2_sensor_readings,
)
from isaaclab_assets.robots.door_manipulation import (
    DoorAutoOpener,
    TeleopControlState,
    compute_door_handle_world_pos,
)
from isaaclab_assets.robots.hospital_door import get_hospital_door_cfg

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent, simultaneous PiPER arm teleop, and Key '1' proximity door opener."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Ensure teleop velocity commands are never overwritten by heading controller or periodic resampling
    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "base_velocity"):
        env_cfg.commands.base_velocity.heading_command = False
        env_cfg.commands.base_velocity.rel_standing_envs = 0.0
        env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)

    # Optionally attach AS2 RGB camera, metric Depth camera, and 16-ch 3D LiDAR tensors to the scene
    if args_cli.enable_sensors:
        attach_as2_sensors_to_scene_cfg(env_cfg.scene, enable_rgbd=True, enable_lidar=True)
        print("[INFO] Attached AS2 RGB camera (640x480), Depth camera, and 16-ch 3D LiDAR to scene.")

    # Optionally spawn an articulated door ~2.0m ahead along +X for WASD walk-up + Key '1' auto-open
    door_pos_local = (2.10, 0.74, 0.0)
    door_quat_local = (0.7071068, 0.0, 0.0, -0.7071068)
    if args_cli.spawn_door:
        if env_cfg.scene.num_envs != 1:
            raise ValueError(
                f"--spawn_door single-operator teleop requires --num_envs 1, got num_envs={env_cfg.scene.num_envs}."
            )
        door_cfg = get_hospital_door_cfg(args_cli.door_type).replace(prim_path="{ENV_REGEX_NS}/Door")
        door_cfg.init_state.pos = door_pos_local
        door_cfg.init_state.rot = door_quat_local
        env_cfg.scene.door = door_cfg
        # Start robot cleanly at origin facing +X toward the door
        if hasattr(env_cfg, "events") and getattr(env_cfg.events, "reset_base", None) is not None:
            env_cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

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

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording evaluation video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if args_cli.enable_ros2:
        existing_rp = None
        if args_cli.enable_sensors and "rgbd_camera" in env.unwrapped.scene.keys():
            rp_paths = getattr(env.unwrapped.scene["rgbd_camera"], "render_product_paths", None)
            if rp_paths:
                existing_rp = str(rp_paths[0])
        if env.unwrapped.num_envs > 1:
            print("[INFO] --enable_ros2 publishes sensor streams for env_0 (/World/envs/env_0/Robot/base).")
        ros2_info = setup_as2_ros2_sensors_graph(
            env.unwrapped.scene.stage,
            simulation_app,
            robot_base_path="/World/envs/env_0/Robot/base",
            existing_camera_rp_path=existing_rp,
        )
        print(f"[INFO] ROS 2 Sensor Bridge initialized. Topics: {', '.join(ros2_info['topics'])}")

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    if version.parse(installed_version) >= version.parse("4.0.0"):
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
        policy_nn = None
    else:
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    # Resolve robot & optional PiPER 8-DOF arm joints + optional Door articulation
    robot: Articulation = env.unwrapped.scene["robot"]
    arm_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "joint8"]
    arm_joint_ids, _ = robot.find_joints(arm_joint_names, preserve_order=True)
    has_piper_arm = len(arm_joint_ids) == 8

    door: Articulation | None = env.unwrapped.scene["door"] if args_cli.spawn_door else None
    door_opener = DoorAutoOpener(door_type=args_cli.door_type) if door is not None else None
    hinge_ids = door.find_joints(["door_hinge"])[0] if door is not None else []
    handle_ids = door.find_joints(["door_handle_joint"])[0] if door is not None else []

    if keyboard_sub is not None:
        print("\n==================== Simultaneous Teleop Controls ====================")
        print(" [Left Hand - AS2 Quadruped Locomotion]")
        print("   W / S : Forward / Backward      |  A / D : Strafe Left / Right")
        print("   Q / E : Yaw Left / Yaw Right")
        if has_piper_arm:
            print(" [Right Hand - AgileX PiPER 6-DOF Arm & Symmetric Gripper]")
            print("   J / L : Base Yaw (joint1)       |  I / K : Shoulder Reach (joint2)")
            print("   U / O : Elbow Pitch (joint3)    |  F / H : Wrist Roll (joint4)")
            print("   T / G : Wrist Pitch (joint5)    |  R / Y : Wrist Yaw (joint6)")
            print("   Z / X : Gripper Open / Close    |  C     : Reset Arm / Abort Macro")
        if door_opener is not None:
            print(f" [Proximity Auto-Door ({args_cli.door_type})]")
            print("   1     : Auto-open door (ONLY activates within 0.28m~0.90m & <=60 deg)")
        print("======================================================================\n")

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    timestep = 0

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            # 1. Base velocity from keyboard (or smooth forward walk in headless video mode)
            if keyboard_sub is None and args_cli.headless:
                vx, vy, wz = 0.5, 0.0, 0.0
            else:
                vx, vy, wz = teleop_state.forward, teleop_state.side, teleop_state.yaw

            # 2. PiPER arm manual control + Key '1' proximity-gated door auto-opener
            if has_piper_arm:
                if teleop_state.reset_arm_requested and door_opener is not None and door_opener.active:
                    print(door_opener.abort())

                if door is not None and door_opener is not None:
                    base_pos_w = robot.data.root_pos_w[0].tolist()
                    base_quat_w = robot.data.root_quat_w[0].tolist()
                    door_pos_w = door.data.root_pos_w[0].tolist()
                    door_quat_w = door.data.root_quat_w[0].tolist()
                    hinge_angle = door.data.joint_pos[0, hinge_ids[0]].item() if hinge_ids else 0.0
                    handle_pos_w = compute_door_handle_world_pos(
                        door_pos_w, door_quat_w, hinge_val=hinge_angle, door_type=args_cli.door_type
                    )

                    zone_msg = door_opener.poll_zone_notification(base_pos_w, base_quat_w, handle_pos_w)
                    if zone_msg is not None:
                        print(zone_msg)

                    if teleop_state.trigger_auto_door_requested:
                        teleop_state.trigger_auto_door_requested = False
                        _, msg = door_opener.try_trigger(
                            base_pos_w, base_quat_w, handle_pos_w, teleop_state.arm_targets
                        )
                        print(msg)

                    if door_opener.active:
                        arm_q8, assist_vel, (hinge_t, handle_t) = door_opener.step(
                            dt, base_pos_w, base_quat_w, handle_pos_w, teleop_state.arm_targets
                        )
                        teleop_state.arm_targets = arm_q8
                        teleop_state.gripper_opening = max(0.0, arm_q8[6])
                        # Blend operator override with auto-door coordinated footstep
                        if abs(vx) < 1e-4 and abs(vy) < 1e-4 and abs(wz) < 1e-4:
                            vx, vy, wz = assist_vel
                        door_targets = door.data.default_joint_pos.clone()
                        if hinge_ids:
                            door_targets[:, hinge_ids[0]] = hinge_t
                        if handle_ids:
                            door_targets[:, handle_ids[0]] = handle_t
                        door.set_joint_position_target(door_targets)
                    else:
                        arm_q8 = teleop_state.step_manual_arm(dt)
                else:
                    if teleop_state.trigger_auto_door_requested:
                        teleop_state.trigger_auto_door_requested = False
                        print("[AUTO-DOOR] 현재 씬에 문이 없습니다. '--spawn_door' 옵션을 켜고 실행하세요.")
                    arm_q8 = teleop_state.step_manual_arm(dt)

                arm_target_tensor = torch.tensor(
                    arm_q8, device=env.unwrapped.device, dtype=torch.float32
                ).unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
                robot.set_joint_position_target(arm_target_tensor, joint_ids=arm_joint_ids)

            # 3. Write updated velocity command BEFORE computing policy observations for zero-lag response
            cmd = torch.tensor([vx, vy, wz], device=env.unwrapped.device, dtype=torch.float32)
            env.unwrapped.command_manager.get_command("base_velocity")[:] = cmd.unsqueeze(0).repeat(
                env.unwrapped.num_envs, 1
            )
            obs = env.get_observations()

            # 4. Step RL locomotion policy & physics environment
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)

        timestep += 1
        if args_cli.enable_sensors and timestep % 100 == 0:
            sensor_stats = summarize_as2_sensor_readings(env.unwrapped.scene)
            if sensor_stats:
                parts = [f"{k}={v:.2f}" for k, v in sensor_stats.items()]
                print(f"[SENSORS step={timestep}] " + " | ".join(parts))

        if args_cli.video and timestep >= args_cli.video_length:
            break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if input_interface is not None and keyboard_sub is not None:
        input_interface.unsubscribe_to_keyboard_events(keyboard, keyboard_sub)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
