# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default=None, help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument(
    "--ml_framework", type=str, default="torch", choices=["torch", "jax", "jax-numpy"], help="The ML framework."
)
parser.add_argument(
    "--algorithm", type=str, default="PPO", choices=["AMP", "PPO", "IPPO", "MAPPO"], help="The RL algorithm."
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import time

import carb
import carb.input
import gymnasium as gym
import omni
import omni.appwindow
import omni.graph.core as og
import omni.usd
from pxr import UsdGeom, Gf
import torch

import skrl
from packaging import version
if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()

# PLACEHOLDER: Extension template (do not remove this comment)


def make_keyboard_state():
    return {
        "forward": 0.0,
        "side": 0.0,
        "yaw": 0.0,
        "reset": False,
        "spawn_obstacle": False,
        "respawn_in_place": False,
        "shift": False,
    }


def update_keyboard_state(state, event):
    pressed = event.type == carb.input.KeyboardEventType.KEY_PRESS
    released = event.type == carb.input.KeyboardEventType.KEY_RELEASE

    if not (pressed or released):
        return

    if event.input == carb.input.KeyboardInput.W:
        state["forward"] = 1.0 if pressed else 0.0
    elif event.input == carb.input.KeyboardInput.S:
        state["forward"] = -1.0 if pressed else 0.0
    elif event.input == carb.input.KeyboardInput.A:
        state["side"] = 1.0 if pressed else 0.0
    elif event.input == carb.input.KeyboardInput.D:
        state["side"] = -1.0 if pressed else 0.0
    elif event.input == carb.input.KeyboardInput.Q:
        state["yaw"] = 1.0 if pressed else 0.0
    elif event.input == carb.input.KeyboardInput.E:
        state["yaw"] = -1.0 if pressed else 0.0
    elif event.input in [carb.input.KeyboardInput.LEFT_SHIFT, carb.input.KeyboardInput.RIGHT_SHIFT]:
        state["shift"] = True if pressed else False
    elif event.input == carb.input.KeyboardInput.R:
        state["reset"] = True if pressed else False
    elif event.input == carb.input.KeyboardInput.T:
        state["spawn_obstacle"] = True if pressed else False
    elif event.input == carb.input.KeyboardInput.F:
        state["respawn_in_place"] = True if pressed else False


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Play with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    agent_cfg["trainer"]["close_environment_at_exit"] = False

    # set the agent and environment seed from command line
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None
    
    log_dir = os.path.dirname(resume_path) if resume_path else os.getcwd()
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # keyboard subscription
    keyboard_state = make_keyboard_state()
    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard()
    input_interface = carb.input.acquire_input_interface()

    def on_keyboard_event(event, *args, **kwargs):
        update_keyboard_state(keyboard_state, event)
        return True

    keyboard_sub = input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    # ==========================================
    # [추가됨] ROS2 RGB & Depth 카메라 퍼블리셔 세팅
    # ==========================================
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import ros2_sensor_setup
    
    stage = omni.usd.get_context().get_stage()
    ros2_sensor_setup.setup_ros2_sensors(stage)
    # ==========================================
    # ==========================================

    # configure and instantiate the skrl runner
    runner = Runner(env, agent_cfg)

    if resume_path:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.agent.act

    print("\n================ Keyboard Control ================")
    print("Click the 3D viewport once, then use:")
    print("W / S : forward / backward")
    print("A / D : left / right")
    print("Q / E : yaw left / yaw right")
    print("=================================================\n")

    dt = env.unwrapped.step_dt

    # Animation state machine variables
    anim_timer = 0.0
    anim_state = 0
    
    # [Pre-spawn] 미리 장애물을 지하(-100m)에 만들어 둠 (런타임 생성 시 프리징 방지)
    try:
        from omni.physx.scripts import physicsUtils
        from pxr import Gf
        stage = omni.usd.get_context().get_stage()
        obstacle_path = "/World/InteractiveObstacle"
        # 로봇 크기만한 직육면체 (길이 0.8m, 폭 0.4m, 높이 0.4m)
        physicsUtils.add_rigid_box(
            stage, obstacle_path, size=Gf.Vec3f(0.8, 0.4, 0.4), 
            position=Gf.Vec3f(0.0, 0.0, -100.0), color=Gf.Vec3f(0.96, 0.96, 0.86), density=100.0
        )
    except Exception as e:
        print(f"[Warning] Failed to pre-spawn obstacle: {e}")

    # reset environment
    obs, _ = env.reset()
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # Get velocity from ROS 2 Twist Subscriber node (Nav2)
            nav_x, nav_y, nav_yaw = 0.0, 0.0, 0.0
            try:
                cmd_node = og.Controller.node("/World/ROS2_Camera_Graph/ROS2CmdVel")
                if cmd_node.is_valid():
                    lin_vel = og.Controller.get(og.Controller.attribute("outputs:linearVelocity", cmd_node))
                    ang_vel = og.Controller.get(og.Controller.attribute("outputs:angularVelocity", cmd_node))
                    if lin_vel is not None and ang_vel is not None:
                        nav_x, nav_y, nav_yaw = float(lin_vel[0]), float(lin_vel[1]), float(ang_vel[2])
            except Exception:
                pass

            # keyboard -> base_velocity (Keyboard override Nav2)
            kb_x, kb_y, kb_yaw = keyboard_state["forward"], keyboard_state["side"], keyboard_state["yaw"]
            
            # [Shift 키 달리기 기능] Shift를 누르고 있으면 속도 2배 증가
            speed_multiplier = 2.0 if keyboard_state.get("shift", False) else 1.0
            kb_x *= speed_multiplier
            kb_y *= speed_multiplier
            kb_yaw *= speed_multiplier
            
            final_x = kb_x if kb_x != 0.0 else nav_x
            final_y = kb_y if kb_y != 0.0 else nav_y
            final_yaw = kb_yaw if kb_yaw != 0.0 else nav_yaw

            cmd = torch.tensor(
                [final_x, final_y, final_yaw],
                device=env.unwrapped.device,
                dtype=torch.float32,
            )
            cmd = cmd.unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
            env.unwrapped.command_manager.get_command("base_velocity")[:] = cmd

            # manual reset via 'R' key
            if keyboard_state["reset"]:
                obs, _ = env.reset()
                keyboard_state["reset"] = False
                print("[INFO] Manual reset triggered. (Origin)")
                continue
            
            # [제자리 리스폰] 제자리에서 똑바로 세우기 via 'F' key
            if keyboard_state["respawn_in_place"]:
                keyboard_state["respawn_in_place"] = False
                print("[INFO] 🔄 Respawning robot in place (F key pressed)!")
                
                try:
                    import math
                    
                    # 1. 카메라 튕김을 막기 위해, R키가 사용하는 공식 reset() 메커니즘을 가로챕니다.
                    robot = env.unwrapped.scene["robot"]
                    
                    # 2. 현재 로봇의 X, Y 좌표 및 Yaw(회전) 가져오기
                    robot_pos = robot.data.root_pos_w[0].clone()
                    robot_quat = robot.data.root_quat_w[0].clone()
                    
                    w, x, y, z = robot_quat.cpu().numpy()
                    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
                    
                    # 3. 새로운 위치(Z=0.5m) 및 회전(Roll=0, Pitch=0) 생성
                    new_pos = robot_pos
                    new_pos[2] = 0.5
                    new_quat = torch.tensor([math.cos(yaw/2.0), 0.0, 0.0, math.sin(yaw/2.0)], device=robot.device)
                    
                    # 4. 초기 상태 덮어쓰기 (IsaacLab 네이티브 버퍼 업데이트)
                    robot.data.default_root_state[0, :3] = new_pos
                    robot.data.default_root_state[0, 3:7] = new_quat
                    
                    # 5. R키와 동일하게 env.reset() 호출 -> 카메라가 절대 튕기지 않고 제자리 리스폰됨!
                    obs, _ = env.reset()
                        
                except Exception as e:
                    print(f"[ERROR] Failed to respawn in place: {e}")
                    
            # spawn dynamic obstacle via 'T' key
            if keyboard_state["spawn_obstacle"]:
                keyboard_state["spawn_obstacle"] = False
                print("[INFO] 📦 Teleporting dynamic obstacle to robot's front (T key pressed)!")
                try:
                    import math
                    from pxr import UsdGeom, UsdPhysics, Gf
                    
                    stage = omni.usd.get_context().get_stage()
                    
                    # 로봇 위치와 회전(쿼터니언 w,x,y,z) 가져오기
                    robot_pos = env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
                    robot_quat = env.unwrapped.scene["robot"].data.root_quat_w[0].cpu().numpy()
                    
                    # 쿼터니언(w,x,y,z)에서 Yaw(Z축 회전) 추출
                    w, x, y, z = robot_quat
                    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
                    
                    # 로봇 정면 방향 계산
                    forward_x = math.cos(yaw)
                    forward_y = math.sin(yaw)
                    
                    # 0.3m 앞, 0.5m 높이
                    target_x = float(robot_pos[0] + forward_x * 0.3)
                    target_y = float(robot_pos[1] + forward_y * 0.3)
                    target_z = float(robot_pos[2] + 0.5)
                    
                    drop_pos = Gf.Vec3d(target_x, target_y, target_z)
                    
                    obstacle_path = "/World/InteractiveObstacle"
                    obstacle_prim = stage.GetPrimAtPath(obstacle_path)
                    
                    # 상자가 아직 없으면 런타임에 즉시 생성!
                    if not obstacle_prim.IsValid():
                        from omni.physx.scripts import physicsUtils
                        physicsUtils.add_rigid_box(
                            stage, obstacle_path, size=Gf.Vec3f(0.8, 0.4, 0.4), 
                            position=Gf.Vec3f(float(target_x), float(target_y), float(target_z)), 
                            color=Gf.Vec3f(0.96, 0.96, 0.86), density=100.0
                        )
                        obstacle_prim = stage.GetPrimAtPath(obstacle_path)
                    
                    # UsdGeom.Xformable을 통한 안전한 장애물 텔레포트
                    if obstacle_prim.IsValid():
                        xform = UsdGeom.Xformable(obstacle_prim)
                        xform.ClearXformOpOrder()
                        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(drop_pos)
                        xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))
                        
                        rb_api = UsdPhysics.RigidBodyAPI(obstacle_prim)
                        if rb_api:
                            rb_api.GetVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                            rb_api.GetAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                            
                        print(f"[INFO] Obstacle dropped exactly 0.3m in front of robot at ({target_x:.2f}, {target_y:.2f})!")
                    else:
                        print(f"[ERROR] Failed to create or find obstacle prim!")
                except Exception as e:
                    print(f"[ERROR] Failed to teleport obstacle: {e}")

            # agent stepping
            actions = policy(obs)
            
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    input_interface.unsubscribe_to_keyboard_events(keyboard, keyboard_sub)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
