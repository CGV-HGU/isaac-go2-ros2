# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--use_pretrained_checkpoint", action="store_true")
parser.add_argument("--real-time", action="store_true", default=False)

# append cli args
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""
import importlib.metadata as metadata
try:
    installed_version = metadata.version("rsl-rl-lib")
except:
    try: installed_version = metadata.version("rsl_rl")
    except: installed_version = "3.0.0"

"""Rest everything follows."""
import time
import random
import math
import csv
import carb
import carb.input
import gymnasium as gym
import omni
import omni.appwindow
import omni.physx
import omni.graph.core as og
from pxr import UsdGeom, Gf, Sdf, UsdPhysics, PhysxSchema
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

def make_keyboard_state():
    return {"forward": 0.0, "side": 0.0, "yaw": 0.0, "reset": False}

def update_keyboard_state(state, event):
    pressed = event.type == carb.input.KeyboardEventType.KEY_PRESS
    released = event.type == carb.input.KeyboardEventType.KEY_RELEASE
    if not (pressed or released): return
    if event.input == carb.input.KeyboardInput.W: state["forward"] = 1.0 if pressed else 0.0
    elif event.input == carb.input.KeyboardInput.S: state["forward"] = -1.0 if pressed else 0.0
    elif event.input == carb.input.KeyboardInput.A: state["side"] = 0.5 if pressed else 0.0
    elif event.input == carb.input.KeyboardInput.D: state["side"] = -0.5 if pressed else 0.0
    elif event.input == carb.input.KeyboardInput.Q: state["yaw"] = 0.8 if pressed else 0.0
    elif event.input == carb.input.KeyboardInput.E: state["yaw"] = -0.8 if pressed else 0.0
    elif event.input == carb.input.KeyboardInput.R: state["reset"] = True if pressed else False

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv): env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # 초기화 및 ROS2 설정
    obs, _ = env.reset()
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import ros2_sensor_setup
    ros2_sensor_setup.setup_ros2_sensors(omni.usd.get_context().get_stage())

    stage = omni.usd.get_context().get_stage()
    GOAL_X, GOAL_Y = 5.0, 0.0
    dt = env.unwrapped.step_dt
    
    keyboard_state = make_keyboard_state()
    input_interface = carb.input.acquire_input_interface()
    def on_keyboard_event(event, *args, **kwargs):
        update_keyboard_state(keyboard_state, event)
        return True
    keyboard_sub = input_interface.subscribe_to_keyboard_events(omni.appwindow.get_default_app_window().get_keyboard(), on_keyboard_event)

    # 결과 기록 파일 초기화
    result_file = os.path.abspath("./eval_results.csv")
    with open(result_file, 'w', newline='') as f:
        csv.writer(f).writerow(["Episode", "Success", "Time_Steps", "Final_Dist"])

    ep_count = 0
    randomize_object = True 

    with torch.inference_mode():
        while simulation_app.is_running():
            print(f"\n🎬 [에피소드 {ep_count+1}] 조용히 재배치 중...")
            
            # [1] 환경 매니저 리셋 (Done 상태 해제 필수)
            env_ids = torch.arange(env.unwrapped.num_envs, device=env.unwrapped.device)
            env.unwrapped._reset_idx(env_ids)
            
            # [2] 로봇 텔레포트 (고정 위치 배치)
            robot_asset = env.unwrapped.scene["robot"]
            env.unwrapped.command_manager.get_command("base_velocity")[:] = 0.0
            root_state = robot_asset.data.default_root_state.clone()
            start_x, start_y = 1.93166, 0.0 # [수정] 고정 위치
            root_state[:, 0], root_state[:, 1], root_state[:, 2] = start_x, start_y, 0.28
            yaw = math.atan2(GOAL_Y - start_y, GOAL_X - start_x)
            root_state[:, 3], root_state[:, 6] = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
            root_state[:, 7:] = 0.0
            robot_asset.reset()
            robot_asset.write_root_pose_to_sim(root_state[:, :7])
            robot_asset.write_root_velocity_to_sim(root_state[:, 7:])
            
            # [3] 장애물 텔레포트 (성공/R시에만 실행, 로봇 앞쪽에 배치)
            drawer_path = "/World/fused_scene/Moveable_Objects/Drawer"
            drawer_prim = stage.GetPrimAtPath(drawer_path)
            if not drawer_prim.IsValid():
                drawer_path = "/World/Moveable_Objects/Drawer"
                drawer_prim = stage.GetPrimAtPath(drawer_path)

            if randomize_object:
                if drawer_prim.IsValid():
                    # [수정] 장애물을 로봇과 더 멀리 배치 (X: 1.5 ~ 4.0)
                    x_rand, y_rand = random.uniform(1.5, 4.0), random.uniform(-0.3, 0.3)
                    drawer_prim.SetActive(False)
                    simulation_app.update()
                    UsdGeom.Xformable(drawer_prim).MakeMatrixXform().Set(Gf.Matrix4d().SetTranslate(Gf.Vec3d(x_rand, y_rand, 0.05)))
                    drawer_prim.SetActive(True)
                    # 잔상 제거 서비스 호출
                    os.system("ros2 service call /local_costmap/clear_entirely_local_costmap nav2_msgs/srv/ClearEntireCostmap '{}' > /dev/null 2>&1")
                    os.system("ros2 service call /global_costmap/clear_entirely_global_costmap nav2_msgs/srv/ClearEntireCostmap '{}' > /dev/null 2>&1")
                    print(f"🎲 [장애물 재배치] 로봇 앞쪽 X={x_rand:.2f}에 배치 완료")
                randomize_object = False

            # [4] 얌전한 안착 (정지 상태에서 물리 엔진 안정화)
            for _ in range(30):
                env.unwrapped.command_manager.get_command("base_velocity")[:] = 0.0
                obs, _, _, _ = env.step(policy(obs)) # AI가 자세만 잡도록 명령

            # [Action Loop]
            step_count = 0
            success = False
            while simulation_app.is_running():
                if keyboard_state["reset"]:
                    print("🔄 [R키 리셋] 로봇과 장애물 모두 새로 배치합니다.")
                    keyboard_state["reset"] = False
                    if drawer_prim.IsValid(): drawer_prim.SetActive(False)
                    randomize_object = True; break
                
                # Nav2 CmdVel 수신
                nav_x, nav_y, nav_yaw = 0.0, 0.0, 0.0
                try:
                    cmd_node = og.Controller.node("/World/ROS2_Camera_Graph/ROS2CmdVel")
                    if cmd_node.is_valid():
                        lin = og.Controller.get(og.Controller.attribute("outputs:linearVelocity", cmd_node))
                        ang = og.Controller.get(og.Controller.attribute("outputs:angularVelocity", cmd_node))
                        nav_x, nav_y, nav_yaw = float(lin[0]), float(lin[1]), float(ang[2])
                except: pass

                kb = keyboard_state
                vx, vy, vyaw = (kb["forward"] or nav_x), (kb["side"] or nav_y), (kb["yaw"] or nav_yaw)
                env.unwrapped.command_manager.get_command("base_velocity")[:] = torch.tensor([vx, vy, vyaw], device=env.unwrapped.device)
                
                obs, _, dones, _ = env.step(policy(obs))
                
                # 거리 기반 성공 판정
                curr_pos = robot_asset.data.root_pos_w[0]
                dist = math.dist((curr_pos[0].item(), curr_pos[1].item()), (GOAL_X, GOAL_Y))
                if dist < 0.5:
                    success = True
                    print(f"🚩 [목적지 도달] 성공! (오차: {dist:.2f}m)")
                    robot_asset.write_root_velocity_to_sim(torch.zeros_like(robot_asset.data.root_vel_w))
                    randomize_object = True; break

                # 로봇 전도 판정
                if dones.any() and step_count > 50: 
                    print("💥 [전도 리셋] 로봇을 다시 시작합니다.")
                    randomize_object = False; break
                
                step_count += 1
            
            # [기록] CSV 파일에 즉시 기록
            with open(result_file, 'a', newline='') as f:
                csv.writer(f).writerow([ep_count+1, str(success), step_count, round(dist, 2)])
                f.flush()
            
            ep_count += 1

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
