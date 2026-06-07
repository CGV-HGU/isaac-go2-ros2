# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import sys
import os
import time
import csv
import math
import random
import torch
import torch.nn as nn
import gymnasium as gym
from isaaclab.app import AppLauncher

# local imports
import cli_args

# argparse 설정
parser = argparse.ArgumentParser(description="Play with Obstacle Randomization and CSV logging.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Unitree-Go2-v0", help="Task name.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# 앱 실행
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 앱 부팅 후 임포트
import omni
import carb
import omni.appwindow
from pxr import Gf, UsdGeom
import omni.graph.core as og
import isaaclab_tasks
from isaaclab_tasks.utils.hydra import hydra_task_config

# 센서 세팅 모듈
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import ros2_sensor_setup

# 공용 정책 네트워크 (ELU 기반)
class UniversalPolicy(nn.Module):
    def __init__(self, input_size=48, output_size=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, output_size)
        )
    def forward(self, obs):
        if isinstance(obs, dict): obs = obs.get("states", obs)
        return self.net(obs)

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg, *args, **kwargs):
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg)

    # 1. 로봇 경로 획득 및 ROS 2 센서 세팅
    robot_asset = env.unwrapped.scene["robot"]
    # robot_prim_path는 보통 "/World/envs/env_0/Robot" 형태입니다.
    robot_prim_path = robot_asset.root_physx_view.prim_paths[0] 

    # [수정] 경로 중복 방지를 위해 조건부로 base 추가
    if robot_prim_path.endswith("/base"):
        robot_base_path = robot_prim_path
    else:
        robot_base_path = f"{robot_prim_path}/base"

    camera_path = f"{robot_base_path}/front_camera"

    
    stage = omni.usd.get_context().get_stage()
    ros2_sensor_setup.setup_ros2_sensors(stage, 
                                       robot_base_path=robot_base_path, 
                                       camera_path=camera_path)

    # 2. 모델 로드 (순수 PyTorch 방식)
    resume_path = args_cli.checkpoint
    if not resume_path:
        from isaaclab_tasks.utils import get_checkpoint_path
        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    policy = UniversalPolicy().to(env.unwrapped.device)
    checkpoint = torch.load(resume_path, map_location=env.unwrapped.device)
    state_dict = checkpoint.get("policy", checkpoint.get("agent", checkpoint.get("state_dict", checkpoint)))
    cleaned_dict = {k.replace("policy.", "").replace("model.", ""): v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
    policy.load_state_dict(cleaned_dict, strict=False)
    policy.eval()

    # 3. 평가 및 장애물 설정
    GOAL_X, GOAL_Y = 5.0, 0.0
    drawer_path = "/World/fused_scene/Moveable_Objects/Drawer"
    result_file = "./play_eval_results.csv"
    
    if not os.path.exists(result_file):
        with open(result_file, 'w', newline='') as f:
            csv.writer(f).writerow(["Timestamp", "Success", "Distance_to_Goal"])

    print("\n[INFO] Play 모드 시작: Nav2로 목표를 주면 자동으로 성공 여부를 기록합니다.")

    # keyboard subscription
    def make_keyboard_state():
        return {"forward": 0.0, "side": 0.0, "yaw": 0.0, "reset": False}

    def update_keyboard_state(state, event):
        pressed = event.type == carb.input.KeyboardEventType.KEY_PRESS
        released = event.type == carb.input.KeyboardEventType.KEY_RELEASE
        if not (pressed or released): return
        if event.input == carb.input.KeyboardInput.W: state["forward"] = 1.0 if pressed else 0.0
        elif event.input == carb.input.KeyboardInput.S: state["forward"] = -1.0 if pressed else 0.0
        elif event.input == carb.input.KeyboardInput.A: state["side"] = 1.0 if pressed else 0.0
        elif event.input == carb.input.KeyboardInput.D: state["side"] = -1.0 if pressed else 0.0
        elif event.input == carb.input.KeyboardInput.Q: state["yaw"] = 1.0 if pressed else 0.0
        elif event.input == carb.input.KeyboardInput.E: state["yaw"] = -1.0 if pressed else 0.0
        elif event.input == carb.input.KeyboardInput.R: state["reset"] = True if pressed else False

    keyboard_state = make_keyboard_state()
    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard()
    input_interface = carb.input.acquire_input_interface()
    def on_keyboard_event(event, *args, **kwargs):
        update_keyboard_state(keyboard_state, event)
        return True
    keyboard_sub = input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)

    print("\n================ Keyboard Control ================")
    print("Click the 3D viewport once, then use:")
    print("W / S : forward / backward")
    print("A / D : left / right")
    print("Q / E : yaw left / yaw right")
    print("R : Manual Reset (Respawn)")
    print("=================================================\n")

    ep_count = 0
    with torch.inference_mode():
        while simulation_app.is_running():
            # 1️⃣ 에피소드 시작 시 환경 리셋
            env.reset()
            
            # [추가] 리셋 직후 로봇이 안정적으로 지면에 내려올 때까지 대기
            for _ in range(20):
                simulation_app.update()
            
            # 2️⃣ 장애물 랜덤 배치
            drawer_prim = stage.GetPrimAtPath(drawer_path)
            if drawer_prim.IsValid():
                x_rand = random.uniform(-1.0, 3.0)
                y_rand = random.uniform(-0.76, 0.76)
                mat = Gf.Matrix4d().SetTranslate(Gf.Vec3d(x_rand, y_rand, 0.01))
                UsdGeom.Xformable(drawer_prim).MakeMatrixXform().Set(mat)
            
            # 3️⃣ 물리 업데이트 및 깨끗한 관측치 추출
            simulation_app.update()
            obs_dict = env.unwrapped.observation_manager.compute()
            obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

            while simulation_app.is_running():
                start_time = time.time()

                # [수동 리셋 (R키)] 
                if keyboard_state["reset"]:
                    print("🔄 [수동 리셋] R키가 입력되어 에피소드를 재시작합니다.")
                    time.sleep(0.5)
                    keyboard_state["reset"] = False
                    break # 내부 루프 탈출 -> 외부 루프에서 리셋됨
                
                # Nav2 명령 수신
                nav_x, nav_y, nav_yaw = 0.0, 0.0, 0.0
                try:
                    cmd_node = og.Controller.node("/World/ROS2_Camera_Graph/ROS2CmdVel")
                    if cmd_node.is_valid():
                        lin_vel = og.Controller.get(og.Controller.attribute("outputs:linearVelocity", cmd_node))
                        ang_vel = og.Controller.get(og.Controller.attribute("outputs:angularVelocity", cmd_node))
                        nav_x, nav_y, nav_yaw = float(lin_vel[0]), float(lin_vel[1]), float(ang_vel[2])
                except: pass

                # keyboard -> base_velocity (Keyboard override Nav2)
                kb_x, kb_y, kb_yaw = keyboard_state["forward"], keyboard_state["side"], keyboard_state["yaw"]
                
                final_x = kb_x if kb_x != 0.0 else nav_x
                final_y = kb_y if kb_y != 0.0 else nav_y
                final_yaw = kb_yaw if kb_yaw != 0.0 else nav_yaw

                # 명령 전달 및 환경 스텝
                cmd = torch.tensor([final_x, final_y, final_yaw], device=env.unwrapped.device).unsqueeze(0)
                env.unwrapped.command_manager.get_command("base_velocity")[:] = cmd

                actions = policy(obs)
                obs_dict, _, terminated, truncated, _ = env.step(actions)
                obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

                # 성공 여부 체크
                curr_pos = robot_asset.data.root_pos_w[0]
                dist = math.dist((curr_pos[0].item(), curr_pos[1].item()), (GOAL_X, GOAL_Y))
                
                if dist < 0.5:
                    print(f"🚩 [목적지 도달] 성공 기록 저장!")
                    with open(result_file, 'a', newline='') as f:
                        csv.writer(f).writerow([time.strftime("%Y-%m-%d %H:%M:%S"), True, round(dist, 2)])
                    time.sleep(1.0); break

                # 자동 실패(terminated/truncated) 시 즉시 리셋 방지, 로그만 남김
                if (terminated | truncated)[0]:
                    # 원래는 여기서 바로 break되어 새 에피소드로 넘어갔으나, 빈번한 자동 리셋을 막기 위해 로그만 찍고 유지
                    # 너무 자주 리셋되는 것을 막고, 사용자가 R키로 제어하도록 유도
                    pass

                if args_cli.real_time:
                    time.sleep(max(0, 0.02 - (time.time() - start_time)))
            
            ep_count += 1
            print(f"🔄 에피소드 {ep_count} 종료.")

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
