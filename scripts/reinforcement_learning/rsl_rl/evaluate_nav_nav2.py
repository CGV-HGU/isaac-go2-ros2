# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import sys
import os
import csv
import math
import random
import torch
import torch.nn as nn
import gymnasium as gym

# 1. ROS 2 환경 설정 (중요: 다른 임포트보다 먼저 실행)
sys.path.insert(0, "/opt/ros/jazzy/lib/python3.12/site-packages")
os.environ["ROS_DISTRO"] = "jazzy"
os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"

from isaaclab.app import AppLauncher

# argparse 설정
parser = argparse.ArgumentParser(description="Integrated Nav2 + Obstacle Eval for Go2.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Unitree-Go2-v0", help="Task name.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# 앱 실행 (이 코드가 실행된 후에 pxr, omni 등을 임포트할 수 있습니다)
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. 앱 실행 후 필요한 Isaac Sim 라이브러리 임포트
from pxr import Gf, UsdGeom
import omni.graph.core as og
import isaaclab_tasks
from isaaclab_tasks.utils.hydra import hydra_task_config

# 센서 세팅 스크립트 경로 추가 및 임포트
sys.path.append(os.path.join(os.getcwd(), "scripts/reinforcement_learning/skrl"))
import ros2_sensor_setup

class PurePyTorchPolicy(nn.Module):
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

@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg)

    # 1. 센서 및 ROS 2 세팅 (ros2_sensor_setup 활용)
    robot_asset = env.unwrapped.scene["robot"]
    robot_base_path = f"{robot_asset.root_physx_view.prim_paths[0]}/base"
    camera_path = f"{robot_base_path}/front_cam"
    
    stage = env.unwrapped.scene.stage
    ros2_sensor_setup.setup_ros2_sensors(stage, 
                                       robot_base_path=robot_base_path, 
                                       camera_path=camera_path)

    # 2. 모델 로드
    policy = PurePyTorchPolicy(input_size=48, output_size=12).to(env.unwrapped.device)    
    checkpoint = torch.load(args_cli.checkpoint, map_location=env.unwrapped.device)    
    state_dict = checkpoint.get("policy", checkpoint.get("agent", checkpoint.get("state_dict", checkpoint)))
    cleaned_dict = {k.replace("policy.", "").replace("model.", ""): v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
    policy.load_state_dict(cleaned_dict, strict=False)
    policy.eval()

    # 3. Nav2 명령 수신을 위한 추가 노드 (Cmd_vel 전용)
    cmd_vel_nav_attr = og.Controller.attribute("/World/ROS2_Camera_Graph/ROS2CmdVel.outputs:linearVelocity")
    ang_vel_nav_attr = og.Controller.attribute("/World/ROS2_Camera_Graph/ROS2CmdVel.outputs:angularVelocity")
    # ros2_sensor_setup에서 생성한 구독 노드의 토픽명을 Nav2 전용으로 변경
    og.Controller.set(og.Controller.attribute("/World/ROS2_Camera_Graph/ROS2CmdVel.inputs:topicName"), "/cmd_vel_nav")

    # 평가 로직
    dt = env.unwrapped.step_dt
    TOTAL_EPISODES = 100
    START_X, START_Y = -1.0, 0.0
    GOAL_X, GOAL_Y = 5.0, 0.0
    drawer_path = "/World/fused_scene/Moveable_Objects/Drawer"

    result_file = "./eval_results_nav2.csv"
    with open(result_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Success", "Time_Steps", "Final_Dist"])

    for ep in range(1, TOTAL_EPISODES + 1):
        if not simulation_app.is_running(): break
        print(f"\n🎬 [에피소드 {ep}/100] 초기화 중...")
        
        env.reset()

        # 로봇 텔레포트
        with torch.inference_mode():
            root_state = robot_asset.data.default_root_state.clone()
            root_state[:, 0] = START_X
            root_state[:, 1] = START_Y
            root_state[:, 2] = 0.42
            env.unwrapped.scene["robot"].write_root_pose_to_sim(root_state[:, :7])
            env.unwrapped.scene["robot"].reset()

        # 장애물 랜덤 배치
        drawer_prim = stage.GetPrimAtPath(drawer_path)
        if drawer_prim.IsValid():
            x_rand = random.uniform(-1.0, 3.0)
            y_rand = random.uniform(-0.76, 0.76)
            mat = Gf.Matrix4d().SetTranslate(Gf.Vec3d(x_rand, y_rand, 0.0))
            UsdGeom.Xformable(drawer_prim).MakeMatrixXform().Set(mat)

        simulation_app.update()
        obs_dict = env.unwrapped.observation_manager.compute()
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

        step_count = 0
        while step_count < 1500 and simulation_app.is_running():
            current_pos = robot_asset.data.root_pos_w[0]
            final_dist = math.dist((current_pos[0].item(), current_pos[1].item()), (GOAL_X, GOAL_Y))

            if final_dist < 0.5:
                print(f"    🎉 성공!"); break

            # Nav2 명령 수신
            nav2_lin = og.Controller.get(cmd_vel_nav_attr)
            nav2_ang = og.Controller.get(ang_vel_nav_attr)
            cmd = torch.tensor([nav2_lin[0], nav2_lin[1], nav2_ang[2]], device=env.unwrapped.device)
            env.unwrapped.command_manager.get_command("base_velocity")[:] = cmd.unsqueeze(0)

            with torch.inference_mode():
                actions = policy(obs)
            obs_dict, _, terminated, truncated, _ = env.step(actions)
            obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
            
            if (terminated | truncated)[0]: print("    💥 실패!"); break
            step_count += 1

        with open(result_file, 'a', newline='') as f:
            csv.writer(f).writerow([ep, final_dist < 0.5, step_count, round(final_dist, 2)])

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
