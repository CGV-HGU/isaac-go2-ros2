# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""skrl 라이브러리 버전 버그를 우회하기 위해 순수 PyTorch 모델로 가중치를 다이렉트 로드하여 Go2 주행을 평가하는 스크립트"""

import argparse
import sys
from isaaclab.app import AppLauncher

# argparse arguments 세팅
parser = argparse.ArgumentParser(description="Pure PyTorch Inference Evaluation for Go2 Navigation.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Unitree-Go2-v0", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained checkpoint.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# 시뮬레이터 앱 부팅
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 필요한 라이브러리 임포트
import os
import csv
import math
import random
import torch
import torch.nn as nn
import gymnasium as gym
import omni.physx
from pxr import Gf, UsdGeom

import isaaclab_tasks
from isaaclab_tasks.utils.hydra import hydra_task_config

# 💡 [핵심] skrl 종속성을 완전히 제거한 순수 PyTorch 신경망 선언
class PurePyTorchPolicy(nn.Module):
    def __init__(self, input_size=48, output_size=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, output_size)
        )
        
    def forward(self, obs):
        if isinstance(obs, dict):
            obs = obs.get("states", obs)
        return self.net(obs)

@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    # 환경 변수 설정
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # Gym 환경 생성 (skrl Wrapper조차 거치지 않는 순수 상태 유지)
    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO]: Loading model weights directly from PyTorch checkpoint: {args_cli.checkpoint}")
    
    # 모델 인스턴스 생성 및 디바이스 할당 (.unwrapped 우회 적용)
    policy = PurePyTorchPolicy(input_size=48, output_size=12).to(env.unwrapped.device)    
    checkpoint = torch.load(args_cli.checkpoint, map_location=env.unwrapped.device)    
    
    # 가중치 딕셔너리 안전 분해 로드
    state_dict = None
    for key in ["policy", "agent", "state_dict"]:
        if key in checkpoint:
            if isinstance(checkpoint[key], dict) and "net.0.weight" in checkpoint[key]:
                state_dict = checkpoint[key]
                break
            elif hasattr(checkpoint[key], "state_dict"):
                state_dict = checkpoint[key].state_dict()
                break
    
    if state_dict is None:
        if isinstance(checkpoint, dict):
            cleaned_dict = {}
            for k, v in checkpoint.items():
                if "linear_layer_" in k or "action_layer" in k or "net" in k:
                    k_clean = k.replace("policy.", "").replace("model.", "")
                    cleaned_dict[k_clean] = v
            state_dict = cleaned_dict if cleaned_dict else checkpoint
        else:
            state_dict = checkpoint.state_dict() if hasattr(checkpoint, "state_dict") else checkpoint

    # 가중치 강제 주입
    try:
        policy.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"[WARNING]: Strict loading failed, trying fallback. Error: {e}")
        adapted_dict = {}
        for (k_old, v_old), (k_new, _) in zip(state_dict.items(), policy.state_dict().items()):
            if v_old.shape == policy.state_dict()[k_new].shape:
                adapted_dict[k_new] = v_old
        policy.load_state_dict(adapted_dict, strict=False)
        
    policy.eval()

    dt = env.unwrapped.step_dt
    TOTAL_EPISODES = 100
    
    START_X, START_Y = -1.0, 0.0
    GOAL_X, GOAL_Y = 5.0, 0.0
    X_MIN, X_MAX = -1.0, 3.0
    Y_BOUND = 0.76

    result_file = "./eval_results_pure.csv"
    with open(result_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Success", "Time_Steps", "Final_Dist"])

    stage = env.unwrapped.scene.stage
    drawer_path = "/World/fused_scene/Moveable_Objects/Drawer"

    print(f"\n================ 총 {TOTAL_EPISODES}회 순수 PyTorch Go2 회피 주행 평가 시작 ================")

    for ep in range(1, TOTAL_EPISODES + 1):
        if not simulation_app.is_running():
            break
        print(f"\n🎬 [실험 {ep}/{TOTAL_EPISODES}] 에피소드 초기화 중...")

        # 1️⃣ [수정] Gym 환경의 타임아웃 및 기본 버퍼를 먼저 비웁니다.
        env.reset()

        # 2️⃣ [수정] 리셋 직후, 우리가 원하는 평가 시작 좌표로 로봇을 텔레포트합니다.
        with torch.inference_mode():
            root_state = env.unwrapped.scene["robot"].data.default_root_state.clone()
            root_state[:, 0] = START_X
            root_state[:, 1] = START_Y
            root_state[:, 2] = 0.42
            
            target_yaw = math.atan2(GOAL_Y - START_Y, GOAL_X - START_X)
            root_state[:, 3] = math.cos(target_yaw / 2.0)
            root_state[:, 6] = math.sin(target_yaw / 2.0)
            
            env.unwrapped.scene["robot"].write_root_pose_to_sim(root_state[:, :7])
            env.unwrapped.scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])
            env.unwrapped.scene["robot"].reset()

        # 3️⃣ 장애물을 무작위 배치합니다.
        drawer_prim = stage.GetPrimAtPath(drawer_path)
        if drawer_prim.IsValid():
            if not drawer_prim.IsActive():
                drawer_prim.SetActive(True)
            x_rand = random.uniform(X_MIN, X_MAX)
            y_rand = random.uniform(-Y_BOUND, Y_BOUND)
            yaw_rand = random.uniform(0, 360)
            mat = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw_rand))
            mat.SetTranslate(Gf.Vec3d(x_rand, y_rand, 0.0))
            UsdGeom.Xformable(drawer_prim).MakeMatrixXform().Set(mat)
            print(f"    🎲 [장애물 무작위 배치] X: {x_rand:.2f}, Y: {y_rand:.2f}")

        # 4️⃣ [수정] 배치가 끝난 시점을 물리 월드에 딱 한 번 동기화합니다. (단종된 옛날함수 제거)
        simulation_app.update()
        env.unwrapped.scene.update(dt=env.unwrapped.step_dt)
        
        # 5️⃣ [수정] 텔레포트가 완료된 현 프레임의 깨끗한 관측치를 추출합니다.
        obs_dict = env.unwrapped.observation_manager.compute()
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

        step_count = 0
        max_steps = 1500
        success = False
        final_dist = 999.0

        while step_count < max_steps and simulation_app.is_running():
            current_pos = env.unwrapped.scene["robot"].data.root_pos_w[0]
            curr_x, curr_y = current_pos[0].item(), current_pos[1].item()
            final_dist = math.dist((curr_x, curr_y), (GOAL_X, GOAL_Y))

            if final_dist < 0.5:
                success = True
                print(f"    🎉 [성공] 장애물 회피 후 목적지 도달! (소요 스텝: {step_count}, 오차: {final_dist:.2f}m)")
                break

            # 💡 [수정] inference_mode는 모델 추론할 때만 정말 좁게 적용합니다.
            with torch.inference_mode():
                actions = policy(obs)
            
            # 액션 클리핑 및 환경 스텝은 inference_mode 밖에서 수행합니다.
            actions = torch.clamp(actions, -1.0, 1.0)
            obs_dict, _, terminated, truncated, _ = env.step(actions)
            obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
            
            dones = terminated | truncated
            if dones[0]:
                print("    💥 [실패] 로봇이 넘어지거나 충돌 판정으로 리셋됨.")
                break

            step_count += 1

        if not success and step_count >= max_steps:
            print(f"    ⏰ [실패] 제한 시간 초과 (남은 거리: {final_dist:.2f}m)")

        with open(result_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([ep, success, step_count, round(final_dist, 2)])

    print("\n================ 🎉 Go2 주행 및 장애물 회피 100회 시나리오 완료 ================")
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()