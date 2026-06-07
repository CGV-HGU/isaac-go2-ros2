import argparse
import sys
import os
import time
import csv
import math
import random
import torch
import numpy as np

from isaaclab.app import AppLauncher

# 1. 인자 설정 및 앱 실행
parser = argparse.ArgumentParser(description="Go2 Nav2 100 Trials Evaluation")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Unitree-Go2-v0", help="Task name")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. Isaac Lab 및 관련 라이브러리 임포트
import gymnasium as gym
import omni.graph.core as og
import omni.kit.app
from pxr import Gf, UsdGeom
import omni.physx
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg

# 하영님의 기존 센서 세팅 스크립트 임포트
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts/reinforcement_learning/skrl"))
import ros2_sensor_setup

def main():
    # 환경 설정 로드
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)

    # 센서 및 통신 세팅
    stage = omni.usd.get_context().get_stage()
    ros2_sensor_setup.setup_ros2_sensors(stage, 
                                       robot_base_path="/World/envs/env_0/Robot/base", 
                                       camera_path="/World/envs/env_0/Robot/base/front_cam")

    # 체크포인트 자동 탐색
    if args_cli.checkpoint is None:
        log_root = "logs/skrl/unitree_go2_flat"
        subdirs = [os.path.join(log_root, d) for d in os.listdir(log_root) if os.path.isdir(os.path.join(log_root, d))]
        latest_run = max(subdirs, key=os.path.getmtime) if subdirs else None
        resume_path = os.path.join(latest_run, "checkpoints/best_agent.pt") if latest_run else None
    else:
        resume_path = args_cli.checkpoint

    if not resume_path:
        print("[ERROR] 모델을 찾을 수 없습니다."); simulation_app.close(); return

    print(f"[INFO] 로딩 모델: {resume_path}")

    # [핵심 수정] skrl 공식 에이전트 로딩 방식 사용
    from skrl.envs.wrappers.torch import wrap_env
    from skrl.agents.torch.ppo import PPO
    from skrl.utils.model_instantiators.torch import gaussian_model
    
    env = wrap_env(env)
    
    # 하영님의 모델 구조와 정확히 일치하는 인스턴시에이터 설정
    # 관찰값(48) -> 512 -> 256 -> 128 -> 액션(12)
    models_cfg = {
        "policy": {
            "class": "GaussianMixin",
            "network": [
                {"name": "net_container", "input": "STATES", "layers": [512, 256, 128], "activation": "elu"},
                {"name": "policy_layer", "input": "net_container", "layers": [env.action_space.shape[0]], "activation": None},
            ],
            "clip_actions": True,
        }
    }
    
    # 저장된 파일 구조에 맞춰 모델 수동 생성 (가장 안전함)
    from skrl.utils.model_instantiators.torch import shape_model_instantiator
    
    # 에러 방지를 위해 하영님의 파일 구조("policy_layer" 존재)를 강제로 생성
    policy = shape_model_instantiator(
        network_cfg=models_cfg["policy"],
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device
    )
    
    checkpoint = torch.load(resume_path, map_location=env.device)
    policy.load_state_dict(checkpoint["policy"] if "policy" in checkpoint else checkpoint)
    policy.eval()

    def get_action(obs):
        with torch.no_grad():
            if isinstance(obs, dict):
                obs = {k: v.to(env.device) if torch.is_tensor(v) else v for k, v in obs.items()}
            elif torch.is_tensor(obs):
                obs = obs.to(env.device)
            return policy.act({"states": obs}, role="policy")[0]

    # 설정 및 결과 파일 초기화
    TOTAL_EPISODES = 100
    START_X, START_Y, GOAL_X, GOAL_Y = -1.0, 0.0, 5.0, 0.0
    RESULT_FILE = "eval_results.csv"
    if not os.path.exists(RESULT_FILE):
        with open(RESULT_FILE, 'w', newline='') as f:
            csv.writer(f).writerow(["Episode", "Success", "Steps", "Final_Dist", "Obs_X", "Obs_Y"])

    cmd_vel_attr_lin = og.Controller.attribute("/World/ROS2_Camera_Graph/ROS2CmdVel.outputs:linearVelocity")
    cmd_vel_attr_ang = og.Controller.attribute("/World/ROS2_Camera_Graph/ROS2CmdVel.outputs:angularVelocity")

    print(f"\n🚀 100회 자동 평가 시작")

    for ep in range(1, TOTAL_EPISODES + 1):
        env.reset()
        with torch.no_grad():
            root_state = env.unwrapped.scene["robot"].data.default_root_state.clone()
            root_state[:, 0], root_state[:, 1], root_state[:, 2] = START_X, START_Y, 0.25
            env.unwrapped.scene["robot"].write_root_pose_to_sim(root_state[:, :7])
            env.unwrapped.scene["robot"].reset()
        
        obs_x, obs_y = random.uniform(0.5, 3.5), random.uniform(-0.7, 0.7)
        drawer_path = "/World/Moveable_Objects/Drawer"
        drawer_prim = stage.GetPrimAtPath(drawer_path)
        if drawer_prim.IsValid():
            UsdGeom.Xformable(drawer_prim).ClearXformOpOrder()
            UsdGeom.Xformable(drawer_prim).AddTranslateOp().Set(Gf.Vec3d(obs_x, obs_y, 0.0))
            omni.physx.get_physx_interface().update_transform(drawer_path)

        obs, _ = env.reset()
        success, step_count, final_dist = False, 0, 999.0

        while step_count < 1500 and simulation_app.is_running():
            lin_vel = og.Controller.get(cmd_vel_attr_lin)
            ang_vel = og.Controller.get(cmd_vel_attr_ang)
            v_x, v_y, v_yaw = (lin_vel[0], lin_vel[1], ang_vel[2]) if lin_vel is not None else (0,0,0)

            with torch.no_grad():
                curr_pos = env.unwrapped.scene["robot"].data.root_pos_w[0]
                final_dist = math.dist((curr_pos[0].item(), curr_pos[1].item()), (GOAL_X, GOAL_Y))
                if final_dist < 0.5:
                    success = True; break
                
                env.unwrapped.command_manager.get_command("base_velocity")[:] = torch.tensor([[v_x, v_y, v_yaw]], device=env.device, dtype=torch.float32)
                
                actions = get_action(obs)
                obs, _, terminated, truncated, _ = env.step(actions)
                if terminated or truncated: break

            step_count += 1
            simulation_app.update()

        print(f"[{ep}/100] {'✅ 성공' if success else '❌ 실패'} ({final_dist:.2f}m)")
        with open(RESULT_FILE, 'a', newline='') as f:
            csv.writer(f).writerow([ep, success, step_count, round(final_dist, 2), obs_x, obs_y])

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
