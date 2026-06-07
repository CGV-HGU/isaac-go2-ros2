# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""
import sys
import os

# 1. Isaac Lab 소스 경로 추가
sys.path.append(os.path.expanduser("~/IsaacLab/source/isaaclab"))

# 2. Isaac Sim 핵심 라이브러리 경로 추가 (버전이나 폴더명 매칭을 위해 여러 개 추가)
sys.path.append(os.path.expanduser("~/.local/share/ov/pkg/isaac-sim-4.0.0"))
sys.path.append(os.path.expanduser("~/.local/share/ov/pkg/isaac_sim-4.0.0"))
sys.path.append(os.path.expanduser("~/.local/share/ov/pkg/isaac-sim-4.0.0/exts/omni.isaac.kit"))

import argparse
from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
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

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import os
import time
import csv
import math
import random

import carb
import carb.input
import gymnasium as gym
import torch
import omni.physx
from pxr import Gf, UsdGeom
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

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

# ==========================================
# 🌐 ROS 2 통신 라이브러리 임포트 및 노드 정의
# ==========================================
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist


class IsaacNav2BridgeNode(Node):
    """Nav2 스택과 목적지/속도 제어 명령을 송수신하는 브릿지 노드"""
    def __init__(self):
        super().__init__('isaac_nav2_bridge')
        # 목적지 발행용 Publisher (Nav2가 수신)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        # Nav2 회피 제어 알고리즘의 결과물 수신용 Subscriber
        self.cmd_vel_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        
        self.latest_cmd_vel = [0.0, 0.0, 0.0]  # [v_x, v_y, v_yaw]

    def cmd_vel_callback(self, msg):
        # Nav2로부터 수신한 속도 명령 저장
        self.latest_cmd_vel = [msg.linear.x, msg.linear.y, msg.angular.z]

    def publish_goal(self, x, y):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = 0.0
        msg.pose.orientation.w = 1.0  # 정방향 바라보기 정렬
        self.goal_pub.publish(msg)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    
    # 🎯 [수정] ROS 2 통신 컨텍스트를 메인 로직 최상단에서 가장 먼저 초기화합니다.
    rclpy.init(args=None)
    nav2_bridge = IsaacNav2BridgeNode()
    
    try:
        task_name = args_cli.task.split(":")[-1]
        train_task_name = task_name.replace("-Play", "")

        agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
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

        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "play"),
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

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

        dt = env.unwrapped.step_dt

        # =========================================================================
        # 🎯 100회 자동 평가 시나리오 주행 루프
        # =========================================================================
        TOTAL_EPISODES = 100   # 🔁 총 실험 횟수
        
        START_X, START_Y = -1.0, 0.0
        GOAL_X, GOAL_Y = 5.0, 0.0

        # 목적지와 출발지 '사이'에 물체를 놓기 위한 범위 정의
        X_MIN, X_MAX = -1.0, 3.0 
        Y_BOUND = 0.76  
        
        result_file = "./eval_results.csv" 
        
        with open(result_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "Success", "Time_Steps", "Final_Dist"])

        stage = env.unwrapped.scene.stage
        drawer_path = "/World/fused_scene/Moveable_Objects/Drawer"
        physx_iface = omni.physx.get_physx_interface()

        print(f"\n================ 총 {TOTAL_EPISODES}회 자동화 Nav2 회피 평가 시작 ================")
        print(f"📍 고정 출발지: ({START_X}, {START_Y}) | 고정 목적지: ({GOAL_X}, {GOAL_Y})")

        for ep in range(1, TOTAL_EPISODES + 1):
            if not simulation_app.is_running():
                break
                
            print(f"\n🎬 [실험 {ep}/{TOTAL_EPISODES}] 에피소드 초기화 중...")

            # [1] 로봇을 지정된 출발지로 강제 이동 (Teleport) 및 낙하 높이 안정화(0.4m)
            with torch.inference_mode():
                root_state = env.unwrapped.scene["robot"].data.default_root_state.clone()
                root_state[:, 0] = START_X
                root_state[:, 1] = START_Y
                root_state[:, 2] = 0.40  # 0.4m 고정 낙하 위치
                
                # 시작 시 목적지를 똑바로 바라보도록 Yaw 각도 계산 및 정렬
                target_yaw = math.atan2(GOAL_Y - START_Y, GOAL_X - START_X)
                root_state[:, 3] = math.cos(target_yaw / 2.0)
                root_state[:, 6] = math.sin(target_yaw / 2.0)
                
                env.unwrapped.scene["robot"].write_root_pose_to_sim(root_state[:, :7])
                env.unwrapped.scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])
                
                # 관절 상태 초기화
                env.unwrapped.scene["robot"].reset()
            
            # [2] 장애물 무작위 배치
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
                
                # PhysX 동기화
                physx_iface.update_transform(drawer_path)
                print(f"    🎲 [장애물 랜덤 배치 완료] X: {x_rand:.2f}, Y: {y_rand:.2f}")

            # [3] Nav2 목적지 토픽 발행
            nav2_bridge.latest_cmd_vel = [0.0, 0.0, 0.0]
            nav2_bridge.publish_goal(GOAL_X, GOAL_Y)
            print(f"    🎯 [Nav2 통신] 목적지 토픽 발행 완료 -> ({GOAL_X}, {GOAL_Y})")

            # 버퍼 업데이트로 텔레포트 확정
            simulation_app.update()
            env.unwrapped.scene.update(dt=env.unwrapped.step_dt)

            # 최신 관측값 획득
            obs = env.get_observations()

            step_count = 0
            max_steps = 1500  
            success = False
            final_dist = 999.0
            
            while step_count < max_steps and simulation_app.is_running():
                start_time = time.time()
                
                # [4] ROS 2 서브스크립션 큐 처리 
                rclpy.spin_once(nav2_bridge, timeout_sec=0.0)
                v_forward, v_lateral, v_yaw = nav2_bridge.latest_cmd_vel
                
                with torch.inference_mode():
                    current_pos = env.unwrapped.scene["robot"].data.root_pos_w[0]
                    curr_x = current_pos[0].item()
                    curr_y = current_pos[1].item()
                    
                    final_dist = math.dist((curr_x, curr_y), (GOAL_X, GOAL_Y))
                    
                    # 50cm 이내 접근 시 성공 판정
                    if final_dist < 0.5:
                        success = True
                        print(f"    🎉 [성공] Nav2 제어로 목적지 도달! (소요 스텝: {step_count}, 오차: {final_dist:.2f}m)")
                        break

                    # [5] 제어 명령 바인딩 및 RL Policy 구동
                    cmd = torch.tensor([v_forward, v_lateral, v_yaw], device=env.unwrapped.device, dtype=torch.float32)
                    cmd = cmd.unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
                    env.unwrapped.command_manager.get_command("base_velocity")[:] = cmd

                    actions = policy(obs)
                    obs, _, dones, _ = env.step(actions)
                    
                    if version.parse(installed_version) >= version.parse("4.0.0"):
                        policy.reset(dones)
                    else:
                        policy_nn.reset(dones)
                        
                    if dones[0]:
                        print("    💥 [실패] 로봇이 넘어지거나 외부 충돌로 인하여 에피소드가 파괴됨.")
                        break
                        
                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)
                    
                step_count += 1
                
            if not success and step_count >= max_steps:
                print(f"    ⏰ [실패] 타임아웃 제한 시간 초과 (남은 거리: {final_dist:.2f}m)")
                
            # CSV 결과 저장
            with open(result_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([ep, success, step_count, round(final_dist, 2)])

        print("\n================ 🎉 100회 자동 시나리오 주행 평가가 최종 완료되었습니다! ================")

    finally:
        # 🧼 자원 해제 보장
        nav2_bridge.destroy_node()
        rclpy.shutdown()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()