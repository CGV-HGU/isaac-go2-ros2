import argparse
import sys
import os
import time
import csv
import math
import random
import torch
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

from isaaclab.app import AppLauncher

# 1. 인자 설정 및 앱 실행
parser = argparse.ArgumentParser(description="Go2 Nav2 100 Trials Auto Evaluation")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Unitree-Go2-v0", help="Task name")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
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

# 센서 세팅 스크립트 임포트
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts/reinforcement_learning/skrl"))
import ros2_sensor_setup

# ROS 2 제어 노드
class Nav2AutoNode(Node):
    def __init__(self):
        super().__init__('nav2_auto_evaluator')
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.curr_pos = [0.0, 0.0]
        self.goal_reached = False

    def odom_callback(self, msg):
        self.curr_pos = [msg.pose.pose.position.x, msg.pose.pose.position.y]

    def send_goal(self, x, y):
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = float(x)
        goal.pose.position.y = float(y)
        goal.pose.orientation.w = 1.0
        self.goal_pub.publish(goal)
        print(f"[ROS2] 목적지 전송: ({x}, {y})")

def main():
    # ROS 2 초기화
    rclpy.init()
    ros_node = Nav2AutoNode()

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

    if not resume_path or not os.path.exists(resume_path):
        print("[ERROR] 모델을 찾을 수 없습니다."); simulation_app.close(); return

    print(f"[INFO] 로딩 모델: {resume_path}")

    # RL 모델 로딩 (skrl)
    from skrl.envs.wrappers.torch import wrap_env
    from skrl.utils.model_instantiators.torch import shape_model_instantiator
    
    env = wrap_env(env)
    models_cfg = {
        "policy": {
            "class": "GaussianMixin",
            "network": [{"name": "net", "input": "STATES", "layers": [512, 256, 128], "activation": "elu"},
                        {"name": "policy", "input": "net", "layers": [env.action_space.shape[0]], "activation": None}],
            "clip_actions": True,
        }
    }
    policy = shape_model_instantiator(network_cfg=models_cfg["policy"], observation_space=env.observation_space, action_space=env.action_space, device=env.device)
    checkpoint = torch.load(resume_path, map_location=env.device)
    policy.load_state_dict(checkpoint["policy"] if "policy" in checkpoint else checkpoint)
    policy.eval()

    # 결과 파일 초기화
    TOTAL_TRIALS = 100
    RESULT_FILE = "eval_results_final.csv"
    with open(RESULT_FILE, 'w', newline='') as f:
        csv.writer(f).writerow(["Trial", "Success", "Distance_to_Goal", "Obstacle_X", "Obstacle_Y", "Goal_X", "Goal_Y"])

    cmd_vel_attr_lin = og.Controller.attribute("/World/ROS2_Camera_Graph/ROS2CmdVel.outputs:linearVelocity")
    cmd_vel_attr_ang = og.Controller.attribute("/World/ROS2_Camera_Graph/ROS2CmdVel.outputs:angularVelocity")

    print(f"\n🚀 100회 자동 평가 시작")

    for trial in range(1, TOTAL_TRIALS + 1):
        # 1. 로봇 위치 초기화 (스폰 높이 0.4m)
        START_X, START_Y = -1.0, 0.0
        env.reset()
        root_state = env.unwrapped.scene["robot"].data.default_root_state.clone()
        root_state[:, 0], root_state[:, 1], root_state[:, 2] = START_X, START_Y, 0.4
        env.unwrapped.scene["robot"].write_root_pose_to_sim(root_state[:, :7])
        env.unwrapped.scene["robot"].reset()

        # 2. 목적지 랜덤 생성
        GOAL_X, GOAL_Y = 5.0, random.uniform(-1.5, 1.5)
        
        # 3. 장애물 랜덤 배치 (출발지와 목적지 사이)
        obs_x, obs_y = random.uniform(1.0, 3.5), random.uniform(-1.0, 1.0)
        drawer_path = "/World/Moveable_Objects/Drawer"
        drawer_prim = stage.GetPrimAtPath(drawer_path)
        if drawer_prim.IsValid():
            UsdGeom.Xformable(drawer_prim).ClearXformOpOrder()
            UsdGeom.Xformable(drawer_prim).AddTranslateOp().Set(Gf.Vec3d(obs_x, obs_y, 0.0))
            omni.physx.get_physx_interface().update_transform(drawer_path)

        # 4. Nav2 목적지 전송
        time.sleep(1.0) # 로봇 안착 대기
        ros_node.send_goal(GOAL_X, GOAL_Y)

        obs, _ = env.reset()
        success = False
        timeout = 600 # 60초 (10Hz 기준)
        
        for step in range(timeout):
            # Nav2 제어 명령 수신
            lin_vel = og.Controller.get(cmd_vel_attr_lin)
            ang_vel = og.Controller.get(cmd_vel_attr_ang)
            v_x, v_y, v_yaw = (lin_vel[0], lin_vel[1], ang_vel[2]) if lin_vel is not None else (0,0,0)

            # RL 에이전트 명령 전달
            env.unwrapped.command_manager.get_command("base_velocity")[:] = torch.tensor([[v_x, v_y, v_yaw]], device=env.device, dtype=torch.float32)
            with torch.no_grad():
                actions = policy.act({"states": obs}, role="policy")[0]
                obs, _, _, _, _ = env.step(actions)

            # 성공 판정
            dist = math.dist(ros_node.curr_pos, [GOAL_X, GOAL_Y])
            if dist < 0.5:
                success = True
                print(f"[{trial}/100] ✅ 성공! (남은 거리: {dist:.2f}m)")
                break

            # ROS 콜백 처리
            rclpy.spin_once(ros_node, timeout_sec=0)
            simulation_app.update()

        if not success:
            print(f"[{trial}/100] ❌ 실패 (남은 거리: {dist:.2f}m)")

        with open(RESULT_FILE, 'a', newline='') as f:
            csv.writer(f).writerow([trial, success, round(dist, 2), obs_x, obs_y, GOAL_X, GOAL_Y])

    print(f"\n✅ 모든 평가 완료. 결과 저장: {RESULT_FILE}")
    env.close()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
    simulation_app.close()
