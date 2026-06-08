# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

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

import carb
import carb.input
import gymnasium as gym
import torch
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


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
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

    # ==========================================
    # [추가됨] ROS2 센서 및 카메라 세팅 (RTAB-Map용)
    # ==========================================
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import ros2_sensor_setup
    ros2_sensor_setup.setup_ros2_sensors(env.unwrapped.scene.stage)
    # ==========================================

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
    # 🎯 100회 자동 평가 시나리오 설정 (최상단 정의)
    # =========================================================================
    import csv
    import math
    import random
    from pxr import Gf, UsdGeom
    import omni.physx
    import omni.graph.core as og

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

    # [중요] stage 및 경로 설정
    stage = env.unwrapped.scene.stage
    drawer_path = "/World/Moveable_Objects/Drawer"
    physx_iface = omni.physx.get_physx_interface()

    # Nav2에 전달할 목적지 Prim 생성
    goal_prim_path = "/World/Navigation_Goal"
    if stage.GetPrimAtPath(goal_prim_path).IsValid():
        omni.kit.commands.execute("DeletePrims", paths=[goal_prim_path])
    
    goal_geom = UsdGeom.Xform.Define(stage, goal_prim_path)
    goal_geom.AddTranslateOp().Set(Gf.Vec3d(GOAL_X, GOAL_Y, 0.0))

    # [중요] ROS2 Bridge 확장 기능 활성화 및 로딩 대기
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    
    ros2_bridge_prefix = None
    if ext_manager.is_extension_enabled("isaacsim.ros2.bridge"):
        ros2_bridge_prefix = "isaacsim.ros2.bridge"
    elif ext_manager.is_extension_enabled("omni.isaac.ros2_bridge"):
        ros2_bridge_prefix = "omni.isaac.ros2_bridge"
    else:
        try:
            ext_manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
            ros2_bridge_prefix = "isaacsim.ros2.bridge"
        except:
            try:
                ext_manager.set_extension_enabled_immediate("omni.isaac.ros2_bridge", True)
                ros2_bridge_prefix = "omni.isaac.ros2_bridge"
            except:
                print("[ERROR] ROS2 Bridge 확장 기능을 로드할 수 없습니다.")

    # 확장 기능 로딩 대기
    print(f"[INFO] ROS2 Bridge ({ros2_bridge_prefix}) 로딩 대기 중...")
    for _ in range(150):
        simulation_app.update()

    # [중요] 기존 그래프가 있으면 확실하게 삭제
    graph_path = "/World/ROS2_Nav_Eval_Graph"
    if stage.GetPrimAtPath(graph_path).IsValid():
        omni.kit.commands.execute("DeletePrims", paths=[graph_path])
        for _ in range(50): simulation_app.update()

    # 노드 타입 설정 (5.1.0 대응)
    possible_pose_nodes = [
        f"{ros2_bridge_prefix}.ROS2PublishPoseStamped",
        f"{ros2_bridge_prefix}.ROS2PublishPose",
        "omni.isaac.ros2_bridge.ROS2PublishPoseStamped",
        "omni.isaac.ros2_bridge.ROS2PublishPose"
    ]
    
    cmd_twist_type = f"{ros2_bridge_prefix}.ROS2SubscribeTwist" if ros2_bridge_prefix else "omni.isaac.ros2_bridge.ROS2SubscribeTwist"

    success_graph = False
    for pose_node_type in possible_pose_nodes:
        if success_graph: break
        try:
            print(f"[INFO] OmniGraph 생성 시도 중... (Pose Node: {pose_node_type})")
            og.Controller.edit(
                {"graph_path": graph_path, "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnTick", "omni.graph.action.OnPlaybackTick"),
                        ("ROSTwistSub", cmd_twist_type),
                        ("ROSGoalPub", pose_node_type),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("ROSTwistSub.inputs:topicName", "/cmd_vel"),
                        ("ROSGoalPub.inputs:topicName", "/goal_pose"),
                        ("ROSGoalPub.inputs:frameId", "map"),
                        ("ROSGoalPub.inputs:targetPrim", goal_prim_path),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnTick.outputs:tick", "ROSTwistSub.inputs:execIn"),
                        ("OnTick.outputs:tick", "ROSGoalPub.inputs:execIn"),
                    ],
                },
            )
            success_graph = True
            print(f"[INFO] OmniGraph 생성 성공! ({pose_node_type})")
        except Exception as e:
            print(f"[WARNING] '{pose_node_type}' 노드로 생성 실패, 다음 시도... ({e})")
            if stage.GetPrimAtPath(graph_path).IsValid():
                omni.kit.commands.execute("DeletePrims", paths=[graph_path])
                for _ in range(20): simulation_app.update()

    if not success_graph:
        print("[ERROR] OmniGraph 생성 최종 실패. Nav2 통신이 불가능할 수 있습니다.")

    # OmniGraph 속성 링크 미리 선언 (루프 내부 최적화용)
    linear_attr = og.Controller.attribute(f"{graph_path}/ROSTwistSub.outputs:linearVelocity")
    angular_attr = og.Controller.attribute(f"{graph_path}/ROSTwistSub.outputs:angularVelocity")

    # [추가] 키보드 입력 인터페이스 (R키 리셋용)
    def make_keyboard_state():
        return {"reset": False}

    def update_keyboard_state(state, event):
        pressed = event.type == carb.input.KeyboardEventType.KEY_PRESS
        if event.input == carb.input.KeyboardInput.R:
            state["reset"] = True if pressed else False

    kb_state = make_keyboard_state()
    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard()
    input_interface = carb.input.acquire_input_interface()
    def on_kb_event(event, *args, **kwargs):
        update_keyboard_state(kb_state, event)
        return True
    kb_sub = input_interface.subscribe_to_keyboard_events(keyboard, on_kb_event)

    print(f"\n================ 총 {TOTAL_EPISODES}회 자동화 Nav2 회피 평가 시작 ================")
    print(f"📍 고정 출발지: ({START_X}, {START_Y}) | 고정 목적지: ({GOAL_X}, {GOAL_Y})")
    print("⌨️  테스트 중 R 키를 누르면 즉시 에피소드를 리셋하고 다음 실험으로 넘어갑니다.")

    with torch.inference_mode():
        for ep in range(1, TOTAL_EPISODES + 1):
            if not simulation_app.is_running():
                break
                
            print(f"\n🎬 [실험 {ep}/{TOTAL_EPISODES}] 에피소드 초기화 중...")
            
            # [1] 로봇을 지정된 출발지로 강제 이동 (Teleport) 및 낙하 높이 안정화
            root_state = env.unwrapped.scene["robot"].data.default_root_state.clone()
            root_state[:, 0] = START_X
            root_state[:, 1] = START_Y
            root_state[:, 2] = 0.20
            
            # 시작 시 목적지를 똑바로 바라보도록 Yaw 각도 계산 및 정렬
            target_yaw = math.atan2(GOAL_Y - START_Y, GOAL_X - START_X)
            root_state[:, 3] = math.cos(target_yaw / 2.0)
            root_state[:, 6] = math.sin(target_yaw / 2.0)
            
            env.unwrapped.scene["robot"].write_root_pose_to_sim(root_state[:, :7])
            env.unwrapped.scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])
            
            env.unwrapped.scene["robot"].reset()
            
            # [2] 장애물 무작위 배치 및 물리 엔진 동기화 리셋
            drawer_prim = stage.GetPrimAtPath(drawer_path)
            if drawer_prim.IsValid():
                if not drawer_prim.IsActive(): 
                    drawer_prim.SetActive(True)
                
                # [수정] 장애물 물리 및 레이저 스캔 인식 설정 (충돌 메시 및 시맨틱 레이블 부여)
                from pxr import UsdPhysics, PhysxSchema, Sdf, Usd
                
                # 모든 하위 메시를 포함하여 충돌 및 시맨틱 레이블 적용
                for p in Usd.PrimRange(drawer_prim):
                    if p.IsA(UsdGeom.Mesh):
                        # 충돌 속성 부여
                        if not p.HasAPI(UsdPhysics.CollisionAPI):
                            UsdPhysics.CollisionAPI.Apply(p)
                        if not p.HasAPI(PhysxSchema.PhysxCollisionAPI):
                            PhysxSchema.PhysxCollisionAPI.Apply(p)
                        
                        # 시맨틱 레이블 부여
                        if not p.HasAttribute("semanticLabel"):
                            p.CreateAttribute("semanticLabel", Sdf.ValueTypeNames.String).Set("obstacle")
                        if not p.HasAttribute("semanticType"):
                            p.CreateAttribute("semanticType", Sdf.ValueTypeNames.String).Set("class")

                x_rand = random.uniform(X_MIN, X_MAX)
                y_rand = random.uniform(-Y_BOUND, Y_BOUND)
                yaw_rand = random.uniform(0, 360)
                
                mat = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw_rand))
                mat.SetTranslate(Gf.Vec3d(x_rand, y_rand, 0.02)) # 바닥에 끼지 않게 살짝 띄움
                
                UsdGeom.Xformable(drawer_prim).MakeMatrixXform().Set(mat)
                
                # 물리 상태 강제 업데이트
                for _ in range(5):
                    simulation_app.update()
                
                try:
                    physx_iface.set_rigid_body_linear_velocity(drawer_path, [0.0, 0.0, 0.0])
                    physx_iface.set_rigid_body_angular_velocity(drawer_path, [0.0, 0.0, 0.0])
                except Exception:
                    pass

                print(f"    🎲 [장애물 세팅 완료] 물리 충돌 및 레이저 인식 활성화 | X: {x_rand:.2f}, Y: {y_rand:.2f}")

            # 초기화 상태 안정화를 위한 5틱 대기
            for _ in range(5):
                simulation_app.update()

            obs = env.get_observations()
            print(f"    🎯 [Nav2 통신] 목적지 토픽 자동 연동 완료 -> ({GOAL_X}, {GOAL_Y})")

            step_count = 0
            max_steps = 1500  
            success = False
            final_dist = 999.0
            
            while step_count < max_steps and simulation_app.is_running():
                start_time = time.time()
                
                # [수동 리셋 (R키)]
                if kb_state["reset"]:
                    print(f"\n🔄 [수동 리셋] R키가 입력되어 {ep}번 실험을 중단하고 다음으로 넘어갑니다.")
                    kb_state["reset"] = False
                    break

                # [4] OmniGraph로부터 ROS 2 /cmd_vel 속도 명령 실시간 패치
                # 그래프 생성이 성공했을 때만 값을 가져오도록 시도
                v_forward, v_lateral, v_yaw = 0.0, 0.0, 0.0
                if success_graph:
                    try:
                        lin_vel = og.Controller.get(linear_attr)
                        ang_vel = og.Controller.get(angular_attr)
                        if lin_vel is not None: v_forward, v_lateral = lin_vel[0], lin_vel[1]
                        if ang_vel is not None: v_yaw = ang_vel[2]
                    except: pass
                
                current_pos = env.unwrapped.scene["robot"].data.root_pos_w[0]
                curr_x = current_pos[0].item()
                curr_y = current_pos[1].item()
                
                final_dist = math.dist((curr_x, curr_y), (GOAL_X, GOAL_Y))
                
                # 목적지 통과 판정 (50cm 이내 접근 시 성공)
                if final_dist < 0.5:
                    success = True
                    print(f"    🎉 [성공] Nav2 제어로 목적지 도달! (소요 스텝: {step_count}, 오차: {final_dist:.2f}m)")
                    break

                # [5] Nav2 명령을 로봇 제어기에 전달
                cmd = torch.tensor([v_forward, v_lateral, v_yaw], device=env.unwrapped.device, dtype=torch.float32)
                cmd = cmd.unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
                env.unwrapped.command_manager.get_command("base_velocity")[:] = cmd

                # RL Policy 구동 및 시뮬레이션 전진
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
                
            # 결과를 CSV 데이터베이스 파일에 기록
            with open(result_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([ep, success, step_count, round(final_dist, 2)])

    print("\n================ 🎉 100회 자동 시나리오 주행 평가가 최종 완료되었습니다! ================")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()