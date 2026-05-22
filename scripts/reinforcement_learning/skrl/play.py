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
import omni
import omni.appwindow
import omni.graph.core as og
import omni.usd
from pxr import UsdGeom, Gf
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

# PLACEHOLDER: Extension template (do not remove this comment)


def make_keyboard_state():
    return {
        "forward": 0.0,
        "side": 0.0,
        "yaw": 0.0,
        "reset": False,
        "spawn_obstacle": False,
        "respawn_in_place": False,
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
    elif event.input == carb.input.KeyboardInput.R:
        state["reset"] = True if pressed else False
    elif event.input == carb.input.KeyboardInput.T:
        state["spawn_obstacle"] = True if pressed else False
    elif event.input == carb.input.KeyboardInput.F:
        state["respawn_in_place"] = True if pressed else False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
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

    # set the log directory for the environment (works for all environment types)
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

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ==========================================
    # [추가됨] ROS2 RGB & Depth 카메라 퍼블리셔 세팅
    # ==========================================
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import ros2_sensor_setup
    
    stage = omni.usd.get_context().get_stage()
    ros2_sensor_setup.setup_ros2_sensors(stage)
    # ==========================================
    # ==========================================

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export the trained policy to JIT and ONNX formats
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    if version.parse(installed_version) >= version.parse("4.0.0"):
        # use the new export functions for rsl-rl >= 4.0.0
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
        policy_nn = None
    else:
        # extract the neural network for rsl-rl < 4.0.0
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

        # extract the normalizer
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        # export to JIT and ONNX
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

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
    obs = env.get_observations()
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
                    # 현재 로봇의 X, Y 좌표 가져오기
                    robot_pos = env.unwrapped.scene["robot"].data.root_pos_w[0].clone()
                    
                    # Z축만 살짝 위로 올리고(0.5m), 회전(Orientation)은 초기화(Roll, Pitch 0 / Yaw 유지 또는 0)
                    new_pos = robot_pos
                    new_pos[2] = 0.5  # 공중에서 살짝 떨어지도록 설정
                    
                    # 쿼터니언을 (w, x, y, z) = (1, 0, 0, 0) 즉 회전 없게 똑바로 세움
                    new_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.unwrapped.device)
                    
                    # 속도 0으로 초기화
                    zero_vel = torch.zeros(3, device=env.unwrapped.device)
                    zero_ang_vel = torch.zeros(3, device=env.unwrapped.device)
                    
                    # 로봇 상태 강제 설정 (root state)
                    env.unwrapped.scene["robot"].write_root_pose_to_sim(torch.cat([new_pos, new_quat]).unsqueeze(0))
                    env.unwrapped.scene["robot"].write_root_velocity_to_sim(torch.cat([zero_vel, zero_ang_vel]).unsqueeze(0))
                    
                    # 환경 내부 상태 업데이트(리셋 효과)를 위해 step()에 필요한 obs 갱신 유도
                    env.unwrapped.scene["robot"].reset()
                    
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
                    
                    import random
                    from pxr import Usd, UsdGeom
                    
                    # [메쉬 맵 기반 자동 범위 계산]
                    # 로드된 캡스톤 메쉬의 실제 크기를 분석하여 스폰 범위를 자동 설정합니다.
                    map_prim = stage.GetPrimAtPath("/World/fused_scene")
                    if map_prim.IsValid():
                        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default'])
                        b_range = bbox_cache.ComputeWorldBound(map_prim).ComputeAlignedBox()
                        min_pt = b_range.GetMin()
                        max_pt = b_range.GetMax()
                        
                        margin = 0.5
                        min_x = float(min_pt[0]) + margin
                        max_x = float(max_pt[0]) - margin
                        min_y = float(min_pt[1]) + margin
                        max_y = float(max_pt[1]) - margin
                    else:
                        min_x, max_x = -3.0, 3.0
                        min_y, max_y = -1.0, 1.0
                    
                    if min_x > max_x: min_x, max_x = max_x, min_x
                    if min_y > max_y: min_y, max_y = max_y, min_y
                    
                    # [Raycast 기반 안전한 지형 판별]
                    # 직사각형 바운딩 박스가 실제 메쉬 모양과 다를 때 허공에 떨어지는 것을 방지합니다.
                    from omni.physx import get_physx_scene_query_interface
                    
                    valid_spawn_found = False
                    max_attempts = 10  # 최대 10번 랜덤 위치 시도
                    final_x, final_y, final_z = 0.0, 0.0, 0.5
                    
                    for _ in range(max_attempts):
                        rand_x = random.uniform(min_x, max_x)
                        rand_y = random.uniform(min_y, max_y)
                        
                        # 하늘(z=10.0m)에서 수직 아래로(0, 0, -1) Ray를 쏩니다.
                        origin = carb.Float3(rand_x, rand_y, 10.0)
                        ray_dir = carb.Float3(0.0, 0.0, -1.0)
                        
                        hit_info = get_physx_scene_query_interface().raycast_closest(origin, ray_dir, 15.0)
                        
                        # Ray가 무언가(메쉬 바닥)에 부딪혔다면 유효한 지형 위라고 판단!
                        if hit_info["hit"]:
                            final_x = rand_x
                            final_y = rand_y
                            final_z = hit_info["position"][2] + 0.5  # 바닥 높이보다 0.5m 위에서 스폰
                            valid_spawn_found = True
                            break
                    
                    if not valid_spawn_found:
                        print("[Warning] Could not find valid ground within mesh bounds after 10 attempts. Spawning at random box location anyway.")
                        final_x = random.uniform(min_x, max_x)
                        final_y = random.uniform(min_y, max_y)
                        final_z = 0.5

                    # 결정된 안전한 위치로 스폰
                    drop_pos = Gf.Vec3d(float(final_x), float(final_y), float(final_z))
                    
                    # 미리 만들어둔 장애물(Prim) 가져와서 위치 강제 이동 (Teleport)
                    obstacle_prim = stage.GetPrimAtPath("/World/InteractiveObstacle")
                    if obstacle_prim.IsValid():
                        xform = UsdGeom.Xformable(obstacle_prim)
                        xform.ClearXformOpOrder()
                        xform.AddTranslateOp().Set(drop_pos)
                        
                        # 속도 초기화 (이전 관성 제거)
                        rb_api = UsdPhysics.RigidBodyAPI(obstacle_prim)
                        if rb_api:
                            rb_api.GetVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                            rb_api.GetAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                            
                        print(f"[INFO] Obstacle dropped randomly at ({rand_x:.2f}, {rand_y:.2f})!")
                    else:
                        print(f"[ERROR] Pre-spawned obstacle not found!")
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
