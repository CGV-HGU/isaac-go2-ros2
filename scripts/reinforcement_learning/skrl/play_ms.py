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
import random
import math

import carb
import carb.input
import gymnasium as gym
import omni
import omni.appwindow
import omni.graph.core as og
from pxr import UsdGeom, Gf, Sdf
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
        "reset": False
    }


def update_keyboard_state(state, event):
    pressed = event.type == carb.input.KeyboardEventType.KEY_PRESS
    released = event.type == carb.input.KeyboardEventType.KEY_RELEASE

    if not (pressed or released):
        return

    # [디버그] 키보드 입력 확인
    # print(f"[Key Event] {event.input} {'Pressed' if pressed else 'Released'}")

    if event.input == carb.input.KeyboardInput.W:
        state["forward"] = 1.0 if pressed else 0.0
    elif event.input == carb.input.KeyboardInput.S:
        state["forward"] = -1.0 if pressed else 0.0
    elif event.input == carb.input.KeyboardInput.A:
        state["side"] = 0.5 if pressed else 0.0
    elif event.input == carb.input.KeyboardInput.D:
        state["side"] = -0.5 if pressed else 0.0
    elif event.input == carb.input.KeyboardInput.Q:
        state["yaw"] = 0.8 if pressed else 0.0
    elif event.input == carb.input.KeyboardInput.E:
        state["yaw"] = -0.8 if pressed else 0.0
    elif event.input == carb.input.KeyboardInput.R:
        state["reset"] = True if pressed else False


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
    # [수정됨] ROS2 RGB & Depth 카메라 퍼블리셔 세팅 (front_camera 사용)
    # ==========================================
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import ros2_sensor_setup
    
    stage = omni.usd.get_context().get_stage()
    # play.py에서는 경로를 직접 주지 않고 ros2_sensor_setup의 기본값(front_camera)을 사용하도록 함
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

    # 3. 평가 및 장애물 설정
    GOAL_X, GOAL_Y = 5.0, 0.0
    drawer_path = "/World/fused_scene/Moveable_Objects/Drawer"
    result_file = "./play_eval_results_ms.csv"
    import csv
    if not os.path.exists(result_file):
        with open(result_file, 'w', newline='') as f:
            csv.writer(f).writerow(["Timestamp", "Success", "Distance_to_Goal"])

    print("\n================ Keyboard Control ================")
    print("Click the 3D viewport once, then use:")
    print("W / S : forward / backward")
    print("A / D : left / right")
    print("Q / E : yaw left / yaw right")
    print("R : Manual Reset (Respawn)")
    print("=================================================\n")

    dt = env.unwrapped.step_dt

    # simulate environment
    ep_count = 0
    with torch.inference_mode():
        while simulation_app.is_running():
            # reset environment (수동/자동 리셋 시 여기로 돌아옴)
            obs, _ = env.reset()
            
            # [추가] 리셋 직후 로봇이 안정적으로 지면에 내려올 때까지 대기
            for _ in range(20):
                simulation_app.update()
            
            # [추가] 장애물 랜덤 배치
            drawer_prim = stage.GetPrimAtPath(drawer_path)
            if drawer_prim.IsValid():
                if not drawer_prim.IsActive(): 
                    drawer_prim.SetActive(True)

                # [수정] 장애물 물리 및 레이저 스캔 인식 설정
                from pxr import UsdPhysics, PhysxSchema
                import omni.syntheticdata._syntheticdata as sd
                
                if not drawer_prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI.Apply(drawer_prim)
                if not drawer_prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                    PhysxSchema.PhysxCollisionAPI.Apply(drawer_prim)
                
                # 시맨틱 레이블 부여 (직접 USD 속성 생성)
                prim = drawer_prim.GetPrim()
                if not prim.HasAttribute("semanticLabel"):
                    prim.CreateAttribute("semanticLabel", Sdf.ValueTypeNames.String).Set("obstacle")
                if not prim.HasAttribute("semanticType"):
                    prim.CreateAttribute("semanticType", Sdf.ValueTypeNames.String).Set("class")

                x_rand = random.uniform(-1.0, 3.0)
                y_rand = random.uniform(-0.76, 0.76)
                mat = Gf.Matrix4d().SetTranslate(Gf.Vec3d(x_rand, y_rand, 0.02))
                UsdGeom.Xformable(drawer_prim).MakeMatrixXform().Set(mat)

                omni.physx.get_physx_interface().update_transform(drawer_path)
            # 물리 업데이트 및 깨끗한 관측치 추출
            simulation_app.update()
            obs = env.get_observations()

            step_count = 0
            while simulation_app.is_running():
                start_time = time.time()
                step_count += 1
                
                # [수동 리셋 (R키)]
                if keyboard_state["reset"]:
                    print("\n🔄 [수동 리셋] R키가 입력되어 에피소드를 재시작합니다.")
                    keyboard_state["reset"] = False
                    break # 내부 루프 탈출 -> 외부 루프에서 env.reset() 호출됨

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

                # agent stepping
                actions = policy(obs)
                # env stepping
                obs, _, dones, _ = env.step(actions)
                
                # [추가] 성공 여부 체크
                robot_asset = env.unwrapped.scene["robot"]
                curr_pos = robot_asset.data.root_pos_w[0]
                dist = math.dist((curr_pos[0].item(), curr_pos[1].item()), (GOAL_X, GOAL_Y))
                
                if dist < 0.5:
                    print(f"\n🚩 [목적지 도달] 성공 기록 저장!")
                    with open(result_file, 'a', newline='') as f:
                        csv.writer(f).writerow([time.strftime("%Y-%m-%d %H:%M:%S"), True, round(dist, 2)])
                    time.sleep(1.0); break

                # reset recurrent states for episodes that have terminated
                if version.parse(installed_version) >= version.parse("4.0.0"):
                    policy.reset(dones)
                else:
                    policy_nn.reset(dones)
                
                # 로봇이 넘어져서 자동 리셋(dones)이 발생한 경우 로그 출력 및 내부 루프 탈출
                # 초기 안정화 기간 50스텝 동안은 무시
                if dones.any() and step_count > 50:
                    print("\n💥 [리셋] 물리 엔진에 의해 로봇이 넘어지거나 조건이 완료되어 자동 리셋됩니다.")
                    with open(result_file, 'a', newline='') as f:
                        csv.writer(f).writerow([time.strftime("%Y-%m-%d %H:%M:%S"), False, round(dist, 2)])
                    break

                # time delay for real-time evaluation
                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)

            ep_count += 1
            print(f"🔄 에피소드 {ep_count} 종료.")

    input_interface.unsubscribe_to_keyboard_events(keyboard, keyboard_sub)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
