# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default=None, help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument(
    "--ml_framework", type=str, default="torch", choices=["torch", "jax", "jax-numpy"], help="The ML framework."
)
parser.add_argument(
    "--algorithm", type=str, default="PPO", choices=["AMP", "PPO", "IPPO", "MAPPO"], help="The RL algorithm."
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
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

import skrl
from packaging import version
if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()

# PLACEHOLDER: Extension template (do not remove this comment)


def make_keyboard_state():
    return {
        "forward": 0.0,
        "side": 0.0,
        "yaw": 0.0,
        "reset": False,
        "spawn_obstacle": False,
        "respawn_in_place": False,
        "shift": False,
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
    elif event.input in [carb.input.KeyboardInput.LEFT_SHIFT, carb.input.KeyboardInput.RIGHT_SHIFT]:
        state["shift"] = True if pressed else False
    elif event.input == carb.input.KeyboardInput.R:
        state["reset"] = True if pressed else False
    elif event.input == carb.input.KeyboardInput.T:
        state["spawn_obstacle"] = True if pressed else False
    elif event.input == carb.input.KeyboardInput.F:
        state["respawn_in_place"] = True if pressed else False


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Play with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    agent_cfg["trainer"]["close_environment_at_exit"] = False

    # set the agent and environment seed from command line
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None
    
    log_dir = os.path.dirname(resume_path) if resume_path else os.getcwd()
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

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    # configure and instantiate the skrl runner
    runner = Runner(env, agent_cfg)

    if resume_path:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.agent.act

    print("\n================ Keyboard Control ================")
    print("Click the 3D viewport once, then use:")
    print("W / S : forward / backward")
    print("A / D : left / right")
    print("Q / E : yaw left / yaw right")
    print("=================================================\n")

    dt = env.unwrapped.step_dt

    # reset environment
    # [시작 위치 고정] 하드코딩된 X, Y 좌표(-1.0, 0.0)를 사용하고 높이는 안전하게 0.0m 설정
    try:
        robot = env.unwrapped.scene["robot"]
        new_pos = torch.tensor([-1.0, 0.0, 0.0], device=robot.device)
        new_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=robot.device)
        zero_vel = torch.zeros(3, device=robot.device)
        
        # IsaacLab의 초기화 버퍼(default_root_state)를 이 좌표로 고정하여 R키 리셋 시에도 여기로 오게 함
        robot.data.default_root_state[0, :3] = new_pos
        robot.data.default_root_state[0, 3:7] = new_quat
        
        robot.write_root_pose_to_sim(torch.cat([new_pos, new_quat]).unsqueeze(0))
        robot.write_root_velocity_to_sim(torch.cat([zero_vel, zero_vel]).unsqueeze(0))
        print(f"[INFO] 🎯 Robot position set to fixed origin: x=-1.0, y=0.0, z=0.0")
    except Exception as e:
        print(f"[Warning] Failed to set initial robot position: {e}")

    obs, _ = env.reset()
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # keyboard -> base_velocity
            kb_x, kb_y, kb_yaw = keyboard_state["forward"], keyboard_state["side"], keyboard_state["yaw"]
            
            # [Shift 키 달리기 기능] 배율 1.2배
            speed_multiplier = 1.2 if keyboard_state.get("shift", False) else 1.0
            kb_x *= speed_multiplier
            kb_y *= speed_multiplier
            kb_yaw *= speed_multiplier
            
            cmd = torch.tensor(
                [kb_x, kb_y, kb_yaw],
                device=env.unwrapped.device,
                dtype=torch.float32,
            )
            cmd = cmd.unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
            
            # 명령어 강제 주입
            if "base_velocity" in env.unwrapped.command_manager.active_terms:
                env.unwrapped.command_manager.get_command("base_velocity")[:] = cmd
            
            # manual reset via 'R' key
            if keyboard_state["reset"]:
                keyboard_state["reset"] = False
                robot = env.unwrapped.scene["robot"]
                new_pos = robot.data.default_root_state[0, :3]
                new_quat = robot.data.default_root_state[0, 3:7]
                zero_vel = torch.zeros(3, device=robot.device)
                
                robot.write_root_pose_to_sim(torch.cat([new_pos, new_quat]).unsqueeze(0))
                robot.write_root_velocity_to_sim(torch.cat([zero_vel, zero_vel]).unsqueeze(0))
                
                # 조인트 초기화
                default_joint_pos = robot.data.default_joint_pos.clone()
                default_joint_vel = robot.data.default_joint_vel.clone() * 0.0
                robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
                continue

            # agent stepping (SKRL format)
            actions, _, _ = policy(obs, timestep=0, timesteps=0)
            
            # env stepping
            obs, _, _, _, _ = env.step(actions)

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
