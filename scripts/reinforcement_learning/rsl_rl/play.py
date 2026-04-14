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
import omni.appwindow
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
    from omni.isaac.core.utils.extensions import enable_extension
    enable_extension("omni.isaac.ros2_bridge")
    
    import omni.graph.core as og
    import omni.isaac.core.utils.prims as prim_utils
    from pxr import UsdGeom, Gf
    
    camera_path = "/World/envs/env_0/Robot/base/front_cam"
    
    # 1. 로봇의 base(몸통) 아래에 카메라 Prim 생성 (앞쪽을 바라보도록 위치/회전 설정)
    if not prim_utils.is_prim_path_valid(camera_path):
        cam = UsdGeom.Camera.Define(env.unwrapped.scene.stage, camera_path)
        # 로봇 머리 위(X축 0.15m, Z축 0.25m) 위치로 수정. 
        cam.AddTranslateOp().Set(Gf.Vec3d(0.15, 0.0, 0.25))
        # 카메라 렌즈 방향(기본 -Z)을 앞쪽(+X)으로 회전 (Pitch 90도)
        cam.AddRotateXYZOp().Set(Gf.Vec3d(90, 0, 90))
        cam.GetFocalLengthAttr().Set(24.0)

    # 2. ROS2 Bridge (RGB & Depth) OmniGraph 생성
    og.Controller.edit(
        {"graph_path": "/World/ROS2_Camera_Graph", "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnTick", "omni.graph.action.OnPlaybackTick"),
                ("RenderProduct", "omni.isaac.core_nodes.IsaacCreateRenderProduct"),
                ("ROS2CameraRGB", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                ("ROS2CameraDepth", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("RenderProduct.inputs:cameraPrim", camera_path),
                ("RenderProduct.inputs:resolution", [640, 480]),
                
                # RGB 설정
                ("ROS2CameraRGB.inputs:type", "rgb"),
                ("ROS2CameraRGB.inputs:topicName", "/go2_camera/rgb"),
                ("ROS2CameraRGB.inputs:frameId", "go2_front_cam"),
                
                # Depth 설정
                ("ROS2CameraDepth.inputs:type", "depth"),
                ("ROS2CameraDepth.inputs:topicName", "/go2_camera/depth"),
                ("ROS2CameraDepth.inputs:frameId", "go2_front_cam"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnTick.outputs:tick", "RenderProduct.inputs:execIn"),
                ("RenderProduct.outputs:execOut", "ROS2CameraRGB.inputs:execIn"),
                ("RenderProduct.outputs:renderProductPath", "ROS2CameraRGB.inputs:renderProductPath"),
                ("RenderProduct.outputs:execOut", "ROS2CameraDepth.inputs:execIn"),
                ("RenderProduct.outputs:renderProductPath", "ROS2CameraDepth.inputs:renderProductPath"),
            ],
        },
    )
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

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # keyboard -> base_velocity
            cmd = torch.tensor(
                [
                    keyboard_state["forward"],
                    keyboard_state["side"],
                    keyboard_state["yaw"],
                ],
                device=env.unwrapped.device,
                dtype=torch.float32,
            )
            cmd = cmd.unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
            env.unwrapped.command_manager.get_command("base_velocity")[:] = cmd

            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

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
