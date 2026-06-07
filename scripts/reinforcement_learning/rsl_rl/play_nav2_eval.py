# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Go2 Nav2 Optimized Play & Evaluation Script (v3.0).
Final attempt: Simplified ROS2 node creation using hardcoded types and direct graph manipulation.
"""

import argparse
import sys
import os
import time
import csv
import math
import random
import torch
import torch.nn as nn

from isaaclab.app import AppLauncher

# 1. CLI Arguments
sys.path.append(os.path.dirname(__file__))
import cli_args

parser = argparse.ArgumentParser(description="Go2 Optimized Nav2 Play & Evaluation")
parser.add_argument("--eval", action="store_true", default=False, help="Run 100-trial evaluation logic.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Unitree-Go2-v0", help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration.")
parser.add_argument("--real_time", action="store_true", default=False, help="Run in real-time.")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Launch Isaac Sim
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. Isaac Lab & Utility Imports
import gymnasium as gym
import omni.graph.core as og
import omni.kit.app
from pxr import Gf, UsdGeom
import omni.physx
from omni.graph.core._impl.errors import OmniGraphError

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# 💡 [구조 수정] 로그에 찍힌 사용자의 skrl 모델 구조와 완벽히 일치하도록 수정 (128, 128, 128, ELU)
class PurePyTorchPolicy(nn.Module):
    def __init__(self, input_size=48, output_size=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, output_size)
        )
        
    def forward(self, obs):
        if isinstance(obs, dict):
            obs = obs.get("states", obs)
        return self.net(obs)

def setup_ros2_communication(stage, robot_base_path, camera_path):
    """Sets up ROS2 Bridge using OmniGraph with hardcoded, version-specific node types."""
    print(f"[INFO] Setting up ROS2 Bridge for Isaac Sim 5.1.0...")
    
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    
    ros2_bridge_prefix = None
    core_nodes_prefix = None

    # Determine ROS2 Bridge and Core Nodes prefixes and ensure they are enabled
    if ext_manager.is_extension_enabled("isaacsim.ros2.bridge"):
        ros2_bridge_prefix = "isaacsim.ros2.bridge"
        core_nodes_prefix = "isaacsim.core.nodes"
        print(f"[INFO] Using ROS2 Bridge extension: {ros2_bridge_prefix}")
    elif ext_manager.is_extension_enabled("omni.isaac.ros2_bridge"):
        ros2_bridge_prefix = "omni.isaac.ros2_bridge"
        core_nodes_prefix = "omni.isaac.core_nodes"
        print(f"[INFO] Using ROS2 Bridge extension: {ros2_bridge_prefix}")
    else:
        print("[WARNING] ROS2 Bridge extension not found. Attempting to enable...")
        try:
            ext_manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
            ros2_bridge_prefix = "isaacsim.ros2.bridge"
            core_nodes_prefix = "isaacsim.core.nodes"
            print(f"[INFO] Enabled 'isaacsim.ros2.bridge'.")
        except Exception as e_isaacsim:
            print(f"[WARNING] Failed to enable 'isaacsim.ros2.bridge': {e_isaacsim}. Trying 'omni.isaac.ros2_bridge'...")
            try:
                ext_manager.set_extension_enabled_immediate("omni.isaac.ros2_bridge", True)
                ros2_bridge_prefix = "omni.isaac.ros2_bridge"
                core_nodes_prefix = "omni.isaac.core_nodes"
                print(f"[INFO] Enabled 'omni.isaac.ros2_bridge'.")
            except Exception as e_omni:
                print(f"[ERROR] Failed to enable any ROS2 Bridge extension: {e_omni}. Exiting.")
                sys.exit(1)

    if not ext_manager.is_extension_enabled(core_nodes_prefix):
        print(f"[WARNING] Core nodes prefix '{core_nodes_prefix}' not enabled. Trying fallback...")
        core_nodes_prefix = "omni.isaac.core_nodes"
        if not ext_manager.is_extension_enabled(core_nodes_prefix):
            print(f"[ERROR] Core nodes extension '{core_nodes_prefix}' not found. Exiting.")
            sys.exit(1)

    # Wait for extensions and their nodes to be registered
    print("[INFO] Waiting for ROS2 nodes to be registered...")
    for _ in range(60): # Increased delay significantly
        simulation_app.update()

    # 1. Create Camera
    if not stage.GetPrimAtPath(camera_path).IsValid():
        cam = UsdGeom.Camera.Define(stage, camera_path)
        cam.AddTranslateOp().Set(Gf.Vec3d(0.15, 0.0, 0.25))
        cam.AddRotateXYZOp().Set(Gf.Vec3d(0, 0, 0))
        cam.GetFocalLengthAttr().Set(24.0)

    # 2. Create Navigation Goal Prim
    goal_prim_path = "/World/Navigation_Goal"
    if not stage.GetPrimAtPath(goal_prim_path).IsValid():
        UsdGeom.Xform.Define(stage, goal_prim_path)

    # 3. Create OmniGraph
    graph_path = "/World/ROS2_Nav_Graph"
    if stage.GetPrimAtPath(graph_path).IsValid():
        omni.kit.commands.execute("DeletePrims", paths=[graph_path])

    try:
        graph = og.Graph.new(graph_path)
        
        # Define and add nodes with hardcoded, version-specific type strings
        # Use 'isaacsim' prefix if available, fallback to 'omni.isaac' if needed
        
        # OnTick Node
        graph.add_node("OnTick", "omni.graph.action.OnPlaybackTick")

        # ROS2 Nodes (preferring isaacsim.ros2.bridge)
        cmd_vel_type = f"{ros2_bridge_prefix}.ROS2SubscribeTwist" if ros2_bridge_prefix else "omni.isaac.ros2_bridge.ROS2SubscribeTwist"
        goal_pub_type = f"{ros2_bridge_prefix}.ROS2PublishPoseStamped" if ros2_bridge_prefix else "omni.isaac.ros2_bridge.ROS2PublishPoseStamped"
        clock_pub_type = f"{ros2_bridge_prefix}.ROS2PublishClock" if ros2_bridge_prefix else "omni.isaac.ros2_bridge.ROS2PublishClock"
        odom_pub_type = f"{ros2_bridge_prefix}.ROS2PublishOdometry" if ros2_bridge_prefix else "omni.isaac.ros2_bridge.ROS2PublishOdometry"
        tf_world_base_type = f"{ros2_bridge_prefix}.ROS2PublishTransformTree" if ros2_bridge_prefix else "omni.isaac.ros2_bridge.ROS2PublishTransformTree"
        tf_base_cam_type = f"{ros2_bridge_prefix}.ROS2PublishTransformTree" if ros2_bridge_prefix else "omni.isaac.ros2_bridge.ROS2PublishTransformTree"
        cam_rgb_type = f"{ros2_bridge_prefix}.ROS2CameraHelper" if ros2_bridge_prefix else "omni.isaac.ros2_bridge.ROS2CameraHelper"

        # Core Nodes
        compute_odom_type = f"{core_nodes_prefix}.IsaacComputeOdometry"
        render_prod_type = f"{core_nodes_prefix}.IsaacCreateRenderProduct"

        # Add nodes to the graph, handling potential TypeErrors if node types are not found
        try:
            graph.add_node("CmdVelSub", cmd_vel_type)
            graph.add_node("GoalPub", goal_pub_type)
            graph.add_node("ClockPub", clock_pub_type)
            graph.add_node("ComputeOdom", compute_odom_type)
            graph.add_node("OdomPub", odom_pub_type)
            graph.add_node("TF_World_Base", tf_world_base_type)
            graph.add_node("TF_Base_Cam", tf_base_cam_type)
            graph.add_node("RenderProduct", render_prod_type)
            graph.add_node("CamRGB", cam_rgb_type)
        except OmniGraphError as e:
            print(f"[ERROR] OmniGraphError during node addition: {e}")
            print("[ERROR] This might indicate an issue with extension loading or node type registration.")
            raise e

        # Set node values
        graph.set_node_attribute(og.Controller.attribute(f"{graph_path}/CmdVelSub", "inputs:topicName"), "/cmd_vel")
        graph.set_node_attribute(og.Controller.attribute(f"{graph_path}/GoalPub", "inputs:topicName"), "/goal_pose")
        graph.set_node_attribute(og.Controller.attribute(f"{graph_path}/GoalPub", "inputs:frameId"), "map")
        graph.set_node_attribute(og.Controller.attribute(f"{graph_path}/GoalPub", "inputs:targetPrim"), goal_prim_path)
        graph.set_node_attribute(og.Controller.attribute(f"{graph_path}/ComputeOdom", "inputs:chassisPrim"), robot_base_path)
        graph.set_node_attribute(og.Controller.attribute(f"{graph_path}/OdomPub", "inputs:topicName"), "/odom")
        graph.set_node_attribute(og.Controller.attribute(f"{graph_path}/OdomPub", "inputs:odomFrameId"), "World")
        graph.set_node_attribute(og.Controller.attribute(f"{graph_path}/OdomPub", "inputs:chassisFrameId"), "base")
        graph.set_node_attribute(og.Controller.attribute(f"{graph_path}/TF_World_Base", "inputs:parentPrim"), "/World")
        graph.set_node_attribute(og.Controller.attribute(f"{graph_path}/TF_World_Base", "inputs:targetPrims"), [robot_base_path])
        graph.set_node_attribute(og.Controller.attribute(f"{graph_path}/TF_Base_Cam", "inputs:parentPrim"), robot_base_path)
        graph.set_node_attribute(og.Controller.attribute(f"{graph_path}/TF_Base_Cam", "inputs:targetPrims"), [camera_path])
        graph.set_node_attribute(og.Controller.attribute(f"{graph_path}/RenderProduct", "inputs:cameraPrim"), camera_path)
        graph.set_node_attribute(og.Controller.attribute(f"{graph_path}/RenderProduct", "inputs:resolution"), [640, 480])
        graph.set_node_attribute(og.Controller.attribute(f"{graph_path}/CamRGB", "inputs:type"), "rgb")
        graph.set_node_attribute(og.Controller.attribute(f"{graph_path}/CamRGB", "inputs:topicName"), "/go2_camera/rgb")
        graph.set_node_attribute(og.Controller.attribute(f"{graph_path}/CamRGB", "inputs:frameId"), "go2_front_cam")

        # Connect nodes
        graph.add_connection("OnTick.outputs:tick", "CmdVelSub.inputs:execIn")
        graph.add_connection("OnTick.outputs:tick", "GoalPub.inputs:execIn")
        graph.add_connection("OnTick.outputs:tick", "ClockPub.inputs:execIn")
        graph.add_connection("OnTick.outputs:tick", "ComputeOdom.inputs:execIn")
        graph.add_connection("ComputeOdom.outputs:execOut", "OdomPub.inputs:execIn")
        graph.add_connection("ComputeOdom.outputs:position", "OdomPub.inputs:position")
        graph.add_connection("ComputeOdom.outputs:orientation", "OdomPub.inputs:orientation")
        graph.add_connection("ComputeOdom.outputs:linearVelocity", "OdomPub.inputs:linearVelocity")
        graph.add_connection("ComputeOdom.outputs:angularVelocity", "OdomPub.inputs:angularVelocity")
        graph.add_connection("OnTick.outputs:tick", "TF_World_Base.inputs:execIn")
        graph.add_connection("OnTick.outputs:tick", "TF_Base_Cam.inputs:execIn")
        graph.add_connection("OnTick.outputs:tick", "RenderProduct.inputs:execIn")
        graph.add_connection("RenderProduct.outputs:execOut", "CamRGB.inputs:execIn")
        graph.add_connection("RenderProduct.outputs:renderProductPath", "CamRGB.inputs:renderProductPath")

    except Exception as e:
        print(f"[ERROR] Failed to setup ROS2 communication: {e}")
        raise e

    return graph_path

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 1
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    import importlib.metadata as metadata
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    
    print(f"[INFO] Loading skrl-trained model into PurePyTorchPolicy: {resume_path}")
    policy = PurePyTorchPolicy(input_size=48, output_size=12).to(env.unwrapped.device)
    checkpoint = torch.load(resume_path, map_location=env.unwrapped.device)
    
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

    policy.load_state_dict(state_dict, strict=False)
    policy.eval()

    stage = env.unwrapped.scene.stage
    robot_base_path = "/World/envs/env_0/Robot/base"
    camera_path = "/World/envs/env_0/Robot/base/front_cam"
    graph_path = setup_ros2_communication(stage, robot_base_path, camera_path)

    linear_attr = og.Controller.attribute(f"{graph_path}/CmdVelSub.outputs:linearVelocity")
    angular_attr = og.Controller.attribute(f"{graph_path}/CmdVelSub.outputs:angularVelocity")

    TOTAL_EPISODES = 100 if args_cli.eval else 1
    START_X, START_Y = -1.0, 0.0
    GOAL_X, GOAL_Y = 5.0, 0.0
    drawer_path = "/World/fused_scene/Moveable_Objects/Drawer"
    if not stage.GetPrimAtPath(drawer_path).IsValid():
        drawer_path = "/World/Moveable_Objects/Drawer"

    result_file = "./eval_results_optimized.csv"
    if args_cli.eval:
        with open(result_file, 'w', newline='') as f:
            csv.writer(f).writerow(["Episode", "Success", "Steps", "Distance"])

    dt = env.unwrapped.step_dt
    obs = env.get_observations()

    print(f"
================ Start {'Evaluation' if args_cli.eval else 'Play'} ================")

    for ep in range(1, TOTAL_EPISODES + 1):
        if not simulation_app.is_running(): break
        
        if args_cli.eval:
            print(f"🎬 [Trial {ep}/{TOTAL_EPISODES}] Resetting...")
            env.reset()
            with torch.inference_mode():
                root_state = env.unwrapped.scene["robot"].data.default_root_state.clone()
                root_state[:, 0], root_state[:, 1], root_state[:, 2] = START_X, START_Y, 0.42
                target_yaw = math.atan2(GOAL_Y - START_Y, GOAL_X - START_X)
                root_state[:, 3] = math.cos(target_yaw / 2.0)
                root_state[:, 6] = math.sin(target_yaw / 2.0)
                env.unwrapped.scene["robot"].write_root_pose_to_sim(root_state[:, :7])
                env.unwrapped.scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])
                env.unwrapped.scene["robot"].reset()
            
            drawer_prim = stage.GetPrimAtPath(drawer_path)
            if drawer_prim.IsValid():
                x_rand = random.uniform(-1.0, 3.0)
                y_rand = random.uniform(-0.76, 0.76)
                mat = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), random.uniform(0, 360)))
                mat.SetTranslate(Gf.Vec3d(x_rand, y_rand, 0.0))
                UsdGeom.Xformable(drawer_prim).MakeMatrixXform().Set(mat)
                omni.physx.get_physx_interface().update_transform(drawer_path)
            
            goal_prim = stage.GetPrimAtPath("/World/Navigation_Goal")
            UsdGeom.Xformable(goal_prim).ClearXformOpOrder()
            UsdGeom.Xformable(goal_prim).AddTranslateOp().Set(Gf.Vec3d(GOAL_X, GOAL_Y, 0.0))
            
            for _ in range(10): simulation_app.update()
            obs = env.get_observations()

        step_count = 0
        max_steps = 1500 if args_cli.eval else 1000000
        success = False
        
        while step_count < max_steps and simulation_app.is_running():
            start_time = time.time()
            lin_vel = og.Controller.get_attr_value(linear_attr)
            ang_vel = og.Controller.get_attr_value(angular_attr)
            v_x, v_y, v_yaw = (lin_vel[0], lin_vel[1], ang_vel[2]) if lin_vel is not None else (0.0, 0.0, 0.0)

            with torch.inference_mode():
                cmd = torch.tensor([[v_x, v_y, v_yaw]], device=env.unwrapped.device, dtype=torch.float32)
                env.unwrapped.command_manager.get_command("base_velocity")[:] = cmd
                
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                
                current_pos = env.unwrapped.scene["robot"].data.root_pos_w[0]
                dist = math.dist((current_pos[0].item(), current_pos[1].item()), (GOAL_X, GOAL_Y))
                if args_cli.eval and dist < 0.5:
                    success = True
                    print(f"    ✅ Success! Steps: {step_count}, Dist: {dist:.2f}m")
                    break
                
                if dones[0] and args_cli.eval:
                    print(f"    💥 Failed (Collision/Fall)")
                    break

            if args_cli.real_time:
                time.sleep(max(0, dt - (time.time() - start_time)))
            step_count += 1
        
        if args_cli.eval:
            with open(result_file, 'a', newline='') as f:
                csv.writer(f).writerow([ep, success, step_count, round(dist, 2)])

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
