# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
AgileX PiPER 6-DOF Robotic Arm - Ball Grasping & Dynamic Motion Demonstration.
Pre-flight Sim-to-Real verification for object gripping, lifting, and vibration/impact shake motion.

Usage:
    ./isaaclab.sh -p experiments/bird_strike/sim_piper_ball_pick_and_move.py
    ./isaaclab.sh -p experiments/bird_strike/sim_piper_ball_pick_and_move.py --as2_mounted
    ./isaaclab.sh -p experiments/bird_strike/sim_piper_ball_pick_and_move.py --ball_radius 0.025
"""

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="AgileX PiPER Ball Grasping and Shake Motion Demo in Isaac Sim.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of simulation environments.")
parser.add_argument("--as2_mounted", action="store_true", default=False,
                    help="Spawn PiPER arm mounted on Unitree AS2 quadruped (Deprecated for BS rig; use standalone pedestal).")
parser.add_argument("--ball_radius", type=float, default=0.020,
                    help="Radius of target ball in meters (default: 0.020m = 4.0cm diameter).")
parser.add_argument("--ball_mass", type=float, default=0.035,
                    help="Mass of target ball in kg (default: 0.035kg).")
parser.add_argument("--trajectory_csv", type=str, default="",
                    help="Optional path to trajectory CSV (e.g. trajectories/trajectory_t1_step.csv) to replay in Isaac Sim.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Omniverse application
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Simulation imports following Omniverse app launch."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.as2_piper import AS2_PIPER_CFG, PIPER_ARM_CFG


def _get_robot_cfg():
    if args_cli.as2_mounted:
        print("[INFO] Loading Unitree AS2 + AgileX PiPER Mobile Manipulator...")
        return AS2_PIPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    else:
        print("[INFO] Loading Standalone AgileX PiPER 6-DOF Robotic Arm (Fixed Pedestal)...")
        return PIPER_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class BallGraspSceneCfg(InteractiveSceneCfg):
    """Interactive scene configuration for PiPER Ball Gripping and Motion."""

    # Ground plane
    ground = sim_utils.AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # Lighting
    dome_light = sim_utils.AssetBaseCfg(
        prim_path="/World/defaultDomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.95, 0.95, 0.98)),
    )

    # Robot arm
    robot = _get_robot_cfg()

    # Small Pedestal / Table for placing the ball
    pedestal = sim_utils.AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Pedestal",
        spawn=sim_utils.CylinderCfg(
            radius=0.07,
            height=0.10,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.28, 0.35), roughness=0.4),
        ),
        init_state=sim_utils.AssetBaseCfg.InitialStateCfg(
            pos=(0.32, 0.0, 0.05),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # Target Small Ball (Rigid body sphere to grip)
    ball: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetBall",
        spawn=sim_utils.SphereCfg(
            radius=args_cli.ball_radius,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.1,
                angular_damping=0.1,
                max_linear_velocity=50.0,
                max_angular_velocity=50.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=args_cli.ball_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                restitution=0.3,
                torsional_patch_radius=0.01,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.95, 0.05, 0.05), # High-visibility bright red for vision tracker
                metallic=0.1,
                roughness=0.3,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.32, 0.0, 0.10 + args_cli.ball_radius + 0.002), # On top of pedestal
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


def main():
    """Main simulation loop demonstrating 5-phase Ball Pick & Motion sequence."""
    print("=" * 72)
    print(" [Isaac Sim 6.2 / Isaac Lab 3.0] PiPER Ball Grasping & Shake Motion Demo")
    print(" Computer Graphics and Vision Lab (CGV)")
    print(" Target: Sim-to-Real Pre-Check for AgileX PiPER Robot Arm")
    print("=" * 72)

    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / 60.0,
        render_interval=1,
        device=args_cli.device if args_cli.device else ("cuda:0" if torch.cuda.is_available() else "cpu"),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0.9, 0.8, 0.65], target=[0.25, 0.0, 0.15])

    # Instantiate Scene
    scene_cfg = BallGraspSceneCfg(num_envs=args_cli.num_envs, env_spacing=1.5)
    scene = InteractiveScene(scene_cfg)

    # Reset simulation
    sim.reset()
    robot: Articulation = scene["robot"]
    ball: RigidObject = scene["ball"]

    # Locate joint indices
    arm_joint_names = [f"joint{i}" for i in range(1, 7)]
    gripper_joint_names = ["joint7", "joint8"]
    all_piper_names = arm_joint_names + gripper_joint_names

    arm_indices, _ = robot.find_joints(all_piper_names)
    print(f"[INFO] PiPER Controlled Joint Indices: {arm_indices}")

    # Gripper limits: joint7 in [0.0, 0.035] (open), joint8 in [-0.035, 0.0] (open)
    GRIPPER_OPEN = [0.035, -0.035]
    GRIPPER_CLOSED_BALL = [0.013, -0.013] # Grasps 4.4cm ball securely

    # Keyframe Joint Poses [joint1, joint2, joint3, joint4, joint5, joint6]
    POSE_HOME = [0.0, 0.45, -0.85, 0.0, 0.40, 0.0]
    POSE_HOVER = [0.0, 0.70, -1.15, 0.0, 0.45, 0.0]
    POSE_GRASP = [0.0, 0.88, -1.35, 0.0, 0.47, 0.0]
    POSE_LIFT = [0.0, 0.50, -0.90, 0.0, 0.40, 0.0]

    def interpolate_pose(p_start, p_end, alpha):
        """Smooth cosine interpolation between two poses."""
        s = 0.5 * (1.0 - math.cos(math.pi * min(max(alpha, 0.0), 1.0)))
        return [p_start[i] + s * (p_end[i] - p_start[i]) for i in range(len(p_start))]

    step_count = 0
    phase = 0
    phase_timer = 0.0
    cycle_count = 1

    print("\n[READY] Starting Automatic 5-Phase Ball Manipulation Loop...")
    print("  - Phase 1: Open Gripper & Hover above Ball")
    print("  - Phase 2: Approach & Lower to Ball Surface")
    print("  - Phase 3: Squeeze Gripper & Firmly Grasp Ball")
    print("  - Phase 4: Lift Up & Execute Dynamic Shake/Impact Motion")
    print("  - Phase 5: Lower Down, Release & Repeat Cycle\n")

    current_arm_target = POSE_HOME.copy()
    current_gripper_target = GRIPPER_OPEN.copy()

    # Optional CSV trajectory playback
    csv_trajectory = []
    if args_cli.trajectory_csv:
        if os.path.isfile(args_cli.trajectory_csv):
            import csv
            with open(args_cli.trajectory_csv, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    csv_trajectory.append({
                        "timestamp": float(row["timestamp"]),
                        "arm_rad": [
                            float(row.get("j1_rad", math.radians(float(row["j1_deg"])))),
                            float(row.get("j2_rad", math.radians(float(row["j2_deg"])))),
                            float(row.get("j3_rad", math.radians(float(row["j3_deg"])))),
                            float(row.get("j4_rad", math.radians(float(row["j4_deg"])))),
                            float(row.get("j5_rad", math.radians(float(row["j5_deg"])))),
                            float(row.get("j6_rad", math.radians(float(row["j6_deg"])))),
                        ],
                        "gripper_rad": [0.013, -0.013],
                    })
            print(f"[INFO] Replaying {len(csv_trajectory)} frames from CSV: {args_cli.trajectory_csv}")
        else:
            print(f"[WARNING] Trajectory CSV file not found: {args_cli.trajectory_csv}. Falling back to default 5-phase demo.")

    while simulation_app.is_running():
        sim_time = step_count * sim.get_physics_dt()
        dt = sim.get_physics_dt()
        phase_timer += dt

        if csv_trajectory:
            # CSV Playback Mode (repeating loop)
            total_duration = csv_trajectory[-1]["timestamp"]
            t_loop = sim_time % max(total_duration, 0.01)
            frame_dt = csv_trajectory[1]["timestamp"] - csv_trajectory[0]["timestamp"] if len(csv_trajectory) > 1 else 0.02
            idx = min(int(round(t_loop / max(frame_dt, 1e-4))), len(csv_trajectory) - 1)
            current_arm_target = csv_trajectory[idx]["arm_rad"]
            current_gripper_target = csv_trajectory[idx]["gripper_rad"]
        else:
            # Default State Machine: 5-Phase Grasping & Moving
            if phase == 0:
                # Phase 0: Home -> Hover (2.0s)
                t_ratio = phase_timer / 2.0
                current_arm_target = interpolate_pose(POSE_HOME, POSE_HOVER, t_ratio)
                current_gripper_target = GRIPPER_OPEN
                if phase_timer >= 2.0:
                    phase = 1
                    phase_timer = 0.0
                    print(f"[Cycle {cycle_count}] Phase 1: Descending towards Ball...")

            elif phase == 1:
                # Phase 1: Hover -> Grasp Position (1.5s)
                t_ratio = phase_timer / 1.5
                current_arm_target = interpolate_pose(POSE_HOVER, POSE_GRASP, t_ratio)
                current_gripper_target = GRIPPER_OPEN
                if phase_timer >= 1.5:
                    phase = 2
                    phase_timer = 0.0
                    print(f"[Cycle {cycle_count}] Phase 2: Gripping the Ball...")

            elif phase == 2:
                # Phase 2: Close Gripper around Ball (1.2s)
                t_ratio = phase_timer / 1.2
                current_arm_target = POSE_GRASP
                current_gripper_target = interpolate_pose(GRIPPER_OPEN, GRIPPER_CLOSED_BALL, t_ratio)
                if phase_timer >= 1.2:
                    phase = 3
                    phase_timer = 0.0
                    print(f"[Cycle {cycle_count}] Phase 3: Lifting Ball into air...")

            elif phase == 3:
                # Phase 3: Grasp -> Lift (1.5s)
                t_ratio = phase_timer / 1.5
                current_arm_target = interpolate_pose(POSE_GRASP, POSE_LIFT, t_ratio)
                current_gripper_target = GRIPPER_CLOSED_BALL
                if phase_timer >= 1.5:
                    phase = 4
                    phase_timer = 0.0
                    print(f"[Cycle {cycle_count}] Phase 4: Dynamic Moving & Shake Scene (Camera Jitter / Bird-Strike Simulation)...")

            elif phase == 4:
                # Phase 4: Dynamic Motion with Ball in Gripper (4.5s)
                # Simulating Bird-Strike shock, yaw sweep, wrist tilt and vibration!
                t = phase_timer
                shake_j1 = POSE_LIFT[0] + 0.35 * math.sin(2.5 * t)              # Base Yaw Sweep
                shake_j2 = POSE_LIFT[1] + 0.15 * math.sin(3.0 * t)              # Shoulder Pitch Wave
                shake_j3 = POSE_LIFT[2] + 0.10 * math.cos(3.0 * t)              # Elbow Pitch
                shake_j4 = POSE_LIFT[3] + 0.40 * math.sin(4.0 * t)              # Wrist Roll Shake
                shake_j5 = POSE_LIFT[4] + 0.25 * math.cos(5.0 * t)              # Wrist Pitch Vibration
                shake_j6 = POSE_LIFT[5] + 0.50 * math.sin(6.0 * t)              # Wrist Tip Impact Spin

                current_arm_target = [shake_j1, shake_j2, shake_j3, shake_j4, shake_j5, shake_j6]
                current_gripper_target = GRIPPER_CLOSED_BALL

                if phase_timer >= 4.5:
                    phase = 5
                    phase_timer = 0.0
                    print(f"[Cycle {cycle_count}] Phase 5: Lowering Ball and Releasing...")

            elif phase == 5:
                # Phase 5: Return to Pedestal & Open Gripper (2.0s)
                t_ratio = phase_timer / 2.0
                if t_ratio < 0.6:
                    current_arm_target = interpolate_pose(POSE_LIFT, POSE_GRASP, t_ratio / 0.6)
                    current_gripper_target = GRIPPER_CLOSED_BALL
                else:
                    release_ratio = (t_ratio - 0.6) / 0.4
                    current_arm_target = POSE_GRASP
                    current_gripper_target = interpolate_pose(GRIPPER_CLOSED_BALL, GRIPPER_OPEN, release_ratio)

                if phase_timer >= 2.0:
                    phase = 0
                    phase_timer = 0.0
                    cycle_count += 1
                    print(f"[SUCCESS] Cycle {cycle_count - 1} completed! Resetting to Home...")

        # ---------------------------------------------------------
        # Apply Targets to PiPER Joints
        # ---------------------------------------------------------
        all_targets = current_arm_target + current_gripper_target
        targets_tensor = torch.tensor(all_targets, device=sim.device, dtype=torch.float32)

        # Retrieve current joint position tensor and update PiPER indices
        joint_pos_target = robot.data.joint_pos_target.clone()
        if len(arm_indices) == 8:
            joint_pos_target[:, arm_indices] = targets_tensor
        elif joint_pos_target.shape[1] >= 20: # If mounted on AS2 (20-DOF)
            joint_pos_target[:, 12:20] = targets_tensor

        robot.set_joint_position_target(joint_pos_target)

        # Step physics and scene
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)
        step_count += 1

    print("[INFO] Simulation finished cleanly.")
    simulation_app.close()


if __name__ == "__main__":
    main()
