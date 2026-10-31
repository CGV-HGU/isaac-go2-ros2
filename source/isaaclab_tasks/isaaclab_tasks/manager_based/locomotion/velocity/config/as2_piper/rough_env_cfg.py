# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.as2_piper import AS2_PIPER_CFG
from isaaclab_assets.robots.as2_sensors import attach_as2_sensors_to_scene_cfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from . import mdp as as2_piper_mdp

LEG_JOINT_NAMES = [
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
]
ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "joint8"]
ORDERED_ALL_20_JOINTS = LEG_JOINT_NAMES + ARM_JOINT_NAMES


@configclass
class UnitreeAS2PiperRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Use Unitree AS2 + AgileX PiPER mobile manipulator
        self.scene.robot = AS2_PIPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

        # Terrain scaling for AS2 (industrial quadruped, 95 Nm peak torque)
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.03, 0.12)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.08)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.015

        # Action scaling & deterministic 20-joint ordering (matches sim2real_bridge_node.py 64-D obs)
        self.actions.joint_pos.joint_names = LEG_JOINT_NAMES
        self.actions.joint_pos.preserve_order = True
        self.actions.joint_pos.scale = 0.25
        self.observations.policy.joint_pos.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=ORDERED_ALL_20_JOINTS, preserve_order=True)
        }
        self.observations.policy.joint_vel.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=ORDERED_ALL_20_JOINTS, preserve_order=True)
        }

        # Rewards (scope joint torque/acc penalties to the 12 leg joints driven by RL)
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ".*_thigh|link[1-8]"
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=LEG_JOINT_NAMES, preserve_order=True
        )
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=LEG_JOINT_NAMES, preserve_order=True
        )

        # Events
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        # Synchronize PiPER arm targets to tucked posture on reset & swing arm periodically during training
        self.events.reset_arm_targets = EventTerm(
            func=as2_piper_mdp.reset_piper_arm_targets,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES, preserve_order=True)},
        )
        self.events.randomize_arm_targets = EventTerm(
            func=as2_piper_mdp.randomize_piper_arm_targets,
            mode="interval",
            interval_range_s=(3.0, 6.0),
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # Terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base|link[1-3]"

        # Optional RGB + Depth + 3D LiDAR attachment when AS2_ENABLE_SENSORS=1 (kept off by default for 4096-env RL)
        if os.getenv("AS2_ENABLE_SENSORS", "0") == "1":
            attach_as2_sensors_to_scene_cfg(self.scene)


@configclass
class UnitreeAS2PiperRoughEnvCfg_PLAY(UnitreeAS2PiperRoughEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # Disable random arm swings during teleop / play so keyboard & auto-door control arm cleanly
        self.events.randomize_arm_targets = None
        # Disable heading overwrite & command resampling so WASD + Q/E teleop works directly
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
