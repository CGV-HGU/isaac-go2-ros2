# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom MDP events for Unitree AS2 + AgileX PiPER loco-manipulation training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_piper_arm_targets(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["joint[1-8]"]),
) -> None:
    """Synchronize uncommanded PiPER arm & gripper position/velocity targets with default posture on reset.

    Because the locomotion policy only outputs actions for the 12 leg joints, Isaac Lab's internal
    ``joint_pos_target`` buffer would otherwise default to ``0.0`` for ``joint1..joint8`` and slam
    ``joint2``/``joint3`` into their mechanical stops on episode step 1.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids_tensor = torch.arange(env.scene.num_envs, device=asset.device)
    else:
        env_ids_tensor = env_ids

    joint_ids = asset_cfg.joint_ids
    if joint_ids == slice(None):
        default_pos = asset.data.default_joint_pos[env_ids_tensor].clone()
        default_vel = asset.data.default_joint_vel[env_ids_tensor].clone()
    else:
        default_pos = asset.data.default_joint_pos[env_ids_tensor[:, None], joint_ids].clone()
        default_vel = asset.data.default_joint_vel[env_ids_tensor[:, None], joint_ids].clone()

    asset.set_joint_position_target(default_pos, joint_ids=joint_ids, env_ids=env_ids_tensor)
    asset.set_joint_velocity_target(default_vel, joint_ids=joint_ids, env_ids=env_ids_tensor)


def randomize_piper_arm_targets(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    joint_ranges: dict[str, tuple[float, float]] | None = None,
    gripper_open_range: tuple[float, float] = (0.0, 0.03),
    tuck_probability: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Sample reachable PiPER arm target postures during training for whole-body CoM disturbance adaptation.

    Trains the 12-DOF quadruped locomotion policy to maintain balance while a human operator or
    autonomous door-opening controller moves the 6-DOF arm and symmetric 2-finger gripper.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids_tensor = torch.arange(env.scene.num_envs, device=asset.device)
    else:
        env_ids_tensor = env_ids

    num_envs = len(env_ids_tensor)
    if num_envs == 0:
        return

    if joint_ranges is None:
        joint_ranges = {
            "joint1": (-1.1, 1.1),
            "joint2": (0.2, 1.35),
            "joint3": (-1.95, -0.4),
            "joint4": (-0.9, 0.9),
            "joint5": (-0.5, 0.75),
            "joint6": (-1.0, 1.0),
        }

    arm_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "joint8"]
    arm_ids, resolved_names = asset.find_joints(arm_names, preserve_order=True)
    if len(arm_ids) != 8:
        return

    targets = asset.data.default_joint_pos[env_ids_tensor[:, None], arm_ids].clone()

    for idx, j_name in enumerate(resolved_names[:6]):
        low, high = joint_ranges[j_name]
        targets[:, idx] = math_utils.sample_uniform(low, high, (num_envs,), device=asset.device)

    # Symmetric gripper opening: joint7 in [0, +0.035], joint8 in [-0.035, 0]
    grip = math_utils.sample_uniform(
        gripper_open_range[0], gripper_open_range[1], (num_envs,), device=asset.device
    )
    targets[:, 6] = grip
    targets[:, 7] = -grip

    # Keep a fraction of environments in the nominal tucked posture so standing/tucked gait stays crisp
    if tuck_probability > 0.0:
        tuck_mask = torch.rand(num_envs, device=asset.device) < tuck_probability
        if torch.any(tuck_mask):
            default_tuck = asset.data.default_joint_pos[env_ids_tensor[tuck_mask][:, None], arm_ids]
            targets[tuck_mask] = default_tuck

    # Clamp to soft joint limits
    soft_limits = asset.data.soft_joint_pos_limits[env_ids_tensor[:, None], arm_ids]
    targets = torch.clamp(targets, soft_limits[..., 0], soft_limits[..., 1])

    asset.set_joint_position_target(targets, joint_ids=arm_ids, env_ids=env_ids_tensor)
