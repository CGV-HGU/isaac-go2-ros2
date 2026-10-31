# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from .rough_env_cfg import UnitreeAS2RoughEnvCfg


@configclass
class UnitreeAS2FlatEnvCfg(UnitreeAS2RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Rewards
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 0.25

        # Flat plane terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # Disable height scan for flat
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None


@configclass
class UnitreeAS2FlatEnvCfg_PLAY(UnitreeAS2FlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # Disable heading overwrite & command resampling so WASD + Q/E teleop works directly
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
