# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab.assets import AssetBaseCfg
import isaaclab.sim as sim_utils

import isaaclab.envs.mdp as mdp
from isaaclab.envs.mdp import RewardTermCfg
from .rough_env_cfg import UnitreeGo2RoughEnvCfg


@configclass
class UnitreeGo2FlatEnvCfg(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        # 1. 부모 클래스(RoughEnv)의 설정 먼저 불러오기
        super().__post_init__()

        # --- [보상 설정] ---
        if hasattr(self.rewards, "track_ang_vel_yaw_exp"):
            self.rewards.track_ang_vel_yaw_exp.weight = 1.0
        
        # 3.0에서는 flat_orientation_l2 등이 이미 부모에 RewardTermCfg로 정의되어 있음
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.base_height_l2.weight = -15.0
        self.rewards.feet_air_time.weight = 0.5

        # --- [명령 설정] ---
        if hasattr(self.commands, "base_velocity"):
            self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
            self.commands.base_velocity.heading_command = False

        # --- [환경 설정: 기본 바닥] ---
        self.scene.terrain = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(),
        )

        # --- [로봇 시작 위치 설정] ---
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.4) 

        # 지형 스캔 관련 불필요한 기능 끄기 (평지이므로)
        self.scene.height_scanner = None
        if hasattr(self.observations.policy, "height_scan"):
            self.observations.policy.height_scan = None
        if hasattr(self.curriculum, "terrain_levels"):
            self.curriculum.terrain_levels = None


@configclass
class UnitreeGo2FlatEnvCfg_PLAY(UnitreeGo2FlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        # 테스트용 환경 설정
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        
        # 테스트 시 불필요한 노이즈 및 랜덤 충격 제거
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
