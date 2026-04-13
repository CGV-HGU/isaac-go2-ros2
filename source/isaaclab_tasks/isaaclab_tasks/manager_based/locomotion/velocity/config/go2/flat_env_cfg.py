# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rough_env_cfg import UnitreeGo2RoughEnvCfg


@configclass
class UnitreeGo2FlatEnvCfg(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        # 부모 클래스(RoughEnv)의 설정 먼저 불러오기
        super().__post_init__()

        # [수정 포인트 1] 회전(Yaw) 추종 보상 가중치 추가/강화
        # 이 보상이 있어야 Q, E를 눌렀을 때 로봇이 회전하려고 노력합니다.
        if hasattr(self.rewards, "track_ang_vel_yaw_exp"):
            self.rewards.track_ang_vel_yaw_exp.weight = 1.0  # 0.5 ~ 1.0 사이 추천
        
        # 기존에 있던 평지용 보상 설정
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 0.25

        # [수정 포인트 2] 명령(Command) 범위 확인 및 설정
        # 학습 중에 AI에게 "이 정도 범위까지 회전해봐"라고 알려주는 설정입니다.
        if hasattr(self.commands, "base_velocity"):
            # ang_vel_z 가 회전(Yaw) 범위입니다.
            self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
            # 연속적인 제자리 회전을 위해 절대 방향(Heading) 제어 끄기
            self.commands.base_velocity.heading_command = False

        # 지형 및 관측치 설정 (기존과 동일)
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None


class UnitreeGo2FlatEnvCfg_PLAY(UnitreeGo2FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
