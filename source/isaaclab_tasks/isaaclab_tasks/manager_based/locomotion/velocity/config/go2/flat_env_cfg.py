# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab.assets import AssetBaseCfg
import isaaclab.sim as sim_utils

from .rough_env_cfg import UnitreeGo2RoughEnvCfg


@configclass
class UnitreeGo2FlatEnvCfg(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        # 1. 부모 클래스(RoughEnv)의 설정 먼저 불러오기
        super().__post_init__()

        # --- [새로운 평지 학습 전략: 회전 및 안정성 극대화] ---
        
        # 회전(Yaw) 성능 강화: 회전 명령 추종 가중치 상향
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        
        # 몸체 수평 유지 강화: 넘어지지 않도록 벌점 대폭 강화
        self.rewards.flat_orientation_l2.weight = -5.0
        
        # 관절 동작의 부드러움: 경련 방지를 위해 가속도 벌점 강화
        self.rewards.dof_acc_l2.weight = -5.0e-7
        
        # 발 체공 시간 최적화: 너무 높게 뛰지 않고 안정적으로 걷게 함
        self.rewards.feet_air_time.weight = 0.5

        # --- [명령 설정: 더 넓은 범위의 조작 학습] ---
        if hasattr(self.commands, "base_velocity"):
            # 전진/후진 및 좌우 이동 범위 (-1.0 ~ 1.0 m/s)
            self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
            self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
            # 회전(Yaw) 범위 대폭 확장: 제자리에서 아주 빠르게 돌 수 있게 함
            self.commands.base_velocity.ranges.ang_vel_z = (-2.0, 2.0) 
            self.commands.base_velocity.heading_command = False

        # --- [환경 설정: 순수 평지(Flat Plane)] ---
        # 3.0.0 (Isaac Lab 2.3.2) 공식 템플릿 규격에 맞추어 TerrainImporterCfg를 유지하며 평지로 설정
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        
        # 가우시안 메쉬 등 불필요한 에셋 삭제
        if hasattr(self.scene, "custom_environment"):
            self.scene.custom_environment = None

        # 로봇 시작 위치 (원점)
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.4) 

        # 지형 스캔 끄기 (평지 모델은 눈이 필요 없음)
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None


@configclass
class UnitreeGo2FlatEnvCfg_PLAY(UnitreeGo2FlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        # 테스트용 환경 설정 (1개 로봇)
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        
        # 테스트 시 불필요한 노이즈 및 랜덤 충격 제거
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # [수동 리스폰 구현]
        self.episode_length_s = 10000.0 
        
        from isaaclab.utils import configclass
        @configclass
        class EmptyTerminationsCfg:
            pass
            
        self.terminations = EmptyTerminationsCfg()
