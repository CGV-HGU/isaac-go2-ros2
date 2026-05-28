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
        # 1. 부모 클래스(RoughEnv)의 설정 불러오기
        super().__post_init__()

        # --- [순수 평지 복구] ---
        # 캡스톤 메쉬 등을 모두 제거하고 순수한 무한 평면(Ground Plane)으로 복구합니다.
        # 이렇게 하면 play.sh 실행 시 로봇의 기초 보행 능력을 정확히 진단할 수 있습니다.
        self.scene.terrain = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(),
        )
        
        # 커스텀 환경(메쉬) 로드 제거
        if hasattr(self.scene, "custom_environment"):
            self.scene.custom_environment = None

        # 로봇 시작 위치 (원점 공중 0.4m)
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.4) 

        # 보상 가중치를 4/14 모델 당시의 평지용 표준값으로 고정
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 0.01

        # 지형 스캔 끄기 (Flat 모델은 눈이 필요 없음)
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None


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
        self.events.add_base_mass = None

        # [수동 리스폰 구현]
        self.episode_length_s = 10000.0 
        
        from isaaclab.utils import configclass
        @configclass
        class EmptyTerminationsCfg:
            pass
            
        self.terminations = EmptyTerminationsCfg()
