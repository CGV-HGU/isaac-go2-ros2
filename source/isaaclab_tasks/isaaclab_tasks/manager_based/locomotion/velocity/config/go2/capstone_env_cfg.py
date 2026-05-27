# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim import UsdFileCfg
import isaaclab.sim as sim_utils

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG


@configclass
class UnitreeGo2CapstoneEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # 1. IsaacLab 순정 지형 주행 클래스(LocomotionVelocityRoughEnvCfg) 설정 불러오기
        super().__post_init__()

        # 2. 로봇 설정 (Go2 순정 설정 사용)
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

        # 3. 환경 설정: 기본 바닥 삭제 및 캡스톤 메쉬 전용 환경
        self.scene.terrain = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=None,
        )
        
        self.scene.custom_environment = AssetBaseCfg(
            prim_path="/World/fused_scene",
            spawn=UsdFileCfg(
                usd_path="/home/hayoung/workspaces/05_08_real.usd",
                scale=(1.0, 1.0, 1.0),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0)
            ),
        )

        # 4. 로봇 시작 위치 설정 (공중 5cm 위에서 낙하)
        self.scene.robot.init_state.pos = (-1.0, 0.0, -0.95) 

        # 5. 불필요한 센서 및 커리큘럼 끄기 (4/14 평지 모델 호환)
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None


class UnitreeGo2CapstoneEnvCfg_PLAY(UnitreeGo2CapstoneEnvCfg):
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

