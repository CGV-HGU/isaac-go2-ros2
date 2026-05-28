# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim import UsdFileCfg
import isaaclab.sim as sim_utils

from .flat_env_cfg import UnitreeGo2FlatEnvCfg


@configclass
class UnitreeGo2CapstoneEnvCfg(UnitreeGo2FlatEnvCfg):
    def __post_init__(self):
        # 1. 방금 우리가 학습시킨 "완벽한 평면 환경(UnitreeGo2FlatEnvCfg)" 설정을 그대로 물려받습니다.
        # 이렇게 하면 뇌(Model)와 몸(Config)이 100% 일치하여 거미처럼 걷는 현상이 사라집니다.
        super().__post_init__()

        # --- [오직 맵 설정만 추가] ---
        # 기존 평면 바닥을 끄고 캡스톤 메쉬만 로드합니다.
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

        # 로봇 시작 위치 (공중 5cm 위)
        self.scene.robot.init_state.pos = (-1.0, 0.0, -0.95) 

        # 험지용 센서는 평지 모델에서 필요 없으므로 부모 클래스 설정을 따라 자동으로 꺼진 상태를 유지합니다.


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
