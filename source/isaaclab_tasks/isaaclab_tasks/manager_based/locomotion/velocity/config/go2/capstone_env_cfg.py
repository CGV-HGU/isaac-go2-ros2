# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim import UsdFileCfg
import isaaclab.sim as sim_utils

from .rough_env_cfg import UnitreeGo2RoughEnvCfg
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG


@configclass
class UnitreeGo2CapstoneEnvCfg(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        # 1. 험지 주행 클래스(RoughEnv)의 설정 먼저 불러오기 (512 신경망 호환)
        super().__post_init__()

        # --- [보상 설정] ---
        # 어제 학습한 뇌(May 26)에서 3가지 고급 옵션을 "비활성화" 하여 테스트
        self.rewards.feet_slide = None
        self.rewards.track_lin_vel_xy_yaw_frame_exp = None
        self.rewards.stand_still_joint_deviation_l1 = None
        
        # 기본 보상 가중치 조정
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 0.25

        # --- [환경 설정: 기본 바닥 삭제 및 메쉬 전용 환경] ---
        
        # 2. 기존 검은색 격자 바닥(Plane) 생성을 완전히 제거
        self.scene.terrain = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=None,
        )
        
        # 3. 캡스톤 맵 (가우시안 복도 등) 메쉬를 유일한 지형으로 등록
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

        # 4. 로봇 시작 위치 설정 (공중 5cm 위)
        self.scene.robot.init_state.pos = (-1.0, 0.0, -0.95) 

        # 5. 지형 스캔 센서 활성화 (May 26 험지 모델은 235개 데이터를 꼭 필요로 함)
        # 단, 커리큘럼은 꺼서 난이도가 변하지 않게 함
        self.curriculum.terrain_levels = None
        # 레이저 스캐너가 캡스톤 메쉬를 쏘도록 설정
        self.scene.height_scanner.mesh_prim_paths = ["/World/fused_scene"]


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

