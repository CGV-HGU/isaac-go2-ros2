# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim import UsdFileCfg
import isaaclab.sim as sim_utils

from .rough_env_cfg import UnitreeGo2RoughEnvCfg


@configclass
class UnitreeGo2FlatEnvCfg(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        # 1. 부모 클래스(RoughEnv)의 설정 먼저 불러오기
        super().__post_init__()

        # --- [보상 설정] 키보드 주행 및 회전(QE) 최적화 ---
        if hasattr(self.rewards, "track_ang_vel_yaw_exp"):
            self.rewards.track_ang_vel_yaw_exp.weight = 1.0
        
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 0.25

        # --- [명령 설정] 회전 범위 확장 ---
        if hasattr(self.commands, "base_velocity"):
            self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
            self.commands.base_velocity.heading_command = False

        # --- [환경 설정: 기본 바닥 삭제 및 메쉬 전용 환경] ---
        
        # 2. 기존 검은색 격자 바닥(Plane) 생성을 완전히 제거
        self.scene.terrain = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=None,  # 이 설정으로 인해 기본 바닥이 깔리지 않습니다.
        )
        
        # 3. 가우시안 복도 메쉬를 유일한 지형으로 등록
        self.scene.custom_environment = AssetBaseCfg(
            prim_path="/World/fused_scene",
            spawn=UsdFileCfg(
                usd_path="/home/hayoung/workspaces/Go2.usd",
                scale=(1.0, 1.0, 1.0),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0), # Nova_Carter가 있던 월드 중심 좌표
                rot=(1.0, 0.0, 0.0, 0.0)
            ),
        )

        # --- [로봇 시작 위치 설정] ---
        # 복도 메쉬의 원점(0,0,0)에서 로봇이 안전하게 착지하도록 Z축만 0.4m 설정
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.4) 

        # 지형 스캔 관련 불필요한 기능 끄기
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
