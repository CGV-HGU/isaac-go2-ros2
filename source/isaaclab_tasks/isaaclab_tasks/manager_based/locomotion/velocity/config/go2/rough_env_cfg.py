# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp.rewards as custom_rewards
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip


@configclass
class UnitreeGo2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        
        # --- [지형 난이도 상향] 험지와 계단을 더 잘 극복하도록 학습 ---
        # 박스(단차)와 거친 지형의 폭을 넓혀 더 다이내믹한 지형에서 구르며 배우게 합니다.
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.05, 0.15) # 단차 상향
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.02, 0.10) # 험지 상향
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.02

        # --- [유연한 관절] 로봇 다리가 더 크고 유연하게 움직이도록 ---
        # Action Scale을 키워서, AI가 관절을 더 넓은 반경으로 힘차게 뻗을 수 있게 합니다.
        self.actions.joint_pos.scale = 0.50 # 기존 0.25에서 두 배 상향 (민첩성 획득)

        # event
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None

        # --- [보상 튜닝] 험지 안정성 및 보행 최적화 ---
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.05 # 발을 더 높이 들고 오래 체공하게 유도 (계단 극복에 필수)
        
        # 험지에서 미끄러짐 방지를 위해 속도 추종 보상 강화 (기존 방식 삭제 후 고급 방식으로 대체)
        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.track_ang_vel_z_exp.weight = 2.0 
        
        # 몸체가 과도하게 흔들리는 것을 방지 (험지 안정성)
        self.rewards.dof_torques_l2.weight = -0.0001 # 토크 페널티를 살짝 줄여서 힘을 강하게 쓰도록 허용
        self.rewards.dof_acc_l2.weight = -1.5e-7 # 가속도 페널티 완화 (민첩한 움직임 허용)

        # [NEW] 1. 발 끌림 방지 (미끄러짐 및 헛디딤 방지)
        self.rewards.feet_slide = RewardTermCfg(
            func=custom_rewards.feet_slide,
            weight=-0.25,
            params={"sensor_cfg": self.scene.contact_forces, "asset_cfg": self.scene.robot}
        )
        
        # [NEW] 2. 중력 보정 속도 추종 (경사로나 계단에서도 똑바로 걷기)
        self.rewards.track_lin_vel_xy_yaw_frame_exp = RewardTermCfg(
            func=custom_rewards.track_lin_vel_xy_yaw_frame_exp,
            weight=2.0,
            params={"command_name": "base_velocity", "std": math.sqrt(0.25), "asset_cfg": self.scene.robot}
        )
        
        # [NEW] 3. 제자리 멈춤 안정성 (명령이 없을 때 덜덜 떨지 않고 차렷 자세 유지)
        self.rewards.stand_still_joint_deviation_l1 = RewardTermCfg(
            func=custom_rewards.stand_still_joint_deviation_l1,
            weight=-0.5,
            params={"command_name": "base_velocity", "command_threshold": 0.1, "asset_cfg": self.scene.robot}
        )

        # --- [명령 설정] 회전 속도 학습 범위 확장 ---
        if hasattr(self.commands, "base_velocity"):
            self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5) # 회전 학습 범위를 -1.5 ~ 1.5 rad/s로 확장
            self.commands.base_velocity.heading_command = False

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"


@configclass
class UnitreeGo2RoughEnvCfg_PLAY(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
