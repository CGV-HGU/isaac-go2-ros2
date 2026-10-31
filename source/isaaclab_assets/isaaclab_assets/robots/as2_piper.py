# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Unitree AS2 quadruped equipped with AgileX PiPER robotic arm.

The following configurations are available:

* :obj:`UNITREE_AS2_CFG`: Unitree AS2 industrial quadruped robot (95 N.m torque)
* :obj:`AS2_PIPER_CFG`: Unitree AS2 quadruped with AgileX PiPER 6-DOF arm and 2-finger gripper
* :obj:`PIPER_ARM_CFG`: Standalone AgileX PiPER robotic arm

Reference:
- Unitree AS2: https://www.unitree.com (Supplied by ROAS)
- AgileX PiPER: https://github.com/agilexrobotics/piper_ros
"""

import logging
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

logger = logging.getLogger(__name__)

_REPO_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../../..",
    )
)
_PIPER_STANDALONE_USD_PATH = os.path.join(
    _REPO_ROOT, "robot_arms", "piper", "piper_isaac_sim", "USD", "piper_v1.usd"
)

##
# AgileX PiPER Standalone Configuration
##

PIPER_ARM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_PIPER_STANDALONE_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=True,
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "joint1": 0.0,
            "joint2": 0.5,
            "joint3": -1.0,
            "joint4": 0.0,
            "joint5": 0.5,
            "joint6": 0.0,
            "joint7": 0.0,
            "joint8": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "arm_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-3]"],
            effort_limit_sim=100.0,
            velocity_limit_sim=5.0,
            stiffness=400.0,
            damping=40.0,
        ),
        "arm_wrist": ImplicitActuatorCfg(
            joint_names_expr=["joint[4-6]"],
            effort_limit_sim=100.0,
            velocity_limit_sim=5.0,
            stiffness=300.0,
            damping=30.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["joint[7-8]"],
            effort_limit_sim=20.0,
            velocity_limit_sim=1.0,
            stiffness=1000.0,
            damping=50.0,
        ),
    },
)
"""Configuration for standalone AgileX PiPER 6-DOF robotic arm (fixed base)."""


##
# Unitree AS2 Base Configuration (Industrial Quadruped, ~18.5kg, 95 N.m torque)
##

_CUSTOM_AS2_USD = os.getenv("UNITREE_AS2_USD_PATH")
if _CUSTOM_AS2_USD:
    UNITREE_AS2_USD_PATH = _CUSTOM_AS2_USD
else:
    UNITREE_AS2_USD_PATH = f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd"
    logger.warning(
        "UNITREE_AS2_USD_PATH is not set; UNITREE_AS2_CFG will use Nucleus Go2 USD geometry "
        "with AS2 95 N.m actuator parameters. Set UNITREE_AS2_USD_PATH when official AS2 USD is available."
    )

UNITREE_AS2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=UNITREE_AS2_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=95.0,  # Unitree AS2 Industrial peak torque (95 N.m)
            saturation_effort=95.0,
            velocity_limit=30.0,
            stiffness=45.0,
            damping=1.5,
            friction=0.0,
        ),
    },
)
"""Configuration of Unitree AS2 industrial quadruped robot."""


##
# Unitree AS2 + AgileX PiPER Composite Mobile Manipulator Configuration
##

_AS2_PIPER_DESCRIPTIONS_DIR = os.path.join(_REPO_ROOT, "robot_arms", "piper", "descriptions")
_AS2_PIPER_URDF_PATH = os.path.join(_AS2_PIPER_DESCRIPTIONS_DIR, "as2_piper.urdf")
# Only load a pre-composed USD if explicitly requested via AS2_PIPER_USD_PATH and valid on disk.
_EXPLICIT_COMPOSITE_USD = os.getenv("AS2_PIPER_USD_PATH", "")
AS2_PIPER_USD_PATH = _EXPLICIT_COMPOSITE_USD if (_EXPLICIT_COMPOSITE_USD and os.path.isfile(_EXPLICIT_COMPOSITE_USD)) else ""

AS2_PIPER_CFG = ArticulationCfg(
    spawn=(
        sim_utils.UsdFileCfg(
            usd_path=AS2_PIPER_USD_PATH,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        )
        if AS2_PIPER_USD_PATH
        else sim_utils.UrdfFileCfg(
            asset_path=_AS2_PIPER_URDF_PATH,
            fix_base=False,
            # Foot joints carry dont_collapse="true" in URDF, so feet remain distinct bodies
            # while zero-mass helper links are safely merged into parent rigid bodies.
            merge_fixed_joints=True,
            make_instanceable=True,
            activate_contact_sensors=True,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        )
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            # Unitree AS2 legs in stable standing posture
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
            # AgileX PiPER arm in tucked/ready posture
            "joint1": 0.0,
            "joint2": 0.6,
            "joint3": -1.2,
            "joint4": 0.0,
            "joint5": 0.6,
            "joint6": 0.0,
            # Gripper fingers (joint7: [0, 0.035], joint8: [-0.035, 0])
            "joint7": 0.0,
            "joint8": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=95.0,  # AS2 High-torque leg motors
            saturation_effort=95.0,
            velocity_limit=30.0,
            stiffness=45.0,
            damping=1.5,
            friction=0.0,
        ),
        "arm_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-3]"],
            effort_limit_sim=100.0,
            velocity_limit_sim=5.0,
            stiffness=400.0,
            damping=40.0,
        ),
        "arm_wrist": ImplicitActuatorCfg(
            joint_names_expr=["joint[4-6]"],
            effort_limit_sim=100.0,
            velocity_limit_sim=5.0,
            stiffness=300.0,
            damping=30.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["joint[7-8]"],
            effort_limit_sim=20.0,
            velocity_limit_sim=1.0,
            stiffness=1000.0,
            damping=50.0,
        ),
    },
)
"""Configuration of Unitree AS2 quadruped with AgileX PiPER 6-DOF arm."""
