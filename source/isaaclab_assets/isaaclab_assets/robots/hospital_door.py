# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for interactive Hospital Articulated Door (revolute hinge and latch handle).

Compatible with NVIDIA Isaac Sim 6.2.0 and Isaac Lab.
"""

import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

_DOOR_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../data/props/door")
)
_DOOR_URDF_PATH = os.path.join(_DOOR_DIR, "hospital_door.urdf")
_DOOR_USDA_PATH = os.path.join(_DOOR_DIR, "hospital_door.usda")

HOSPITAL_DOOR_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=_DOOR_URDF_PATH,
        fix_base=True,
        merge_fixed_joints=False,
        make_instanceable=False,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.1,
            angular_damping=0.5,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(1.2, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={"door_hinge": 0.0, "door_handle_joint": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "hinge": ImplicitActuatorCfg(
            joint_names_expr=["door_hinge"],
            effort_limit=50.0,
            velocity_limit=2.5,
            stiffness=0.5,  # slight hydraulic closer effect
            damping=4.0,    # smooth damping
        ),
        "handle": ImplicitActuatorCfg(
            joint_names_expr=["door_handle_joint"],
            effort_limit=10.0,
            velocity_limit=5.0,
            stiffness=15.0, # spring return for handle
            damping=1.0,
        ),
    },
)
"""Configuration for interactive articulated Hospital Door."""
