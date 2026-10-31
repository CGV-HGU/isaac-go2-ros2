# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for interactive Hospital Articulated Doors across multiple door mechanisms.

Supported door types (`DOOR_TYPES`):
* ``lever_push``: Standard inward-opening hospital door with spring lever handle.
* ``lever_pull``: Outward-opening hospital door requiring pull-back coordination.
* ``sliding``: Horizontal rail-mounted ICU/ward sliding door with vertical bar handle.
* ``push_bar``: Latchless push-bar / panic-bar swing door.

Compatible with NVIDIA Isaac Sim 6.2.0 / 5.x and Isaac Lab.
"""

import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

_DOOR_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../data/props/door")
)
_DOOR_URDF_PUSH = os.path.join(_DOOR_DIR, "hospital_door.urdf")
_DOOR_URDF_PULL = os.path.join(_DOOR_DIR, "hospital_door_pull.urdf")
_DOOR_URDF_SLIDING = os.path.join(_DOOR_DIR, "hospital_door_sliding.urdf")

DOOR_TYPES = ("lever_push", "lever_pull", "sliding", "push_bar")


def get_hospital_door_cfg(door_type: str = "lever_push") -> ArticulationCfg:
    """Return an :class:`ArticulationCfg` tailored to the requested ``door_type``."""
    if door_type not in DOOR_TYPES:
        raise ValueError(f"Unsupported door_type '{door_type}'. Expected one of {DOOR_TYPES}.")

    if door_type == "lever_pull":
        urdf_path = _DOOR_URDF_PULL
    elif door_type == "sliding":
        urdf_path = _DOOR_URDF_SLIDING
    else:
        # Both 'lever_push' and 'push_bar' share the push-swing door structure
        urdf_path = _DOOR_URDF_PUSH

    hinge_vel_limit = 1.2 if door_type == "sliding" else 2.5
    hinge_stiffness = 30.0 if door_type == "sliding" else 25.0
    hinge_damping = 8.0 if door_type == "sliding" else 6.0

    return ArticulationCfg(
        spawn=sim_utils.UrdfFileCfg(
            asset_path=urdf_path,
            fix_base=True,
            merge_fixed_joints=True,
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
            pos=(1.1, -0.45, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={"door_hinge": 0.0, "door_handle_joint": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "hinge": ImplicitActuatorCfg(
                joint_names_expr=["door_hinge"],
                effort_limit_sim=80.0,
                velocity_limit_sim=hinge_vel_limit,
                stiffness=hinge_stiffness,
                damping=hinge_damping,
            ),
            "handle": ImplicitActuatorCfg(
                joint_names_expr=["door_handle_joint"],
                effort_limit_sim=15.0,
                velocity_limit_sim=5.0,
                stiffness=15.0,
                damping=1.0,
            ),
        },
    )


HOSPITAL_DOOR_CFG = get_hospital_door_cfg("lever_push")
"""Default configuration for interactive inward-pushing Hospital Door."""
