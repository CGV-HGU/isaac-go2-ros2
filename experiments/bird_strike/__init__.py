# Copyright (c) 2022-2026, Computer Graphics and Vision Lab (CGV).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
AgileX PiPER 6-DOF Robot Arm Moving-Target Experiment Rig.
Part of the Computer Graphics and Vision Lab (CGV) test suite.
"""

from .piper_teach_playback import (
    FIELDNAMES,
    GRIPPER_BALL_GRASP_UM,
    GRIPPER_MAX_UM,
    GRIPPER_MIN_UM,
    PIPER_JOINT_LIMITS_DEG,
    MockPiperInterface,
    compute_trajectory_stats,
    generate_t1_step_trajectory,
    generate_t2_sweep_trajectory,
    load_trajectory_csv,
    resolve_piper_interface,
    save_trajectory_csv,
    smooth_s_curve,
)

__all__ = [
    "FIELDNAMES",
    "GRIPPER_BALL_GRASP_UM",
    "GRIPPER_MAX_UM",
    "GRIPPER_MIN_UM",
    "PIPER_JOINT_LIMITS_DEG",
    "MockPiperInterface",
    "compute_trajectory_stats",
    "generate_t1_step_trajectory",
    "generate_t2_sweep_trajectory",
    "load_trajectory_csv",
    "resolve_piper_interface",
    "save_trajectory_csv",
    "smooth_s_curve",
]
