# Copyright (c) 2022-2026, Computer Graphics and Vision Lab (CGV).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ROS 2 Humble perception and Sim-to-Real bringup package for Unitree AS2 + AgileX PiPER."""

from .sim2real_bridge_node import (
    AS2_DEFAULT_LEG_POS_12,
    AS2_LEG_JOINT_NAMES,
    PIPER_ARM_JOINT_NAMES,
    PIPER_DEFAULT_ARM_POS_8,
    actions_to_leg_joint_targets,
    build_locomotion_observation,
    compute_projected_gravity_wxyz,
    load_door_manipulation_module,
    propagate_latched_handle_in_base,
    quat_rotate_inverse_wxyz,
)
from .vision_door_detector import (
    DEFAULT_CAM_POS_BASE,
    DEFAULT_LIDAR_POS_BASE,
    DoorDetectionResult,
    LidarDoorPlaneEstimate,
    backproject_pixel_to_base_frame,
    decode_pointcloud2_xyz,
    decode_ros_image_msg,
    detect_door_and_handle_rgbd_lidar,
    estimate_door_plane_from_lidar_scan,
    extract_door_plane_from_pointcloud_xyz,
    fit_door_plane_points_xy,
    normalize_depth_to_meters,
)

__all__ = [
    "AS2_DEFAULT_LEG_POS_12",
    "AS2_LEG_JOINT_NAMES",
    "PIPER_ARM_JOINT_NAMES",
    "PIPER_DEFAULT_ARM_POS_8",
    "actions_to_leg_joint_targets",
    "build_locomotion_observation",
    "compute_projected_gravity_wxyz",
    "load_door_manipulation_module",
    "propagate_latched_handle_in_base",
    "quat_rotate_inverse_wxyz",
    "DEFAULT_CAM_POS_BASE",
    "DEFAULT_LIDAR_POS_BASE",
    "DoorDetectionResult",
    "LidarDoorPlaneEstimate",
    "backproject_pixel_to_base_frame",
    "decode_pointcloud2_xyz",
    "decode_ros_image_msg",
    "detect_door_and_handle_rgbd_lidar",
    "estimate_door_plane_from_lidar_scan",
    "extract_door_plane_from_pointcloud_xyz",
    "fit_door_plane_points_xy",
    "normalize_depth_to_meters",
]
