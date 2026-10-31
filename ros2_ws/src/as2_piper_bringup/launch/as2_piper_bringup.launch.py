# Copyright (c) 2022-2026, CGV Lab (Handong Global University).
# SPDX-License-Identifier: BSD-3-Clause

"""Launch RGB-D + 3D LiDAR door perception and whole-body Sim-to-Real controller for AS2 + PiPER."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("as2_piper_bringup")
    default_params = os.path.join(pkg_share, "config", "as2_piper_params.yaml")

    onnx_policy_arg = DeclareLaunchArgument(
        "onnx_policy_path",
        default_value="",
        description="Optional path to exported/policy.onnx from RSL-RL training.",
    )
    door_type_arg = DeclareLaunchArgument(
        "door_type",
        default_value="lever_push",
        description="Initial door mechanism type (lever_push, lever_pull, sliding, push_bar).",
    )
    enable_piper_can_arg = DeclareLaunchArgument(
        "enable_piper_can",
        default_value="false",
        description="Set true on physical robot to stream 6-DoF + gripper commands over SocketCAN.",
    )

    detector_node = Node(
        package="as2_piper_bringup",
        executable="vision_door_detector",
        name="vision_door_detector",
        output="screen",
        parameters=[default_params],
    )

    bridge_node = Node(
        package="as2_piper_bringup",
        executable="sim2real_bridge_node",
        name="as2_piper_sim2real_bridge",
        output="screen",
        parameters=[
            default_params,
            {
                "onnx_policy_path": LaunchConfiguration("onnx_policy_path"),
                "door_type": LaunchConfiguration("door_type"),
                "enable_piper_can": LaunchConfiguration("enable_piper_can"),
            },
        ],
    )

    return LaunchDescription(
        [
            onnx_policy_arg,
            door_type_arg,
            enable_piper_can_arg,
            detector_node,
            bridge_node,
        ]
    )
