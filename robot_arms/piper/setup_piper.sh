#!/usr/bin/env bash
# ==============================================================================
# Setup script for AgileX PiPER Robotic Arm in isaac-go2-ros2
# Supports: Python SDK, ROS 2 (Humble), and Isaac Sim assets
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================="
echo " [AgileX PiPER] Initializing Robotic Arm Environment for As2"
echo "================================================================="

# 1. Python SDK 설치
echo ">> [1/3] Installing Python CAN and piper_sdk via pip..."
pip install --upgrade python-can piper_sdk || {
    echo "Warning: pip install failed. Please ensure Python environment is active."
}

# 2. PiPER ROS 2 (Humble branch) 클론
if [ ! -d "piper_ros" ]; then
    echo ">> [2/3] Cloning piper_ros (branch: humble)..."
    git clone -b humble https://github.com/agilexrobotics/piper_ros.git piper_ros
else
    echo ">> [2/3] piper_ros already exists. Pulling latest humble branch..."
    cd piper_ros && git pull origin humble && cd ..
fi

# 3. PiPER Isaac Sim USD/URDF 자산 클론
if [ ! -d "piper_isaac_sim" ]; then
    echo ">> [3/3] Cloning piper_isaac_sim (Isaac Sim USD / Descriptions)..."
    git clone https://github.com/agilexrobotics/piper_isaac_sim.git piper_isaac_sim
else
    echo ">> [3/3] piper_isaac_sim already exists. Pulling latest..."
    cd piper_isaac_sim && git pull origin master && cd ..
fi

echo "================================================================="
echo " [AgileX PiPER] Setup completed successfully!"
echo " Refer to robot_arms/piper/README.md for CAN activation & usage."
echo "================================================================="
