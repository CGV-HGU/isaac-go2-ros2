#!/usr/bin/env bash
# ==============================================================================
# Setup script for AgileX PiPER Robotic Arm in isaac-go2-ros2
# Supports: Python SDK, ROS 2 (Humble), and Isaac Sim assets
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================="
echo " [AgileX PiPER] Initializing Robotic Arm Environment for As2"
echo "================================================================="

# 1. Install Python CAN and piper_sdk into Isaac Lab Python (or active Python)
echo ">> [1/3] Installing Python CAN and piper_sdk..."
if [ -x "$REPO_ROOT/isaaclab.sh" ]; then
    "$REPO_ROOT/isaaclab.sh" -p -m pip install --upgrade python-can piper_sdk
else
    python3 -m pip install --upgrade python-can piper_sdk
fi

# 2. Clone PiPER ROS 2 (Humble branch)
if [ ! -d "piper_ros" ] || [ ! -e "piper_ros/.git" ]; then
    echo ">> [2/3] Cloning piper_ros (branch: humble)..."
    rm -rf piper_ros
    git clone -b humble https://github.com/agilexrobotics/piper_ros.git piper_ros
else
    echo ">> [2/3] Updating piper_ros (humble)..."
    git -C piper_ros pull --ff-only origin humble
fi

# 3. Initialize pinned piper_isaac_sim submodule cleanly
echo ">> [3/3] Initializing piper_isaac_sim submodule..."
if git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 && [ -f "$REPO_ROOT/.gitmodules" ]; then
    git -C "$REPO_ROOT" submodule update --init --recursive robot_arms/piper/piper_isaac_sim
elif [ ! -e "piper_isaac_sim/USD/piper_v1.usd" ]; then
    rm -rf piper_isaac_sim
    git clone https://github.com/agilexrobotics/piper_isaac_sim.git piper_isaac_sim
fi

echo "================================================================="
echo " [AgileX PiPER] Setup completed successfully!"
echo " Refer to robot_arms/piper/README.md for CAN activation & usage."
echo "================================================================="
