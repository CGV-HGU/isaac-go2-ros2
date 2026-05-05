# Sim-to-Real Autonomous Navigation for Quadruped Robots

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![ROS 2](https://img.shields.io/badge/ROS_2-Jazzy-blue.svg)](https://docs.ros.org/en/jazzy/)
[![Robot](https://img.shields.io/badge/Hardware-Unitree_Go2_Edu_Plus-orange.svg)]()

This repository contains the codebase for our capstone project and ICCAS paper submission, focusing on a **Zero-Code Sim-to-Real Architecture** for quadruped robots. 

By integrating Reinforcement Learning (RL) based locomotion with industry-standard ROS 2 autonomy stacks (RTAB-Map & Nav2), this project demonstrates how to train and validate a quadruped robot in a highly realistic virtual environment, and seamlessly deploy the exact same autonomy code to the physical hardware.

---

## 🌟 Key Features (The 3 Core Pillars)

1. **High-Fidelity Virtual Environment (fVDB + Gaussian Splatting)**
   - Utilizes advanced photorealistic rendering to minimize the visual reality gap.
   - Provides identical visual features for V-SLAM algorithms in both simulation and reality.

2. **Robust RL Locomotion (SKRL)**
   - The robot's base walking capabilities are governed by a robust Neural Network policy trained via Reinforcement Learning.
   - Includes a **Safety-Critical Velocity Override** system that prioritizes human keyboard input over autonomous commands to prevent collisions.

3. **Decoupled ROS 2 Autonomy Stack (RTAB-Map & Nav2)**
   - Converts RGB-D data into 2D laser scans via `depthimage_to_laserscan` for efficient, LiDAR-free 2D obstacle avoidance.
   - Implements a seamless **Virtual Sensor Bridge (OmniGraph & Static TF)** to resolve coordinate frame conflicts (`World` vs `map -> odom`).
   - The Nav2 stack calculates the path and publishes `/cmd_vel_nav`, which is directly translated into joint torques by the RL policy.

---

## 🏗️ System Architecture

For a detailed breakdown of the system data flow, coordinate frames, and the chronological research methodology pipeline, please refer to our dedicated architecture document:

👉 **[View the Architecture Documentation](./docs/architecture.md)**

---

## 🚀 Quick Start Guide

### Prerequisites
- **OS:** Ubuntu 24.04
- **Simulation:** NVIDIA Isaac Sim 5.1.0
- **ROS 2:** Jazzy Jalisco
- **Dependencies:** `nav2_bringup`, `rtabmap_ros`, `depthimage_to_laserscan`

### 1. Launch the Simulation (Isaac Sim + RL Policy)
This script launches Isaac Sim, loads the Gaussian Splatting environment, and runs the SKRL policy in inference mode. It also activates the OmniGraph ROS 2 bridge.

```bash
# Terminal 1
./play.sh
```

### 2. Launch the Autonomy Stack (V-SLAM + Nav2)
Once the robot is spawned in the simulation, run the following script. This will start the map server, depth-to-laser conversion, RTAB-Map localization, and the Nav2 behavior tree.

```bash
# Terminal 2
source /opt/ros/jazzy/setup.bash
./rtabmap_localization.sh
```

### 3. Command the Robot (RViz2)
The `rtabmap_localization.sh` script will automatically open RViz2.
1. Wait for the `[lifecycle_manager]: Managed nodes are active` message in the terminal.
2. In RViz2, click the **`2D Goal Pose`** button in the top toolbar.
3. Click and drag on the map to set a destination. The robot will automatically navigate to the target, avoiding dynamic obstacles dropped in the simulation.

---

## 🧠 Retraining the RL Policy

If the robot's locomotion behavior needs tuning (e.g., handling sharper turns or rougher terrain), you can retrain the base policy using massively parallel environments.

```bash
./train_go2.sh
```

---

## 👥 Contributors
- **Minseok** - Architecture Design & Sim-to-Real Strategy
- **Hayoung** - ROS 2 Integration & Simulation Operation
