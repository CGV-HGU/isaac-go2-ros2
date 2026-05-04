# Sim-to-Real Architecture for Quadruped Robots

This document outlines the conceptual framework for our ICCAS paper. It highlights the seamless Sim-to-Real transfer of autonomous navigation for a quadruped robot (Unitree Go2) using NVIDIA Isaac Sim and ROS 2.

## Concept Diagram: Bridging Simulation and Reality

The architecture is divided into two primary axes: the **Simulation Realm** and the **Real-World Realm**. The simulation environment acts as a robust proving ground, built upon three core pillars, which are then seamlessly transferred to the physical robot.

```mermaid
flowchart TB
    %% Styling
    classDef simRealm fill:#e8f4f8,stroke:#0277bd,stroke-width:3px,color:#000;
    classDef realRealm fill:#f1f8e9,stroke:#2e7d32,stroke-width:3px,color:#000;
    classDef corePillar fill:#fff8e1,stroke:#f57c00,stroke-width:2px,color:#000,stroke-dasharray: 5 5;
    classDef default fill:#ffffff,stroke:#333,stroke-width:1px;

    %% Axis 1: Simulation
    subgraph Sim_Realm [Axis 1: Simulation Realm Isaac Sim]
        direction TB
        
        %% Pillar 1
        subgraph Pillar1 [Core 1: High-Fidelity Environment]
            direction LR
            SfM[Structure from Motion SfM] --> fVDB[fVDB Processing]
            fVDB --> GS[Gaussian Splatting Map]
        end
        class Pillar1 corePillar
        
        %% Pillar 2
        subgraph Pillar2 [Core 2: RL Locomotion]
            direction LR
            Train[Parallel RL Training SKRL] --> Policy[Robust Walking Policy]
        end
        class Pillar2 corePillar

        %% Pillar 3
        subgraph Pillar3 [Core 3: ROS 2 Autonomy Stack]
            direction LR
            SLAM[RTAB-Map V-SLAM] <--> Nav[Nav2 Autonomous Navigation]
        end
        class Pillar3 corePillar

        %% Connections within Sim
        GS --> |"Virtual RGB-D & Odometry"| Pillar3
        Pillar3 --> |"/cmd_vel"| Pillar2
        Pillar2 --> |"Joint Torques"| SimRobot[Simulated Quadruped]
        GS -.-> |"Physics & Collision"| SimRobot
    end
    class Sim_Realm simRealm

    %% Sim-to-Real Transfer Arrows
    Pillar2 ===>|"Zero-Shot Transfer"| Policy_Real
    Pillar3 ===>|"Seamless Code Transfer"| Autonomy_Real

    %% Axis 2: Real-World
    subgraph Real_Realm [Axis 2: Real-World Realm]
        direction TB
        
        RealEnv[Physical Environment Real Corridor] -.-> |"Real Camera & IMU"| Autonomy_Real
        
        subgraph RealAutonomy [Deployed ROS 2 Autonomy]
            direction LR
            Autonomy_Real[Same RTAB-Map & Nav2 Stack]
        end
        class RealAutonomy default
        
        subgraph RealPolicy [Deployed Locomotion]
            direction LR
            Policy_Real[Same RL Policy]
        end
        class RealPolicy default
        
        Autonomy_Real --> |"/cmd_vel"| Policy_Real
        Policy_Real --> |"Joint Torques"| RealRobot[Unitree Go2 Hardware]
        RealEnv -.-> |"Physical Collision"| RealRobot
    end
    class Real_Realm realRealm
```

## The 3 Core Pillars of the Simulation (For Paper Narrative)

To maximize the paper's impact at ICCAS, the narrative should focus on how these three distinct technologies were integrated to bridge the Sim-to-Real gap without performance degradation.

### 1. High-Fidelity Environment via Gaussian Splatting (fVDB + SfM)
Traditional mesh-based simulations often fail to capture complex lighting, textures, and thin structures, leading to the "reality gap" in visual SLAM. By utilizing **Structure from Motion (SfM)** and processing the point clouds via **fVDB**, we generated highly optimized **Gaussian Splatting** maps. This provided the simulated RGB-D cameras with photorealistic data, allowing the RTAB-Map algorithm to extract visual features identical to what it would see in the real world.

### 2. Robust RL Locomotion Policy
Instead of relying on fragile analytical kinematics, the robot's base movement is governed by a Neural Network trained via Reinforcement Learning (RL) in Isaac Sim. The policy was trained across massively parallel environments to handle diverse terrains and velocity commands, ensuring that when the Nav2 stack commands a sudden turn or acceleration, the robot maintains stability.

### 3. Seamless ROS 2 Autonomy Integration (RTAB-Map + Nav2)
The autonomy stack was built using industry-standard ROS 2 frameworks. A critical contribution is the **Virtual Sensor Bridge architecture** (OmniGraph & Static TF patching). This bridge translates the absolute ground truth of the simulation (`World`) into the continuous, standard coordinate frames required by ROS 2 (`map -> odom -> base_link`). Consequently, the exact same Docker container running RTAB-Map and Nav2 can be unplugged from the simulation and plugged into the physical robot without rewriting a single line of autonomy code.