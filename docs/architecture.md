# Sim-to-Real Architecture for Quadruped Robots

This document details the framework for our ICCAS paper. It is divided into the **Chronological Research Methodology** and the **Separated System Architectures** (Simulation vs. Real-World).

## 1. Research Methodology Pipeline (Chronological Steps)

This flowchart illustrates the 5-step process we followed to achieve Sim-to-Real autonomous navigation.

```mermaid
flowchart LR
    %% Styling
    classDef phase fill:#e8eaf6,stroke:#1565c0,stroke-width:2px,color:#000;
    classDef deploy fill:#e8f5e9,stroke:#2e7d32,stroke-width:3px,color:#000;

    P1["Phase 1: Environment Generation<br>(fVDB + Gaussian Splatting)"]:::phase
    P2["Phase 2: RL Locomotion Training<br>(SKRL + Isaac Sim)"]:::phase
    P3["Phase 3: V-SLAM Integration<br>(RTAB-Map Mapping)"]:::phase
    P4["Phase 4: Autonomous Navigation<br>(Nav2 + Costmaps)"]:::phase
    P5["Phase 5: Sim-to-Real Deployment<br>(Physical Go2 Robot)"]:::deploy

    P1 --> P2
    P2 --> P3
    P3 --> P4
    P4 --> P5
```

---

## 2. System Architecture: Simulation Realm

This diagram shows the complete data flow entirely within the Isaac Sim environment. The ROS 2 autonomy stack runs flawlessly inside the simulation using the OmniGraph sensor bridge.

```mermaid
flowchart TD
    %% Styling
    classDef simEnv fill:#e3f2fd,stroke:#0288d1,stroke-width:2px;
    classDef rosStack fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef topic fill:#ffffff,stroke:#757575,stroke-width:1px,stroke-dasharray: 3 3;

    subgraph Simulation_Architecture [Simulation System Architecture]
        direction TB
        
        %% Environment & Hardware Mock
        subgraph Sim_Hardware [Isaac Sim 5.1.0]
            Env[Gaussian Splatting Corridor]
            Cam[Virtual RGB-D Camera]
            Odom[Isaac Compute Odometry]
            Robot[Simulated Go2 Robot]
            
            Env -->|Render| Cam
            Env -.->|Physics| Odom
        end
        class Sim_Hardware simEnv

        %% ROS 2 Autonomy Stack
        subgraph ROS2_Stack ["ROS 2 Autonomy Stack (Zero-Code Transferable)"]
            Depth2Scan[depthimage_to_laserscan]
            RTABMap[RTAB-Map V-SLAM]
            Nav2[Nav2 Autonomous Navigation]
            RL_Policy[SKRL Policy Loop]
            
            Depth_Topic([/depth/image_raw]):::topic
            Scan_Topic([/scan]):::topic
            Odom_Topic([/odom]):::topic
            CmdVel_Topic([/cmd_vel_nav]):::topic
            
            Depth_Topic --> Depth2Scan --> Scan_Topic
            Depth_Topic & Odom_Topic --> RTABMap
            RTABMap -->|map to odom TF| Nav2
            Scan_Topic --> Nav2
            Nav2 --> CmdVel_Topic
            CmdVel_Topic --> RL_Policy
        end
        class ROS2_Stack rosStack
        
        %% Connections
        Cam ==>|Publishes| Depth_Topic
        Odom ==>|Publishes| Odom_Topic
        RL_Policy ==>|Applies Torques| Robot
    end
```

---

## 3. System Architecture: Real-World Deployment

This diagram shows the architecture deployed on the physical robot. Notice that the **ROS 2 Autonomy Stack** remains exactly the same as in the Simulation phase. Only the hardware inputs and outputs change.

```mermaid
flowchart TD
    %% Styling
    classDef realEnv fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;
    classDef rosStack fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef topic fill:#ffffff,stroke:#757575,stroke-width:1px,stroke-dasharray: 3 3;

    subgraph Real_Architecture [Real-World System Architecture]
        direction TB
        
        %% Real Hardware
        subgraph Real_Hardware [Physical Robot Hardware]
            Env[Real Physical Corridor]
            Cam[Intel Realsense D435i]
            Odom[Go2 Hardware Kinematics/IMU]
            Robot[Physical Go2 Robot Motors]
            
            Env -->|Optical| Cam
            Env -.->|Physical Movement| Odom
        end
        class Real_Hardware realEnv

        %% ROS 2 Autonomy Stack (IDENTICAL TO SIM)
        subgraph ROS2_Stack_Real ["ROS 2 Autonomy Stack (Zero-Code Transferable)"]
            Depth2Scan[depthimage_to_laserscan]
            RTABMap[RTAB-Map V-SLAM]
            Nav2[Nav2 Autonomous Navigation]
            RL_Policy[SKRL Policy Loop]
            
            Depth_Topic([/depth/image_raw]):::topic
            Scan_Topic([/scan]):::topic
            Odom_Topic([/odom]):::topic
            CmdVel_Topic([/cmd_vel_nav]):::topic
            
            Depth_Topic --> Depth2Scan --> Scan_Topic
            Depth_Topic & Odom_Topic --> RTABMap
            RTABMap -->|map to odom TF| Nav2
            Scan_Topic --> Nav2
            Nav2 --> CmdVel_Topic
            CmdVel_Topic --> RL_Policy
        end
        class ROS2_Stack_Real rosStack
        
        %% Connections
        Cam ==>|Publishes| Depth_Topic
        Odom ==>|Publishes| Odom_Topic
        RL_Policy ==>|Sends PWM/Torque| Robot
    end
```

---

## 4. Key Takeaways for the Paper

By presenting these two diagrams side-by-side in your paper, you can effectively demonstrate the **"Seamless Sim-to-Real Transfer"**. 

Reviewers will clearly see that the orange box (`ROS 2 Autonomy Stack`) is perfectly identical in both Figure 2 (Simulation) and Figure 3 (Real-World). The only elements that change are the blue box (Virtual Sensors) swapping out for the green box (Physical Sensors). This proves the robustness and high fidelity of your Isaac Sim Gaussian environment setup.