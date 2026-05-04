# Isaac Sim to Real-World Architecture for Quadruped Robots

This document outlines the Sim-to-Real architecture integrating Reinforcement Learning (RL) based locomotion with the ROS 2 Navigation Stack (Nav2) and RTAB-Map SLAM in NVIDIA Isaac Sim 5.1.0.

## Architecture Diagram

```mermaid
graph TD
    %% 스타일 정의
    classDef sim fill:#e1f5fe,stroke:#333,stroke-width:2px;
    classDef ros fill:#e8f5e9,stroke:#333,stroke-width:2px;
    classDef policy fill:#fff3e0,stroke:#333,stroke-width:2px;
    classDef user fill:#ffebee,stroke:#333,stroke-width:2px;

    %% 1. Isaac Sim 영역
    subgraph Isaac_Sim [1. Isaac Sim 5.1.0 Environment]
        direction TB
        Env[Gaussian Splatting Map] --> Robot[Unitree Go2 Robot]
        
        subgraph OmniGraph [OmniGraph Sensor Bridge]
            Robot -->|Physics| IsaacOdom[Isaac Compute Odometry]
            Robot -->|Vision| IsaacCam[RGB-D Camera]
            
            IsaacOdom -->|/odom, /tf (World to base)| ROS2OdomPub[ROS 2 Publisher]
            IsaacCam -->|/rgb, /depth| ROS2CamPub[ROS 2 Camera Helper]
        end
    end
    class Isaac_Sim sim;

    %% 2. ROS 2 영역 (Perception & SLAM)
    subgraph ROS2_Perception [2. ROS 2 Perception & SLAM]
        direction TB
        ROS2CamPub -->|/depth/image_raw| DepthToScan[depthimage_to_laserscan]
        DepthToScan -->|/scan| Nav2Costmap
        
        ROS2CamPub -->|/rgb, /depth| RTABMap[RTAB-Map V-SLAM]
        ROS2OdomPub -->|/odom| RTABMap
        
        StaticTF[Static TF: odom to World] -.-> |Fixes TF Tree| RTABMap
    end
    class ROS2_Perception ros;

    %% 3. ROS 2 영역 (Navigation)
    subgraph ROS2_Navigation [3. ROS 2 Navigation2 Stack]
        MapServer[Map Server] -->|/map yaml| Nav2Costmap[Global / Local Costmap]
        RTABMap -->|map to odom TF| Nav2Costmap
        
        Nav2Costmap --> Planner[Planner Server]
        Planner --> Controller[Controller Server]
        
        Controller -->|/cmd_vel_nav| CmdVelOut((cmd_vel_nav))
    end
    class ROS2_Navigation ros;

    %% 4. 강화학습(RL) 제어 영역
    subgraph RL_Control [4. RL Policy Control Loop]
        direction TB
        CmdVelOut -->|Subscribe Twist| VelocitySelector{Velocity Override}
        
        Keyboard[Keyboard Input W,A,S,D] -->|User Override| VelocitySelector
        
        VelocitySelector -->|final_cmd_vel| SKRL[SKRL Policy 'best_agent.pt']
        SKRL -->|Joint Torques| Robot
    end
    class RL_Control policy;
    class Keyboard user;
```

## Key Contributions (For ICCAS Paper)

1. **Modular Sim-to-Real Design:**
   The architecture strictly decouples the simulation environment (Isaac Sim) from the ROS 2 autonomy stack. For real-world deployment, the `Isaac_Sim` block can be seamlessly replaced with physical sensor drivers (e.g., RealSense camera, hardware odometry) without altering a single line of the ROS 2 perception or navigation code.

2. **Safety-Critical Velocity Override:**
   To address the inherent unpredictability of RL policies and dynamic environments, a safety override mechanism is embedded directly within the control loop. User inputs (`Keyboard`) strictly take precedence over autonomous commands (`/cmd_vel_nav`). This ensures immediate human intervention is possible, a critical requirement for real-world quadruped deployments.

3. **Efficient Perception Pipeline (Depth-to-Scan):**
   By utilizing a single RGB-D camera and converting depth data to 2D laser scans (`depthimage_to_laserscan`), the system satisfies the rigorous obstacle detection requirements of Nav2's Costmaps without the need for expensive and heavy 2D/3D LiDAR sensors, optimizing the payload for agile quadruped robots.

4. **Bridging the TF Tree Gap:**
   Overcoming the standard coordinate frame mismatch between Isaac Sim's default (`World -> base`) and ROS 2's REP-105 standard (`map -> odom -> base_link`), the architecture utilizes a strategic `static_transform_publisher`. This seamlessly welds the two distinct TF trees together, allowing RTAB-Map and Nav2 to function flawlessly within the simulation environment.