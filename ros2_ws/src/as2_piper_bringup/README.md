# `as2_piper_bringup` — ROS 2 Humble Perception & Sim-to-Real Bridge

Unitree AS2 + AgileX PiPER 모바일 매니퓰레이터의 **RGB 카메라 + Depth 카메라 + 3D LiDAR** 센서 융합 문/손잡이 인식 노드(`vision_door_detector`)와 **ONNX 보행 정책 + 근접 게이트 자동 문 열기(`DoorAutoOpener`) 통합 제어 브리지(`sim2real_bridge_node`)** 패키지입니다.

## 1. 빌드 방법 (ROS 2 Humble)

```bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
```

## 2. Isaac Sim 연동 실행

터미널 1 (Isaac Sim — RGB/Depth/3D LiDAR ROS 2 브리지 활성화):
```bash
./play_as2_piper.sh --flat --spawn_door --door_type lever_push --enable_ros2 --real-time
```

터미널 2 (ROS 2 비전/LiDAR 문 인식 + Sim-to-Real 제어 노드):
```bash
source ros2_ws/install/setup.bash
ros2 launch as2_piper_bringup as2_piper_bringup.launch.py \
    onnx_policy_path:=/abs/path/to/exported/policy.onnx \
    door_type:=lever_push
```

자동 문 열기 트리거 토픽 송신 (문 앞 `0.28m ~ 0.90m`, `±60°` 이내 접근 시):
```bash
ros2 topic pub --once /as2_piper/auto_door_trigger std_msgs/msg/Bool "{data: true}"
```

## 3. 실기(Sim-to-Real) 배포 (Unitree AS2 + RealSense D435i + AgileX PiPER CAN)

```bash
sudo ip link set can0 up type can bitrate 1000000
ros2 launch as2_piper_bringup as2_piper_bringup.launch.py \
    onnx_policy_path:=/abs/path/to/exported/policy.onnx \
    enable_piper_can:=true
```
