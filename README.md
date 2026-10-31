# Unitree AS2 + AgileX PiPER Mobile Manipulator
## NVIDIA Isaac Sim 6.2.0 & Isaac Lab Simulation Framework
### Computer Graphics and Vision Lab (CGV)

[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-6.2.0-76B900.svg?logo=nvidia)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-v2.3+-blue.svg)](https://isaac-sim.github.io/IsaacLab)
[![ROS 2](https://img.shields.io/badge/ROS%202-Humble-22314E.svg?logo=ros)](https://docs.ros.org/en/humble/)
[![Branch](https://img.shields.io/badge/Branch-As2-green.svg)](https://github.com/CGV-HGU/isaac-go2-ros2/tree/As2)
[![Base Robot](https://img.shields.io/badge/Base%20Robot-Unitree%20AS2%20(95Nm)-orange.svg)](https://www.unitree.com)
[![Robotic Arm](https://img.shields.io/badge/Robotic%20Arm-AgileX%20PiPER-blueviolet.svg)](https://github.com/agilexrobotics)

---

## 1. 개요 (Overview)

본 저장소의 **`As2`** 브랜치는 CGV 연구실(Computer Graphics and Vision Lab)에 새로 도입된 **Unitree AS2 산업용 4족 보행 로봇**과 **AgileX PiPER 6자유도 경량 로봇 암**(공급사: (주)로아스 ROAS)을 결합한 모바일 매니퓰레이터(Mobile Manipulator)를 최신 **NVIDIA Isaac Sim 6.2.0 / 5.x** 및 **Isaac Lab** 환경에서 구동·학습하기 위한 연구 플랫폼입니다.

온보드 **RGB 카메라 + Depth 카메라 + 3D LiDAR** 센서 스위트, 왼손 보행(`WASD/QE`) + 오른손 6축 팔 동시 조종, `[1]`번 키 근접 조건부 4종 도어 자동 개방, 그리고 **ROS 2 Humble 비전/LiDAR 문 인식 및 Sim-to-Real 제어 패키지(`ros2_ws/src/as2_piper_bringup`)**가 통합되어 있습니다.

---

## 2. 주요 기능 및 지원 환경 (Key Features)

### 1) 온보드 센서 스위트: RGB 카메라 + Depth 카메라 + 3D LiDAR
* **물리 센서 마운트 ([`as2_piper.urdf`](robot_arms/piper/descriptions/as2_piper.urdf))**:
  - `front_camera` (`base` 기준 `0.30, 0.0, 0.10 m`): `640x480` RGB + Metric Depth (`distance_to_image_plane`)
  - `lidar_link` (`base` 기준 `0.285, 0.0, 0.045 m`): 16채널 360° 3D RayCaster 및 RTX 3D LiDAR
* **시뮬레이션 텐서 및 ROS 2 브리지 ([`as2_sensors.py`](source/isaaclab_assets/isaaclab_assets/robots/as2_sensors.py))**:
  - `--enable_sensors`: `scene["rgbd_camera"]` 및 `scene["lidar"]` GPU 텐서 실시간 활성화
  - `--enable_ros2`: `/as2_camera/rgb`, `/as2_camera/depth`, `/as2_camera/camera_info`, `/as2_lidar/point_cloud`, `/as2_lidar/scan`, `/tf`, `/clock` 자동 발행

### 2) 병원 환경(`hospital.usd`) 및 4종 도어 근접 자동 개방 (`[1]`번 키)
* **지원 문 타입 (`--door_type`)**: `lever_push` (밀어 여는 레버문), `lever_pull` (당겨 여는 레버문), `sliding` (슬라이딩 도어), `push_bar` (패닉바 스윙문)
* **근접 활성화 게이트 ([`door_manipulation.py`](source/isaaclab_assets/isaaclab_assets/robots/door_manipulation.py))**:
  - 손잡이 평면 거리 `0.28m ~ 0.90m` 및 정면 정렬 오차 `≤ 60°` 이내에 진입했을 때만 `[1]`번 키 자동 개방 시퀀스가 활성화됩니다.

### 3) Isaac Lab 강화학습(RL) Task & ROS 2 Sim-to-Real 패키지
* `Isaac-Velocity-Flat-Unitree-AS2-v0` / `Isaac-Velocity-Rough-Unitree-AS2-v0`: AS2 단독 평지/험지 보행 학습
* `Isaac-Velocity-Flat-Unitree-AS2-Piper-v0` / `Isaac-Velocity-Rough-Unitree-AS2-Piper-v0`: AS2 + PiPER 팔 무작위 스윙 외란 적응 전신 보행 학습
* **ROS 2 Humble 브링업 ([`ros2_ws/src/as2_piper_bringup`](ros2_ws/src/as2_piper_bringup))**: RGB-D + 3D LiDAR 문/손잡이 검출 노드(`vision_door_detector`) 및 48D/64D ONNX 정책 + `piper_sdk` CAN 통합 제어 노드(`sim2real_bridge_node`)
* **정량 벤치마크 도구 ([`evaluate_loco_manipulation.py`](scripts/tools/evaluate_loco_manipulation.py))**: 팔 고정 vs 스윙 보행 안정성 및 4종 도어 개방 성공률 CSV/Markdown 표 자동 생성

---

## 3. 빠른 시작 가이드 (Quick Start)

상세 단계별 학습 및 실행 순서는 **[`docs/TRAINING_RUNBOOK.md`](docs/TRAINING_RUNBOOK.md)** 를 참고하세요.

```bash
# 1. 저장소 클론 및 서브모듈 초기화
git clone https://github.com/CGV-HGU/isaac-go2-ros2.git -b As2
cd isaac-go2-ros2
bash robot_arms/piper/setup_piper.sh

# 2. 강화학습 실행 (평지 1차 -> 험지+팔 외란 본 학습)
./train_as2_piper.sh --flat
./train_as2_piper.sh --rough

# 3. RGB + Depth + 3D LiDAR + 동시 조종 + [1]키 자동 문 열기 시연
./play_as2_piper.sh --rough --spawn_door --door_type lever_push --enable_sensors --enable_ros2 --real-time

# 4. 병원 환경 도어 조작 데모
./spawn_as2_piper_hospital.sh --door_type lever_push --enable_sensors --enable_ros2
```

---

## 4. 디렉터리 구조 (Repository Structure)

```text
isaac-go2-ros2/
├── docs/
│   ├── TRAINING_RUNBOOK.md                    # GPU 서버 학습·센서·시연 Step-by-Step 런북
│   └── isaac_sim_6_2_as2_piper_guide.md       # 연구실 서버 및 Isaac Sim 종합 기술 매뉴얼
├── robot_arms/piper/
│   ├── descriptions/as2_piper.urdf(.xacro)    # 20-DOF 통합 URDF + RGB-D/3D LiDAR 센서 마운트
│   └── piper_isaac_sim/                       # AgileX 공식 Isaac Sim USD/Mesh 서브모듈
├── ros2_ws/src/as2_piper_bringup/             # ROS 2 Humble 비전/LiDAR 문 인식 & Sim-to-Real 패키지
├── scripts/
│   ├── demos/spawn_as2_piper(_hospital).py    # 기본 및 병원 도어 모바일 매니퓰레이션 데모
│   ├── reinforcement_learning/rsl_rl/play.py  # 동시 원격조종 + [1]키 자동 문열기 + ROS 2 센서 브리지
│   └── tools/
│       ├── preflight_check.py                 # 5초 학습 환경 사전 진단 도구
│       └── evaluate_loco_manipulation.py      # 논문/캡스톤 정량 벤치마크 자동 산출 도구
├── source/
│   ├── isaaclab_assets/isaaclab_assets/robots/
│   │   ├── as2_piper.py                       # AS2_PIPER_CFG, UNITREE_AS2_CFG, PIPER_ARM_CFG
│   │   ├── as2_sensors.py                     # RGB 카메라, Depth 카메라, 3D LiDAR 통합 설정
│   │   ├── door_manipulation.py               # 6축 DLS IK, 동시 텔레옵, 4종 도어 자동 개방 상태머신
│   │   └── hospital_door.py                   # 4종 도어 Articulation 팩토리
│   └── isaaclab_tasks/.../velocity/config/
│       ├── as2/                               # AS2 단독 보행 RL 설정
│       └── as2_piper/                         # AS2 + PiPER 팔 스윙 적응 전신 RL 설정
└── tests/test_as2_piper_audit.py              # 오프라인 전체 기구학·센서·ROS 2·상태머신 회귀 테스트 (20개)
```

---

## 5. 연구실 정보 (Laboratory)

* **연구실**: **Computer Graphics and Vision Lab (CGV)** — 한동대학교 (Handong Global University)
* **연구 주제**: 모바일 매니퓰레이션(Mobile Manipulation), Sim-to-Real 사족보행 로봇 제어, 강화학습 및 로봇 비전
