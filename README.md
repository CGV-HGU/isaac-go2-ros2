# Unitree AS2 + AgileX PiPER Mobile Manipulator
## NVIDIA Isaac Sim 6.2.0 & Isaac Lab 3.0 Simulation Framework
### Computer Graphics and Vision Lab (CGV)

[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-6.2.0-76B900.svg?logo=nvidia)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-3.0-blue.svg)](https://isaac-sim.github.io/IsaacLab)
[![ROS 2](https://img.shields.io/badge/ROS%202-Humble-22314E.svg?logo=ros)](https://docs.ros.org/en/humble/)
[![Branch](https://img.shields.io/badge/Branch-As2-green.svg)](https://github.com/CGV-HGU/isaac-go2-ros2/tree/As2)
[![Base Robot](https://img.shields.io/badge/Base%20Robot-Unitree%20AS2%20(95Nm)-orange.svg)](https://www.unitree.com)
[![Robotic Arm](https://img.shields.io/badge/Robotic%20Arm-AgileX%20PiPER-blueviolet.svg)](https://github.com/agilexrobotics)

---

## 1. 개요 (Overview)

본 저장소의 **`As2`** 브랜치는 CGV 연구실(Computer Graphics and Vision Lab)에 새로 도입된 **Unitree AS2 산업용 4족 보행 로봇**과 **AgileX PiPER 6자유도 경량 로봇 암**(공급사: (주)로아스 ROAS)을 결합한 모바일 매니퓰레이터(Mobile Manipulator)를 최신 **NVIDIA Isaac Sim 6.2.0** 및 **Isaac Lab 3.0** 가상 시뮬레이션 환경에서 구동·학습하기 위한 연구 플랫폼입니다.

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

## 4. 디렉터리 구조 및 폴더 가이드 (Repository Structure & Navigation)

방문자와 연구원이 저장소를 쉽게 탐색할 수 있도록 기능별로 직관적인 폴더링 체계를 제공합니다.

### 1) 디렉터리 트리 맵 (Directory Tree Map)
```text
isaac-go2-ros2/
├── launchers/                                 # 🚀 [원클릭 실행 스크립트 모음]
│   ├── train_as2_piper.sh                     # AS2 + PiPER 전신 보행 강화학습 (--flat / --rough)
│   ├── train_as2.sh                           # AS2 단독 보행 강화학습
│   ├── play_as2_piper.sh                      # 동시 원격조종 + [1]키 도어 개방 시연
│   ├── spawn_as2_piper.sh                     # 단일 로봇 스폰 및 기구학 확인
│   ├── spawn_as2_piper_hospital.sh            # 병원 월드 + 4종 도어 환경 스폰
│   ├── run_piper_ball_demo.sh                 # PiPER 공 파지 & 다이내믹 외란 모션 검증
│   └── README.md                              # 런처 상세 사용법
│
├── experiments/                               # 🔬 [정량 검증 및 전문 실험 패키지]
│   └── bird_strike/                           # 🎯 로봇팔 이동 표적 리그 (비전 추적 벤치마크)
│       ├── run_sim_check.sh                   # Isaac Sim 사전 검증 런처
│       ├── sim_piper_ball_pick_and_move.py    # 공 파지·이동 시뮬레이션 (초안, 미실행)
│       ├── trajectories/                      # 실물 PiPER 드래그 티칭 궤적 저장소 (.csv)
│       └── README.md                          # 실험 설계, 현재 상태, 안전 수칙
│
├── docs/                                      # 📚 [연구 및 기술 문서 허브]
│   ├── COGNITIVE_BIO_AI_AS2_MASTER_PLAN.md    # 🌟 DAMP 프루닝 & 생체 모사 인지 마스터 플랜 (RA-L)
│   ├── TRAINING_RUNBOOK.md                    # GPU 서버 학습·센서·시연 Step-by-Step 런북
│   ├── isaac_sim_6_2_as2_piper_guide.md       # 연구실 서버 및 Isaac Sim 종합 기술 매뉴얼
│   ├── plans/                                 # 세부 연구 기획 및 논문 로드맵
│   ├── guides/                                # 시뮬레이션 및 센서 상세 매뉴얼
│   └── README.md                              # 문서 디렉토리 색인
│
├── robot_arms/piper/                          # 🦾 [하드웨어 로봇 암 에셋]
│   ├── descriptions/as2_piper.urdf(.xacro)    # 20-DOF 통합 URDF + RGB-D/3D LiDAR 마운트
│   ├── piper_isaac_sim/                       # AgileX 공식 Isaac Sim USD/Mesh 서브모듈
│   └── setup_piper.sh                         # PiPER 에셋 초기화 스크립트
│
├── ros2_ws/src/as2_piper_bringup/             # 🤖 [ROS 2 Humble 패키지]
│   ├── launch/as2_piper_bringup.launch.py     # 비전 검출 + Sim-to-Real 브릿지 통합 런치
│   └── as2_piper_bringup/
│       ├── vision_door_detector.py            # RGB-D + 3D LiDAR 도어/손잡이 인식 노드
│       └── sim2real_bridge_node.py            # ONNX 정책 + piper_sdk CAN 통합 제어기
│
├── source/                                    # 🧠 [핵심 시뮬레이션 로직 - Isaac Lab 패키지]
│   ├── isaaclab_assets/isaaclab_assets/robots/
│   │   ├── as2_piper.py                       # AS2_PIPER_CFG (20-DOF 일체형 로봇 설정)
│   │   ├── as2_sensors.py                     # RGB-D 카메라 + 3D LiDAR 센서 텐서 및 ROS 2 발행
│   │   ├── door_manipulation.py               # 6축 DLS IK, 동시 텔레옵, 4종 도어 자동 개방 FSM
│   │   └── hospital_door.py                   # 4종 도어 Articulation 팩토리
│   └── isaaclab_tasks/.../velocity/config/
│       ├── as2/                               # AS2 단독 보행 RL 설정
│       └── as2_piper/                         # AS2 + PiPER 팔 스윙 적응 전신 RL 설정
│
├── scripts/                                   # 🛠️ [실행 및 유틸리티 도구]
│   ├── demos/                                 # 인터랙티브 데모 파이썬 스크립트
│   ├── reinforcement_learning/rsl_rl/         # RSL-RL 학습(train.py) 및 플레이(play.py)
│   └── tools/
│       ├── preflight_check.py                 # 5초 학습 환경 사전 진단 도구
│       └── evaluate_loco_manipulation.py      # 논문/캡스톤 정량 벤치마크 자동 산출 도구
│
├── tools/                                     # ⚙️ [CI/CD 인프라 및 환경 도구]
│   ├── run_all_tests.py                       # 전체 테스트 일괄 실행
│   └── install_deps.py                        # 파이썬 의존성 설치
│
└── tests/                                     # 🧪 [무결성 감사 테스트]
    └── test_as2_piper_audit.py                # 20개 전방위 기구학·센서·도어 FSM 무결성 테스트
```

### 2) 목적별 폴더 내비게이션 (Directory Cheat Sheet)

| 하고자 하는 작업 (Goal) | 찾아가야 할 위치 (Path) | 주요 파일 / 스크립트 |
| :--- | :--- | :--- |
| **강화학습을 바로 돌리고 싶다** | `launchers/` 또는 루트 `./` | [`train_as2_piper.sh`](launchers/train_as2_piper.sh) |
| **로봇을 키보드로 조종하고 문을 열어보고 싶다** | `launchers/` 또는 루트 `./` | [`play_as2_piper.sh`](launchers/play_as2_piper.sh) |
| **DAMP 메모리 프루닝 논문/연구 계획을 보고 싶다** | `docs/` | [`COGNITIVE_BIO_AI_AS2_MASTER_PLAN.md`](docs/COGNITIVE_BIO_AI_AS2_MASTER_PLAN.md) |
| **GPU 서버 학습 및 센서 설정법을 보고 싶다** | `docs/` | [`TRAINING_RUNBOOK.md`](docs/TRAINING_RUNBOOK.md) |
| **로봇 센서(RGB-D/LiDAR) 또는 문 조작 FSM을 수정하고 싶다** | `source/isaaclab_assets/.../` | [`as2_sensors.py`](source/isaaclab_assets/isaaclab_assets/robots/as2_sensors.py), [`door_manipulation.py`](source/isaaclab_assets/isaaclab_assets/robots/door_manipulation.py) |
| **강화학습 보상 함수 및 보행 환경을 수정하고 싶다** | `source/isaaclab_tasks/.../` | `as2_piper/velocity_env_cfg.py` |
| **ROS 2 문 검출 노드 또는 실물 로봇 제어기를 수정하고 싶다** | `ros2_ws/src/as2_piper_bringup/` | [`vision_door_detector.py`](ros2_ws/src/as2_piper_bringup/as2_piper_bringup/vision_door_detector.py) |
| **로봇 3D CAD/URDF 메쉬를 확인하고 싶다** | `robot_arms/piper/descriptions/` | [`as2_piper.urdf`](robot_arms/piper/descriptions/as2_piper.urdf) |
| **실행 전 환경에 이상이 없는지 5초 진단하고 싶다** | `scripts/tools/` | [`preflight_check.py`](scripts/tools/preflight_check.py) |

---

## 5. 연구실 정보 (Laboratory)

* **연구실**: **Computer Graphics and Vision Lab (CGV)** — 한동대학교 (Handong Global University)
* **연구 주제**: 모바일 매니퓰레이션(Mobile Manipulation), Sim-to-Real 사족보행 로봇 제어, 강화학습 및 로봇 비전
