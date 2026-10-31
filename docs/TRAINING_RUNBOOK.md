# Unitree AS2 + AgileX PiPER 강화학습·센서(RGB/Depth/LiDAR)·통합 시연 런북 (CGV Lab)

이 문서는 GPU 서버(`cgv-server-02`) 접속 직후 **무엇부터 어떤 순서로 실행해야 하는지**, **RGB 카메라·Depth 카메라·3D LiDAR 센서를 어떻게 켜는지**, **ROS 2 문 인식 및 정량 벤치마크는 어떻게 실행하는지** 정리한 팀 공용 실행 가이드입니다.

---

## 1. 필요한 강화학습 요약

- **필수 1단계 (`--flat`, 약 12분 / 500 iter)**: `Isaac-Velocity-Flat-Unitree-AS2-Piper-v0` — 4.6kg PiPER 팔이 얹힌 AS2(95 N·m)의 평지 기본 보행 수렴 확인 (64차원 전신 관측: 12 다리 관절 + 8 팔 관절 상태 포함).
- **필수 2단계 (`--rough`, 약 30분 / 1500 iter)**: `Isaac-Velocity-Rough-Unitree-AS2-Piper-v0` — 험지 보행 + 3~6초 주기로 로봇팔을 무작위로 휘두르는 외란(`randomize_piper_arm_targets`)을 함께 학습하여 **보행 중 팔 조작 및 문 열기 시 넘어지지 않는 최종 정책** 생성.
- **선택 3단계 (논문 비교용)**: `./train_as2.sh --rough` — 팔 없는 순수 12자유도 AS2 보행 비교군 학습.
- **별도 학습 불필요**: 오른손 로봇팔 수동 조종, `[1]`번 키 문 타입별 자동 개방(`source/isaaclab_assets/isaaclab_assets/robots/door_manipulation.py`), 그리고 RGB-D + 3D LiDAR 문/손잡이 인식(`ros2_ws/src/as2_piper_bringup`)은 해석적 역기구학 및 기하 융합으로 즉시 동작합니다.
  > **참고**: 4096개 병렬 환경 고속 학습 시 VRAM 병목을 막기 위해 학습 중에는 카메라 렌더링을 끄고, 시연/추론 시 `--enable_sensors` 또는 `--enable_ros2` 플래그로 즉시 활성화합니다 (학습 중에도 센서를 붙이려면 `AS2_ENABLE_SENSORS=1` 환경변수 지정).

---

## 2. AS2 탑재 센서 구성 및 ROS 2 토픽 명세

`robot_arms/piper/descriptions/as2_piper.urdf` 및 `source/isaaclab_assets/isaaclab_assets/robots/as2_sensors.py`에 정의된 센서 프레임(`dont_collapse="true"` 적용, 자체 레이 간섭 방지를 위해 광학 원점 충돌체 제외):

| 센서 | URDF 링크 (`base` 기준 좌표) | Isaac Lab 텐서 (`--enable_sensors`) | ROS 2 토픽 (`--enable_ros2`) |
| :--- | :--- | :--- | :--- |
| **전방 RGB 카메라** | `front_camera` `(0.30, 0.0, 0.10)` | `scene["rgbd_camera"].data.output["rgb"]` (`640x480 uint8`) | `/as2_camera/rgb`, `/as2_camera/camera_info` |
| **전방 Depth 카메라** | `front_camera` `(0.30, 0.0, 0.10)` | `scene["rgbd_camera"].data.output["distance_to_image_plane"]` (`float32 m`) | `/as2_camera/depth` (`32FC1`, 단위: m) |
| **전방 3D LiDAR** | `lidar_link` `(0.285, 0.0, 0.045)` | `scene["lidar"].data.ray_hits_w` (16채널 360° 3D RayCaster) | `/as2_lidar/point_cloud` (`PointCloud2`), `/as2_lidar/scan` (`LaserScan`) |
| **TF / 시뮬레이션 시각** | `base`, `front_cam`, `lidar_link`, `imu`, `radar` | `robot.data.root_pos_w / root_quat_w` | `/tf`, `/clock` |

---

## 3. 서버 오픈 즉시 실행 순서 (Step-by-Step)

```bash
# Step 1. 브랜치 동기화 및 5초 사전 점검 (2분)
git fetch origin && git checkout As2 && git pull origin As2
bash robot_arms/piper/setup_piper.sh
./isaaclab.sh -p scripts/tools/preflight_check.py --require-gpu

# Step 2. 평지(Flat) 1차 보행 학습 (~12분)
tmux new -s as2_train
./train_as2_piper.sh --flat

# Step 3. SSH 원격 보행 영상 검증 (1분, GUI 없이 MP4 생성)
./play_as2_piper.sh --flat --headless --video

# Step 4. 험지(Rough) + 팔 스윙 외란 적응 본 학습 (~30분)
./train_as2_piper.sh --rough

# Step 5. RGB + Depth + 3D LiDAR + 동시 조종 + [1]키 자동 문 열기 통합 실행 (GUI)
./play_as2_piper.sh --rough --spawn_door --door_type lever_push --enable_sensors --enable_ros2 --real-time

# Step 6. (선택) ROS 2 RGB-D/LiDAR 문 인식 + Sim-to-Real 제어 브리지 실행 (새 터미널)
cd ros2_ws && colcon build --symlink-install && source install/setup.bash
ros2 launch as2_piper_bringup as2_piper_bringup.launch.py door_type:=lever_push

# Step 7. 논문/캡스톤 발표용 정량 평가 표(CSV + Markdown) 자동 산출
./isaaclab.sh -p scripts/tools/evaluate_loco_manipulation.py \
    --task Isaac-Velocity-Rough-Unitree-AS2-Piper-Play-v0 \
    --headless --episodes 10 --mode all
```

> **지원 문 타입 (`--door_type`)**: `lever_push` (밀어 여는 레버문) | `lever_pull` (당겨 여는 레버문) | `sliding` (수평 슬라이딩 도어) | `push_bar` (패닉바 스윙문)

---

## 4. 동시 조종 키보드 매핑표

| 구분 | 키 | 기능 |
| :--- | :---: | :--- |
| **로봇개 이동 (왼손)** | `W` / `S` | 전진 / 후진 |
| **로봇개 이동 (왼손)** | `A` / `D` | 좌 / 우 게걸음 |
| **로봇개 회전 (왼손)** | `Q` / `E` | 좌 / 우 제자리 회전 |
| **로봇팔 대관절 (오른손)** | `J` / `L` | 베이스 좌/우 회전 (`joint1`) |
| **로봇팔 대관절 (오른손)** | `I` / `K` | 어깨 앞으로 뻗기 / 당기기 (`joint2`) |
| **로봇팔 대관절 (오른손)** | `U` / `O` | 팔꿈치 펴기 / 접기 (`joint3`) |
| **로봇팔 손목 (오른손)** | `F/H`, `T/G`, `R/Y` | 손목 롤(`joint4`), 피치(`joint5`), 요(`joint6`) |
| **대칭 그리퍼 (오른손)** | `Z` / `X` | 양쪽 집게 대칭 열기 / 닫기 (`joint7`, `joint8`) |
| **자동 문 열기 매크로** | `1` | **근접 조건(`0.28m~0.90m`, 정면 `≤60°`) 만족 시** 현재 문 타입 자동 개방 |
| **비상 중단 / 원위치** | `C` | 자동 문 열기 즉시 취소 및 팔 기본 접힘 자세 복귀 |
