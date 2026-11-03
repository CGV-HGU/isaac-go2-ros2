# Robot-Arm Moving-Target Rig (로봇팔 이동 표적 리그)
**Computer Graphics and Vision Lab (CGV)**

> [!NOTE]
> **폴더 명칭 및 목적**: 본 모듈은 야외 조류 퇴치 UGV(`bs_ws`)의 **카메라 및 짐벌 비전 추적 알고리즘 성능을 객관적으로 검증하기 위해 AgileX PiPER 6-DOF 로봇팔을 '결정론적 이동 표적 리그'로 활용하는 전용 실험 패키지**입니다. 사람이 손으로 공을 흔들 때 발생하는 인간 편향(Human Bias)과 비재현성을 원천 차단하고, 수학적으로 정의된 정량 벤치마크 데이터를 제공합니다.

---

## 1. 왜 로봇팔 이동 표적 리그인가?

사람이 손으로 공을 들고 움직이면 실험할 때마다 속도, 가속도, 경로가 제각각 달라져 추적기 알고리즘의 개선 여부를 신뢰도 있게 입증하기 어렵습니다. 로봇팔을 표적 장치로 사용하면 다음과 같은 핵심 이점이 확보됩니다:

1. **완벽한 재현성 (Repeatability)**:
   - 동일한 속도, 진폭, 가속도의 표적 궤적을 100회 이상 오차 없이 반복 실행 가능.
2. **지면 참값 (Ground Truth) 제공**:
   - 로봇팔 관절 엔코더 각도와 정기구학(Forward Kinematics)을 통해 매 순간 표적 공의 3D 공간 좌표 $(X, Y, Z)$를 1mm 미만 오차로 산출.
   - 추적 카메라 영상 평면 상의 정답 픽셀 좌표 $(u^*, v^*)$와 오차(Pixel Tracking Error, RMSE)를 정량 계산.

---

## 2. 표적 궤적 체계 (Benchmark Trajectories)

| ID | 궤적 명칭 | 동작 특성 | 측정 목적 및 평가지표 |
| :---: | :--- | :--- | :--- |
| **T1** | **스텝 응답 (Step Response)** | 중심 정지 $\rightarrow$ $\pm 15^\circ$ 방위각(Yaw) 급격 스텝 이동 후 4초 정지 $\rightarrow$ $+12^\circ$ 고도각(Pitch) 스텝 | 추적기의 **상승 시간($t_r$)**, **정정 시간($t_s$)**, **오버슈트($M_p$)**, **정상상태 추적 오차** |
| **T2** | **등속 스윕 (Velocity Sweep)** | 저속(15°/s), 중속(39°/s), 고속(78°/s) 방위각 왕복 및 2D 8자(Lissajous) 복합 기동 | **목표 각속도에 따른 추적 지연(Lag)**, **추적 락(Lock) 이탈 한계 속도** 도출 |
| **T3** | **드래그 티칭 (Drag-Teaching)** | 연구자가 로봇팔을 직접 손으로 잡고 임의의 자연스러운 비행 궤적을 실시간 교시 녹화 | 정성적 데모 및 필드 유사 복합 기동 검증 (정량 지표는 T1/T2 기반) |

---

## 3. 디렉터리 구성

```
experiments/bird_strike/
├── README.md                          # 본 실험 설명서
├── run_sim_check.sh                   # Isaac Sim 사전 확인 런처 (--t1, --t2 지원)
├── sim_piper_ball_pick_and_move.py    # Isaac Sim 시뮬레이션 (공 파지, 5단계 모션 & CSV 재생)
├── piper_teach_playback.py            # PiPER 드래그 티칭 녹화, 실기 재생, T1/T2 생성, info 분석
└── trajectories/                      # 궤적 CSV 데이터셋
    ├── trajectory_t1_step.csv         # [생성완료] T1 스텝 응답 궤적 (24.0초, 50Hz, 1201프레임)
    └── trajectory_t2_sweep.csv        # [생성완료] T2 등속 스윕 궤적 (28.0초, 50Hz, 1401프레임)
```

---

## 4. 실물 로봇 제어 및 사용법 (`piper_teach_playback.py`)

`piper_teach_playback.py`는 실물 AgileX PiPER 하드웨어 제어(`piper_sdk`), 궤적 녹화, 안전 재생, 모의 시험(`--mock`)을 모두 지원합니다.

### 4.1. 벤치마크 궤적 재합성 및 분석
```bash
# T1 스텝 궤적 생성
python3 experiments/bird_strike/piper_teach_playback.py generate-t1

# T2 속도 스윕 궤적 생성
python3 experiments/bird_strike/piper_teach_playback.py generate-t2

# 궤적 통계 및 각 관절별 최고 속도(Max Velocity) 검사
python3 experiments/bird_strike/piper_teach_playback.py info --input experiments/bird_strike/trajectories/trajectory_t1_step.csv
python3 experiments/bird_strike/piper_teach_playback.py info --input experiments/bird_strike/trajectories/trajectory_t2_sweep.csv
```

### 4.2. 사람 직접 교시 (드래그 티칭) 녹화
실제 PiPER 로봇팔의 팔목 티칭 버튼을 누른 후 손으로 부드럽게 표적 공을 움직여 궤적을 기록합니다:
```bash
# 50Hz 주기로 CAN(can0)에서 관절각 및 그리퍼 위치 녹화
python3 experiments/bird_strike/piper_teach_playback.py record \
    --can can0 \
    --rate-hz 50 \
    --output experiments/bird_strike/trajectories/recorded_drag_01.csv
```

### 4.3. 실물 로봇 궤적 안전 재생 (Playback)
급작스러운 튕김을 방지하기 위해 **현재 로봇 자세에서 궤적 시작 자세까지 2.5초간 코사인 S-커브로 부드럽게 사전 보간(Pre-flight ramp)** 후 본 궤적을 실행합니다:
```bash
# 50% 감속 안전 모드로 T1 스텝 응답 재생
python3 experiments/bird_strike/piper_teach_playback.py playback \
    --input experiments/bird_strike/trajectories/trajectory_t1_step.csv \
    --can can0 \
    --speed 0.5 \
    --safe-disable

# 실물 장비 연결 없이 로직 테스트 (Mock 모드)
python3 experiments/bird_strike/piper_teach_playback.py playback \
    --input experiments/bird_strike/trajectories/trajectory_t1_step.csv \
    --mock --speed 2.0
```

---

## 5. Isaac Sim 시뮬레이션 사전 검증

실물 하드웨어 구동 전 NVIDIA Isaac Sim 6.2.0 환경에서 로봇 동작과 공 파지 상태를 시각적으로 확인합니다.

```bash
# 기본 5단계 공 파지 및 다축 가진(Dynamic Shake) 시뮬레이션
./experiments/bird_strike/run_sim_check.sh

# 실제 표적 궤적(T1 스텝 응답)을 가상 환경에서 재현 검증
./experiments/bird_strike/run_sim_check.sh --t1

# 실제 표적 궤적(T2 속도 스윕)을 가상 환경에서 재현 검증
./experiments/bird_strike/run_sim_check.sh --t2
```

---

## 6. 하드웨어 설정 및 안전 수칙

1. **CAN 통신 초기화**:
   ```bash
   sudo ip link set can0 type can bitrate 1000000
   sudo ip link set up can0
   ```
2. **비상 정지(E-Stop)**:
   - 첫 구동 시에는 반드시 `--speed 0.2`~`0.3` 수준의 저속으로 시작합니다.
   - 키보드 `Ctrl + C` 입력 시 즉시 CAN 위치 명령을 중단하고 모터를 안전 모드로 전환합니다.
3. **표적 공 사양**:
   - 지름: 4.0 cm (반경 0.020 m)
   - 중량: 약 35 g (경량 탁구공 또는 비전 트래킹용 고대비 무반사 레드 볼)
   - 그리퍼 체결: 핑거 패드 사이 간격 2.6 cm (`gripper_um=26000`)로 견고 파지 또는 막대 고정대 체결 권장.

---
*Computer Graphics and Vision Lab (CGV)*
