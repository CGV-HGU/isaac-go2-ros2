# Scripts Directory (Isaac Lab Execution & Utilities)
**Computer Graphics and Vision Lab (CGV)**

본 디렉터리는 NVIDIA Isaac Lab 시뮬레이션 환경에서 실행되는 파이썬 실행 스크립트 모음입니다.

```text
scripts/
├── demos/                      # 🎮 인터랙티브 데모 및 시뮬레이션 스폰 스크립트
│   ├── spawn_as2_piper.py      # AS2 + PiPER 기본 평지 스폰 및 조종
│   └── spawn_as2_piper_hospital.py # 병원 환경 도어 개방 데모
├── reinforcement_learning/     # 🧠 RSL-RL 강화학습 train/play 구현체
│   └── rsl_rl/                 # train.py, play.py
├── tools/                      # 🛠️ 시뮬레이션 분석, 변환 및 사전 진단 도구
│   ├── preflight_check.py      # 5초 오프라인 무결성 진단 도구
│   ├── evaluate_loco_manipulation.py # 도어 조작 및 보행 성공률 벤치마크
│   └── convert_urdf.py         # URDF to USD 변환 유틸리티
└── benchmarks/                 # 📊 프레임레이트 및 물리 엔진 성능 벤치마크
```
