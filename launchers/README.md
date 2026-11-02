# Execution Launchers (실행 런처 모음)
**Computer Graphics and Vision Lab (CGV)**

본 디렉터리는 Unitree AS2 + AgileX PiPER 시뮬레이션 및 학습을 위한 원클릭 쉘 스크립트 모음입니다.

| 스크립트 | 용도 및 설명 | 실행 예시 |
| :--- | :--- | :--- |
| [`train_as2_piper.sh`](train_as2_piper.sh) | AS2 + PiPER 팔 외란 적응 전신 강화학습 | `./launchers/train_as2_piper.sh --rough` |
| [`train_as2.sh`](train_as2.sh) | AS2 단독 보행 강화학습 | `./launchers/train_as2.sh --flat` |
| [`play_as2_piper.sh`](play_as2_piper.sh) | 학습된 정책 플레이 (키보드 조종 + 자동 문 열기) | `./launchers/play_as2_piper.sh --spawn_door --door_type lever_push --real-time` |
| [`spawn_as2_piper.sh`](spawn_as2_piper.sh) | 기본 환경 로봇 스폰 및 기구학 확인 | `./launchers/spawn_as2_piper.sh` |
| [`spawn_as2_piper_hospital.sh`](spawn_as2_piper_hospital.sh) | 병원(`hospital.usd`) 환경 로봇 + 센서 구동 | `./launchers/spawn_as2_piper_hospital.sh` |
| [`run_piper_ball_demo.sh`](run_piper_ball_demo.sh) | **AgileX PiPER 공 파지 & 다이내믹 외란 모션 사전 검증 데모** | `./launchers/run_piper_ball_demo.sh` |

---
> [!TIP]
> 저장소 루트에서도 하위 호환성을 위해 동일한 이름의 래퍼 스크립트(`./train_as2_piper.sh` 등)를 계속 지원합니다.
