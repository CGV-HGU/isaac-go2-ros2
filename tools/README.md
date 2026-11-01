# Tools Directory (CI/CD & Environment Setup)
**Computer Graphics and Vision Lab (CGV)**

본 디렉터리는 저장소의 **CI/CD 테스트, 파이썬 의존성 설치, 전체 테스트 실행**을 위한 개발 인프라 도구들을 보관합니다.

| 스크립트 | 용도 | 실행 방법 |
| :--- | :--- | :--- |
| `run_all_tests.py` | 저장소 전체 단위 테스트 일괄 실행 | `python tools/run_all_tests.py` |
| `install_deps.py` | Isaac Lab 필수 의존성 패키지 설치 | `python tools/install_deps.py` |
| `run_train_envs.py` | 환경 구동 안정성 점검 | `python tools/run_train_envs.py` |

> [!NOTE]
> * **`tools/` (루트)**: CI/CD 테스트 및 파이썬 환경 설정용 인프라 스크립트
> * **`scripts/tools/`**: URDF 변환, 사전 진단(`preflight_check.py`), 벤치마크 평가(`evaluate_loco_manipulation.py`) 등 **시뮬레이션 전용 에셋/데이터 도구**
