# Documentation Hub (문서 허브)
**Computer Graphics and Vision Lab (CGV)**

본 디렉터리는 **Isaac Sim 6.2.0 / Isaac Lab 3.0 시뮬레이션** 환경의 학습 런북 및 기술 매뉴얼을 제공합니다.

---

## 📚 주요 문서 바로가기 (Quick Links)

### 1. 저장소 역할 분리 및 비공개 문서 이관 안내
* **미발표 연구 기획 문서 이관**: 이전 버전에서 `docs/`에 포함되었던 미발표 DAMP 관련 연구 초안은 공개 저장소 노출 방지를 위해 제거되었으며, 연구실 내부 비공개 문서/저장소로 이관되었습니다.
* **실물 로봇 제어 워크스페이스 (`CGV-HGU/quadruped_ws`, Private)**: 실물 **Unitree Go2 및 Unitree AS2(단독 주행)** 통합 제어·브링업은 별도 비공개 저장소 `CGV-HGU/quadruped_ws`에서 관리합니다.
* **공식 Unitree AS2 리소스**:
  - URDF/XML/메쉬: `unitreerobotics/unitree_ros` (`robots/as2_description/`)
  - C++/Python SDK: `unitreerobotics/unitree_sdk2` (`example/as2/`), `unitreerobotics/unitree_sdk2_python` (`unitree_sdk2py/as2/`)
  - MuJoCo / RL: `unitreerobotics/unitree_mujoco` (`unitree_robots/as2/`), `unitreerobotics/unitree_rl_mjlab` (`src/assets/robots/unitree_as2/`)

### 2. 가이드 및 런북 (Guides & Runbooks)
* **[`TRAINING_RUNBOOK.md`](TRAINING_RUNBOOK.md) / [`guides/`](guides/)**:  
  GPU 서버에서의 강화학습(RSL-RL Flat $\rightarrow$ Rough), 체크포인트 평가, 온보드 센서(RGB-D/3D LiDAR), 도어 조작 시연 단계별 매뉴얼.
* **[`isaac_sim_6_2_as2_piper_guide.md`](isaac_sim_6_2_as2_piper_guide.md)**:  
  NVIDIA Isaac Sim 6.2.0 & Isaac Lab 3.0 환경 설정 및 문제 해결(Troubleshooting) 가이드.

---

<details>
<summary><strong>Sphinx 기술 문서 빌드 가이드 (Sphinx Build Documentation)</strong></summary>

### Building Documentation with Sphinx

We use [Sphinx](https://www.sphinx-doc.org/en/master/) with the [Book Theme](https://sphinx-book-theme.readthedocs.io/en/stable/) for generating HTML documentation.

```bash
# 1. Install dependencies
cd docs
pip install -r requirements.txt

# 2. Build HTML documentation
make current-docs

# 3. View docs
xdg-open _build/current/index.html
```
</details>
