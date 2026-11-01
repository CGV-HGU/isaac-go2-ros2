# Documentation Hub (문서 허브)
**Computer Graphics and Vision Lab (CGV)**

본 디렉터리는 Unitree AS2 + AgileX PiPER 모바일 매니퓰레이터 프로젝트의 연구 마스터 플랜, 강화학습 런북, 시뮬레이터 가이드 문서를 체계적으로 제공합니다.

---

## 📚 주요 문서 바로가기 (Quick Links)

### 1. 연구 및 마스터 플랜 (Research & Master Plans)
* **[`COGNITIVE_BIO_AI_AS2_MASTER_PLAN.md`](COGNITIVE_BIO_AI_AS2_MASTER_PLAN.md) / [`plans/`](plans/)**:  
  현우 학생의 **DAMP (Safety-Preserving 3D Scene Graph Pruning)** 알고리즘과 생체 모사 인지 아키텍처, IEEE RA-L 타겟 연구 마스터 플랜.

### 2. 가이드 및 런북 (Guides & Runbooks)
* **[`TRAINING_RUNBOOK.md`](TRAINING_RUNBOOK.md) / [`guides/`](guides/)**:  
  GPU 서버에서의 강화학습(RSL-RL Flat $\rightarrow$ Rough), 체크포인트 평가, 온보드 센서(RGB-D/3D LiDAR), 도어 조작 시연 단계별 매뉴얼.
* **[`isaac_sim_6_2_as2_piper_guide.md`](isaac_sim_6_2_as2_piper_guide.md)**:  
  NVIDIA Isaac Sim 6.2.0 / Isaac Lab v2.3 환경 설정 및 문제 해결(Troubleshooting) 가이드.

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
