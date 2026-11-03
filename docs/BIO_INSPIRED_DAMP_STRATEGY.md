---
title: "DAMP 연구 전략 메모 (Driving-Aware Memory Pruning)"
author: "Computer Graphics and Vision Lab (CGV)"
date: 2026-11-03
version: "5.0.0 (정정판)"
status: "전략 초안 - 실험 데이터 없음"
tags:
  - damp-memory-pruning
  - hyunwoo-research
  - 3d-scene-graph
  - strategy-draft
  - cgv-lab
aliases:
  - Bio-DAMP-Strategy
  - DAMP-Strategy
---

# DAMP 연구 전략 메모 (정정판 v5.0)
**Computer Graphics and Vision Lab (CGV)** · 연구 주관: 현우 (Hyunwoo)

> [!WARNING]
> **이 문서는 측정된 결과가 아니라 '전략 초안'입니다.**
> * v1.0~v4.0에 들어 있던 성능 수치(메모리 16.8 GB → 2.6 GB, 84.5% 절감, 12 ms, 충돌률 0.0%, 완주율 98.5% 등)는 **실제 실험으로 측정한 값이 아니라 작성 중에 임의로 넣은 예시값**이었습니다. v5.0에서 모두 제거하고 `[측정 예정]`으로 바꿨습니다. **논문·발표 자료에 옮기지 마세요.**
> * DAMP의 실제 알고리즘, 코드, 데이터의 주인은 현우 연구원입니다. 이 메모는 현우 연구원의 현재 메모리 구조를 확인하지 않은 상태에서 작성되었습니다.

---

## 1. 연구 질문 (확정된 부분)

로봇이 장거리 주행을 할 때 3D Scene Graph 메모리가 계속 쌓이는 문제를, 주행에 방해되지 않게 줄이는 방법:

1. **What to prune**: 어떤 노드를 지우거나, 합치거나, 반드시 남길 것인가?
2. **When to prune**: 주행 연산을 방해하지 않으려면 언제 정리할 것인가?

---

## 2. 제안 구조 (가설 단계)

### 2.1 What: 안전 집합은 지우지 않는다
안전에 필요한 노드 집합 $\mathcal{N}_{\text{safe}}$를 정의하고, 그 안의 노드는 정리 대상에서 뺍니다.

$$d_{\text{stop}}(v) = v\,t_{\text{react}} + \frac{v^2}{2a_{\max}}, \qquad
\tau_i = \frac{\|\mathbf{r}_i\|}{\max\!\left(\epsilon,\; -\dot{\mathbf{r}}_i\cdot\hat{\mathbf{r}}_i\right)}$$

$$\mathcal{N}_{\text{safe}} = \{\, n_i \mid \|\mathbf{r}_i\| \le d_{\text{stop}}(v) \;\lor\; \tau_i \le \tau_{\text{th}} \,\}, \qquad n_i \in \mathcal{N}_{\text{safe}} \Rightarrow \text{prune 금지}$$

그 밖의 노드는 (a) 로봇이 도달할 수 없는 영역이면 삭제, (b) 같은 의미·같은 평면의 정적 노드면 하나의 바운딩 박스로 병합, (c) 오래 관측되지 않은 노드는 점진 삭제합니다.

> [!IMPORTANT]
> **보장할 수 있는 것과 없는 것을 구분하세요.**
> * 보장 가능: "$\mathcal{N}_{\text{safe}}$ 안의 노드는 프루닝 때문에 사라지지 않는다." (정의에 의해 성립)
> * 보장 불가: "충돌률 0%". 충돌은 인식 누락, 플래너 한계, $\tau_{\text{th}}$ 설정, 정리 후 새로 나타나는 물체 등에 따라 달라집니다.
> * 논문 주장은 **"No Pruning 대비, 프루닝 때문에 늘어난 충돌이 통계적으로 유의하지 않다"** 형태로 써야 하고, 그 근거는 실험으로 보여야 합니다.

### 2.2 When: 두 단계로 나눠 정리
| 단계 | 조건 (예시, 조정 필요) | 하는 일 |
| :--- | :--- | :--- |
| Minor (온라인) | 주행 중, 매 주기 | 가벼운 삭제만 (시간 예산 `[측정 예정] ms`) |
| Major (유휴) | 정차, 저속 | 병합, 그래프 구조 정리 |
| Fallback | 메모리 사용률 ≥ 임계값, 또는 직진 순항 중 CPU 여유 | 멈추지 않는 장거리 주행 대비 강제 정리 |

---

## 3. 서론 프레이밍: 하나만 고르기

v3.0("인간 운전자 인지 + 시스템 공학")과 v4.0("초파리 + 쥐")이 서로 다른 방향을 제시했습니다. **현우 연구원이 하나를 골라야 합니다.**

| 선택지 | 장점 | 위험 |
| :--- | :--- | :--- |
| **A. 순수 공학** (Reachable set + 세대별 GC) | 방법론 섹션과 서론이 일치함 | 이야기로서의 인상은 약함 |
| B. 인간 운전자 인지 | *Driving-Aware*라는 이름과 잘 맞음 | 인지과학 근거를 인용해야 함 |
| C. 동물 비유 (초파리, 쥐) | 첫인상이 강함 | 두 동물을 섞으면 "억지 비유" 공격을 받기 쉬움 |

**권장**: 방법론은 A로 쓰고, 서론의 한두 문장에서만 생물학적 동기를 짧게 언급합니다. 인용할 만한 실제 연구 (인용 전 원문 확인 필수):
* D. N. Lee (1976): 광학적 팽창률에서 얻는 time-to-contact $\tau$. TTC 항을 생물학적으로 뒷받침하는 정통 근거.
* G. Tononi & C. Cirelli: 수면 중 시냅스 항상성 가설(SHY). 유휴 시간 정리의 비유로 사용.
* R. L. Davis & Y. Zhong (2017): 능동적 망각의 생물학 (*Neuron* 리뷰).
* RatSLAM(Milford & Wyeth)과는 **다른 연구**임을 명시하세요. RatSLAM은 지도를 *만드는* 연구이고, DAMP는 지도를 *줄이는* 연구입니다.

---

## 4. 반드시 비교해야 할 선행 연구 (Novelty 위치 잡기)

| 연구 | 관련성 |
| :--- | :--- |
| **Clio** (Maggio et al., 2024): task-driven 3D scene graph | 정보 병목(Information Bottleneck)으로 노드를 묶음. **DAMP의 병합과 가장 가까운 경쟁 연구.** 차별점(동역학 기반 안전 집합, 실행 시점 스케줄링)을 분명히 해야 함 |
| **Hydra** (Hughes et al., RSS 2022) | 실시간 3D scene graph의 표준 baseline |
| **Khronos** (Schmid et al., RSS 2024) | 시공간·동적 장면의 장기 표현 |
| Sliding window / Voxel downsampling | 약한 baseline 비판(strawman)을 피하려면 꼭 포함 |

---

## 5. 측정 계획 (아직 측정값 없음)

| 기법 | 메모리 (GB) | 쿼리 지연 (ms) | 프루닝 유발 충돌 증가 | 완주율 |
| :--- | :--- | :--- | :--- | :--- |
| No Pruning | `[측정 예정]` | `[측정 예정]` | 기준 | `[측정 예정]` |
| Sliding Window | `[측정 예정]` | `[측정 예정]` | `[측정 예정]` | `[측정 예정]` |
| Voxel Downsampling | `[측정 예정]` | `[측정 예정]` | `[측정 예정]` | `[측정 예정]` |
| Clio 계열 병합 | `[측정 예정]` | `[측정 예정]` | `[측정 예정]` | `[측정 예정]` |
| **DAMP** | `[측정 예정]` | `[측정 예정]` | `[측정 예정]` | `[측정 예정]` |

**첫 단계 (DAMP 구현보다 먼저)**: 현우 연구원의 현재 메모리 구조로 1~2 km 주행 로그를 돌려 **메모리가 시간에 따라 얼마나 늘어나는지, 어떤 종류의 노드가 많은지**를 측정합니다. 이 그래프가 논문의 동기(Figure 1)이자, 무엇을 지울지 고르는 근거가 됩니다.

---

## 6. 영문 서론 템플릿 (수치는 비워 둠)

> *"Long-horizon robot navigation causes 3D scene graph memory to grow without bound on embedded computers. We propose DAMP (Driving-Aware Memory Pruning), which (i) never prunes nodes inside a kinodynamic safety set defined by stopping distance and time-to-contact, and (ii) schedules lightweight online pruning during motion and structural consolidation during idle periods, with a memory-watermark fallback for non-stop driving. On [datasets / platforms], DAMP reduces memory by [X]% and query latency by [Y]% with no statistically significant increase in pruning-induced collisions compared to an unpruned baseline."*

---
*관련 문서: [[COGNITIVE_BIO_AI_AS2_MASTER_PLAN]]*
