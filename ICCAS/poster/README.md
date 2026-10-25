# ICCAS 2026 Academic Poster Project

This folder contains the official A0 poster generation pipeline, source assets, and **final print deliverables** for the ICCAS 2026 conference paper:
**"Bridging the Sim-to-Real Gap in Quadrupedal Visual Navigation via 3D Gaussian Splatting"**

---

## 🚀 Quick Access: Final Deliverables (최종 결과물 바로가기)
> **최종 출력 및 발표용 파일은 [`final/`](final/) 폴더에 모두 정리되어 있습니다.**
> - 📄 **인쇄 제출용 PDF**: [`final/ICCAS_자율주행팀포스터.pdf`](final/)
> - 📊 **편집용 PPTX**: [`final/ICCAS2026_Poster_Quadruped.pptx`](final/ICCAS2026_Poster_Quadruped.pptx)
> - 🖼️ **고해상도 프리뷰**: [`final/poster_preview.png`](final/poster_preview.png)

---

## 📁 Directory Structure (폴더 구조)

```
ICCAS/poster/
├── final/                              # ★ 최종본 전용 폴더 (접근 용이)
│   ├── ICCAS_자율주행팀포스터.pdf       # 이미지스테이션 인쇄용 PDF (한글명)
│   ├── ICCAS2026_Poster_Quadruped.pdf  # 인쇄용 PDF (영문명)
│   ├── ICCAS2026_Poster_Quadruped.pptx # 최종 수정본 파워포인트
│   ├── poster_preview.png              # 최종 고해상도 미리보기
│   └── README.md
├── sample/                             # 학회 공식 샘플 템플릿
│   └── ICCAS2026_Poster-Sample.pptx
├── assets/                             # 고해상도 그림, 크롭 이미지, 공식 로고
├── make_poster.py                      # python-pptx 기반 A0 포스터 빌드 스크립트
├── render.sh                           # LibreOffice / COM 렌더링 스크립트
├── ICC2026_Poster_Quadruped.pptx       # 루트 동기화 PPTX
├── ICCAS2026_Poster_Quadruped.pdf      # 루트 동기화 PDF
└── README.md
```

---

## 🖼️ Poster Preview (최신 최종본)
![ICCAS 2026 Poster Preview](poster_preview_75dpi.png)

---

## ⚙️ How to Re-generate / Modify
- **Session Code**: `FrPo6P.32`
- **Slide Dimensions**: A0 portrait (`84.10 cm × 118.00 cm`)
- 직접 파워포인트에서 편집하거나, `make_poster.py` 스크립트를 통해 프로그래밍 방식으로 재생성할 수 있습니다.
