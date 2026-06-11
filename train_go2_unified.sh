#!/bin/bash
# Go2 로봇 평지/험지 통합 만능 보행 강화학습(RL) 학습 스크립트
# 터미널에서 ./train_go2_unified.sh 를 입력하면 학습이 시작됩니다.

echo "========================================================"
echo "🧠 Unitree Go2 통합(Unified) 보행 학습을 시작합니다..."
echo "평지 고속 회전 및 장애물/계단 극복 능력을 동시에 훈련합니다."
echo "--------------------------------------------------------"

# 1. 험지 및 평지 통합 환경(Rough)에서 3000만 스텝 학습 진행
# 2. 뇌 크기는 [512, 256, 128]을 활용하여 고차원 거동을 학습합니다.
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
    --task Isaac-Velocity-Rough-Unitree-Go2-v0 \
    --headless \
    --num_envs 4096 

# 3. 학습 완료 후 최종 훈련 분석 이미지 리포트 자동 생성
python generate_report.py
