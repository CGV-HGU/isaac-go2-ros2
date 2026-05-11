#!/bin/bash
# Go2 로봇 험지/계단/빠른회전 다목적 강화학습(RL) 학습 스크립트
# 터미널에서 ./train_go2_rough.sh 를 입력하면 백그라운드 머신러닝 학습이 시작됩니다.

echo "🧠 Unitree Go2 (SKRL - Rough Terrain) 훈련을 시작합니다..."
echo "평지뿐만 아니라 계단, 경사로 등 험지를 돌파하고 더 빠른 회전을 배우도록 훈련합니다."
echo "--------------------------------------------------------"

./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
    --task Isaac-Velocity-Rough-Unitree-Go2-v0 \
    --headless \
    --num_envs 4096 
