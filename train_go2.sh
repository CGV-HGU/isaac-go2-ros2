#!/bin/bash
# Go2 로봇 평지 보행 강화학습(RL) 학습(Train) 스크립트
# 터미널에서 ./train_go2.sh 를 입력하면 백그라운드 머신러닝 학습이 시작됩니다.

echo "🧠 Unitree Go2 (SKRL) 훈련을 시작합니다..."
echo "빠른 학습을 위해 화면 없는(Headless) 모드로 수천 마리의 로봇이 동시에 달립니다."
echo "--------------------------------------------------------"

# 1. 기존의 평지 환경(Flat)에서 SKRL 라이브러리를 사용하여 학습 시작
# 2. --headless 옵션을 넣으면 Isaac Sim 화면이 안 뜨고 GPU 성능을 100% 학습에만 몰빵합니다.
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
    --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
    --headless \
    --num_envs 4096 
