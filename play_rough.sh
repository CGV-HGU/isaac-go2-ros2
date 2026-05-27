#!/bin/bash
# Go2 로봇 험지/계단/빠른회전 다목적 강화학습(RL) Play 실행 스크립트
# 터미널에서 ./play_rough.sh 를 입력하면 바로 실행됩니다.

echo "🚀 Unitree Go2 험지(Rough) 시뮬레이션 테스트를 시작합니다..."

# 학습된 험지 모델을 테스트하는 명령어
./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py --task Isaac-Velocity-Rough-Unitree-Go2-Play-v0 --num_envs 1 +checkpoint="/home/hayoung/IsaacLab/logs/skrl/unitree_go2_rough/2026-05-26_19-00-21_ppo_torch/checkpoints/best_agent.pt"
