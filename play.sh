#!/bin/bash
# Go2 로봇 평지 보행 강화학습(RL) Play 실행 스크립트
# 터미널에서 ./play.sh 를 입력하면 바로 실행됩니다.

echo "🚀 Unitree Go2 시뮬레이션(Play)을 시작합니다..."

# 사용자님이 요청하신 정확한 명령어
./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py --task Isaac-Velocity-Flat-Unitree-Go2-v0 --num_envs 1 +checkpoint="logs/skrl/unitree_go2_flat/2026-04-1413-12-51_ppo_torch/checkpoints/best_agent.pt"
