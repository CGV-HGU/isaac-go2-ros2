#!/bin/bash
# Go2 로봇 캡스톤 환경(Custom USD) Play 실행 스크립트
# 터미널에서 ./play_capstone.sh 를 입력하면 바로 실행됩니다.

echo "🚀 Unitree Go2 캡스톤 커스텀 시뮬레이션(Play)을 시작합니다..."

# 4월 14일 가장 안정적이었던 모델을 절대 경로로 불러옵니다.
./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py --task Isaac-Velocity-Capstone-Unitree-Go2-Play-v0 --num_envs 1 --checkpoint "/home/hayoung/IsaacLab/logs/skrl/unitree_go2_flat/2026-04-14_00-20-39_ppo_torch/checkpoints/best_agent.pt"
