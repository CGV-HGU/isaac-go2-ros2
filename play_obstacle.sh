#!/bin/bash
# Go2 로봇 장애물 투척 테스트 전용 Play 스크립트
# 터미널에서 ./play_obstacle.sh 를 입력하면 실행됩니다. (T키로 장애물 스폰)

echo "🚀 Unitree Go2 동적 장애물 테스트 환경(Play)을 시작합니다..."

# 사용자님이 요청하신 정확한 명령어 (캡스톤 전용 환경 로드)
./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py --task Isaac-Velocity-Capstone-Unitree-Go2-Play-v0 --num_envs 1 +checkpoint="logs/skrl/unitree_go2_flat/2026-04-1413-12-51_ppo_torch/checkpoints/best_agent.pt"
