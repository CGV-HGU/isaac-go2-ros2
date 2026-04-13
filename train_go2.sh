#!/bin/bash
# Go2 로봇 평지 보행 강화학습(RL) 실행 스크립트
# 5.1.0 데스크탑에서 이 파일을 실행하면 바로 학습이 시작됩니다.

echo "🚀 Unitree Go2 강화학습(Training)을 시작합니다..."
python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-Unitree-Go2-v0 --headless
