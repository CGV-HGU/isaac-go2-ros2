#!/bin/bash
# [Blackwell Optimized] Go2 RL Training Script for cgv-server-02
# This script is tuned for ultra-high performance on NVIDIA RTX PRO 6000 Blackwell.

echo "========================================="
echo "🚀 Starting Blackwell Optimized Go2 Training"
echo "🔧 GPU: Dual NVIDIA RTX PRO 6000 (96GB VRAM)"
echo "🤖 Environments: 16384 (Massive Parallelism)"
echo "========================================="

# --num_envs 16384: Blackwell의 96GB VRAM을 활용해 대규모 병렬 학습 수행
# --headless: 서버 환경 최적화
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
    --num_envs 16384 \
    --headless
