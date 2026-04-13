#!/bin/bash
# Go2 RL Training Script
# Run this on your 5.1.0 Desktop

echo "Starting Unitree Go2 Training..."
python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-Unitree-Go2-v0 --headless
