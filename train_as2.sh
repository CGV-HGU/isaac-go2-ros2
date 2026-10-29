#!/usr/bin/env bash
# ==============================================================================
# Train Standalone Unitree AS2 with RSL-RL in Isaac Lab (Isaac Sim 6.2.0)
# Computer Graphics and Vision Lab (CGV)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo " Starting Standalone Unitree AS2 RL Training (Isaac Lab / RSL-RL)..."
echo "======================================================================"

if [ -f "./isaaclab.sh" ]; then
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Velocity-Flat-Unitree-AS2-v0 \
        --headless "$@"
else
    python scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Velocity-Flat-Unitree-AS2-v0 \
        --headless "$@"
fi
