#!/usr/bin/env bash
# ==============================================================================
# Train Unitree AS2 + AgileX PiPER with RSL-RL in Isaac Lab (Isaac Sim 6.2.0)
# Computer Graphics and Vision Lab (CGV)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo " Starting Unitree AS2 + AgileX PiPER RL Training (Isaac Lab / RSL-RL)..."
echo "======================================================================"

if [ -f "./isaaclab.sh" ]; then
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Velocity-Flat-Unitree-AS2-Piper-v0 \
        --headless "$@"
else
    python scripts/reinforcement_learning/rsl_rl/train.py \
        --task Isaac-Velocity-Flat-Unitree-AS2-Piper-v0 \
        --headless "$@"
fi
