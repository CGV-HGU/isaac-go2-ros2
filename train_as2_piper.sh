#!/usr/bin/env bash
# ==============================================================================
# Train Unitree AS2 + AgileX PiPER with RSL-RL in Isaac Lab
# Usage:
#   ./train_as2_piper.sh --flat               # Stage 1: Flat baseline (500 iter, ~12 min)
#   ./train_as2_piper.sh --rough              # Stage 2: Rough + Arm swing balance (1500 iter, ~30 min)
#   ./train_as2_piper.sh --rough --num_envs 2048
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TASK="Isaac-Velocity-Flat-Unitree-AS2-Piper-v0"
EXTRA_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --flat)
            TASK="Isaac-Velocity-Flat-Unitree-AS2-Piper-v0"
            ;;
        --rough)
            TASK="Isaac-Velocity-Rough-Unitree-AS2-Piper-v0"
            ;;
        *)
            EXTRA_ARGS+=("$arg")
            ;;
    esac
done

echo "======================================================================"
echo " [CGV Lab] Unitree AS2 + AgileX PiPER RSL-RL Training"
echo " Task : ${TASK}"
echo "======================================================================"

if [ -f "./isaaclab.sh" ]; then
    ./isaaclab.sh -p scripts/tools/preflight_check.py --require-gpu
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task "${TASK}" \
        --headless "${EXTRA_ARGS[@]}"
else
    python scripts/tools/preflight_check.py --require-gpu
    python scripts/reinforcement_learning/rsl_rl/train.py \
        --task "${TASK}" \
        --headless "${EXTRA_ARGS[@]}"
fi
