#!/usr/bin/env bash
# ==============================================================================
# Train Standalone Unitree AS2 with RSL-RL in Isaac Lab
# Usage:
#   ./train_as2.sh --flat
#   ./train_as2.sh --rough
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TASK="Isaac-Velocity-Flat-Unitree-AS2-v0"
EXTRA_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --flat)
            TASK="Isaac-Velocity-Flat-Unitree-AS2-v0"
            ;;
        --rough)
            TASK="Isaac-Velocity-Rough-Unitree-AS2-v0"
            ;;
        *)
            EXTRA_ARGS+=("$arg")
            ;;
    esac
done

echo "======================================================================"
echo " [CGV Lab] Standalone Unitree AS2 RSL-RL Training"
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
