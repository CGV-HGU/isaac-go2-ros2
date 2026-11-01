#!/usr/bin/env bash
# ==============================================================================
# Train Unitree AS2 (Solo Base Robot) with RSL-RL in Isaac Lab
# Usage:
#   ./launchers/train_as2.sh --flat
#   ./launchers/train_as2.sh --rough
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

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
echo " [CGV Lab] Unitree AS2 (Solo) RSL-RL Training"
echo " Task : ${TASK}"
echo "======================================================================"

if [ -f "./isaaclab.sh" ]; then
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task "${TASK}" \
        --headless "${EXTRA_ARGS[@]}"
else
    python scripts/reinforcement_learning/rsl_rl/train.py \
        --task "${TASK}" \
        --headless "${EXTRA_ARGS[@]}"
fi
