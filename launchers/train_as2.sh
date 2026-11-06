#!/usr/bin/env bash
# ==============================================================================
# Train Unitree AS2 (Solo Base Robot) with RSL-RL in Isaac Lab
# Computer Graphics and Vision Lab (CGV)
#
# Usage:
#   ./launchers/train_as2.sh --flat
#   ./launchers/train_as2.sh --rough
#   ./launchers/train_as2.sh --rough --num_envs 2048
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

TASK="Isaac-Velocity-Flat-Unitree-AS2-v0"
EXTRA_ARGS=()
HEADLESS_FLAG="--headless"
SKIP_PREFLIGHT=0

for arg in "$@"; do
    case "$arg" in
        --flat)
            TASK="Isaac-Velocity-Flat-Unitree-AS2-v0"
            ;;
        --rough)
            TASK="Isaac-Velocity-Rough-Unitree-AS2-v0"
            ;;
        --gui|--no-headless)
            HEADLESS_FLAG=""
            ;;
        --headless)
            HEADLESS_FLAG="--headless"
            ;;
        --skip-preflight|--no-preflight)
            SKIP_PREFLIGHT=1
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

ISAACLAB_CMD=()
if [[ -x "${REPO_ROOT}/isaaclab.sh" ]]; then
    ISAACLAB_CMD=("${REPO_ROOT}/isaaclab.sh" "-p")
elif [[ -f "${REPO_ROOT}/isaaclab.sh" ]]; then
    ISAACLAB_CMD=("bash" "${REPO_ROOT}/isaaclab.sh" "-p")
elif command -v isaaclab &>/dev/null; then
    ISAACLAB_CMD=("isaaclab" "-p")
elif command -v python3 &>/dev/null; then
    ISAACLAB_CMD=("python3")
else
    ISAACLAB_CMD=("python")
fi

# Run preflight GPU verification unless bypassed
if [[ "${SKIP_PREFLIGHT}" -eq 0 ]]; then
    "${ISAACLAB_CMD[@]}" scripts/tools/preflight_check.py --require-gpu
fi

RUN_ARGS=("--task" "${TASK}")
if [[ -n "${HEADLESS_FLAG}" ]]; then
    RUN_ARGS+=("${HEADLESS_FLAG}")
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    RUN_ARGS+=("${EXTRA_ARGS[@]}")
fi

exec "${ISAACLAB_CMD[@]}" scripts/reinforcement_learning/rsl_rl/train.py "${RUN_ARGS[@]}"
