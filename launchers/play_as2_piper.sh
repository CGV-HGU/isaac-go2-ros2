#!/usr/bin/env bash
# ==============================================================================
# Play Trained Unitree AS2 + AgileX PiPER Policy (WASD + Arm Teleop + Key '1' Door)
# Computer Graphics and Vision Lab (CGV)
#
# Usage:
#   ./launchers/play_as2_piper.sh --flat --spawn_door --door_type lever_push --real-time
#   ./launchers/play_as2_piper.sh --rough --spawn_door --door_type sliding --real-time
#   ./launchers/play_as2_piper.sh --flat --headless --video   # SSH headless MP4 verification
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

TASK="Isaac-Velocity-Flat-Unitree-AS2-Piper-Play-v0"
EXTRA_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --flat)
            TASK="Isaac-Velocity-Flat-Unitree-AS2-Piper-Play-v0"
            ;;
        --rough)
            TASK="Isaac-Velocity-Rough-Unitree-AS2-Piper-Play-v0"
            ;;
        *)
            EXTRA_ARGS+=("$arg")
            ;;
    esac
done

echo "======================================================================"
echo " [CGV Lab] Unitree AS2 + AgileX PiPER Teleop & Door Manipulation Play"
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

RUN_ARGS=("--task" "${TASK}")
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    RUN_ARGS+=("${EXTRA_ARGS[@]}")
fi

exec "${ISAACLAB_CMD[@]}" scripts/reinforcement_learning/rsl_rl/play.py "${RUN_ARGS[@]}"
