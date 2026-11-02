#!/usr/bin/env bash
# ==============================================================================
# AgileX PiPER Robot Arm - Ball Grasping & Dynamic Motion Pre-Check Launcher
# NVIDIA Isaac Sim 6.2.0 & Isaac Lab 3.0
# Computer Graphics and Vision Lab (CGV)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

# Locate Isaac Lab Python launcher
ISAACLAB_EXE=""
if [[ -x "${REPO_ROOT}/isaaclab.sh" ]]; then
    ISAACLAB_EXE="${REPO_ROOT}/isaaclab.sh"
elif command -v isaaclab &>/dev/null; then
    ISAACLAB_EXE="isaaclab"
else
    # Fallback to python3 if running within active Isaac Lab conda/docker environment
    ISAACLAB_EXE="python3"
fi

echo "========================================================================"
echo " [Isaac Lab 3.0] AgileX PiPER Ball Grasping & Dynamic Motion Demo"
echo " Pre-flight Verification before Real Hardware Deployment"
echo " CGV Lab - Computer Graphics and Vision Lab"
echo "========================================================================"

EXTRA_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --as2)
            EXTRA_ARGS+=(--as2_mounted)
            ;;
        --headless)
            EXTRA_ARGS+=(--headless)
            ;;
        *)
            EXTRA_ARGS+=("$arg")
            ;;
    esac
done

if [[ "${ISAACLAB_EXE}" == *"isaaclab.sh"* ]]; then
    exec "${ISAACLAB_EXE}" -p scripts/demos/piper_ball_pick_and_move.py "${EXTRA_ARGS[@]}"
else
    exec "${ISAACLAB_EXE}" scripts/demos/piper_ball_pick_and_move.py "${EXTRA_ARGS[@]}"
fi
