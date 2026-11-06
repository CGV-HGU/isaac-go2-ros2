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

echo "========================================================================"
echo " [Isaac Lab 3.0] AgileX PiPER Ball Grasping & Dynamic Motion Demo"
echo " Pre-flight Verification before Real Hardware Deployment"
echo " Computer Graphics and Vision Lab (CGV)"
echo "========================================================================"

EXTRA_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --as2)
            EXTRA_ARGS+=(--as2_mounted)
            ;;
        --t1)
            EXTRA_ARGS+=(--trajectory_csv "experiments/bird_strike/trajectories/trajectory_t1_step.csv")
            ;;
        --t2)
            EXTRA_ARGS+=(--trajectory_csv "experiments/bird_strike/trajectories/trajectory_t2_sweep.csv")
            ;;
        --headless)
            EXTRA_ARGS+=(--headless)
            ;;
        *)
            EXTRA_ARGS+=("$arg")
            ;;
    esac
done

exec "${ISAACLAB_CMD[@]}" experiments/bird_strike/sim_piper_ball_pick_and_move.py "${EXTRA_ARGS[@]}"
