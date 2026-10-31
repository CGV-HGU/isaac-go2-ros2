#!/usr/bin/env bash
# ==============================================================================
# Play Trained Unitree AS2 + AgileX PiPER Policy (WASD + Arm Teleop + Key '1' Door)
# Usage:
#   ./play_as2_piper.sh --flat --spawn_door --door_type lever_push --real-time
#   ./play_as2_piper.sh --rough --spawn_door --door_type sliding --real-time
#   ./play_as2_piper.sh --flat --headless --video   # SSH headless MP4 verification
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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

if [ -f "./isaaclab.sh" ]; then
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
        --task "${TASK}" "${EXTRA_ARGS[@]}"
else
    python scripts/reinforcement_learning/rsl_rl/play.py \
        --task "${TASK}" "${EXTRA_ARGS[@]}"
fi
