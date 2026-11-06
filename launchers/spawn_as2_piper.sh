#!/usr/bin/env bash
# ==============================================================================
# Launch Unitree AS2 + AgileX PiPER in NVIDIA Isaac Sim 6.2.0 / Isaac Lab
# Computer Graphics and Vision Lab (CGV)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "======================================================================"
echo " Starting Unitree AS2 + AgileX PiPER in Isaac Sim 6.2.0..."
echo " Computer Graphics and Vision Lab (CGV)"
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

exec "${ISAACLAB_CMD[@]}" scripts/demos/spawn_as2_piper.py "$@"
