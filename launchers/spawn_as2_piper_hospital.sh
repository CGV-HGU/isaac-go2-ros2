#!/usr/bin/env bash
# ==============================================================================
# Launch Unitree AS2 + AgileX PiPER in Hospital World with Sensors
# Computer Graphics and Vision Lab (CGV)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "======================================================================"
echo " Starting Unitree AS2 + AgileX PiPER in Hospital Environment..."
echo "======================================================================"

if [ -f "./isaaclab.sh" ]; then
    ./isaaclab.sh -p scripts/demos/spawn_as2_piper_hospital.py "$@"
else
    python scripts/demos/spawn_as2_piper_hospital.py "$@"
fi
