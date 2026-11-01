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
echo "======================================================================"

if [ -f "./isaaclab.sh" ]; then
    ./isaaclab.sh -p scripts/demos/spawn_as2_piper.py "$@"
else
    python scripts/demos/spawn_as2_piper.py "$@"
fi
