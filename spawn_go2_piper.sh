#!/usr/bin/env bash
# ==============================================================================
# Launch Unitree Go2 + AgileX PiPER in NVIDIA Isaac Sim 6.2.0 / Isaac Lab
# Computer Graphics and Vision Lab (CGV)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo " Starting Unitree Go2 + AgileX PiPER in Isaac Sim 6.2.0..."
echo "======================================================================"

# If isaaclab.sh exists, use it; otherwise fallback to python
if [ -f "./isaaclab.sh" ]; then
    ./isaaclab.sh -p scripts/demos/spawn_go2_piper.py "$@"
else
    python scripts/demos/spawn_go2_piper.py "$@"
fi
