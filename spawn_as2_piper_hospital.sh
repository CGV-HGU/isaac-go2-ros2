#!/usr/bin/env bash
# ==============================================================================
# Launch Unitree AS2 + AgileX PiPER in Hospital Environment with Articulated Door
# Simulation of Mobile Manipulation (Door Opening & Closing) in Isaac Sim 6.2.0
# Computer Graphics and Vision Lab (CGV)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo " Launching Unitree AS2 + AgileX PiPER in Hospital Simulation..."
echo " Scene: Environments/Hospital/hospital.usd + Interactive Articulated Door"
echo " Platform: NVIDIA Isaac Sim 6.2.0 & Isaac Lab"
echo " Computer Graphics and Vision Lab (CGV)"
echo "======================================================================"

if [ -f "./isaaclab.sh" ]; then
    ./isaaclab.sh -p scripts/demos/spawn_as2_piper_hospital.py "$@"
else
    python scripts/demos/spawn_as2_piper_hospital.py "$@"
fi
