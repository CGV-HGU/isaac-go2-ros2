#!/usr/bin/env bash
# ==============================================================================
# Bird Strike (BS) Experiment - Isaac Sim Pre-Check Launcher
# NVIDIA Isaac Sim 6.2.0 & Isaac Lab 3.0
# Computer Graphics and Vision Lab (CGV)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"
exec ./launchers/run_piper_ball_demo.sh "$@"
