# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""5-second pre-flight readiness check for GPU training server (cgv-server-02).

Run before starting RL training:
    python scripts/tools/preflight_check.py
    ./isaaclab.sh -p scripts/tools/preflight_check.py
"""

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET


def _check(label: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    suffix = f" ({detail})" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    return ok


def main() -> int:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    print("=" * 72)
    print(" [Pre-Flight Check] Unitree AS2 + AgileX PiPER Training Readiness")
    print(f" Repository : {repo_root}")
    print("=" * 72)

    all_ok = True

    # 1. Unified 20-DOF URDF & Door URDFs
    urdf_path = os.path.join(repo_root, "robot_arms", "piper", "descriptions", "as2_piper.urdf")
    if os.path.isfile(urdf_path):
        tree = ET.parse(urdf_path)
        active = [
            j.attrib["name"]
            for j in tree.findall("joint")
            if j.attrib.get("type") in ("revolute", "prismatic", "continuous")
        ]
        all_ok &= _check("Unified AS2+PiPER URDF", len(active) == 20, f"{len(active)} active joints")
        sensor_links = {
            lnk.attrib["name"]: lnk
            for lnk in tree.findall("link")
            if lnk.attrib["name"] in ("front_camera", "lidar_link", "imu", "radar")
        }
        sensors_ok = len(sensor_links) == 4 and all(lnk.find("inertial") is not None for lnk in sensor_links.values())
        all_ok &= _check("AS2 Sensor Mounts (RGB-D + 3D LiDAR + IMU + Radar)", sensors_ok, "4 links verified")
    else:
        all_ok &= _check("Unified AS2+PiPER URDF", False, f"Missing: {urdf_path}")

    for door_file in ("hospital_door.urdf", "hospital_door_pull.urdf", "hospital_door_sliding.urdf"):
        p = os.path.join(repo_root, "source", "isaaclab_assets", "data", "props", "door", door_file)
        all_ok &= _check(f"Door asset '{door_file}'", os.path.isfile(p))

    for mod_label, rel_path in (
        ("Isaac Lab Sensor Suite (as2_sensors.py)", "source/isaaclab_assets/isaaclab_assets/robots/as2_sensors.py"),
        ("ROS 2 RGB-D/LiDAR Door Detector", "ros2_ws/src/as2_piper_bringup/as2_piper_bringup/vision_door_detector.py"),
        ("ROS 2 Sim-to-Real Whole-Body Bridge", "ros2_ws/src/as2_piper_bringup/as2_piper_bringup/sim2real_bridge_node.py"),
        ("Paper/Capstone Benchmark Evaluator", "scripts/tools/evaluate_loco_manipulation.py"),
    ):
        all_ok &= _check(mod_label, os.path.isfile(os.path.join(repo_root, *rel_path.split("/"))))

    # 2. Submodule piper_v1.usd
    piper_usd = os.path.join(repo_root, "robot_arms", "piper", "piper_isaac_sim", "USD", "piper_v1.usd")
    all_ok &= _check(
        "PiPER Submodule USD",
        os.path.isfile(piper_usd),
        "OK" if os.path.isfile(piper_usd) else "Run 'bash robot_arms/piper/setup_piper.sh'",
    )

    # 3. Shell script LF line-ending check
    sh_files = ["train_as2_piper.sh", "train_as2.sh", "play_as2_piper.sh", "spawn_as2_piper.sh", "spawn_as2_piper_hospital.sh"]
    crlf_found = []
    for sh in sh_files:
        sp = os.path.join(repo_root, sh)
        if os.path.isfile(sp):
            with open(sp, "rb") as f:
                if b"\r\n" in f.read():
                    crlf_found.append(sh)
    all_ok &= _check("Linux LF line endings on .sh scripts", len(crlf_found) == 0, f"CRLF: {crlf_found}" if crlf_found else "all LF")

    # 4. PyTorch & CUDA availability
    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "No CUDA GPU detected (offline host)"
        _check("PyTorch + CUDA GPU", cuda_ok, f"torch {torch.__version__}, {gpu_name}")
        if not cuda_ok and "--require-gpu" in sys.argv:
            all_ok = False
    except ImportError:
        _check("PyTorch import", False, "Activate Isaac Lab Python environment")
        if "--require-gpu" in sys.argv:
            all_ok = False

    print("=" * 72)
    if all_ok:
        print(" [READY] Pre-flight check passed. Ready to launch './train_as2_piper.sh --flat'!")
        return 0
    print(" [ACTION REQUIRED] Resolve FAIL items above before launching training.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
