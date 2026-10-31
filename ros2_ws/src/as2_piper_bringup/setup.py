from glob import glob
import os
from setuptools import find_packages, setup

package_name = "as2_piper_bringup"

# Bundle door_manipulation.py into share/as2_piper_bringup/ when building inside the repo
# so installed deployments on the physical robot PC run standalone without Isaac Lab source.
extra_share_files = ["package.xml", "README.md"]
repo_door_manip = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "source",
        "isaaclab_assets",
        "isaaclab_assets",
        "robots",
        "door_manipulation.py",
    )
)
if os.path.isfile(repo_door_manip):
    extra_share_files.append(repo_door_manip)

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", extra_share_files),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="CGV Lab",
    maintainer_email="cgv@handong.edu",
    description="ROS 2 Humble RGB-D + 3D LiDAR door perception & Sim-to-Real controller for Unitree AS2 + PiPER.",
    license="BSD-3-Clause",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "vision_door_detector = as2_piper_bringup.vision_door_detector:main",
            "sim2real_bridge_node = as2_piper_bringup.sim2real_bridge_node:main",
        ],
    },
)
