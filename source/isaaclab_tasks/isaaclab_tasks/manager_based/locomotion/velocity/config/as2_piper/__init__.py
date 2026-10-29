# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
from . import agents

##
# Register Gym environments for Unitree AS2 + AgileX PiPER
##

gym.register(
    id="Isaac-Velocity-Flat-Unitree-AS2-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeAS2PiperFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeAS2PiperFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Unitree-AS2-Piper-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeAS2PiperFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeAS2PiperFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Unitree-AS2-Piper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeAS2PiperRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeAS2PiperRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Unitree-AS2-Piper-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeAS2PiperRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeAS2PiperRoughPPORunnerCfg",
    },
)
