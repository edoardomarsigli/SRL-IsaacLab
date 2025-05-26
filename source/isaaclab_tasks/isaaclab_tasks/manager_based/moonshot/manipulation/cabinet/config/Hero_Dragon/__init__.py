# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

##
# Joint Position Control - HeroDragon
##

gym.register(
    id="Manipulation-HeroDragon-v0",
    entry_point="isaaclab_tasks.manager_based.moonshot.manipulation.cabinet.HeroDragonGraspEnv:HeroDragonGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:HeroDragonGraspEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HeroDragonGraspPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Manipulation-HeroDragon-Play-v0",
    entry_point="isaaclab_tasks.manager_based.moonshot.manipulation.cabinet.HeroDragonGraspEnv:HeroDragonGraspEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:HeroDragonGraspEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HeroDragonGraspPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)


gym.register(
    id="Manipulation-HeroDragon-Play-v0-madrl",
    entry_point="isaaclab_tasks.manager_based.moonshot.manipulation.cabinet.HeroDragonGraspEnv:HeroDragonGraspEnvMadrl",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:HeroDragonGraspEnvCfg_PLAYMadrl",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HeroDragonGraspPPORunnerCfgMadrl",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfgMadrl.yaml",
    },
    disable_env_checker=True,
)