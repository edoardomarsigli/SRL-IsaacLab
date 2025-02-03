# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())

def scale_reward_weight_episodal(env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str, init_weight: float, final_weight: float):
    """Curriculum that modifies a reward weight linearly between two weights as episode progresses.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        init_weight: The weight of the reward term to begin with.
        final_weight: The weight of the reward at the end of the episode
    """
    # determine episode progress
    # maximum_steps = env.step_dt * env.max_episode_length_s 
    progress = env.common_step_counter/env.max_episode_length 
    
    scaled_reward_weight = (1 - progress) * init_weight + progress * final_weight
    
    term_cfg = env.reward_manager.get_term_cfg(term_name)
    # update term settings
    term_cfg.weight = scaled_reward_weight
    env.reward_manager.set_term_cfg(term_name, term_cfg)

def scale_reward_weight_iteration(env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str, init_weight: float, final_weight: float):
    """Curriculum that modifies a reward weight linearly between two weights as episode progresses.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        init_weight: The weight of the reward term to begin with.
        final_weight: The weight of the reward at the end of the episode
    """
    # determine iteration progress
    # env.scene.cfg.
    maximum_steps = env.step_dt * env.max_episode_length_s 
    progress = env.common_step_counter/maximum_steps 
    
    scaled_reward_weight = (1 - progress) * init_weight + progress * final_weight
    
    term_cfg = env.reward_manager.get_term_cfg(term_name)
    # update term settings
    term_cfg.weight = scaled_reward_weight
    env.reward_manager.set_term_cfg(term_name, term_cfg)