from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers.command_manager import CommandTerm


def root_height_above_maximum(
    env: ManagerBasedRLEnv, maximum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's root height is above the maximum height.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] > maximum_height

def root_roll_above_threshold(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's roll is above the threshold.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    quat = asset.data.root_quat_w
    if quat.ndim == 1:  
        quat = quat.unsqueeze(0) 
    roll, _, _ = math_utils.euler_xyz_from_quat(quat)

    return torch.abs(math_utils.wrap_to_pi(roll)) > threshold

def get_to_goal(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate when goal has been reached."""
    command = env.command_manager.get_command(command_name)
    asset: RigidObject = env.scene[asset_cfg.name]
    goal = command[:, :2]
    if torch.abs(goal - asset.data.root_pose_w[:, :2]) > 0.05:
        return 1
    else:
        return 0