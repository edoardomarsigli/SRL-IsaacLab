from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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

def joint_vel_out_of_manual_limit_vehicle(
    env: ManagerBasedRLEnv, max_velocity: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's joint velocities are outside the provided limits."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    body_joint_names = ["leg1joint[1-7]"] 
    body_joint_idx =  [asset.find_joints(name)[0][0] for name in body_joint_names]
    # compute any violations
    return torch.any(torch.abs(asset.data.joint_vel[:, body_joint_idx]) > max_velocity, dim=1)

def terminate_low_joint(env: ManagerBasedRLEnv) -> torch.Tensor:

    joint4_pos = env.scene["joint4_frame"].data.target_pos_w[..., 0, :]    # [N, 3]
    joint4_z = joint4_pos[:, 2]  # [N]

    return joint4_z < 0.10  # Terminate if joint4 is lower than 0.1m from the ground

def terminate_wheel_z(env: ManagerBasedRLEnv) -> torch.Tensor:
    try:
        front_pos = env.scene["front_wheel_frame"].data.target_pos_w[..., 0, :]
        front_z = front_pos[..., 2]  # prende asse z

        rear_pos = env.scene["rear_wheel_frame"].data.target_pos_w[..., 0, :]
        rear_z = rear_pos[..., 2]  # prende asse z

        pen_front = front_z > 0.4
        pen_rear = rear_z > 0.4
        
        return (pen_front | pen_rear)
    except KeyError:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
