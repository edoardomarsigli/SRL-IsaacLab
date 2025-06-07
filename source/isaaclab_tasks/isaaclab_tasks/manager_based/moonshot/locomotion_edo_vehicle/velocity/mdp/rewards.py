# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster

from . import observations as obs

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv



class power_consumption(ManagerTermBase):
    """Penalty for the power consumed by the actions to the environment.

    This is computed as commanded torque times the joint velocity.
    """

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # add default argument
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]

        # resolve the gear ratio for each joint
        self.gear_ratio = torch.ones(env.num_envs, asset.num_joints, device=env.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(
            cfg.params["gear_ratio"], asset.joint_names
        )
        self.gear_ratio[:, index_list] = torch.tensor(value_list, device=env.device)
        self.gear_ratio_scaled = self.gear_ratio / torch.max(self.gear_ratio)

    def __call__(
        self, env: ManagerBasedRLEnv, gear_ratio: dict[str, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # return power = torque * velocity (here actions: joint torques)
        return torch.sum(torch.abs(env.action_manager.action * asset.data.joint_vel * self.gear_ratio_scaled), dim=-1)




def track_lin_vel_xy_exp_vehicle(
    env: ManagerBasedRLEnv, std: float, command_name: str, body_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel of specific body."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    body_link_idx = asset.find_bodies(body_name)[0][0]
    body_lin_vel_w = asset.data.body_lin_vel_w[:, body_link_idx, :]
    body_quat_w = asset.data.body_quat_w[:, body_link_idx, :]
    if body_name == "leg1link2":
        tf_d_matrix = torch.tensor([[-1,0,0],[0,0,1],[0,1,0]], device = env.device)
        tf_d_matrix_expanded = tf_d_matrix.unsqueeze(0).expand(env.num_envs, -1, -1)
        tf_d_quat = math_utils.quat_from_matrix(tf_d_matrix_expanded)
    elif body_name == "leg1link4":
        tf_d_matrix = torch.tensor([[0,0,-1],[-1,0,0],[0,1,0]], device = env.device)
        tf_d_matrix_expanded = tf_d_matrix.unsqueeze(0).expand(env.num_envs, -1, -1)
        tf_d_quat = math_utils.quat_from_matrix(tf_d_matrix_expanded)
    else:
        raise ValueError(f"Unexpected link name: {body_name}")

    quat_w_d = math_utils.quat_mul(body_quat_w, tf_d_quat)
    
    body_lin_vel_d = math_utils.quat_rotate_inverse(
        quat_w_d, body_lin_vel_w
    )

    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - body_lin_vel_d[:, :2]),
        dim=1,
    )
    
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_exp_vehicle(env: ManagerBasedRLEnv, std: float, command_name: str, body_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    body_link_idx = asset.find_bodies(body_name)[0][0]
    body_ang_vel_w = asset.data.body_ang_vel_w[:, body_link_idx, :]

    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - body_ang_vel_w[:, 2])

    return torch.exp(-ang_vel_error / std**2)


def joint_torques_vehicle_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize leg joint torques applied on the articulation using L2 squared kernel.

    Excludes any wheel joint torques.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    all_joint_indices = range(len(asset.joint_names))
    
    wheel_joint_names = [
        "wheel11_left_joint",
        "wheel11_right_joint",
        "wheel12_left_joint",
        "wheel12_right_joint",
    ] 

    wheel_joint_idx = {asset.find_joints(name)[0][0] for name in wheel_joint_names}
    # if not a wheel joint, then it's a leg joint, which makes it applicable for any number of leg joints
    leg_joint_idx = [idx for idx in all_joint_indices if idx not in wheel_joint_idx]

    return torch.sum(torch.square(asset.data.applied_torque[:, leg_joint_idx]), dim=1)



def upright_wheel_bodies_angle(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    # upright height difference (desired height difference)

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    front_body_link_idx = asset.find_bodies("leg1link2")[0][0]
    rear_body_link_idx = asset.find_bodies("leg1link6")[0][0]
    wheel_joint_names = ["wheel11_left","wheel11_right",
                         "wheel12_left","wheel12_right"] 
    front_wheel_joint_idx =  [asset.find_bodies(name)[0][0] for name in wheel_joint_names[:2]]
    front_wheel_center = torch.mean(asset.data.body_pos_w[:, front_wheel_joint_idx, :], dim = 1)
    front_body = asset.data.body_pos_w[:,front_body_link_idx, :]
    front_diff = front_body - front_wheel_center
    front_diff_norm = torch.norm(front_diff,dim=1, keepdim=True)
    # avoid division by 0
    epsilon = 1e-8
    front_diff_normalized = front_diff / (front_diff_norm + epsilon)
    front_z = front_diff_normalized[:, 2]
    front_angles = torch.acos(torch.clamp(front_z, -1.0, 1.0))
    
    rear_wheel_joint_idx =  [asset.find_bodies(name)[0][0] for name in wheel_joint_names[2:]]    
    rear_wheel_center = torch.mean(asset.data.body_pos_w[:, rear_wheel_joint_idx, :], dim = 1)
    rear_body = asset.data.body_pos_w[:,rear_body_link_idx, :]
    rear_diff = rear_body - rear_wheel_center
    rear_diff_norm = torch.norm(rear_diff,dim=1, keepdim=True)
    # avoid division by 0
    epsilon = 1e-8
    rear_diff_normalized = rear_diff / (rear_diff_norm + epsilon)
    rear_z = rear_diff_normalized[:, 2]
    rear_angles = torch.acos(torch.clamp(rear_z, -1.0, 1.0))

    error = torch.abs(front_angles) + torch.abs(rear_angles)
    return error
    return torch.exp(-error / std**2)