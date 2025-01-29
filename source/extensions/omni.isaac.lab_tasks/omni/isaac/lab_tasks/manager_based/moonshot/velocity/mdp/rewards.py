# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
import omni.isaac.lab.utils.string as string_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor

from . import observations as obs

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def upright_posture_bonus(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for maintaining an upright posture."""
    up_proj = obs.base_up_proj(env, asset_cfg).squeeze(-1)
    return (up_proj > threshold).float()


def move_to_target_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    target_pos: tuple[float, float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for moving to the target heading."""
    heading_proj = obs.base_heading_proj(env, target_pos, asset_cfg).squeeze(-1)
    return torch.where(heading_proj > threshold, 1.0, heading_proj / threshold)


class progress_reward(ManagerTermBase):
    """Reward for making progress towards the target."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # initialize the base class
        super().__init__(cfg, env)
        # create history buffer
        self.potentials = torch.zeros(env.num_envs, device=env.device)
        self.prev_potentials = torch.zeros_like(self.potentials)

    def reset(self, env_ids: torch.Tensor):
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = self._env.scene["robot"]
        # compute projection of current heading to desired heading vector
        target_pos = torch.tensor(self.cfg.params["target_pos"], device=self.device)
        to_target_pos = target_pos - asset.data.root_pos_w[env_ids, :3]
        # reward terms
        self.potentials[env_ids] = -torch.norm(to_target_pos, p=2, dim=-1) / self._env.step_dt
        self.prev_potentials[env_ids] = self.potentials[env_ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        target_pos: tuple[float, float, float],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # compute vector to target
        target_pos = torch.tensor(target_pos, device=env.device)
        to_target_pos = target_pos - asset.data.root_pos_w[:, :3]
        to_target_pos[:, 2] = 0.0
        # update history buffer and compute new potential
        self.prev_potentials[:] = self.potentials[:]
        self.potentials[:] = -torch.norm(to_target_pos, p=2, dim=-1) / env.step_dt

        return self.potentials - self.prev_potentials


class joint_limits_penalty_ratio(ManagerTermBase):
    """Penalty for violating joint limits weighted by the gear ratio."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # add default argument
        if "asset_cfg" not in cfg.params:
            cfg.params["asset_cfg"] = SceneEntityCfg("robot")
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # resolve the gear ratio for each joint
        self.gear_ratio = torch.ones(env.num_envs, asset.num_joints, device=env.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(
            cfg.params["gear_ratio"], asset.joint_names
        )
        self.gear_ratio[:, index_list] = torch.tensor(value_list, device=env.device)
        self.gear_ratio_scaled = self.gear_ratio / torch.max(self.gear_ratio)

    def __call__(
        self, env: ManagerBasedRLEnv, threshold: float, gear_ratio: dict[str, float], asset_cfg: SceneEntityCfg
    ) -> torch.Tensor:
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # compute the penalty over normalized joints
        joint_pos_scaled = math_utils.scale_transform(
            asset.data.joint_pos, asset.data.soft_joint_pos_limits[..., 0], asset.data.soft_joint_pos_limits[..., 1]
        )
        # scale the violation amount by the gear ratio
        violation_amount = (torch.abs(joint_pos_scaled) - threshold) / (1 - threshold)
        violation_amount = violation_amount * self.gear_ratio_scaled

        return torch.sum((torch.abs(joint_pos_scaled) > threshold) * violation_amount, dim=-1)


class power_consumption(ManagerTermBase):
    """Penalty for the power consumed by the actions to the environment.

    This is computed as commanded torque times the joint velocity.
    """

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # add default argument
        if "asset_cfg" not in cfg.params:
            cfg.params["asset_cfg"] = SceneEntityCfg("robot")
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # resolve the gear ratio for each joint
        self.gear_ratio = torch.ones(env.num_envs, asset.num_joints, device=env.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(
            cfg.params["gear_ratio"], asset.joint_names
        )
        self.gear_ratio[:, index_list] = torch.tensor(value_list, device=env.device)
        self.gear_ratio_scaled = self.gear_ratio / torch.max(self.gear_ratio)

    def __call__(self, env: ManagerBasedRLEnv, gear_ratio: dict[str, float], asset_cfg: SceneEntityCfg) -> torch.Tensor:
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # return power = torque * velocity (here actions: joint torques)
        return torch.sum(torch.abs(env.action_manager.action * asset.data.joint_vel * self.gear_ratio_scaled), dim=-1)

def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_rotate_inverse(math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)

def joint_deviation_vehicle_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # do not include the grippers revolute joints
    joint_names = [f"leg1joint{i}" for i in range(2,7)] 
    leg_joint_idx = [asset.find_joints(name)[0][0] for name in joint_names]
    vehicle_cfg_angles = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    
    vehicle_cfg_angles[:, leg_joint_idx[0]] = 0
    vehicle_cfg_angles[:, leg_joint_idx[1]] = 0
    vehicle_cfg_angles[:, leg_joint_idx[2]] = -math.pi/2
    vehicle_cfg_angles[:, leg_joint_idx[3]] = 0
    vehicle_cfg_angles[:, leg_joint_idx[4]] = 0
    
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids][:,leg_joint_idx] - vehicle_cfg_angles[:,leg_joint_idx]
    
    # leg_joint_idx4 = asset.find_joints("leg1joint4")[0]
    
    # angle = asset.data.joint_pos[:, leg_joint_idx4] - vehicle_cfg_angles[:, leg_joint_idx4]
    return torch.sum(torch.abs(angle), dim=1)

def joint_deviation_vehicle_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # do not include the grippers revolute joints
    joint_names = [f"leg1joint{i}" for i in range(2,7)] 
    leg_joint_idx = [asset.find_joints(name)[0][0] for name in joint_names]
    vehicle_cfg_angles = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    
    vehicle_cfg_angles[:, leg_joint_idx[0]] = 0
    vehicle_cfg_angles[:, leg_joint_idx[1]] = 0
    vehicle_cfg_angles[:, leg_joint_idx[2]] = -math.pi/2
    vehicle_cfg_angles[:, leg_joint_idx[3]] = 0
    vehicle_cfg_angles[:, leg_joint_idx[4]] = 0
    
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids][:,leg_joint_idx] - vehicle_cfg_angles[:,leg_joint_idx]
    
    # leg_joint_idx4 = asset.find_joints("leg1joint4")[0]
    
    # angle = asset.data.joint_pos[:, leg_joint_idx4] - vehicle_cfg_angles[:, leg_joint_idx4]
    return torch.sum(torch.square(angle), dim=1)


def wheel_vel_deviation_front(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize wheel velocities that are different, pairwise."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    
    wheel_joint_names = ["wheel11_left_joint","wheel11_right_joint"] 
    wheel_joint_idx =  [asset.find_joints(name)[0][0] for name in wheel_joint_names]
    wheel_joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids][:, wheel_joint_idx]

    # pairwise wheel velocity diff
    front_wheel_vel_diff = wheel_joint_vel[:,0] - wheel_joint_vel[:,1]
    # rear_wheel_vel_diff = wheel_joint_vel[:,2] - wheel_joint_vel[:,3]
    
    total_wheel_vel_diff = torch.sum(torch.square(front_wheel_vel_diff))
    
    return total_wheel_vel_diff

def wheel_vel_deviation_rear(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize wheel velocities that are different, pairwise."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    
    wheel_joint_names = ["wheel12_left_joint","wheel12_right_joint"] 
    wheel_joint_idx =  [asset.find_joints(name)[0][0] for name in wheel_joint_names]
    wheel_joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids][:, wheel_joint_idx]

    # pairwise wheel velocity diff
    # front_wheel_vel_diff = wheel_joint_vel[:,0] - wheel_joint_vel[:,1]
    rear_wheel_vel_diff = wheel_joint_vel[:,0] - wheel_joint_vel[:,1]
    
    total_wheel_vel_diff = torch.sum(torch.square(rear_wheel_vel_diff))
    
    return total_wheel_vel_diff