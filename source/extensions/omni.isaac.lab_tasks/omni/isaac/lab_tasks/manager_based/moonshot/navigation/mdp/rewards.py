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
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

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

def position_command_error_tanh(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    distance = torch.norm(des_pos_b, dim=1)
    return 1 - torch.tanh(distance / std)


def heading_command_error_abs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize tracking orientation error."""
    command = env.command_manager.get_command(command_name)
    heading_b = command[:, 3]
    return heading_b.abs()

def move_to_target_bonus_command(
    env: ManagerBasedRLEnv,
    threshold: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for moving to the target heading."""
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    heading_proj = obs.base_heading_proj(env, des_pos_b, asset_cfg).squeeze(-1)
    return torch.where(heading_proj > threshold, 1.0, heading_proj / threshold)

class progress_reward_command(ManagerTermBase):
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
        command_name = self.cfg.params["command_name"]
        command = self._env.command_manager.get_command(command_name)
        target_pos = torch.tensor(command[:, :3], device = self.device)
        to_target_pos = target_pos[env_ids, :3] - asset.data.root_pos_w[env_ids, :3]
        # reward terms
        self.potentials[env_ids] = -torch.norm(to_target_pos, p=2, dim=-1) / self._env.step_dt
        self.prev_potentials[env_ids] = self.potentials[env_ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # compute vector to target
        command_name = self.cfg.params["command_name"]
        command = self._env.command_manager.get_command(command_name)
        target_pos = torch.tensor(command[:, :3], device = self.device)
        to_target_pos = target_pos[:, :3] - asset.data.root_pos_w[:, :3]
        to_target_pos[:, 2] = 0.0
        # update history buffer and compute new potential
        self.prev_potentials[:] = self.potentials[:]
        self.potentials[:] = -torch.norm(to_target_pos, p=2, dim=-1) / env.step_dt

        return self.potentials - self.prev_potentials

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
    

def lin_ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base linear/angular velocity product using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2] * asset.data.root_lin_vel_b[:, :2]), dim=1)


def get_to_goal_reward(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """WORK IN PROGRESS"""
    command = env.command_manager.get_command(command_name)
    asset: RigidObject = env.scene[asset_cfg.name]
    goal = command[:, :2]
    if torch.abs(goal - asset.data.root_pose_w[:, :2]) > 0.05:
        return 1
    else:
        return 0
    
def joint_deviation_leg_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # do not include the grippers revolute joints
    joint_names = [f"leg1joint{i}" for i in range(2, 7)] 
    leg_joint_idx = [asset.find_joints(name)[0][0] for name in joint_names]
    
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids][:,leg_joint_idx] - asset.data.default_joint_pos[:, asset_cfg.joint_ids][:,leg_joint_idx]
    return torch.sum(torch.abs(angle), dim=1)


def joint_deviation_vehicle_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # do not include the grippers revolute joints
    joint_names = [f"leg1joint{i}" for i in [2,4,6]] 
    leg_joint_idx = [asset.find_joints(name)[0][0] for name in joint_names]
    vehicle_cfg_angles = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    
    vehicle_cfg_angles[:, leg_joint_idx[0]] = 0
    vehicle_cfg_angles[:, leg_joint_idx[1]] = -math.pi/2
    vehicle_cfg_angles[:, leg_joint_idx[2]] = 0
    # vehicle_cfg_angles[:, leg_joint_idx[3]] = 0
    # vehicle_cfg_angles[:, leg_joint_idx[4]] = 0
    
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids][:,leg_joint_idx] - vehicle_cfg_angles[:,leg_joint_idx]
    
    # leg_joint_idx4 = asset.find_joints("leg1joint4")[0]
    
    # angle = asset.data.joint_pos[:, leg_joint_idx4] - vehicle_cfg_angles[:, leg_joint_idx4]
    return torch.sum(torch.abs(angle), dim=1)

def joint_deviation_vehicle_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # do not include the grippers revolute joints
    joint_names = [f"leg1joint{i}" for i in [2,4,6]] 
    leg_joint_idx = [asset.find_joints(name)[0][0] for name in joint_names]
    vehicle_cfg_angles = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    
    vehicle_cfg_angles[:, leg_joint_idx[0]] = 0
    vehicle_cfg_angles[:, leg_joint_idx[1]] = -math.pi/2
    vehicle_cfg_angles[:, leg_joint_idx[2]] = 0
    # vehicle_cfg_angles[:, leg_joint_idx[3]] = 0
    # vehicle_cfg_angles[:, leg_joint_idx[4]] = 0
    
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

# def wheel_vel_deviation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), front_weight: float, rear_weight: float) -> torch.Tensor:
#     """Penalize wheel velocities that are different (SSE), pairwise ."""
#     # extract the used quantities (to enable type-hinting)
#     asset: Articulation = env.scene[asset_cfg.name]
    
#     wheel_joint_names = ["wheel11_left_joint","wheel11_right_joint","wheel12_left_joint","wheel12_right_joint"] 
#     wheel_joint_idx =  [asset.find_joints(name)[0][0] for name in wheel_joint_names]
#     wheel_joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids][:, wheel_joint_idx]

#     # pairwise wheel velocity diff
#     front_wheel_vel_diff = wheel_joint_vel[:,0] - wheel_joint_vel[:,1]
#     rear_wheel_vel_diff = wheel_joint_vel[:,2] - wheel_joint_vel[:,3]
    
#     front_wheel_vel_sse = front_weight * torch.sum(torch.square(front_wheel_vel_diff))
#     rear_wheel_vel_sse = rear_weight * torch.sum(torch.square(rear_wheel_vel_diff))
    
    
#     total_wheel_vel_sse = front_wheel_vel_sse + rear_wheel_vel_sse
    
#     return total_wheel_vel_sse

def wheel_vel_reward(env: ManagerBasedRLEnv, target_vel: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward wheel velocities close to target vel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    
    wheel_joint_names = ["wheel11_left_joint","wheel11_right_joint", "wheel12_left_joint","wheel12_right_joint"] 
    wheel_joint_idx =  [asset.find_joints(name)[0][0] for name in wheel_joint_names]
    wheel_joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids][:, wheel_joint_idx]
    target_error = wheel_joint_vel - target_vel

    target_error_mse = torch.mean(torch.square(target_error))
    
    return target_error_mse