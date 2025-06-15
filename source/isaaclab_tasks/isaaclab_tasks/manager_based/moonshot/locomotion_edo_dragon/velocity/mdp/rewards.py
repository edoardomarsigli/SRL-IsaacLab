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


def upright_posture_bonus(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for maintaining an upright posture."""
    up_proj = obs.base_up_proj(env, asset_cfg).squeeze(-1)
    return torch.square(up_proj - threshold).float()


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







def lin_vel_z_body_l2(env: ManagerBasedRLEnv, body_names: list, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    body_link_idx = [asset.find_bodies(name)[0][0] for name in body_names]

    return torch.sum(torch.square(asset.data.body_lin_vel_w[:, body_link_idx, 2]), dim=1)

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




def lin_acc_body_l2(env: ManagerBasedRLEnv, body_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize linear acceleration of bodies using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    body_link_idx = asset.find_bodies(body_name)[0][0]

    return torch.sum(torch.square(asset.data.body_lin_acc_w[:, body_link_idx, :]), dim=1)

def body_height_l2(
    env: ManagerBasedRLEnv,
    body_name: str,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    body_link_idx = asset.find_bodies(body_name)[0][0]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    return torch.square(asset.data.body_pos_w[:,body_link_idx, 2] - adjusted_target_height)

def wheel_air_time(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time) * first_contact, dim=1)
    # no reward for zero command
    return reward

def body_height_above_wheels_l2(
    env: ManagerBasedRLEnv,
    body_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    body_link_idx = asset.find_bodies(body_name)[0][0]
    wheel_joint_names = ["wheel11_left","wheel11_right",
                         "wheel12_left","wheel12_right"] 
    wheel_joint_idx =  [asset.find_bodies(name)[0][0] for name in wheel_joint_names]
    wheel_center_height = torch.mean(asset.data.body_pos_w[:, wheel_joint_idx, 2], dim = 1)
    wheel_center_height = wheel_center_height + 0.15
    body_height = asset.data.body_pos_w[:,body_link_idx, 2]
    difference = wheel_center_height - body_height
    # punish by square difference only if below 
    result = torch.where(difference > 0, difference ** 2, torch.zeros_like(difference))

    return result

# def upright_wheel_bodies(
#     env: ManagerBasedRLEnv,
#     std: float,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# ) -> torch.Tensor:
#     # upright height difference (desired height difference)
#     target_height = 0.3786

#     # extract the used quantities (to enable type-hinting)
#     asset: RigidObject = env.scene[asset_cfg.name]
#     front_body_link_idx = asset.find_bodies("leg1link2")[0][0]
#     rear_body_link_idx = asset.find_bodies("leg1link6")[0][0]
#     wheel_joint_names = ["wheel11_left","wheel11_right",
#                          "wheel12_left","wheel12_right"] 
#     front_wheel_joint_idx =  [asset.find_bodies(name)[0][0] for name in wheel_joint_names[:2]]
#     front_wheel_center_height = torch.mean(asset.data.body_pos_w[:, front_wheel_joint_idx, 2], dim = 1)
#     front_body_height = asset.data.body_pos_w[:,front_body_link_idx, 2]
#     front_height_diff = front_body_height - front_wheel_center_height
#     front_deviation = front_height_diff - target_height
    
#     rear_wheel_joint_idx =  [asset.find_bodies(name)[0][0] for name in wheel_joint_names[2:]]    
#     rear_wheel_center_height = torch.mean(asset.data.body_pos_w[:, rear_wheel_joint_idx, 2], dim = 1)    
#     rear_body_height = asset.data.body_pos_w[:,rear_body_link_idx, 2]
#     rear_height_diff = rear_body_height - rear_wheel_center_height
#     rear_deviation = rear_height_diff - target_height

#     error = torch.abs(front_deviation) + torch.abs(rear_deviation)

#     return torch.exp(-error / std**2)

# def upright_wheel_bodies_angle(
#     env: ManagerBasedRLEnv,
#     std: float,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# ) -> torch.Tensor:
#     # upright height difference (desired height difference)

#     # extract the used quantities (to enable type-hinting)
#     asset: RigidObject = env.scene[asset_cfg.name]
#     front_body_link_idx = asset.find_bodies("leg4link2")[0][0]
#     rear_body_link_idx = asset.find_bodies("leg4link6")[0][0]
#     wheel_joint_names = ["wheel11_left","wheel11_right",
#                          "wheel12_left","wheel12_right"] 
    
#     front_wheel_joint_idx =  [asset.find_bodies(name)[0][0] for name in wheel_joint_names[:2]]
#     front_wheel_center = torch.mean(asset.data.body_pos_w[:, front_wheel_joint_idx, :], dim = 1)
#     front_body = asset.data.body_pos_w[:,front_body_link_idx, :]
#     front_diff = front_body - front_wheel_center
#     front_diff_norm = torch.norm(front_diff,dim=1, keepdim=True)
#     # avoid division by 0
#     epsilon = 1e-8
#     front_diff_normalized = front_diff / (front_diff_norm + epsilon)
#     front_z = front_diff_normalized[:, 2]
#     front_angles = torch.acos(torch.clamp(front_z, -1.0, 1.0))
    
#     rear_wheel_joint_idx =  [asset.find_bodies(name)[0][0] for name in wheel_joint_names[2:]]    
#     rear_wheel_center = torch.mean(asset.data.body_pos_w[:, rear_wheel_joint_idx, :], dim = 1)
#     rear_body = asset.data.body_pos_w[:,rear_body_link_idx, :]
#     rear_diff = rear_body - rear_wheel_center
#     rear_diff_norm = torch.norm(rear_diff,dim=1, keepdim=True)
#     # avoid division by 0
#     epsilon = 1e-8
#     rear_diff_normalized = rear_diff / (rear_diff_norm + epsilon)
#     rear_z = rear_diff_normalized[:, 2]
#     rear_angles = torch.acos(torch.clamp(rear_z, -1.0, 1.0))

#     error = torch.abs(front_angles) + torch.abs(rear_angles)
#     return error
#     return torch.exp(-error / std**2)

def track_lin_vel_xy_exp_vehicle(
    env: ManagerBasedRLEnv, std: float, command_name: str, body_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel of specific body."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    body_link_idx = asset.find_bodies(body_name)[0][0]
    body_lin_vel_w = asset.data.body_lin_vel_w[:, body_link_idx, :]
    body_quat_w = asset.data.body_quat_w[:, body_link_idx, :]
    if body_name == "leg4link2":
        tf_d_matrix = torch.tensor([[-1,0,0],[0,0,1],[0,1,0]], device = env.device)
        tf_d_matrix_expanded = tf_d_matrix.unsqueeze(0).expand(env.num_envs, -1, -1)
        tf_d_quat = math_utils.quat_from_matrix(tf_d_matrix_expanded)
    elif body_name == "leg4link4":
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

    reward = 1.0 / (1.0 + lin_vel_error)
    
    return reward

def track_ang_vel_z_exp_vehicle(env: ManagerBasedRLEnv, std: float, command_name: str, body_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    body_link_idx = asset.find_bodies(body_name)[0][0]
    body_ang_vel_w = asset.data.body_ang_vel_w[:, body_link_idx, :]

    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - body_ang_vel_w[:, 2])

    reward = 1.0 / (1.0 + ang_vel_error)

    return reward

def joint_torques_dragon_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize leg joint torques applied on the articulation using L2 squared kernel.

    Excludes any wheel joint torques.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    all_joint_indices = range(len(asset.joint_names))
    
    wheel_joint_names = [
        "wheel12_left_joint",
        "wheel12_right_joint",
        "wheel14_left_joint",
        "wheel14_right_joint",
    ] 

    wheel_joint_idx = {asset.find_joints(name)[0][0] for name in wheel_joint_names}
    # if not a wheel joint, then it's a leg joint, which makes it applicable for any number of leg joints
    leg_joint_idx = [idx for idx in all_joint_indices if idx not in wheel_joint_idx]

    return torch.sum(torch.square(asset.data.applied_torque[:, leg_joint_idx]), dim=1)



import torch.nn.functional as F

def upright_wheel_bodies_angle(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Penalizza:
    - il discostamento da 45° del vettore front_wheel → joint2
    - il discostamento da 0° del vettore rear_wheel → joint6 (cioè deve essere verticale)
    """

    # Posizioni frame nel mondo
    front_wheel_pos = env.scene["front_wheel_frame"].data.target_pos_w[..., 0, :]
    rear_wheel_pos = env.scene["rear_wheel_frame"].data.target_pos_w[..., 0, :]
    joint2_pos = env.scene["joint2_frame"].data.target_pos_w[..., 0, :]
    joint6_pos = env.scene["joint6_frame"].data.target_pos_w[..., 0, :]

    # Vettori normalizzati ruota → giunto
    front_vec = F.normalize(joint2_pos - front_wheel_pos, dim=-1)
    rear_vec = F.normalize(joint6_pos - rear_wheel_pos, dim=-1)

    # Asse Z mondo
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=front_vec.device)

    # Angoli tra vettore e asse Z
    angle_front = torch.acos(torch.clamp(torch.sum(front_vec * z_axis, dim=-1), -1.0, 1.0))
    angle_rear  = torch.acos(torch.clamp(torch.sum(rear_vec  * z_axis, dim=-1), -1.0, 1.0))

    # Errore rispetto a target: front → 45°, rear → 45°
    target_front = torch.pi / 4  # 45°
    target_rear = torch.pi / 4           
    error_front = torch.abs(angle_front - target_front)
    error_rear  = torch.abs(angle_rear  - target_rear)

    # Reward: massimo quando angoli sono nei target desiderati
    reward = (1.0 / (1 +(error_front *3 + error_rear *2)))

    return reward


# # 3. Vettori differenza: da frame ruota → corpo
# front_diff = joint2__pos - front_wheel_pos
# rear_diff  = joint6_pos  - rear_wheel_pos

# # 4. Normalizzazione per evitare divisione per 0
# front_diff_norm = torch.norm(front_diff, dim=1, keepdim=True) + 1e-6
# rear_diff_norm  = torch.norm(rear_diff,  dim=1, keepdim=True) + 1e-6

# front_unit = front_diff / front_diff_norm
# rear_unit  = rear_diff  / rear_diff_norm

# # 5. Calcolo angoli rispetto all’asse Z (verticale)
# front_z = front_unit[:, 2]
# rear_z  = rear_unit[:, 2]

# front_angle = torch.acos(torch.clamp(front_z, -1.0, 1.0))
# rear_angle  = torch.acos(torch.clamp(rear_z,  -1.0, 1.0))

# # 6. Somma angoli → errore totale
# error = front_angle + rear_angle

def penalize_low_joint(env: ManagerBasedRLEnv) -> torch.Tensor:

    joint4_pos = env.scene["joint4_frame"].data.target_pos_w[..., 0, :]    # [N, 3]
    joint4_z = joint4_pos[:, 2]  # [N]

    joint2_pos = env.scene["joint2_frame"].data.target_pos_w[..., 0, :]    # [N, 3]
    joint2_z = joint2_pos[:, 2]  # [N]

    joint6_pos = env.scene["joint6_frame"].data.target_pos_w[..., 0, :]    # [N, 3]
    joint6_z = joint6_pos[:, 2]  # [N]

    pen4= torch.where(joint4_z<0.3, 0.5-joint4_z, 0)  # Penalize if joint4 is lower than 0.1m from the ground
    pen2= torch.where(joint2_z<0.3, 0.3-joint2_z, 0)  # Penalize if joint2 is lower than 0.1m from the ground
    pen6= torch.where(joint6_z<0.3, 0.3-joint6_z, 0)  # Penalize if joint6 is lower than 0.1m from the ground

    return pen2+ pen4 + pen6  # Terminate if joint4 is lower than 0.1m from the ground

def penalize_narrow_wheel(env: ManagerBasedRLEnv) -> torch.Tensor:
    front_pos = env.scene["front_wheel_frame"].data.target_pos_w[..., 0, :]  # [N, 3]
    rear_pos = env.scene["rear_wheel_frame"].data.target_pos_w[..., 0, :]    # [N, 3]
    distance = torch.norm(front_pos - rear_pos, dim=1)  # [N]

    return torch.where(distance < 1.3, 1.3 - distance, torch.zeros_like(distance))  # Penalize if distance is less than 0.5m

def wheel_z(env: ManagerBasedRLEnv) -> torch.Tensor:
    # Ottieni la posizione z del frame
    front_pos = env.scene["front_wheel_frame"].data.target_pos_w[..., 0, :]
    front_z = front_pos[..., 2]  # prende asse z

    rear_pos = env.scene["rear_wheel_frame"].data.target_pos_w[..., 0, :]
    rear_z = rear_pos[..., 2]  # prende asse z

    front = torch.where(front_z < 0.3, 0.3-front_z, torch.zeros_like(front_z)) 
    rear = torch.where(rear_z < 0.3, 0.3-rear_z, torch.zeros_like(rear_z))
    
    return front + rear 


