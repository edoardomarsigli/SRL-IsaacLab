# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv



def approach_ee_handle(env: ManagerBasedRLEnv, threshold: float = 0.2) -> torch.Tensor:
    try:
        handle_pos = env.scene["handle_frame"].data.target_pos_w[..., 0, :]
        ee_pos = env.scene["handle_frame"].data.target_pos_w[..., 0, :]
        distance = torch.norm(handle_pos - ee_pos, dim=-1, p=2)
        reward = 1.0 / (1.0 + distance**2)
        reward = torch.pow(reward, 2)
        return torch.where(distance <= threshold, 2 * reward, reward)
    except (KeyError, AttributeError):
        # Se uno dei frame non è pronto, ritorna reward 0 temporanea
        return torch.zeros(env.num_envs, device=env.device)


def align_ee_handle(env: ManagerBasedRLEnv, align_threshold) -> torch.Tensor:
    """
    Reward for aligning the end-effector with the handle.

    Main reward is shaped via squared dot products between relevant axes.
    Bonus is added when alignment exceeds a threshold.
    """
    ee_quat = env.scene["ee_frame"].data.target_quat_w[..., 0, :]
    handle_quat = env.scene["handle_frame"].data.target_quat_w[..., 0, :]

    ee_rot_mat = matrix_from_quat(ee_quat)
    handle_mat = matrix_from_quat(handle_quat)

    # EE directions
    ee_x, ee_z = ee_rot_mat[..., 0], ee_rot_mat[..., 2]
    # Handle directions
    handle_x, handle_z = handle_mat[..., 0], handle_mat[..., 2]

    # Dot products (alignment measures)
    align_z = torch.bmm(ee_z.unsqueeze(1), -handle_z.unsqueeze(-1)).squeeze(-1).squeeze(-1)  # (N,)
    align_x = torch.bmm(ee_x.unsqueeze(1), -handle_x.unsqueeze(-1)).squeeze(-1).squeeze(-1)  # (N,)

    # Main shaped reward
    shaped_reward = 0.6 * torch.sign(align_z) * align_z**2 + 0.4 * torch.sign(align_x) * align_x**2

    # Binary bonus: if alignment exceeds threshold
    z_bonus = (align_z >= align_threshold).float() * 0.5
    x_bonus = (align_x >= align_threshold).float() * 0.3

    return shaped_reward + z_bonus + x_bonus




def approach_xy_alignment(env: ManagerBasedRLEnv, xy_threshold, align_threshold) -> torch.Tensor:
    """
    Reward for aligning the EE and handle origins in the XY plane (ignores Z axis).
    """
    try:

        ee_quat = env.scene["ee_frame"].data.target_quat_w[..., 0, :]
        handle_quat = env.scene["handle_frame"].data.target_quat_w[..., 0, :]

        ee_rot_mat = matrix_from_quat(ee_quat)
        handle_mat = matrix_from_quat(handle_quat)

        # EE directions
        ee_x, ee_z = ee_rot_mat[..., 0], ee_rot_mat[..., 2]
        # Handle directions
        handle_x, handle_z = handle_mat[..., 0], handle_mat[..., 2]

        # Dot products (alignment measures)
        align_z = torch.bmm(ee_z.unsqueeze(1), -handle_z.unsqueeze(-1)).squeeze(-1).squeeze(-1)  # (N,)
        align_x = torch.bmm(ee_x.unsqueeze(1), -handle_x.unsqueeze(-1)).squeeze(-1).squeeze(-1)  # (N,)
        handle_pos = env.scene["handle_frame"].data.target_pos_w[..., 0, :]
        ee_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]

        # Take only X and Y
        handle_xy = handle_pos[..., :2]
        ee_xy = ee_pos[..., :2]

        # Euclidean distance in XY plane
        distance_xy = torch.norm(handle_xy - ee_xy, dim=-1, p=2)

        # Reward shaping
        reward = 1.0 / (1.0 + distance_xy**2)
        reward = torch.pow(reward, 2)

        xy_bonus = (distance_xy <= xy_threshold).float()

        # Bonus if close enough
        return torch.where(align_z >= align_threshold, reward, torch.zeros_like(reward))+xy_bonus

    except (KeyError, AttributeError):
        return torch.zeros(env.num_envs, device=env.device)
    

def approach_z_conditional_on_xy(env: ManagerBasedRLEnv, xy_threshold) -> torch.Tensor:
    """
    Reward for minimizing the vertical (Z-axis) distance between EE and handle,
    but only if their XY positions are close (within xy_threshold).
    """
    try:
        handle_pos = env.scene["handle_frame"].data.target_pos_w[..., 0, :]  # (N, 3)
        ee_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]          # (N, 3)

        # Step 1: Calcola la distanza in XY
        handle_xy = handle_pos[..., :2]  # (N, 2)
        ee_xy = ee_pos[..., :2]          # (N, 2)
        distance_xy = torch.norm(handle_xy - ee_xy, dim=-1, p=2)  # (N,)

        # Step 2: Calcola la distanza lungo l'asse Z (modulo)
        z_distance = torch.abs(handle_pos[..., 2] - ee_pos[..., 2])  # (N,)

        # Step 3: Reward shaping sulla distanza Z (solo se XY è sotto soglia)
        z_reward = 1.0 / (1.0 + z_distance**2)
        z_reward = torch.pow(z_reward, 2)

        # Step 4: Applica solo se XY è vicino
        return torch.where(distance_xy <= xy_threshold, z_reward, torch.zeros_like(z_reward))

    except (KeyError, AttributeError):
        return torch.zeros(env.num_envs, device=env.device)




def align_grasp(env: ManagerBasedRLEnv) -> torch.Tensor: # bonus per rf e lf sopra e sotto ee non handle senno partono subito ad aprirsi come matti
    """Bonus for correct hand orientation.

    The correct hand orientation is when the right finger is above the handle and the left finger is below the handle.
    """
    # Target object position: (num_envs, 3)
    ee_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]  
    # Fingertips position: (num_envs, n_fingertips, 3)

    lf_pos = env.scene["lf_frame"].data.target_pos_w[..., 0, :]
    rf_pos = env.scene["rf_frame"].data.target_pos_w[..., 0, :]

    # Check if hand is in a graspable pose
    is_graspable = (rf_pos[:, 2] > ee_pos[:, 2]) & (lf_pos[:, 2] < ee_pos[:, 2])

    # bonus if left finger is above the drawer handle and right below
    return is_graspable


# def approach_gripper_handle(env: ManagerBasedRLEnv, offset: float = 0.04) -> torch.Tensor:
#     """Reward the robot's gripper reaching the drawer handle with the right pose.

#     This function returns the distance of fingertips to the handle when the fingers are in a grasping orientation
#     (i.e., the left finger is above the handle and the right finger is below the handle). Otherwise, it returns zero.
#     """
#     # Target object position: (num_envs, 3)
#     handle_pos = env.scene["handle_frame"].data.target_pos_w_named["handle_target"]
#     # Fingertips position: (num_envs, n_fingertips, 3)
#     lf_pos = env.scene["lf_frame"].data.target_pos_w_named["lf"]
#     rf_pos = env.scene["rf_frame"].data.target_pos_w_named["rf"]

#     # Compute the distance of each finger from the handle
#     lf_dist = torch.abs(lf_pos[:, 2] - handle_pos[:, 2])
#     rf_dist = torch.abs(rf_pos[:, 2] - handle_pos[:, 2])

#     # Check if hand is in a graspable pose
#     is_graspable = (rf_pos[:, 2] > handle_pos[:, 2]) & (lf_pos[:, 2] < handle_pos[:, 2])

#     return is_graspable * ((offset - lf_dist) + (offset - rf_dist))


# def grasp_handle(
#     env: ManagerBasedRLEnv, threshold: float, open_joint_pos: float, asset_cfg: SceneEntityCfg
# ) -> torch.Tensor:
#     """Reward for closing the fingers when being close to the handle.

#     The :attr:`threshold` is the distance from the handle at which the fingers should be closed.
#     The :attr:`open_joint_pos` is the joint position when the fingers are open.

#     Note:
#         It is assumed that zero joint position corresponds to the fingers being closed.
#     """
#     ee_pos = env.scene["ee_frame"].data.target_pos_w_named["ee"] 
#     handle_pos = env.scene["handle_frame"].data.target_pos_w_named["handle_target"]
# lf_pos = env.scene["lf_frame"].data.target_pos_w_named["lf"]
#     rf_pos = env.scene["rf_frame"].data.target_pos_w_named["rf"]

#     gripper_joint_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]

#     distance = torch.norm(handle_pos - ee_pos, dim=-1, p=2)
#     is_close = distance <= threshold

#     return is_close * torch.sum(open_joint_pos - gripper_joint_pos, dim=-1)


def grasp_handle(
    env: ManagerBasedRLEnv, threshold: float, open_joint_pos: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward for closing the fingers when being close to the handle.

    The :attr:`threshold` is the distance from the handle at which the fingers should be closed.
    The :attr:`open_joint_pos` is the joint position when the fingers are open.

    Note:
        It is assumed that zero joint position corresponds to the fingers being closed.
    """
    ee_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    handle_pos = env.scene["handle_frame"].data.target_pos_w[..., 0, :]
    lf_pos = env.scene["lf_frame"].data.target_pos_w[..., 0, :]
    rf_pos = env.scene["rf_frame"].data.target_pos_w[..., 0, :]

    #gripper_joint_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]

    distance_lf = torch.norm(handle_pos - lf_pos, dim=-1, p=2)
    distance_rf = torch.norm(handle_pos - rf_pos, dim=-1, p=2)
    finger_error= distance_lf + distance_rf
    alpha = 5
    distance = torch.norm(handle_pos - ee_pos, dim=-1, p=2)
    is_close = distance <= threshold
    
    return is_close * torch.exp(-alpha * finger_error)
