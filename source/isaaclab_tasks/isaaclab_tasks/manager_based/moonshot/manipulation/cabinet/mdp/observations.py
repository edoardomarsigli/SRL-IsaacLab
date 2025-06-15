# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationData
from isaaclab.sensors import FrameTransformerData

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv



# def rel_ee__distance(env: ManagerBasedRLEnv) -> torch.Tensor:
#     """The distance between the end-effector and the object."""
#     ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
#     cabinet_tf_data: FrameTransformerData = env.scene["handle_frame"].data
#     return cabinet_tf_data.target_pos_w[..., 0, :] - ee_tf_data.target_pos_w[..., 0, :]

def rel_ee__distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Returns distance between EE and handle; safe even if handle_frame not ready."""
    try:
        ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
        handle_tf_data: FrameTransformerData = env.scene["handle_frame"].data
        return handle_tf_data.target_pos_w[..., 0, :] - ee_tf_data.target_pos_w[..., 0, :]
        # return handle_tf_data.target_pos_w_named["handle_target"] - ee_tf_data.target_pos_w_named["ee"]
    except (KeyError, AttributeError) as e:
        # Se il frame non è pronto, ritorna zeri temporanei
        return torch.zeros((env.num_envs, 3), device=env.device)


def ee_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Returns the position of the end-effector relative to the environment origins.
    If the frame is not yet available, returns zero tensors as fallback."""
    try:
        ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
        ee_pos = ee_tf_data.target_pos_w[..., 0, :] - env.scene.env_origins
        return ee_pos
    except (KeyError, AttributeError):
        # Se il frame non è pronto, ritorna zeri temporanei
        return torch.zeros((env.num_envs, 3), device=env.device)
    

def handle_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Returns the position of the handle relative to the environment origins.
    If the frame is not yet available, returns zero tensors as fallback."""
    try:
        handle_tf_data: FrameTransformerData = env.scene["handle_frame"].data
        handle_pos = handle_tf_data.target_pos_w[..., 0, :]  - env.scene.env_origins
        return handle_pos
    except (KeyError, AttributeError):
        # Se il frame non è pronto, ritorna zeri temporanei
        return torch.zeros((env.num_envs, 3), device=env.device)



def rel_rf__distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Returns distance between right finger and handle; safe even if handle_frame not ready."""
    try:
        ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
        rf_tf_data: FrameTransformerData = env.scene["rf_frame"].data
        return rf_tf_data.target_pos_w[..., 0, :] - ee_tf_data.target_pos_w[..., 0, :]
        # return rf_tf_data.target_pos_w_named["rf"] - ee_tf_data.target_pos_w_named["ee"]
    except (KeyError, AttributeError) as e:
        # Se il frame non è pronto, ritorna zeri temporanei
        return torch.zeros((env.num_envs, 3), device=env.device)
    

def rel_lf__distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Returns distance between left finger and handle; safe even if handle_frame not ready."""
    try:
        ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
        rf_tf_data: FrameTransformerData = env.scene["lf_frame"].data
        return rf_tf_data.target_pos_w[..., 0, :] - ee_tf_data.target_pos_w[..., 0, :]
        # return rf_tf_data.target_pos_w_named["lf"] - ee_tf_data.target_pos_w_named["ee"]
    except (KeyError, AttributeError) as e:
        # Se il frame non è pronto, ritorna zeri temporanei
        return torch.zeros((env.num_envs, 3), device=env.device)


def ee_quat(env: ManagerBasedRLEnv, make_quat_unique: bool = True) -> torch.Tensor:
    """Returns the orientation (quaternion) of the end-effector in the world frame.
    If `make_quat_unique` is True, ensures the quaternion has a positive real part to avoid discontinuities.
    If the frame is not yet available, returns identity quaternions.
    """
    try:
        ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
        ee_quat = ee_tf_data.target_quat_w[..., 0, :] 
        return math_utils.quat_unique(ee_quat) if make_quat_unique else ee_quat
    except (KeyError, AttributeError):
        # Se il frame non è pronto, ritorna quaternioni identità (0, 0, 0, 1)
        return torch.tensor([[0.0, 0.0, 0.0, 1.0]] * env.num_envs, device=env.device)
    

def handle_quat(env: ManagerBasedRLEnv, make_quat_unique: bool = True) -> torch.Tensor:
    """Returns the orientation (quaternion) of the end-effector in the world frame.
    If `make_quat_unique` is True, ensures the quaternion has a positive real part to avoid discontinuities.
    If the frame is not yet available, returns identity quaternions.
    """
    try:
        handle_tf_data: FrameTransformerData = env.scene["handle_frame"].data
        handle_quat = handle_tf_data.target_quat_w[..., 0, :] 
        return math_utils.quat_unique(handle_quat) if make_quat_unique else handle_quat
    except (KeyError, AttributeError):
        # Se il frame non è pronto, ritorna quaternioni identità (0, 0, 0, 1)
        return torch.tensor([[0.0, 0.0, 0.0, 1.0]] * env.num_envs, device=env.device)
    

def gripper1_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Returns the joint position of gripper1 (leg2grip1)."""
    try:
        joint_idx = env.scene["robot"].joint_names.index("leg2grip1")
        return env.scene["robot"].data.joint_pos[:, joint_idx].unsqueeze(-1)
    except (KeyError, AttributeError, ValueError):
        return torch.zeros((env.num_envs, 1), device=env.device)




def gripper2_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Returns the joint position of gripper2 (leg2grip2)."""
    try:
        joint_idx = env.scene["robot"].joint_names.index("leg2grip2")
        return env.scene["robot"].data.joint_pos[:, joint_idx].unsqueeze(-1)
    except (KeyError, AttributeError, ValueError):
        return torch.zeros((env.num_envs, 1), device=env.device)