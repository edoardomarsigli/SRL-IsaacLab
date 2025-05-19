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



def threshold_curriculum(env, start_value, end_value, start_episode, end_episode):
    """
    Calcola un valore threshold che decresce linearmente da start_value a end_value
    tra start_episode e end_episode.
    """
    if not hasattr(env, "episode_counter"):
        return start_value
    
    global_episode = env.episode_counter.min().float() #prendo il min per sincronizzare gli envs
    episode_progress = torch.clamp(
        (global_episode - start_episode) / (end_episode - start_episode),
        min=0.0,
        max=1.0,
    )

    # Normalizzazione tra 0 e 1
    # episode_progress = torch.clamp(
    #     (env.episode_counter.float() - start_episode) / (end_episode - start_episode),
    #     min=0.0,
    #     max=1.0,
    # )
    # Interpolazione lineare tra start e end
    return start_value + (end_value - start_value) * episode_progress

def get_curriculum_thresholds(env):
    if not hasattr(env, "episode_counter"):
        episode = torch.zeros(env.num_envs, device="cuda")
    else:
        episode = env.episode_counter.float()

    align_threshold = threshold_curriculum(env, 0.7, 0.9, start_episode=0, end_episode=10000) #per bonus di align 
    align_threshold1 = threshold_curriculum(env, 0.7, 0.9, start_episode=100, end_episode=10000) #per start di zy a 200
    zy_threshold = threshold_curriculum(env, 0.1, 0.02, start_episode=100, end_episode=10000) #per bonus zy
    zy_threshold1 = threshold_curriculum(env, 0.1, 0.02, start_episode=400, end_episode=10400) #per start di x a 400
    x_threshold = threshold_curriculum(env, 0.15, 0.05, start_episode=400, end_episode=1000) #per bonus di x
    grasp_threshold = threshold_curriculum(env, 0.50, 0.01, start_episode=600, end_episode=10600) #per start di grasp a 600
    # z_limit = threshold_curriculum(env, 0.5, 0.2, start_episode=0, end_episode=10000) #per start di grasp a 600

    return align_threshold, align_threshold1, zy_threshold, zy_threshold1 , x_threshold, grasp_threshold


def weight_curriculum(env, start_value, end_value, start_episode, end_episode):
    """
    Come threshold_curriculum ma restituisce un peso crescente nel tempo.
    """
    if not hasattr(env, "episode_counter"):
        return start_value

    global_episode = env.episode_counter.min().float() #prendo il min per sincronizzare gli envs
    progress = torch.clamp(
        (global_episode - start_episode) / (end_episode - start_episode),
        min=0.0,
        max=1.0,
    )
    return start_value + (end_value - start_value) * progress




# def approach_ee_handle(env: ManagerBasedRLEnv, threshold: float = 0.2) -> torch.Tensor:
#     try:
#         handle_pos = env.scene["handle_frame"].data.target_pos_w[..., 0, :]
#         ee_pos = env.scene["handle_frame"].data.target_pos_w[..., 0, :]
#         distance = torch.norm(handle_pos - ee_pos, dim=-1, p=2)
#         reward = 1.0 / (1.0 + distance**2)
#         reward = torch.pow(reward, 2)
#         return torch.where(distance <= threshold, 2 * reward, reward)
#     except (KeyError, AttributeError):
#         # Se uno dei frame non è pronto, ritorna reward 0 temporanea
#         return torch.zeros(env.num_envs, device=env.device)


def align_ee_handle_z(env: ManagerBasedRLEnv, align_threshold) -> torch.Tensor:
    """
    Reward for aligning the end-effector Z axis with the handle Z axis.
    """
    ee_quat = env.scene["ee_frame"].data.target_quat_w[..., 0, :]
    handle_quat = env.scene["handle_frame"].data.target_quat_w[..., 0, :]

    ee_rot_mat = matrix_from_quat(ee_quat)
    handle_mat = matrix_from_quat(handle_quat)

    ee_z = ee_rot_mat[..., 2]
    handle_z = handle_mat[..., 2]

    align_z = torch.bmm(ee_z.unsqueeze(1), -handle_z.unsqueeze(-1)).squeeze(-1).squeeze(-1)  # (N,)

    reward = torch.sign(align_z) * align_z**2
    z_bonus = (align_z >= align_threshold).float() * 0.5

    return reward + z_bonus

def align_ee_handle_z_curriculum_wrapped(env: ManagerBasedRLEnv) -> torch.Tensor:
    _, align_threshold, _,_,_,_ = get_curriculum_thresholds(env)
    return align_ee_handle_z(env, align_threshold)

def align_ee_handle_x(env: ManagerBasedRLEnv, align_threshold) -> torch.Tensor:
    """
    Reward for aligning the end-effector X axis with the handle X axis.
    """
    ee_quat = env.scene["ee_frame"].data.target_quat_w[..., 0, :]
    handle_quat = env.scene["handle_frame"].data.target_quat_w[..., 0, :]

    ee_rot_mat = matrix_from_quat(ee_quat)
    handle_mat = matrix_from_quat(handle_quat)

    ee_x = ee_rot_mat[..., 0]
    handle_x = handle_mat[..., 0]

    align_x = torch.bmm(ee_x.unsqueeze(1), handle_x.unsqueeze(-1)).squeeze(-1).squeeze(-1)  # (N,)

    reward = torch.sign(align_x) * align_x**4
    x_bonus = (align_x >= align_threshold).float() * 0.5

    return reward + x_bonus

def align_ee_handle_x_curriculum_wrapped(env: ManagerBasedRLEnv) -> torch.Tensor:
    _, align_threshold, _,_,_,_ = get_curriculum_thresholds(env)
    return align_ee_handle_x(env, align_threshold)

def penalize_joint7(env: ManagerBasedRLEnv, weight: float = -3.0) -> torch.Tensor:
    """
    Penalizza se la posizione del giunto 7 è fuori dal range [-π, π].
    """
    # Indice del giunto 7 nella lista dei joint names
    joint_idx = env.scene["robot"].joint_names.index("leg2joint7")

    # Posizione attuale del giunto 7
    joint7_pos = env.scene["robot"].data.joint_pos[..., joint_idx]  # (N,)

    # Calcolo quanto esce dal range


    violation = (joint7_pos.abs() > 3.1).float()  # (N,)
    in_range = (joint7_pos.abs() <= 3.1).float()  # (N,)

    return violation * weight + in_range

def penalize_low_joints(env, threshold4, threshold_ee)-> torch.Tensor:

    try:
    # Ottieni le posizioni dei giunti
        joint4_pos = env.scene["joint4_frame"].data.target_pos_w[..., 0, :]
        joint4_z = joint4_pos[...,  2]

        # Calcola maschera: True se sotto soglia

        below_threshold = joint4_z < threshold4

        ee_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
        ee_z = ee_pos[...,  2]

        below_threshold_ee = ee_z < threshold_ee

        # Penalità proporzionale alla distanza sotto soglia (opzionale)
        penalty_magnitudes_ee = torch.where(below_threshold_ee, threshold_ee - ee_z, torch.zeros_like(ee_z),
        )        
        penalty_magnitudes = torch.where(below_threshold, threshold4 - joint4_z, torch.zeros_like(joint4_z),
        )

        # Moltiplica per un coefficiente forte negativo
        return -10.0 * (penalty_magnitudes+penalty_magnitudes_ee) # <-- qui controlli quanto vuoi punire
    except (KeyError, AttributeError):
            return torch.zeros(env.num_envs, device=env.device)
    
    
def penalize_low_joints_curriculum(env: ManagerBasedRLEnv, threshold4, threshold_ee) -> torch.Tensor:
    base_reward = penalize_low_joints(env, threshold4, threshold_ee)
    weight = weight_curriculum(env, start_value=0.5, end_value=10, start_episode=0, end_episode=10000)
    return base_reward * weight




#PHASE1

def reward_joint4_height(env: ManagerBasedRLEnv, threshold: float, z_limit:float) -> torch.Tensor: #reward e penalità per giunto 4 altezza

    threshold = 0.25
    z_limit = 0.5

    # Ottieni posizione mondiale del giunto 4
    joint4_pos = env.scene["joint4_frame"].data.target_pos_w[..., 0, :]
    joint4_z = joint4_pos[..., 2]  # solo asse Z

    # Reward continuo: maggiore è la distanza sopra la soglia, più reward
    in_range = (joint4_z >= threshold) & (joint4_z <= z_limit)
    below = (joint4_z < threshold).float()

    # Linear shaping
    positive_reward = (joint4_z - threshold).clamp(min=0.0) * in_range.float()  # positivo solo se sopra soglia
    negative_penalty = (threshold - joint4_z).clamp(min=0.0) * below  # positivo solo se sotto soglia

    weight = weight_curriculum(env, start_value=1, end_value=0.1, start_episode=0, end_episode=300)
    global_episode_counter = env.episode_counter.min()

    if global_episode_counter >= 400:
        return (positive_reward>=0).float()*weight
    else:
        return weight * positive_reward - weight * negative_penalty  # pesi scalabili



#PHASE2



def approach_zy_alignment(env: ManagerBasedRLEnv, zy_threshold, align_threshold1, only_if_over) -> torch.Tensor: #allineamento xy e bonus solo se align ee handle buono
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
        handle_yz = handle_pos[..., [1, 2]]
        ee_yz = ee_pos[...,  [1, 2]]

        # Euclidean distance in XY plane
        distance_yz = torch.norm(handle_yz - ee_yz, dim=-1, p=2)

        # Reward shaping
        reward = 1.0 / (1.0 + distance_yz**2)
        reward = torch.pow(reward, 2)

        yz_bonus = (distance_yz <= zy_threshold).float()

        if only_if_over:
            mask = (ee_pos[..., 2] >= handle_pos[..., 2])
            reward = (reward + yz_bonus) * mask.float()

        # Bonus if close enough
        # return torch.where((align_z >= align_threshold1) & (align_x >= align_threshold1), reward, torch.zeros_like(reward))+yz_bonus
        return reward+yz_bonus

    except (KeyError, AttributeError):
        return torch.zeros(env.num_envs, device=env.device)

def approach_zy_alignment_curriculum_wrapped(env: ManagerBasedRLEnv) -> torch.Tensor:
 
    if not hasattr(env, "episode_counter"):
        return torch.zeros(env.num_envs, device=env.device)

    _, align_threshold1, zy_threshold, _, _, _  = get_curriculum_thresholds(env)

    global_episode_counter = env.episode_counter.min() #attivazione sincronizzata tra gli envs

    only_if_over = global_episode_counter >= 400

    if global_episode_counter >= 30:
        return approach_zy_alignment(env, zy_threshold, align_threshold1, only_if_over)
    else:
        return torch.zeros_like(approach_zy_alignment(env, zy_threshold, align_threshold1, only_if_over))


    

def approach_x_conditional_on_yz(env: ManagerBasedRLEnv, zy_threshold1, x_threshold) -> torch.Tensor: #allineamento z solo se xy buono
    """
    Reward for minimizing the vertical (Z-axis) distance between EE and handle,
    but only if their XY positions are close (within xy_threshold).
    """
    try:
        handle_pos = env.scene["handle_frame"].data.target_pos_w[..., 0, :]
        ee_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]

        # Take only X and Y
        handle_yz = handle_pos[..., [1, 2]]
        ee_yz = ee_pos[...,  [1, 2]]

        # Euclidean distance in XY plane
        distance_yz = torch.norm(handle_yz - ee_yz, dim=-1, p=2)

        # Step 2: Calcola la distanza lungo l'asse x (modulo)
        x_distance = torch.abs(handle_pos[..., 0] - ee_pos[..., 0])  # (N,)

        # Step 3: Reward shaping sulla distanza Z (solo se XY è sotto soglia)
        x_reward = 1.0 / (1.0 + x_distance**2)
        x_reward = torch.pow(x_reward, 2)

        x_bonus = (x_distance <= x_threshold).float()

        adaptive_threshold = 0.8 * x_distance + zy_threshold1 #cono di threshold
        # Step 4: Applica solo se XY è vicino
        return torch.where(distance_yz <= adaptive_threshold, x_reward, torch.zeros_like(x_reward))+x_bonus

    except (KeyError, AttributeError):
        return torch.zeros(env.num_envs, device=env.device)

def approach_x_conditional_on_yz_curriculum_wrapped(env: ManagerBasedRLEnv) -> torch.Tensor:

    if not hasattr(env, "episode_counter"):
        return torch.zeros(env.num_envs, device=env.device)
    _,_,_, zy_threshold1, x_threshold, _ = get_curriculum_thresholds(env)

    reward= approach_x_conditional_on_yz(env, zy_threshold1, x_threshold)

    global_episode_counter = env.episode_counter.min() #attivazione sincronizzata tra gli envs
    mask = global_episode_counter >= 200
    # mask = env.episode_counter >= 400
    if mask:
        return reward
    else:
        return torch.zeros_like(reward)

def collision_penalty(env: ManagerBasedRLEnv, weight:float) -> torch.Tensor:
    
    try:
        sensor = env.scene.sensors["contact_sensor_left1"]
        contact_forces = sensor.data.force_matrix_w  # shape: (N, B, F, 3)
        magnitudes = torch.norm(contact_forces, dim=-1).sum(dim=(-1, -2))  # sum across bodies and filters

        sensor2 = env.scene.sensors["contact_sensor_right1"]
        contact_forces2 = sensor2.data.force_matrix_w  # shape: (N, B, F, 3)
        magnitudes2 = torch.norm(contact_forces2, dim=-1).sum(dim=(-1, -2))  # sum across bodies and filters

        sensor3 = env.scene.sensors["contact_sensor_left2"]
        contact_forces3 = sensor3.data.force_matrix_w  # shape: (N, B, F, 3)
        magnitudes3 = torch.norm(contact_forces3, dim=-1).sum(dim=(-1, -2))  # sum across bodies and filters

        sensor4 = env.scene.sensors["contact_sensor_right2"]
        contact_forces4 = sensor4.data.force_matrix_w  # shape: (N, B, F, 3)
        magnitudes4 = torch.norm(contact_forces4, dim=-1).sum(dim=(-1, -2))  # sum across bodies and filters

        return weight*(magnitudes + magnitudes2 + magnitudes3 + magnitudes4)# penalità
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)

def penalty_x(env: ManagerBasedRLEnv) -> torch.Tensor: #allineamento z solo se xy buono
    try:
        handle_pos = env.scene["handle_frame"].data.target_pos_w[..., 0, :]
        ee_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]

        handle_x = handle_pos[..., 0] 
        ee_x = ee_pos[..., 0]

        penalty_x = (handle_x < ee_x).float()   # (N,)

        return penalty_x 

    except (KeyError, AttributeError):
        return torch.zeros(env.num_envs, device=env.device)    





#GRIPPER2


# def grasp_handle(env: ManagerBasedRLEnv, grasp_threshold, closed2_threshold:float=-0.02) -> torch.Tensor: #reward e penalità per gripper2
#     ee_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
#     handle_pos = env.scene["handle_frame"].data.target_pos_w[..., 0, :]

#     #lf_pos = env.scene["lf2_frame"].data.joint_pos[..., 0]
#     # rf_pos = env.scene["rf2_frame"].data.joint_pos[..., 0, :]
#     joint_idx = env.scene["robot"].joint_names.index("leg2grip2")
#     lf_pos = env.scene["robot"].data.joint_pos[:, joint_idx]


#     distance = torch.norm(handle_pos - ee_pos, dim=-1, p=2)
#     is_close = distance <= grasp_threshold

#     is_closed = (lf_pos <= closed2_threshold)
#     reward = torch.where(is_closed, torch.full_like(lf_pos, 1), torch.full_like(lf_pos, -1))
    
#     return is_close * reward

# def grasp_handle_curriculum_wrapped(env: ManagerBasedRLEnv) -> torch.Tensor:

#     if not hasattr(env, "episode_counter"):
#         return torch.zeros(env.num_envs, device=env.device)

#     _,_,_,_,_, grasp_threshold = get_curriculum_thresholds(env)
#     reward= grasp_handle(env, grasp_threshold)

#     global_episode_counter = env.episode_counter.min() #attivazione sincronizzata tra gli envs
#     mask = global_episode_counter >= 600
#     # mask = env.episode_counter >= 600
#     if mask:
#         return reward
#     else:
#         return torch.zeros_like(reward)


#GRIPPER1

def keep_gripper1_closed(env: ManagerBasedRLEnv, closed1_threshold: float=-0.02) -> torch.Tensor:  #reward e penalità per gripper1
    
    try:
        # Ottieni le posizioni dei giunti gripper (es. indice 7 e 8, da adattare ai tuoi indici)

        joint_idx = env.scene["robot"].joint_names.index("leg2grip1")
        gripper_right_pos = env.scene["robot"].data.joint_pos[:, joint_idx]
        # Verifica se entrambi sono sotto soglia (cioè chiusi)
        # is_closed = (gripper_left_pos <= closed_threshold) & (gripper_right_pos <= closed_threshold)
        is_closed = (gripper_right_pos <= closed1_threshold)

        # Reward: +0.1 se chiuso, -0.1 se aperto
        reward = torch.where(is_closed, torch.full_like(gripper_right_pos, 1), torch.full_like(gripper_right_pos, -1))

        return reward
    except (KeyError, AttributeError):
        return torch.zeros(env.num_envs, device=env.device)
    

def keep_gripper1_closed_curriculum_wrapped(env: ManagerBasedRLEnv,) -> torch.Tensor:

    if not hasattr(env, "episode_counter"):
        return torch.zeros(env.num_envs, device=env.device)

    # _,_,_,_,_, _, closed_threshold = get_curriculum_thresholds(env)
    #
    reward= keep_gripper1_closed(env,closed1_threshold=-0.02)

    weight = weight_curriculum(env, start_value=0.2, end_value=4, start_episode=0, end_episode=800)

    return reward * weight








def grasp_handle(env: ManagerBasedRLEnv, closed2_threshold:float) -> torch.Tensor: #reward e penalità per gripper2
    ee_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    handle_pos = env.scene["handle_frame"].data.target_pos_w[..., 0, :]

    #lf_pos = env.scene["lf2_frame"].data.joint_pos[..., 0]
    # rf_pos = env.scene["rf2_frame"].data.joint_pos[..., 0, :]
    joint_idx = env.scene["robot"].joint_names.index("leg2grip2")
    lf_pos = env.scene["robot"].data.joint_pos[:, joint_idx]

    is_closed = (lf_pos <= closed2_threshold)
    reward = torch.where(is_closed, torch.full_like(lf_pos, 1), torch.full_like(lf_pos, -1))

    return reward

def grasp_handle_curriculum_wrapped(env: ManagerBasedRLEnv) -> torch.Tensor:

    if not hasattr(env, "episode_counter"):
        return torch.zeros(env.num_envs, device=env.device)

    reward= grasp_handle(env, closed2_threshold=-0.02)

    global_episode_counter = env.episode_counter.min() #attivazione sincronizzata tra gli envs
    mask = global_episode_counter >= 0
    # mask = env.episode_counter >= 600
    if mask:
        return reward
    else:
        return torch.zeros_like(reward)
    
def grasp2(env: ManagerBasedRLEnv) -> torch.Tensor:

    try:
        lf_pos = env.scene["lf2_frame"].data.target_pos_w[..., 0, :]  # (N, 3)
        rf_pos = env.scene["rf2_frame"].data.target_pos_w[..., 0, :]  # (N, 3)

        # Distanze da handle
        dist= torch.norm(lf_pos - rf_pos, dim=-1, p=2)  # (N,)

        # Penalità sulle distanze
        base_reward = 1.0 / (1.0 + dist**4)

        return base_reward

    except (KeyError, AttributeError):
        return torch.zeros(env.num_envs, device=env.device)

def grasp2_curriculum_wrapped(env: ManagerBasedRLEnv) -> torch.Tensor:

    if not hasattr(env, "episode_counter"):
        return torch.zeros(env.num_envs, device=env.device)

    reward= grasp2(env)
    global_episode_counter = env.episode_counter.min() #attivazione sincronizzata tra gli envs
    mask = global_episode_counter >= 0
    # mask = env.episode_counter >= 600
    if mask:
        return reward
    else:
        return torch.zeros_like(reward)