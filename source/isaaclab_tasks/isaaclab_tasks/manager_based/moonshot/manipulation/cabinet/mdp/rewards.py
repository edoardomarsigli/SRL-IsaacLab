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

    grasp_threshold = threshold_curriculum(env, 0.50, 0.01, start_episode=600, end_episode=2600) #per start di grasp a 600
    # z_limit = threshold_curriculum(env, 0.5, 0.2, start_episode=0, end_episode=10000) #per start di grasp a 600

    return grasp_threshold


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

def align_ee_handle(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Reward for aligning both the end-effector Z and X axes with the handle's Z and X axes.
    Combines the alignment in Z (tip direction) and X (gripper facing).
    """
    ee_quat = env.scene["ee_frame"].data.target_quat_w[..., 0, :]
    handle_quat = env.scene["handle_frame"].data.target_quat_w[..., 0, :]

    ee_rot_mat = matrix_from_quat(ee_quat)
    handle_rot_mat = matrix_from_quat(handle_quat)

    # Z alignment
    ee_z = ee_rot_mat[..., 2]
    handle_z = handle_rot_mat[..., 2]
    align_z = torch.bmm(ee_z.unsqueeze(1), handle_z.unsqueeze(-1)).squeeze(-1).squeeze(-1)  # (N,)
    reward_z = torch.sign(align_z) * align_z**2

    # X alignment
    ee_x = ee_rot_mat[..., 0]
    handle_x = handle_rot_mat[..., 0]
    align_x = torch.bmm(ee_x.unsqueeze(1), handle_x.unsqueeze(-1)).squeeze(-1).squeeze(-1)  # (N,)
    reward_x = torch.sign(align_x) * align_x**4

    # Total reward
    return reward_z + reward_x

def align_ee_handle_curriculum_wrapped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Wrapper to apply curriculum-based thresholds for the alignment reward.
    """
    if not hasattr(env, "episode_counter"):
        return torch.zeros(env.num_envs, device=env.device)


    global_episode_counter = env.episode_counter.min()  # attivazione sincronizzata tra gli envs

    if global_episode_counter >= 0:
        return align_ee_handle(env)
    else:
        return torch.zeros_like(align_ee_handle(env))  # ritorna zero se non ancora attivo

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
    weight = weight_curriculum(env, start_value=0.5, end_value=5, start_episode=0, end_episode=10000)
    return base_reward * weight

def reward_joint4_zyx(env: ManagerBasedRLEnv) -> torch.Tensor: #reward e penalità per giunto 4 altezza


    # Ottieni posizione mondiale del giunto 4
    joint4_pos = env.scene["joint4_frame"].data.target_pos_w[..., 0, :]
    joint4_z = joint4_pos[..., 2]  # solo asse Z
    joint4_y= joint4_pos[..., 1]  # solo asse Y
    joint4_x= joint4_pos[..., 0]  # solo asse X

    handle_pos = env.scene["handle_frame"].data.target_pos_w[..., 0, :]
    handle_z = handle_pos[..., 2]  # solo asse Z
    handle_y = handle_pos[..., 1]  # solo asse Y
    handle_x = handle_pos[..., 0]  # solo asse X

    # Reward continuo: maggiore è la distanza sopra la soglia, più reward
    in_range = (joint4_z >= handle_z)

    # Linear shaping
    reward_z= (joint4_z - handle_z)* in_range  
    
    distance_y = (joint4_y - (handle_y)).abs()  # distanza tra giunto 4 e handle lungo Y
    reward_y = torch.where(in_range, 1/(1+distance_y*4), torch.zeros_like(distance_y))  
    
    distance_x = (joint4_x - handle_x).abs()  # distanza tra giunto 4 e handle lungo X
    reward_x= torch.where(in_range, 1/(1+distance_x*4), torch.zeros_like(distance_x))

    if env.episode_counter.min() < 400:
        return reward_z + reward_y

    # weight = weight_curriculum(env, start_value=1, end_value=0.1, start_episode=0, end_episode=200)

    return reward_z + reward_y*1.25 + reward_x*1.25 #* weight  # reward positivo se sopra soglia, zero altrimenti

def approach_zy(env: ManagerBasedRLEnv) -> torch.Tensor: #allineamento xy e bonus solo se align ee handle buono
    """
    Reward for aligning the EE and handle origins in the XY plane (ignores Z axis).
    """
    try:

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


        mask = (ee_pos[..., 2] >= handle_pos[..., 2])
        reward = (reward) * mask.float()

        x_distance = torch.abs(handle_pos[..., 0] - ee_pos[..., 0])  # (N,)

        if env.episode_counter.min() < 400:
            return torch.where(x_distance > 0.3 , reward, torch.zeros_like(reward))

        return reward

    except (KeyError, AttributeError):
        return torch.zeros(env.num_envs, device=env.device)

def approach_zy_curriculum_wrapped(env: ManagerBasedRLEnv) -> torch.Tensor:
 
    if not hasattr(env, "episode_counter"):
        return torch.zeros(env.num_envs, device=env.device)

    global_episode_counter = env.episode_counter.min() #attivazione sincronizzata tra gli envs

    if global_episode_counter >= 0:
        return approach_zy(env)
    else:
        return torch.zeros_like(approach_zy(env))    

def approach_x(env: ManagerBasedRLEnv) -> torch.Tensor: #allineamento z solo se xy buono
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
        x_distance = torch.abs((handle_pos[..., 0]) - ee_pos[..., 0])  # (N,)

        # Step 3: Reward shaping sulla distanza Z (solo se XY è sotto soglia)
        x_reward = 1.0 / (1.0 + x_distance*2)
        x_reward = torch.pow(x_reward, 2)

        adaptive_threshold = 0.8 * x_distance + 0.05 #cono di threshold
        # Step 4: Applica solo se XY è vicino
        return torch.where(distance_yz <= adaptive_threshold, x_reward, torch.zeros_like(x_reward))

    except (KeyError, AttributeError):
        return torch.zeros(env.num_envs, device=env.device)

def approach_x_curriculum_wrapped(env: ManagerBasedRLEnv) -> torch.Tensor:

    if not hasattr(env, "episode_counter"):
        return torch.zeros(env.num_envs, device=env.device)

    reward= approach_x(env)

    global_episode_counter = env.episode_counter.min() #attivazione sincronizzata tra gli envs
    mask = global_episode_counter >= 50
    # mask = env.episode_counter >= 400
    if mask:
        return reward
    else:
        return torch.zeros_like(reward)



   
    
def penalty_touch(env: ManagerBasedRLEnv) -> torch.Tensor:  # penalità per tocco
    try:
        handle_pos = env.scene["handle_frame"].data.target_pos_w[..., 0, :]
        ee_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]

        ee_x = ee_pos[..., 0]
        handle_x = handle_pos[..., 0]
        close = torch.abs(handle_x - ee_x) < 0.3  # bool

        ee_y = ee_pos[..., 1]
        handle_y = handle_pos[..., 1]
        touch = torch.abs(handle_y - ee_y) > 0.05  # bool

        weight = weight_curriculum(env, start_value=1, end_value=10, start_episode=20, end_episode=10000)

        # Penalità -1.0 solo dove la condizione è vera
        return weight*(close & touch).float()

    except (KeyError, AttributeError):
        return torch.zeros(env.num_envs, device=env.device)
    
# def penalty_wheel(env: ManagerBasedRLEnv) -> torch.Tensor:
#     try:
#         step = env.episode_length_buf  # (num_envs,)
#         env_ids = torch.arange(env.num_envs, device=step.device)

#         wheel_quat = env.scene["wheel_frame"].data.target_quat_w[..., 0, :]
#         rot_mat = matrix_from_quat(wheel_quat)
#         wheel_x = rot_mat[..., 0]

#         # Inizializza la reference se non esiste
#         if not hasattr(env, "wheel_ref_dir"):
#             env.wheel_ref_dir = {
#                 "x": torch.zeros_like(wheel_x),
#                 "stored": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
#             }

#         # Salva la direzione al 250° step
#         save_mask = (step == 250)
#         env.wheel_ref_dir["x"][save_mask] = wheel_x[save_mask]
#         env.wheel_ref_dir["stored"][save_mask] = True

#         valid_mask = (step > 250) & env.wheel_ref_dir["stored"]

#         # Calcola l'angolo (dot product tra direzioni salvate e attuali)
#         dot = torch.sum(wheel_x * env.wheel_ref_dir["x"], dim=-1)
#         angle_diff = 1.0 - torch.abs(dot)  # distanza da perfetto allineamento

#         # Penalità solo per env validi
#         penalty = torch.zeros(env.num_envs, device=env.device)
#         penalty[valid_mask] = angle_diff[valid_mask] * 2.0  # peso regolabile

#         return torch.where(penalty >= 0.01, penalty, 0)

#     except (KeyError, AttributeError):
#         return torch.zeros(env.num_envs, device=env.device)
    
def penalty_wheel(env: ManagerBasedRLEnv) -> torch.Tensor:
    try:
        wheel_quat = env.scene["wheel_frame"].data.target_quat_w[..., 0, :]
        rot_mat = matrix_from_quat(wheel_quat)
        wheel_x = rot_mat[..., 0]  # (N, 3)

        origin_x = torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, 3)


        # Allineamento via dot product (batch)
        align_x = torch.bmm(wheel_x.unsqueeze(1), origin_x.unsqueeze(-1)).squeeze(-1).squeeze(-1)  # (N,)


        return 1- align_x

    except (KeyError, AttributeError):
        return torch.zeros(env.num_envs, device=env.device)

    

#APPROACH

def align_grasp(env: ManagerBasedRLEnv) -> torch.Tensor:
    try:
        ee_quat = env.scene["ee_frame"].data.target_quat_w[..., 0, :]
        handle_quat = env.scene["handle_frame"].data.target_quat_w[..., 0, :]

        ee_rot_mat = matrix_from_quat(ee_quat)
        handle_rot_mat = matrix_from_quat(handle_quat)

        # Z alignment
        ee_z = ee_rot_mat[..., 2]
        handle_z = handle_rot_mat[..., 2]
        align_z = torch.bmm(ee_z.unsqueeze(1), handle_z.unsqueeze(-1)).squeeze(-1).squeeze(-1)  # (N,)

        # X alignment
        ee_x = ee_rot_mat[..., 0]
        handle_x = handle_rot_mat[..., 0]
        align_x = torch.bmm(ee_x.unsqueeze(1), handle_x.unsqueeze(-1)).squeeze(-1).squeeze(-1)  # (N,)

        handle_pos = env.scene["handle_frame"].data.target_pos_w[..., 0, :]
        ee_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
        x_distance = torch.abs((handle_pos[..., 0]-0.01) - ee_pos[..., 0])  # (N,)
        threshold= 0.98 - 0.4 * x_distance


        close = torch.abs(handle_pos[..., 0] - ee_pos[..., 0]) < 0.1
        aligned = (align_z > threshold) & (align_x > threshold)
        reward = 1 / (10*x_distance +1)  

        if not hasattr(env, "episode_counter"):
            return torch.zeros(env.num_envs, device=env.device)

        global_episode_counter = env.episode_counter.min() #attivazione sincronizzata tra gli envs
        mask = global_episode_counter >= 50
        # mask = env.episode_counter >= 400
        if mask:
            return torch.where(close, torch.where(aligned, reward, -2.0), 0.0)
        else:
            return torch.zeros(env.num_envs, device=env.device)

    except (KeyError, AttributeError):
        # Se uno dei frame non è pronto, ritorna reward 0 temporanea
        return torch.zeros(env.num_envs, device=env.device)
    
def approach_grasp(env: ManagerBasedRLEnv) -> torch.Tensor:
    try:
        handle_pos = env.scene["handle_frame"].data.target_pos_w[..., 0, :]
        ee_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]

        handle_yz = handle_pos[..., [1, 2]]
        ee_yz = ee_pos[...,  [1, 2]]

        # Euclidean distance in XY plane
        distance_yz = torch.norm(handle_yz - ee_yz, dim=-1, p=2)
        close= torch.abs(handle_pos[..., 0] - ee_pos[..., 0]) < 0.1  # (N,)
        x_distance = torch.abs((handle_pos[..., 0]) - ee_pos[..., 0])  # (N,)
        threshold= 0.2 * x_distance +0.01
        reward_pos = 1 / (10*x_distance +1)  
    
        global_episode_counter = env.episode_counter.min() #attivazione sincronizzata tra gli envs
        mask = global_episode_counter >= 50
        # mask = env.episode_counter >= 400
        if mask:
            return torch.where(close, torch.where(distance_yz <= threshold, reward_pos, -2), 0) 
        else:
            return torch.zeros(env.num_envs, device=env.device)

    except (KeyError, AttributeError):
        # Se uno dei frame non è pronto, ritorna reward 0 temporanea
        return torch.zeros(env.num_envs, device=env.device)
    

# inizializza il buffer nel tuo ambiente, ad es. in __init__ o reset()
handle_ref_pose = None

def penalize_handle_drift(env:ManagerBasedRLEnv) -> torch.Tensor:
    """
    Penalizza il drift della maniglia rispetto alla sua posa al 40° step.
    Confronta posizione e orientamento (quaternion).
    """
    step = env.episode_length_buf  # (num_envs,)
    env_ids = torch.arange(env.num_envs, device=step.device)

    # Ottieni posa corrente della maniglia
    handle_pos = env.scene["handle_frame"].data.target_pos_w[..., 0, :]  # (N, 3)
    handle_quat = env.scene["handle_frame"].data.target_quat_w[..., 0, :]  # (N, 4)

    # Inizializza reference se non esiste
    if not hasattr(env, "handle_ref_pose"):
        env.handle_ref_pose = {
            "pos": torch.zeros_like(handle_pos),
            "quat": torch.zeros_like(handle_quat),
            "stored": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        }

    # Salva la posa al 40° step
    save_mask = (step == 250)
    env.handle_ref_pose["pos"][save_mask] = handle_pos[save_mask]
    env.handle_ref_pose["quat"][save_mask] = handle_quat[save_mask]
    env.handle_ref_pose["stored"][save_mask] = True

    # Applica penalità solo dopo step 40 e se reference è salvata
    valid_mask = (step > 250) & env.handle_ref_pose["stored"]

    # Calcola deviazione
    pos_diff = handle_pos - env.handle_ref_pose["pos"]
    quat_dot = torch.sum(handle_quat * env.handle_ref_pose["quat"], dim=-1)
    rot_diff = 1.0 - torch.abs(quat_dot)  # distanza tra quaternion

    # Penalità solo dove valido
    penalty = torch.zeros(env.num_envs, device=env.device)
    penalty[valid_mask] = (
        1.0 * torch.norm(pos_diff[valid_mask], dim=-1) +
        0.5 * rot_diff[valid_mask]
    )

    return torch.where(penalty >= 0.002, penalty, 0)

def penalize_collision(env)-> torch.Tensor:
    try:
    # Ottieni le posizioni dei giunti
        joint2_pos = env.scene["joint2_frame"].data.target_pos_w[..., 0, :]
        joint6_pos = env.scene["joint6_frame"].data.target_pos_w[..., 0, :]

        distance = torch.norm(joint2_pos - joint6_pos, dim=-1, p=2)

        # Penalità proporzionale alla distanza sotto soglia (opzionale)
        return torch.where(distance < 0.4, 1/(1+distance) ,torch.zeros(env.num_envs, device=env.device))

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
        ee_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
        handle_pos = env.scene["handle_frame"].data.target_pos_w[..., 0, :]
    
        lf_pos = env.scene["lf2_frame"].data.target_pos_w[..., 0, :]  # (N, 3)
        rf_pos = env.scene["rf2_frame"].data.target_pos_w[..., 0, :]  # (N, 3)

        # Distanze da handle
        dist= torch.norm(lf_pos - rf_pos, dim=-1, p=2)  # (N,)

        distance = torch.norm(handle_pos - ee_pos, dim=-1, p=2)
        is_close = distance <= 0.03


        # Penalità sulle distanze
        base_reward = 1.0 / (1.0 + dist**8)

        return base_reward*is_close

    except (KeyError, AttributeError):
        return torch.zeros(env.num_envs, device=env.device)

def grasp2_curriculum_wrapped(env: ManagerBasedRLEnv) -> torch.Tensor:

    if not hasattr(env, "episode_counter"):
        return torch.zeros(env.num_envs, device=env.device)

    reward= grasp2(env)
    global_episode_counter = env.episode_counter.min() #attivazione sincronizzata tra gli envs
    mask = global_episode_counter >= 600
    # mask = env.episode_counter >= 600
    if mask:
        return reward
    else:
        return torch.zeros_like(reward)
    


def keep_g1_closed(env: ManagerBasedRLEnv) -> torch.Tensor:

    try:
        lf_pos = env.scene["lf1_frame"].data.target_pos_w[..., 0, :]  # (N, 3)
        rf_pos = env.scene["rf1_frame"].data.target_pos_w[..., 0, :]  # (N, 3)

        # Distanze da handle
        dist= torch.norm(lf_pos - rf_pos, dim=-1, p=2)  # (N,)

        # Penalità sulle distanze
        base_reward = 1.0 / (1.0 + dist**8)

        return base_reward

    except (KeyError, AttributeError):
        return torch.zeros(env.num_envs, device=env.device)

def keep_g1_closed_curriculum_wrapped(env: ManagerBasedRLEnv) -> torch.Tensor:

    if not hasattr(env, "episode_counter"):
        return torch.zeros(env.num_envs, device=env.device)

    reward= keep_g1_closed(env)
    weight = weight_curriculum(env, start_value=0.2, end_value=4, start_episode=0, end_episode=800)

    global_episode_counter = env.episode_counter.min() #attivazione sincronizzata tra gli envs
    mask = global_episode_counter >= 0
    # mask = env.episode_counter >= 600
    if mask:
        return reward*weight    
    else:
        return torch.zeros_like(reward)
    
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

    align_z = torch.bmm(ee_z.unsqueeze(1), handle_z.unsqueeze(-1)).squeeze(-1).squeeze(-1)  # (N,)

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

def penalty_joint1_3(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Penalizza la deviazione dei giunti 1 e 3 dalla posizione 0.
    """
    try:
        joint_names = env.scene["robot"].joint_names
        joint_pos = env.scene["robot"].data.joint_pos  # (N, num_joints)

        idx1 = joint_names.index("leg2joint1")
        idx3 = joint_names.index("leg2joint3")

        j1_pos = joint_pos[:, idx1]
        j3_pos = joint_pos[:, idx3]

        penalty = torch.abs(j1_pos) + torch.abs(j3_pos)

        return penalty  # reward negativa crescente in base alla deviazione
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

def collision_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    
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

        weight = weight_curriculum(env, start_value=-500, end_value=-1000, start_episode=0, end_episode=10000)


        return weight*(magnitudes + magnitudes2 + magnitudes3 + magnitudes4)# penalità
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)