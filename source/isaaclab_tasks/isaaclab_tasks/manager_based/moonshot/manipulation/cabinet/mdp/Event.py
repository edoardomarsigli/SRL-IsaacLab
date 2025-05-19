# hero_dragon_events.py

# Importiamo il base events ufficiale
from isaaclab.envs.mdp import events as base_events

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import matrix_from_quat


# ======================================================================================
# --- Re-importiamo tutte le funzioni standard (forwarding)


randomize_rigid_body_material = base_events.randomize_rigid_body_material

reset_joints_by_offset = base_events.reset_joints_by_offset

reset_scene_to_default_original = base_events.reset_scene_to_default


# def is_grasp_successful(env: ManagerBasedRLEnv,              #posso usarlo nei reward di gripper1 ma non nelle sue observation senno non piu madrl
#                         threshold_distance: float = 0.01, 
#                         threshold_finger_distance: float = 0.015, 
#                         required_duration_s: float = 1.0) -> torch.Tensor:
#     """
#     Grasp è considerato riuscito se:
#     - EE vicino alla handle (distanza <= threshold_distance)
#     - LF e RF vicini a EE (distanza <= threshold_finger_distance)
#     - Tutto questo mantenuto per almeno `required_duration_s` secondi.
#     """

#     # Posizioni
#     handle_pos = env.scene["handle_frame"].data.target_pos_w[..., 0, :]
#     ee_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
#     lf_pos = env.scene["lf_frame"].data.target_pos_w[..., 0, :]
#     rf_pos = env.scene["rf_frame"].data.target_pos_w[..., 0, :]

#     # Distanze
#     ee_handle_dist = torch.norm(handle_pos - ee_pos, dim=-1, p=2)
#     lf_ee_dist = torch.norm(lf_pos - ee_pos, dim=-1, p=2)
#     rf_ee_dist = torch.norm(rf_pos - ee_pos, dim=-1, p=2)

#     # Condizioni di distanza
#     is_ee_close = ee_handle_dist <= threshold_distance
#     is_lf_close = lf_ee_dist <= threshold_finger_distance
#     is_rf_close = rf_ee_dist <= threshold_finger_distance

#     grasp_success = is_ee_close & is_lf_close & is_rf_close

#     return grasp_success

def is_grasp_successful(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Restituisce un BoolTensor (N_envs,) True dove il grasp è valido.
    """
    # Esempio fittizio:
    # check se gripper è chiuso e la distanza con handle è bassa
    distance = torch.norm(env.scene["ee_frame"].data.target_pos_w[..., 0, :] - env.scene["handle_frame"].data.target_pos_w[..., 0, :], dim=-1)
    #grip_closed = env.actions.gripper_action.current_actions < -0.9

    #return (distance < 0.03) & grip_closed

    return (distance < 0.03)


def terminate_if_low(env, threshold4: float, threshold6: float, threshold_ee: float) -> torch.Tensor:
    """
    Terminate the episode if joint4_frame goes below a given z threshold.
    """
    try:
        # Ottieni la posizione z del frame
        joint4_pos = env.scene["joint4_frame"].data.target_pos_w[..., 0, :]
        joint4_z = joint4_pos[..., 2]  # prende asse z
        
        ee_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
        ee_z = ee_pos[..., 2]  # prende asse z


        joint6_pos = env.scene["joint6_frame"].data.target_pos_w[..., 0, :]
        joint6_z = joint6_pos[..., 2]  # prende asse z

        # Restituisci True dove la posizione z è sotto la soglia
        joint4_below = joint4_z < threshold4
        joint6_below = joint6_z < threshold6
        ee_below = ee_z < threshold_ee

        below_threshold = joint4_below | ee_below | joint6_below
        # if torch.any(below_threshold):
        #     print(f"⚠️ Termination triggered on envs: {torch.nonzero(below_threshold)} with z={joint4_z}")
        
        return below_threshold  # tensor (num_envs,), dtype=bool
    except (KeyError, AttributeError):
        # Se fallisce, non termina nessuno
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    


def terminate_wheel_z(env, threshold: float) -> torch.Tensor:

    try:
        # Ottieni la posizione z del frame
        wheel_pos = env.scene["wheel_frame"].data.target_pos_w[..., 0, :]
        wheel_z = wheel_pos[..., 2]  # prende asse z

        wheel_below = wheel_z > threshold

        # if torch.any(below_threshold):
        #     print(f"⚠️ Termination triggered on envs: {torch.nonzero(below_threshold)} with z={joint4_z}")
        
        return wheel_below  # tensor (num_envs,), dtype=bool
    except (KeyError, AttributeError):
        # Se fallisce, non termina nessuno
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)    
    

def terminate_wheel(env: ManagerBasedRLEnv, limit: float) -> torch.Tensor:
    try:

        wheel_quat = env.scene["wheel_frame"].data.target_quat_w[..., 0, :]  

        rot_mat = matrix_from_quat(wheel_quat)

        wheel_x = rot_mat[..., 0]
        
        x_env = torch.tensor([1.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1) 
        align_x = torch.bmm(wheel_x.unsqueeze(1), x_env.unsqueeze(-1)).squeeze(-1).squeeze(-1) 

        return align_x < limit 
    except (KeyError, AttributeError):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

def collision_termination(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    try:
        sensor = env.scene.sensors["contact_sensor_left1"]
        contact_forces = sensor.data.force_matrix_w  # shape: (N, B, F, 3)
        magnitude = torch.norm(contact_forces, dim=-1).sum(dim=(-1, -2))  # sum across bodies and filters

        sensor2 = env.scene.sensors["contact_sensor_right1"]
        contact_forces2 = sensor2.data.force_matrix_w  # shape: (N, B, F, 3)
        magnitude2 = torch.norm(contact_forces2, dim=-1).sum(dim=(-1, -2))  # sum across bodies and filters

        sensor3 = env.scene.sensors["contact_sensor_left2"]
        contact_forces3 = sensor3.data.force_matrix_w  # shape: (N, B, F, 3)
        magnitude3 = torch.norm(contact_forces3, dim=-1).sum(dim=(-1, -2))  # sum across bodies and filters

        sensor4 = env.scene.sensors["contact_sensor_right2"]
        contact_forces4 = sensor4.data.force_matrix_w  # shape: (N, B, F, 3)
        magnitude4 = torch.norm(contact_forces4, dim=-1).sum(dim=(-1, -2))  # sum across bodies and filters

        mask = ((magnitude > threshold)|(magnitude2 > threshold)|(magnitude3 > threshold)|(magnitude4 > threshold)  ).any(dim=-1).any(dim=-1)  # (N,)
        return mask
    except Exception:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)