# hero_dragon_events.py

# Importiamo il base events ufficiale
from isaaclab.envs.mdp import events as base_events

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs import ManagerBasedRLEnv

# ======================================================================================
# --- Re-importiamo tutte le funzioni standard (forwarding)


randomize_rigid_body_material = base_events.randomize_rigid_body_material

reset_joints_by_offset = base_events.reset_joints_by_offset

reset_scene_to_default_original = base_events.reset_scene_to_default


# ======================================================================================
# --- Override o aggiunte nostre

def reset_scene_to_default(env: ManagerBasedEnv, env_ids: torch.Tensor):
    """Reset scene di default + reset grasp_completed"""
    # Primo: chiama la funzione originale per resettare tutto normalmente
    reset_scene_to_default_original(env, env_ids)

    # Poi: resetta il flag grasp_completed
    if hasattr(env, "grasp_completed"):
        env.grasp_completed[env_ids] = False
    if hasattr(env, "episode_counter"):
        env.episode_counter[env_ids] += 1

def grasp_completed(env: ManagerBasedEnv) -> torch.Tensor:
    """Termination condition: end episode if grasp is completed."""
    if hasattr(env, "grasp_completed") and env.grasp_completed is not None:
        return env.grasp_completed
    else:
        return torch.zeros((env.num_envs,), dtype=torch.bool, device="cuda")
    
# Devi aggiungere un nuovo buffer persistente nel tuo ambiente!
# es: self.grasp_counter = torch.zeros((self.num_envs,), dtype=torch.int, device="cuda")

def is_grasp_successful(env: ManagerBasedRLEnv,              #posso usarlo nei reward di gripper1 ma non nelle sue observation senno non piu madrl
                        threshold_distance: float = 0.01, 
                        threshold_finger_distance: float = 0.015, 
                        required_duration_s: float = 1.0) -> torch.Tensor:
    """
    Grasp è considerato riuscito se:
    - EE vicino alla handle (distanza <= threshold_distance)
    - LF e RF vicini a EE (distanza <= threshold_finger_distance)
    - Tutto questo mantenuto per almeno `required_duration_s` secondi.
    """

    # Posizioni
    handle_pos = env.scene["handle_frame"].data.target_pos_w[..., 0, :]
    ee_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    lf_pos = env.scene["lf_frame"].data.target_pos_w[..., 0, :]
    rf_pos = env.scene["rf_frame"].data.target_pos_w[..., 0, :]

    # Distanze
    ee_handle_dist = torch.norm(handle_pos - ee_pos, dim=-1, p=2)
    lf_ee_dist = torch.norm(lf_pos - ee_pos, dim=-1, p=2)
    rf_ee_dist = torch.norm(rf_pos - ee_pos, dim=-1, p=2)

    # Condizioni di distanza
    is_ee_close = ee_handle_dist <= threshold_distance
    is_lf_close = lf_ee_dist <= threshold_finger_distance
    is_rf_close = rf_ee_dist <= threshold_finger_distance

    grasp_success = is_ee_close & is_lf_close & is_rf_close

    return grasp_success

