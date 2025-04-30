from isaaclab.envs import ManagerBasedRLEnv
import torch
import atexit

from isaaclab_tasks.manager_based.moonshot.manipulation.cabinet.mdp import Event as mdp_events

class HeroDragonGraspEnv(ManagerBasedRLEnv):

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)

        # Inizializza grasp_completed una volta sola qui!
        self.grasp_completed = torch.zeros((self.num_envs,), dtype=torch.bool, device="cuda")
        print("✅ HeroDragonGraspEnv constructor CALLED", flush=True)
        self.episode_counter = torch.zeros((self.num_envs,), dtype=torch.long, device="cuda")
        if self.episode_counter.max().item() % 100 == 0:
            print(f"▶️ Episode counters: {self.episode_counter}")



    def pre_physics_step(self, actions):
        # super().pre_physics_step(actions)
        
        # 1. Aggiorna grasp_completed a ogni step
        if hasattr(self, "grasp_completed"):
            grasp_success = mdp_events.is_grasp_successful(self)
            self.grasp_completed |= grasp_success  # aggiorna dove serve

            # 2. Blocca il gripper se grasp non ancora completato
            gripper_indices = [7, 8]  # <-- tuoi indici gripper
            actions[self.grasp_completed == False][:, gripper_indices] = -1.0

    # def pre_physics_step(self, actions: dict):
    #     self._apply_actions(actions["arm"], self.actions_cfg.arm)
    #     self._apply_actions(actions["gripper2"], self.actions_cfg.gripper2)
    #     self._apply_actions(actions["gripper1"], self.actions_cfg.gripper1)
