from isaaclab.envs import ManagerBasedRLEnv
import torch
import atexit

from isaaclab_tasks.manager_based.moonshot.manipulation.cabinet.mdp import Event as mdp_events
from isaaclab_tasks.manager_based.moonshot.manipulation.cabinet.mdp.rewards import get_curriculum_thresholds



class HeroDragonGraspEnv(ManagerBasedRLEnv):

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)


        print("✅ HeroDragonGraspEnv constructor CALLED", flush=True)

        # self.grasp_manager: GraspManager = self.scene.grasp_managers["grasp"]
        self.episode_counter = torch.zeros((self.num_envs,), dtype=torch.long, device="cuda")
        self.grasp_completed = torch.zeros((self.num_envs,), dtype=torch.bool, device="cuda")

        # self.grasp_manager = SimpleGraspManager(
    #     env=self,
    #     gripper_prim_paths=[f"/World/envs/env_{i}/hero_dragon/leg2gripper2_jaw_left" for i in range(self.num_envs)],
    #     object_prim_paths=[f"/World/envs/env_{i}/hero_wheel/wheel11_out" for i in range(self.num_envs)],
    # )


    def pre_physics_step(self, actions):
        new_grasp = mdp_events.is_grasp_successful(self)
        self.grasp_completed = new_grasp

        if self.episode_counter[0] % 1 == 0:
            env_ids = torch.nonzero(self.grasp_completed).squeeze(-1).tolist()
            if env_ids:
                print(f"[GRASP] grasp_completed=True in envs: {env_ids}", flush=True)

        contact_tensor = self.scene.sensors["contact_sensor_left1"].data.force_matrix_w  # (N, B, F, 3)
        magnitudes = torch.norm(contact_tensor, dim=-1)  # (N, B, F)

        for env_id in range(self.num_envs):
            mag = magnitudes[env_id]
            if (mag >= 0.01).any():  # soglia minima per evitare stampe vuote
                print(f"[CONTACT][env {env_id}] force_matrix_w =\n{contact_tensor[env_id].cpu().numpy()}", flush=True)


        # print("grip1 actual pos:", self.scene["robot"].data.joint_pos[:, grip1_index])
        # print("grip1 target pos:", self.scene["robot"].data.joint_target[:, grip1_index])








