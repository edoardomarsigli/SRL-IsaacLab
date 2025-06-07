from isaaclab.envs import ManagerBasedRLEnv
import torch



class HeroDragonEnv(ManagerBasedRLEnv):

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)


        print("✅ HeroDragonGraspEnv constructor CALLED", flush=True)

        # self.grasp_manager: GraspManager = self.scene.grasp_managers["grasp"]
        self.episode_counter = torch.zeros((self.num_envs,), dtype=torch.long, device="cuda")
