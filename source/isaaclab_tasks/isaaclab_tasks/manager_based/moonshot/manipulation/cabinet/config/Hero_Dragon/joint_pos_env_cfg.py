# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab.assets import AssetBaseCfg
import isaaclab.sim as sim_utils
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg


from isaaclab_tasks.manager_based.moonshot.manipulation.cabinet import mdp

from isaaclab_tasks.manager_based.moonshot.manipulation.cabinet.cabinet_env_cfg import (  # isort: skip
    FRAME_MARKER_SMALL_CFG,
    DragonGraspEnvCfg,
)


##
# Pre-defined configs
##
import isaaclab_tasks.manager_based.moonshot.utils as moonshot_utils

ISAAC_LAB_PATH = moonshot_utils.find_isaaclab_path().replace("\\","/")


@configclass
class HeroDragonGraspEnvCfg(DragonGraspEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["leg2joint1", "leg2joint2", "leg2joint3", "leg2joint4", "leg2joint5", "leg2joint6", "leg2joint7"],
        #joint_pos_limits=[(-3.1,3.1),(-3.1,3.1),(-3.1,3.1),(-3.1,3.1),(-3.1,3.1),(-3.1,3.1),(-3.1,3.1)],
        scale=1,
        )
        self.actions.arm_action.joint_pos_limits=[(-3.1,3.1),(-3.1,3.1),(-3.1,3.1),(-3.1,3.1),(-3.1,3.1),(-3.1,3.1),(-3.1,3.1)]


        self.actions.gripper_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["leg2grip1", "leg2grip2"],
        #joint_pos_limits=[(0.0, -0.04), (0.0, -0.04)],
        scale=1,
        )
        # reduce the number of terrains to save memory
        # if self.scene.terrain.terrain_generator is not None:
        #     self.scene.terrain.terrain_generator.num_rows = 5
        #     self.scene.terrain.terrain_generator.num_cols = 5
        #     self.scene.terrain.terrain_generator.curriculum = False
        

        # self.scene.terrain = AssetBaseCfg(
        # prim_path="/World/Terrain",
        # spawn=UsdFileCfg(
        #     usd_path=ISAAC_LAB_PATH + "/source/isaaclab_tasks/isaaclab_tasks/manager_based/moonshot/descriptions/usd/terrain/sagamihara_v2.usd",  # cambia path qui!
        #     scale=(1.0, 1.0, 1.0),
        # ),
        # init_state=AssetBaseCfg.InitialStateCfg(
        #     pos=(0.0, 0.0, 0.0),
        #     rot=(0.0, 0.0, 0.0, 1.0),
        # ),
        # collision_group=-1,
        # )

        # Listens to the required transforms
        # IMPORTANT: The order of the frames in the list is important. The first frame is the tool center point (TCP)
        # the other frames are the fingers

        # override rewards
        # self.rewards.approach_gripper_handle.params["offset"] = 0.04
        # self.rewards.grasp_handle.params["open_joint_pos"] = 0.04
        # self.rewards.grasp_handle.params["asset_cfg"].joint_names = ["panda_finger_.*"]

@configclass
class HeroDragonGraspEnvCfg_PLAY(HeroDragonGraspEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        env_spacing = 2.5
        self.observations.policy.enable_corruption = False

        # self.scene.light = AssetBaseCfg(
        #     prim_path="/World/light",
        #     spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=15000.0),
        #     init_state=AssetBaseCfg.InitialStateCfg(rot = (0.52133, 0.47771, 0.47771, 0.52133))
        # )
        self.replicate_physics = True

        # # reduce the number of terrains to save memory
        # if self.scene.terrain.terrain_generator is not None:
        #     self.scene.terrain.terrain_generator.num_rows = 6
        #     self.scene.terrain.terrain_generator.num_cols = 6
        #     self.scene.terrain.terrain_generator.curriculum = False


# Dove mettere i joint_pos_limits allora?
# Se hai bisogno di limitare i range di azione per evitare che il RL generi valori fuori range, puoi:

# Farlo internamente nella funzione reward/termination, oppure

# Applicare normalizzazione manuale nella tua policy RL, o

# (se stai usando skrl o rsl-rl) impostare limiti nell’action space della policy.