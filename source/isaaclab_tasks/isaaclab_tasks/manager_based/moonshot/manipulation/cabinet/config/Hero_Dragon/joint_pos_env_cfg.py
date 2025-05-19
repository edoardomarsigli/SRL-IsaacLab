# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab_tasks.manager_based.moonshot.descriptions.config.moonbot_cfgs_edo import DRAGON_ARTICULATED_CFG, WHEEL_WITH_HANDLE_CFG







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

# @configclass
# class MultiAgentActionsCfg:
#     arm: mdp.JointPositionActionCfg = MISSING
#     gripper2: mdp.JointPositionActionCfg = MISSING
#     gripper1: mdp.JointPositionActionCfg = MISSING


@configclass
class HeroDragonGraspEnvCfg(DragonGraspEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.grasp_completed = None

        self.scene.robot=DRAGON_ARTICULATED_CFG
        self.scene.wheel_with_handle=WHEEL_WITH_HANDLE_CFG
        self.scene.ee_frame=FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/hero_dragon/leg2gripper2_base",
            debug_vis=True,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EEFrame"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/hero_dragon/leg2gripper2_base",
                    name="ee",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.057)),
                )
            ],
        )


        # Set Actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["leg2joint1", "leg2joint2", "leg2joint3", "leg2joint4", "leg2joint5", "leg2joint6", "leg2joint7"],
        scale=3.1,
        clip={"leg2joint1": (-3.1, 3.1), "leg2joint2": (-3.1, 3.1), "leg2joint3": (-3.1, 3.1),"leg2joint4": (-3.1, 3.1), "leg2joint5": (-3.1, 3.1), "leg2joint6": (-3.1, 3.1),"leg2joint7": (-3.1, 3.1)},
        )


        self.actions.gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["leg2grip1", "leg2grip2"],
            scale=0.029,
            clip={"leg2grip1": (-0.029, 0.0),"leg2grip2": (-0.029, 0.0),}
        )

        # self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
        #     asset_name="robot",
        #     joint_names=["leg2grip1"],
        #     open_command_expr={"leg2grip1": 0.01,},
        #     close_command_expr={"leg2grip1": -0.029, },
        # )
        # self.actions.gripper2_action = mdp.BinaryJointPositionActionCfg(
        #     asset_name="robot",
        #     joint_names=["leg2grip2"],
        #     open_command_expr={"leg2grip2": 0.01},
        #     close_command_expr={"leg2grip2": -0.029},
        # )
        # self.actions.arm.arm_action = mdp.JointPositionActionCfg(  #actions madrl
        #         asset_name="robot", 
        #         joint_names=["leg2joint1", "leg2joint2", "leg2joint3", "leg2joint4", "leg2joint5", "leg2joint6", "leg2joint7"],
        #         scale=1.0,
        #         clip={"leg2joint1": (-3.1, 3.1), "leg2joint2": (-3.1, 3.1), "leg2joint3": (-3.1, 3.1),
        #               "leg2joint4": (-3.1, 3.1), "leg2joint5": (-3.1, 3.1), "leg2joint6": (-3.1, 3.1),
        #               "leg2joint7": (-3.1, 3.1)},
        #     )

        # self.actions.gripper1.gripper1 = mdp.JointPositionActionCfg(
        #         asset_name="robot",
        #         joint_names=["leg2grip1", "leg2grip1bis"],
        #         scale=0.015,
        #         clip={"leg2grip1": (-0.029, 0.0), "leg2grip1bis": (-0.029, 0.0)},
        #     )

        # self.actions.gripper2.gripper2 = mdp.JointPositionActionCfg(
        #         asset_name="robot",
        #         joint_names=["leg2grip2", "leg2grip2bis"],
        #         scale=0.015,
        #         clip={"leg2grip2": (-0.029, 0.0), "leg2grip2bis": (-0.029, 0.0)},
        #     )




@configclass
class HeroDragonGraspEnvCfg_PLAY(HeroDragonGraspEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
    
        self.scene.env_spacing = 6

        self.scene.terrain.max_init_terrain_level = None
        
        self.observations.policy.enable_corruption = False
        self.viewer.resolution = (2540,1440)
        self.viewer.eye = (0.8, 2, 0.8) # basic view