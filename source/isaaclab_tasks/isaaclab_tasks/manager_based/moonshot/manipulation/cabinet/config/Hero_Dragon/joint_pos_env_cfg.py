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

from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR



ISAAC_LAB_PATH = moonshot_utils.find_isaaclab_path().replace("\\","/")


from pathlib import Path


@configclass
class HeroDragonGraspEnvCfg(DragonGraspEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.grasp_completed = None

        self.scene.env_spacing = 6

        self.scene.terrain.max_init_terrain_level = None
    
        self.viewer.resolution = (2540,1440)
        self.viewer.eye = (0.8, 2, 0.8) # basic view

        self.scene.robot=DRAGON_ARTICULATED_CFG
        self.scene.wheel_with_handle=WHEEL_WITH_HANDLE_CFG


        self.scene.ee_frame=FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/hero_dragon/leg2gripper2",
            debug_vis=True,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EEFrame"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/hero_dragon/leg2gripper2",
                    name="ee",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
                )
            ],
        )
        # self.scene.ee_frame=FrameTransformerCfg(
        #     prim_path="{ENV_REGEX_NS}/hero_dragon/leg2gripper2_base",
        #     debug_vis=True,
        #     visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EEFrame"),
        #     target_frames=[
        #         FrameTransformerCfg.FrameCfg(
        #             prim_path="{ENV_REGEX_NS}/hero_dragon/leg2gripper2_base",
        #             name="ee",
        #             offset=OffsetCfg(pos=(0.0, 0.0, 0.057)),
        #         )
        #     ],
        # )

        # Set Actions for the specific robot type (franka)

        # self.actions.arm_action = mdp.JointPositionActionCfg(
        # asset_name="robot",
        # joint_names=["leg2joint1", "leg2joint2", "leg2joint3", "leg2joint4", "leg2joint5", "leg2joint6", "leg2joint7"],
        # scale=3.1,
        # clip={"leg2joint1": (-3.1, 3.1), "leg2joint2": (-3.1, 3.1), "leg2joint3": (-3.1, 3.1),
        #       "leg2joint4": (-3.1, 3.1), "leg2joint5": (-3.1, 3.1), "leg2joint6": (-3.1, 3.1),"leg2joint7": (-3.1, 3.1)},
        # )

        self.actions.j1_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["leg2joint1"],
            scale=3.1,
            clip={"leg2joint1": (-3.1, 3.1)},
        )

        self.actions.j2_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["leg2joint2"],
            scale=3.1,
            clip={"leg2joint2": (-3.1, 3.1)},
        )

        self.actions.j3_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["leg2joint3"],
            scale=3.1,
            clip={"leg2joint3": (-3.1, 3.1)},
        )

        self.actions.j4_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["leg2joint4"],
            scale=3.1,
            clip={"leg2joint4": (-3.1, 3.1)},
        )

        self.actions.j5_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["leg2joint5"],
            scale=3.1,
            clip={"leg2joint5": (-3.1, 3.1)},
        )

        self.actions.j6_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["leg2joint6"],
            scale=3.1,
            clip={"leg2joint6": (-3.1, 3.1)},
        )

        self.actions.j7_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["leg2joint7"],
            scale=3.1,
            clip={"leg2joint7": (-3.1, 3.1)},
        )


        # self.actions.gripper_action = mdp.JointPositionActionCfg(
        #     asset_name="robot",
        #     joint_names=["leg2grip1", "leg2grip2"],
        #     scale=0.015,
        #     offset=-0.015,
        #     clip={"leg2grip1": (-0.029, 0.0),"leg2grip2": (-0.029, 0.0)}
        # )

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





@configclass
class HeroDragonGraspEnvCfg_PLAY(HeroDragonGraspEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 32
    
        self.scene.env_spacing = 6

        self.scene.terrain.max_init_terrain_level = None

        # self.scene.light = AssetBaseCfg(
        #     prim_path="/World/light",
        #     spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=15000.0),
        #     init_state=AssetBaseCfg.InitialStateCfg(rot = (0.52133, 0.47771, 0.47771, 0.52133))
        # )

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 8
            self.scene.terrain.terrain_generator.num_cols = 4
            self.scene.terrain.terrain_generator.curriculum = False
        
        self.viewer.resolution = (2540,1440)
        self.viewer.eye = (0.5, 0, 0.8) # basic view
        self.viewer.lookat = (0.8, -2.0, 0.5)