# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from datetime import datetime
from typing import Union
from dataclasses import MISSING, _MISSING_TYPE

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg


import isaaclab_tasks.manager_based.moonshot.locomotion_edo_dragon.velocity.mdp as mdp
import isaaclab_tasks.manager_based.moonshot.utils as moonshot_utils

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

##
# Pre-defined configs
##
from isaaclab_tasks.manager_based.moonshot.descriptions.config.terrain_edo_nav.rough import ROUGH_TERRAINS_CFG
from isaaclab.markers.config import FRAME_MARKER_CFG 

##
# Path
##

ISAAC_LAB_PATH = moonshot_utils.find_isaaclab_path().replace("\\","/")

FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)

##
# Scene definition
##


@configclass
class HeroDragonSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a Moonbot robot."""

    # num_envs: int = 4096  # ⬅️ AGGIUNGI QUESTA LINEA
    # env_spacing: float = 10.0

    # terrain (rough)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=True,
    )
    robot: ArticulationCfg = MISSING
    # wheel_with_handle: ArticulationCfg= MISSING

    # joint2_frame=FrameTransformerCfg(
    #     prim_path="{ENV_REGEX_NS}/hero_dragon_realsense/leg1link2",
    #     debug_vis=True,
    #     visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/LFFrame"),
    #     target_frames=[
    #         FrameTransformerCfg.FrameCfg(
    #             prim_path="{ENV_REGEX_NS}/hero_dragon_realsense/leg1link2",
    #             name="leg1link2",
    #             offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
    #         )
    #     ],
    # )
    # joint4_frame=FrameTransformerCfg(
    #     prim_path="{ENV_REGEX_NS}/hero_dragon_realsense/leg1link4",
    #     debug_vis=True,
    #     visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/LFFrame"),
    #     target_frames=[
    #         FrameTransformerCfg.FrameCfg(
    #             prim_path="{ENV_REGEX_NS}/hero_dragon_realsense/leg1link4",
    #             name="leg1link4",
    #             offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
    #         )
    #     ],
    # )
    # joint6_frame=FrameTransformerCfg(
    #     prim_path="{ENV_REGEX_NS}/hero_dragon_realsense/leg1link6",
    #     debug_vis=True,
    #     visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/LFFrame"),
    #     target_frames=[
    #         FrameTransformerCfg.FrameCfg(
    #             prim_path="{ENV_REGEX_NS}/hero_dragon_realsense/leg1link6",
    #             name="leg1link6",
    #             offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
    #         )
    #     ],
    # )
    # rear_wheel_frame=FrameTransformerCfg(
    #     prim_path="{ENV_REGEX_NS}/hero_dragon_realsense/wheel12_body",
    #     debug_vis=True,
    #     visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/LFFrame"),
    #     target_frames=[
    #         FrameTransformerCfg.FrameCfg(
    #             prim_path="{ENV_REGEX_NS}/hero_dragon_realsense/wheel12_body",
    #             name="rear_wheel",
    #             offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
    #         )
    #     ],
    # )

    # front_wheel_frame=FrameTransformerCfg(
    #     prim_path="{ENV_REGEX_NS}/hero_dragon_realsense/wheel11_body",
    #     debug_vis=True,
    #     visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/LFFrame"),
    #     target_frames=[
    #         FrameTransformerCfg.FrameCfg(
    #             prim_path="{ENV_REGEX_NS}/hero_dragon_realsense/wheel11_body",
    #             name="front_wheel",
    #             offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
    #         )
    #     ],
    # )

    joint2_frame=FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/hero_dragon_realsense/leg4link2",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/LFFrame"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/hero_dragon_realsense/leg4link2",
                name="leg4link2",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            )
        ],
    )
    joint4_frame=FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/hero_dragon_realsense/leg4link4",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/LFFrame"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/hero_dragon_realsense/leg4link4",
                name="leg4link4",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            )
        ],
    )
    joint6_frame=FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/hero_dragon_realsense/leg4link6",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/LFFrame"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/hero_dragon_realsense/leg4link6",
                name="leg4link6",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            )
        ],
    )
    rear_wheel_frame=FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/hero_dragon_realsense/wheel14_body",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/LFFrame"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/hero_dragon_realsense/wheel14_body",
                name="rear_wheel",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            )
        ],
    )

    front_wheel_frame=FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/hero_dragon_realsense/wheel12_body",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/LFFrame"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/hero_dragon_realsense/wheel12_body",
                name="front_wheel",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            )
        ],
    )

    # # robots
    # robot: Union[ArticulationCfg, _MISSING_TYPE] = MISSING
    # # sensors
    # contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    body_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(30, 30),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformBodyVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.12, 1.12), 
            ang_vel_z=(-math.pi/12, math.pi/12), 
            lin_vel_y=(-0.0, 0.0),
            heading=(-math.pi, math.pi)
        ),
    )

    

    # uniformpose2dcommandscfg preso da navigation target_pos

@configclass
class ActionsCfg:
    """Action specifications for the MDP.
    Note: due to naming convention of joints in Vehicle/Dragon, we can use these shorthands for names:

    Leg joints: leg.*
    Wheel joints .*_joint

    which is due to all leg joints being called legXjointY and wheel joints wheelXX_{left/right}_joint.
    However, to ensure correct sorting, please overwrite the joint names in the env cfg for specific robot
    """
    # joint_vel_action = mdp.JointVelocityActionCfg(
    #     asset_name="robot", 
    #     joint_names= ["wheel.*"], 
    #     scale=5.0
    # )
    # joint_pos_action = mdp.JointPositionActionCfg(
    #     asset_name="robot", 
    #     joint_names=["leg.*"],
    #     scale = 0.5,
    #     use_default_offset=True
    # )

    j1_action: mdp.JointPositionActionCfg = MISSING
    j2_action: mdp.JointPositionActionCfg = MISSING
    # j3_action: mdp.JointPositionActionCfg = MISSING
    j4_action: mdp.JointPositionActionCfg = MISSING    
    # j5_action: mdp.JointPositionActionCfg = MISSING
    j6_action: mdp.JointPositionActionCfg = MISSING
    j7_action: mdp.JointPositionActionCfg = MISSING
    
    # wheel14r_action: mdp.JointVelocityActionCfg = MISSING
    # wheel14l_action: mdp.JointVelocityActionCfg = MISSING
    # wheel12r_action: mdp.JointVelocityActionCfg = MISSING
    # wheel12l_action: mdp.JointVelocityActionCfg = MISSING


    wheels_action: mdp.JointVelocityActionCfg = MISSING
    

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""

        body_lin_vel = ObsTerm(
            func=mdp.body_lin_vel,
            noise = Unoise(n_min=-0.01, n_max=0.01),
            params = {"body_name": MISSING}
        )
        body_ang_vel = ObsTerm(
            func=mdp.body_ang_vel,
            noise = Unoise(n_min=-0.01, n_max=0.01),
            params = {"body_name": MISSING}
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, 
            noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, 
            noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, 
            params={"command_name": "body_velocity"}
        )

        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset_steering_joints,
        mode="reset",
        params={
            "steering_joints": MISSING,
            "position_range": (-math.pi/6, math.pi/6),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    
    # # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp_vehicle, 
        weight=3.0, 
        params={
            "command_name": "body_velocity",
            "body_name": MISSING, 
            "std": math.sqrt(0.01)
        }
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp_vehicle, 
        weight=3.0, 
        params={
            "command_name": "body_velocity", 
            "body_name": MISSING,
            "std": math.sqrt(0.01)
        }
    )
    # -- penalties
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-2000)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # dof_torques_l2 = RewTerm(func=mdp.joint_torques_dragon_l2, weight=-1.0e-4)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-2)
    # energy = RewTerm(func=mdp.power_consumption, weight=-2.5e-2, params={"gear_ratio": {".*": 1.0}})
    upright_wheel_bodies = RewTerm(func=mdp.upright_wheel_bodies_angle,weight = 2)
    low_joint = RewTerm(func=mdp.penalize_low_joint,weight=-0.3)
    # narrow_wheels = RewTerm(func=mdp.penalize_narrow_wheel, weight=-0.5)
    wheel_z= RewTerm(func=mdp.wheel_z, weight=15.0)






@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)



    # base_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    # )

    low_joint= DoneTerm(func=mdp.terminate_low_joint)

    wheel_z = DoneTerm(func=mdp.terminate_wheel_z)
    terminate_angle= DoneTerm(func=mdp.terminate_angle)  


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(
        func=mdp.terrain_levels_vel,
        params={"body_name": MISSING}
    )


@configclass
class RecorderCfg(ActionStateRecorderManagerCfg):
    """Configuration for recorder."""

    dataset_export_dir_path = "isaaclab_recordings_navigation/"
    dataset_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_action_states"

##
# Environment configuration
##


@configclass
class LocomotionDragonEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: HeroDragonSceneCfg = HeroDragonSceneCfg(num_envs=4096, env_spacing=10)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    # recorders: RecorderCfg = RecorderCfg() # enable this to record data. WILL SLOW DOWN TRAINING BY A LOT

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 30.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        # if self.scene.contact_forces is not None:
        #     self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        # if getattr(self.curriculum, "terrain_levels", None) is not None:
        #     if self.scene.terrain.terrain_generator is not None:
        #         self.scene.terrain.terrain_generator.curriculum = True
        # else:
        #     if self.scene.terrain.terrain_generator is not None:
        #         self.scene.terrain.terrain_generator.curriculum = False
