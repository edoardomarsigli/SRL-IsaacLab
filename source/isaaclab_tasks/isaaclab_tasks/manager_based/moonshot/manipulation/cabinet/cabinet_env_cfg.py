# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.moonshot.utils as moonshot_utils
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab_tasks.manager_based.moonshot.descriptions.config.terrain.rough import ROUGH_TERRAINS_CFG
from isaaclab.managers import SceneEntityCfg


ISAAC_LAB_PATH = moonshot_utils.find_isaaclab_path().replace("\\","/")

from isaaclab.sim import UsdFileCfg


from . import mdp
from isaaclab_tasks.manager_based.moonshot.descriptions.config.moonbot_cfgs_edo import DRAGON_ARTICULATED_CFG, WHEEL_WITH_HANDLE_CFG


##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)


##
# Scene definition
##

@configclass
class DragonGraspSceneCfg(InteractiveSceneCfg):
    """Scene with hero_dragon robot and a fixed wheel with a handle."""

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
        debug_vis=True
        
    )

    # Robot (popolato dall'env cfg)
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    handle_frame: FrameTransformerCfg = MISSING
    rf_frame: FrameTransformerCfg = MISSING
    lf_frame: FrameTransformerCfg = MISSING
    # Ruota fissa con maniglia
    wheel_with_handle = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/hero_wheel/wheel11_out",

        spawn=UsdFileCfg(
            usd_path=ISAAC_LAB_PATH + "/source/isaaclab_tasks/isaaclab_tasks/manager_based/moonshot/descriptions/usd/robot/HERO_wheel/hero_wheel.usd",
            scale=(1.0, 1.0, 1.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.8, 0.0, 0.3),  # allinea con il braccio leg3
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )

    # Lights
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
class ActionsCfg:
    """Action specifications for the MDP."""
    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.JointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        rel_ee__distance = ObsTerm(func=mdp.rel_ee__distance)

        ee_pos = ObsTerm(func=mdp.ee_pos)

        handle_pos = ObsTerm(func=mdp.handle_pos)

        rel_rf__distance = ObsTerm(func=mdp.rel_rf__distance)

        rel_lf__distance = ObsTerm(func=mdp.rel_lf__distance)

        ee_quat = ObsTerm(func=mdp.ee_quat)

        #actions = ObsTerm(func=mdp.last_action) #da aggiungere piu avanti

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    robot_physics_material = EventTerm(      #serve per domain randomization per far imparare a muovere il robot con fisica variabile
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1),
            "dynamic_friction_range": (0.5, 0.9),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 16,
        },
    )

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_robot_joints = EventTerm(     #serve per il reset della posizione del robot in posizione casuale intorno alla posizione iniziale
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    approach_ee_handle = RewTerm(func=mdp.approach_ee_handle, weight=2.0, params={"threshold": 0.03})

    # align_ee_handle = RewTerm(func=mdp.align_ee_handle, weight=5, params={"align_threshold": 0.99})

    # approach_xy_alignment = RewTerm(func=mdp.approach_xy_alignment, weight=0.125, params={"xy_threshold": 0.03, "align_threshold": 0.99})

    # approach_z_conditional_on_xy = RewTerm(func=mdp.approach_z_conditional_on_xy, weight=0.125, params={"xy_threshold": 0.03})
    
    # align_grasp = RewTerm(func=mdp.align_grasp, weight=0.125)

    # # approach_gripper_handle = RewTerm(func=mdp.approach_gripper_handle, weight=5.0, params={"offset": MISSING})
    # grasp_handle = RewTerm(
    #     func=mdp.grasp_handle,
    #     weight=0.5,
    #     params={
    #         "threshold": 0.03,
    #         "open_joint_pos": None,
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=None),
    #     },
    # )


    # # 4. Penalize actions for cosmetic reasons
    # action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-2)
    # joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0001)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Environment configuration
##


@configclass
class DragonGraspEnvCfg(ManagerBasedRLEnvCfg):
    scene: DragonGraspSceneCfg = DragonGraspSceneCfg(
        num_envs=4096,
        env_spacing=3.0,
        robot=DRAGON_ARTICULATED_CFG,
        wheel_with_handle=WHEEL_WITH_HANDLE_CFG,
        ee_frame=FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/hero_dragon/leg2gripper2_base",
            debug_vis=True,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EEFrame"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    # prim_path="{ENV_REGEX_NS}/hero_wheel/wheel11_in/wheel11_center/wheel11_out",
                    prim_path="{ENV_REGEX_NS}/hero_dragon/leg2gripper2_base",
                     name="ee",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.055)), #devo aggiungere l offset dalla base del gripper al punto di contatto con handle
                )
            ],
        ),
        handle_frame=FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/hero_wheel/wheel11_out",
            debug_vis=True,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/HandleFrame"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/hero_wheel/wheel11_out",
                    name="handle_target",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
                )
            ],
        ),
        rf_frame=FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/hero_dragon/leg2gripper2_jaw_right",
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/RFFrame"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/hero_dragon/leg2gripper2_jaw_right",
                    name="rf",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
                )
            ],
        ),
        lf_frame=FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/hero_dragon/leg2gripper2_jaw_left",
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/LFFrame"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/hero_dragon/leg2gripper2_jaw_left",
                    name="lf",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
                )
            ],
        ),

    )

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        # self.decimation = 4
        # self.episode_length_s = 5.0
        # self.viewer.eye = (-2.0, 2.0, 2.0)
        # self.viewer.lookat = (0.8, 0.0, 0.5)
        # self.sim.dt = 0.005
        # self.sim.render_interval = self.decimation

        # # simulation settings

        # self.sim.physx.bounce_threshold_velocity = 0.5
        # self.sim.physx.friction_correlation_distance = 0.01


        # # 
        self.sim.physx.solver_position_iteration_count = 8 # + iterazioni posizione = risoluzione contatti rigidi
        self.sim.physx.solver_velocity_iteration_count = 1 
        # # self.sim.disable_contact_processing = False  #Disabilita il contact processing PhysX (cioè niente eventi di contatto nei sensori).Utile per migliorare performance se non usi sensori di contatto.Ma se hai rewards basati sul contatto o vuoi logging preciso → lascialo False.
        # # self.sim.physics_material = self.scene.terrain.physics_material
        # # self.sim.physx.gpu_max_rigid_patch_count = 50 * 2**15

        self.decimation = 1
        self.episode_length_s = 4.0
        self.viewer.eye = (-2.0, 2.0, 2.0)
        self.viewer.lookat = (0.8, 0.0, 0.5)
        # simulation settings
        self.sim.dt = 0.005 # 
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.5
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.enable_ccd = True  # se hai movimenti molto veloci o piccoli oggetti, abilita CCD (collision detection continua)
