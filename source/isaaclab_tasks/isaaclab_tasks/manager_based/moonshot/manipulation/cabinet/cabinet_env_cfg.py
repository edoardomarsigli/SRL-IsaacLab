# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from dataclasses import MISSING
from isaaclab.managers import RecorderManagerBaseCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.moonshot.utils as moonshot_utils
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim import UsdFileCfg
#from isaaclab.terrains.config.rough import PERLIN_TERRAIN_CFG






ISAAC_LAB_PATH = moonshot_utils.find_isaaclab_path().replace("\\","/")




from isaaclab_tasks.manager_based.moonshot.manipulation.cabinet import mdp
from isaaclab_tasks.manager_based.moonshot.manipulation.cabinet.mdp import Event as mdp_events




##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg

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
        terrain_type="plane",
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
            texture_scale=(0.25, 0.25)),
        debug_vis=True
    )

    contact_sensor_left1 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/hero_dragon/leg2gripper2_jaw_left",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/hero_wheel/wheel11_left"],  # o "{ENV_REGEX_NS}/hero_wheel/wheel11_out" se vuoi la maniglia specifica
        track_air_time=True,
        track_pose=True,
        force_threshold=1.0,
    )
    contact_sensor_left2 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/hero_dragon/leg2gripper2_jaw_left",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/hero_wheel/wheel11_right"],  # o "{ENV_REGEX_NS}/hero_wheel/wheel11_out" se vuoi la maniglia specifica
        track_air_time=True,
        track_pose=True,
        force_threshold=1.0,
    )
    contact_sensor_right1 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/hero_dragon/leg2gripper2_jaw_right",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/hero_wheel/wheel11_left"],  # o "{ENV_REGEX_NS}/hero_wheel/wheel11_out" se vuoi la maniglia specifica
        track_air_time=True,
        track_pose=True,
        force_threshold=1.0,
    )
    contact_sensor_right2 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/hero_dragon/leg2gripper2_jaw_right",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/hero_wheel/wheel11_right"],  # o "{ENV_REGEX_NS}/hero_wheel/wheel11_out" se vuoi la maniglia specifica
        track_air_time=True,
        track_pose=True,
        force_threshold=1.0,
    )



    # Robot (popolato dall'env cfg)
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    wheel_with_handle: ArticulationCfg= MISSING

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
    )

    rf2_frame=FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/hero_dragon/leg2gripper2_jaw_right",  #associato a giunto grip2bis
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/RFFrame"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/hero_dragon/leg2gripper2_jaw_right",
                name="rf2",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            )
        ],
    )

    lf2_frame=FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/hero_dragon/leg2gripper2_jaw_left",   #associato a giunto grip2
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/LFFrame"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/hero_dragon/leg2gripper2_jaw_left",
                name="lf2",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            )
        ],
    )

    rf1_frame=FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/hero_dragon/leg2gripper1_jaw_right",  #associato a giunto grip1
        debug_vis=True, 
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/RFFrame"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/hero_dragon/leg2gripper1_jaw_right",
                name="rf1",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            )
        ],
    )

    lf1_frame=FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/hero_dragon/leg2gripper1_jaw_left", #associato a giunto grip1bis
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/LFFrame"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/hero_dragon/leg2gripper1_jaw_left",
                name="lf1",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            )
        ],
    )

    joint4_frame=FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/hero_dragon/leg2link4",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/LFFrame"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/hero_dragon/leg2link4",
                name="joint4",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            )
        ],
    )
    joint6_frame=FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/hero_dragon/leg2link6",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/LFFrame"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/hero_dragon/leg2link6",
                name="joint6",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            )
        ],
    )
    wheel_frame=FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/hero_dragon/base_link",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/LFFrame"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/hero_dragon/base_link",
                name="wheel",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            )
        ],
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

        # rel_rf__distance = ObsTerm(func=mdp.rel_rf__distance)

        # rel_lf__distance = ObsTerm(func=mdp.rel_lf__distance)

        ee_quat = ObsTerm(func=mdp.ee_quat)

        gripper1= ObsTerm(func=mdp.gripper1_pos)
        gripper2= ObsTerm(func=mdp.gripper2_pos)


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
        func=mdp_events.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1),
            "dynamic_friction_range": (0.5, 0.9),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 16,
        },
    )


    # reset_all = EventTerm(func=mdp_events.reset_scene_to_default, mode="reset")

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_robot_joints = EventTerm(     #serve per il reset della posizione del robot in posizione casuale intorno alla posizione iniziale
        func=mdp_events.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (0.0, 0.0),
        },
    )


# @configclass   #reward senza curriculum
# class RewardsCfg:
#     """Reward terms for the MDP."""

#     #approach_ee_handle = RewTerm(func=mdp.approach_ee_handle, weight=2.0, params={"threshold": 0.03})

#     align_ee_handle = RewTerm(func=mdp.align_ee_handle, weight=2, params={"align_threshold": 0.9})

#     approach_zy_alignment = RewTerm(func=mdp.approach_zy_alignment, weight=1, params={"zy_threshold": 0.05, "align_threshold1": 0.9})

#     approach_x_conditional_on_yz = RewTerm(func=mdp.approach_x_conditional_on_yz, weight=1, params={"zy_threshold1": 0.05, 'x_threshold': 0.1})

#     penalize_low_joints = RewTerm(func=mdp.penalize_low_joints, weight=0.2, params={"threshold": 0.2})


#     # approach_gripper_handle = RewTerm(func=mdp.approach_gripper_handle, weight=5.0, params={"offset": MISSING})
#     # grasp_handle = RewTerm(
#     #     func=mdp.grasp_handle,
#     #     weight=1,
#     #     params={
#     #         "grasp_threshold": 0.05
#     #     },
#     # )


#     # 4. Penalize actions for cosmetic reasons
#     action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
#     joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.005)



@configclass
class RewardsCfg:   #reward con curriculum

    # align_ee_handle = RewTerm(func=mdp.align_ee_handle_curriculum_wrapped, weight=2.5)

    align_ee_handlez = RewTerm(func=mdp.align_ee_handle_z_curriculum_wrapped, weight=1)

    align_ee_handlex = RewTerm(func=mdp.align_ee_handle_x_curriculum_wrapped, weight=1)

    penalize_joint7 = RewTerm(func=mdp.penalize_joint7, weight=0.05)
    
    penalize_low_joints = RewTerm(func=mdp.penalize_low_joints_curriculum, weight=1, params={"threshold4": 0.25, "threshold_ee": 0.25})
    
    reward_joint4_height = RewTerm(func=mdp.reward_joint4_height, weight=1, params={"threshold": 0.5, "z_limit": 0.75})
    
    # approach_zy = RewTerm(func=mdp.approach_zy_alignment_curriculum_wrapped, weight=1)
        
    # approach_x = RewTerm(func=mdp.approach_x_conditional_on_yz_curriculum_wrapped, weight=2)


    collision_penalty = RewTerm(func=mdp.collision_penalty, weight=1.0, params={"weight": -3.0})

    penalty_x = RewTerm(func=mdp.penalty_x, weight=-1.0)




    # approach_gripper_handle = RewTerm(func=mdp.approach_gripper_handle, weight=5.0, params={"offset": MISSING})

    # grasp_handle = RewTerm(func=mdp.grasp_handle_curriculum_wrapped,weight=1)

    grasp_handle2 = RewTerm(func=mdp.grasp2_curriculum_wrapped,weight=1)

    keep_gripper1_closed = RewTerm(func=mdp.keep_gripper1_closed_curriculum_wrapped, weight=1,)    

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.005)



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    grasp_completed = DoneTerm(func=mdp_events.is_grasp_successful) #uso la funzione grasp_completed del mio events.py customizzato

    low_arm = DoneTerm(func=mdp_events.terminate_if_low, params={"threshold4": 0.15, "threshold6": 0.15, "threshold_ee": 0.15})

    wheel_z = DoneTerm(func=mdp_events.terminate_wheel_z, params={"threshold": 0.31})

    wheel = DoneTerm(func=mdp_events.terminate_wheel, params={"limit": 0.95})

    collision = DoneTerm(func=mdp_events.collision_termination, params={"threshold": 1})


##MADRL

#from isaaclab.managers import MultiAgentActionsCfg

# @configclass
#  class ActionsCfgMADRL(MultiAgentActionsCfg):  #actions per madrl
#     """Multi-agent action configuration for MADRL."""
        # @configclass
        # class ArmActionsCfg:
        #     arm_action: mdp.JointPositionActionCfg = MISSING

        # @configclass
        # class Gripper1ActionsCfg:
        #     gripper1: mdp.JointPositionActionCfg = MISSING

        # @configclass
        # class Gripper2ActionsCfg:
        #     gripper2: mdp.JointPositionActionCfg = MISSING

#     arm = ArmActionsCfg()
#     gripper1 = Gripper1ActionsCfg()
#     gripper2 = Gripper2ActionsCfg()


# @configclass
# class ObservationsCfgMADRL:   #osservazioni per madrl
#     class ArmPolicyCfg(ObsGroup):
#         joint_pos = ObsTerm(func=mdp.joint_pos_rel)
#         joint_vel = ObsTerm(func=mdp.joint_vel_rel)
#         rel_ee__distance = ObsTerm(func=mdp.rel_ee__distance)
#         ee_quat = ObsTerm(func=mdp.ee_quat)
#         ee_pos = ObsTerm(func=mdp.ee_pos)
#         handle_pos = ObsTerm(func=mdp.handle_pos)

#     class Gripper2PolicyCfg(ObsGroup):
#         lf_dist = ObsTerm(func=mdp.rel_lf__distance)
#         rf_dist = ObsTerm(func=mdp.rel_rf__distance)
#         gripper2= ObsTerm(func=mdp.gripper2_pos)

#     class Gripper1PolicyCfg(ObsGroup):
#         gripper1= ObsTerm(func=mdp.gripper1_pos)

#     arm: ArmPolicyCfg = ArmPolicyCfg()
#     gripper2: Gripper2PolicyCfg = Gripper2PolicyCfg()
#     gripper1: Gripper1PolicyCfg = Gripper1PolicyCfg()

# @configclass #reward per madrl
# class RewardsCfgMADRL:
#     class ArmRewards:
        # align_ee_handlez = RewTerm(func=mdp.align_ee_handle_z_curriculum_wrapped, weight=1)

        # align_ee_handlex = RewTerm(func=mdp.align_ee_handle_x_curriculum_wrapped, weight=1)

        # approach_zy_alignment = RewTerm(func=mdp.approach_zy_alignment_curriculum_wrapped, weight=1)

        # approach_x_conditional_on_yz = RewTerm(func=mdp.approach_x_conditional_on_yz_curriculum_wrapped, weight=1)

        # penalize_low_joints = RewTerm(func=mdp.penalize_low_joints, weight=1, params={"threshold": 0.2})
    
#     class Gripper2Rewards:
#         grasp_handle = RewTerm(...)
    
#     class Gripper1Rewards:
#         stay_closed = RewTerm(func=mdp.keep_gripper1_closed_curriculum_wrapped, weight=1.0)

#     arm = ArmRewards()
#     gripper2 = Gripper2Rewards()
#     gripper1 = Gripper1Rewards()



# @configclass
# class TerminationsCfgMADRL: #terminations per madrl
#     """Termination terms per MADRL."""

#     @configclass
#     class ArmTerminations:
#         time_out = DoneTerm(func=mdp.time_out, time_out=True)
#         low_arm = DoneTerm(func=mdp_events.terminate_if_low, params={"threshold4": 0.15, "threshold6": 0.1, "threshold_ee": 0.1})
#         wheel = DoneTerm(func=mdp_events.terminate_wheel, params={"threshold": 0.31})

#     @configclass
#     class Gripper1Terminations:
#         # es. se vuoi terminare se gripper è troppo aperto o altro
#         pass

#     @configclass
#     class Gripper2Terminations:

#     arm = ArmTerminations()
#     gripper1 = Gripper1Terminations()
#     gripper2 = Gripper2Terminations()



@configclass
class DragonGraspEnvCfg(ManagerBasedRLEnvCfg):
    scene: DragonGraspSceneCfg = DragonGraspSceneCfg(num_envs=4096,env_spacing=6)
    # recorders: RecorderCfg = RecorderCfg(dataset_export_mode=DatasetExportMode.EXPORT_ALL,  # o altro modo, vedi sotto
    # dataset_export_dir_path="/path/di/output",
    # dataset_filename="nome_file_output")
    recorders: RecorderManagerBaseCfg = RecorderManagerBaseCfg()


    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    # observations: ObservationsCfg = ObservationsCfgMADRL()
    # actions: ActionsCfg = ActionsCfgMADRL()
    # rewards: RewardsCfg = RewardsCfgMADRL()
    # terminations: TerminationsCfg = TerminationsCfgMADRL()


    def __post_init__(self):

        # # 
        self.sim.physx.solver_position_iteration_count = 8 # + iterazioni posizione = risoluzione contatti rigidi
        self.sim.physx.solver_velocity_iteration_count = 1 
        # # self.sim.disable_contact_processing = False  #Disabilita il contact processing PhysX (cioè niente eventi di contatto nei sensori).Utile per migliorare performance se non usi sensori di contatto.Ma se hai rewards basati sul contatto o vuoi logging preciso → lascialo False.
        # # self.sim.physics_material = self.scene.terrain.physics_material
        # # self.sim.physx.gpu_max_rigid_patch_count = 50 * 2**15

        self.decimation = 1
        self.episode_length_s = 10
        self.viewer.eye = (-2.0, 2.0, 2.0)
        self.viewer.lookat = (0.8, 0.0, 0.5)
        # simulation settings
        self.sim.dt = 0.01 # 100 hz
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.5
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.enable_ccd = True  # se hai movimenti molto veloci o piccoli oggetti, abilita CCD (collision detection continua)

