# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
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
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns, FrameTransformerCfg, ImuCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.moonshot.velocity.mdp as mdp
import isaaclab_tasks.manager_based.moonshot.utils as moonshot_utils

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
ISAAC_LAB_PATH = moonshot_utils.find_isaaclab_path().replace("\\","/") #  
##
# Pre-defined configs
##
from isaaclab_tasks.manager_based.moonshot.descriptions.config.moonbot_cfgs import VEHICLE_ARTICULATED_CFG  # isort: skip
from isaaclab_tasks.manager_based.moonshot.velocity.terrain.rough import ROUGH_TERRAINS_CFG, ROUGH_HILLY_TERRAINS_CFG

##
# Robot Base Link Name (desired base)
##
BASE_NAME = "leg1link4"

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with an Moonbot Wheel robot."""
    
    # terrain (flat plane)
    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="plane",
    #     collision_group=-1,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="average",
    #         restitution_combine_mode="average",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #         restitution=0.0,
    #     ),
    #     debug_vis=False,
    # )
    # terrain (rough)
    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="generator",
    #     terrain_generator=ROUGH_TERRAINS_CFG,
    #     max_init_terrain_level=5,
    #     collision_group=-1,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #     ),
    #     visual_material=sim_utils.MdlFileCfg(
    #         mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
    #         project_uvw=True,
    #         texture_scale=(0.25, 0.25),
    #     ),
    #     debug_vis=False,
    # )
    # terrain (moon terrain)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path = ISAAC_LAB_PATH + "/source/isaaclab_tasks/isaaclab_tasks/manager_based/moonshot/descriptions/usd/terrain/petavius_crater.usd",
        collision_group=-1,
        # physics_material=sim_utils.RigidBodyMaterialCfg(
        #     friction_combine_mode="average",
        #     restitution_combine_mode="average",
        #     static_friction=1.0,
        #     dynamic_friction=1.0,
        #     restitution=0.0,
        # ),
        debug_vis=False,
    )
    # robot
    robot = VEHICLE_ARTICULATED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors

    # IMU currently located at the Lattepanda in Link 4
    # imu = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/leg1link4", debug_vis=False)

    # height_scanner = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base_link",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    #     attach_yaw_only=True,World
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    #     debug_vis=True,
    #     mesh_prim_paths=["/World/ground"],
    # )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    # base_to_link_transform = FrameTransformerCfg(
    #     prim_path = "{ENV_REGEX_NS}/Robot/base_link",
    #     target_frames = [
    #         FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/leg1link2"),
    #         # FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/base_link")
    #     ],
    #     debug_vis=True
    # )

    # lights
    # light = AssetBaseCfg(
        # prim_path="/World/light",
        # spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    # )
    # lights (moon terrain)
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=15000.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot = (0.52133, 0.47771, 0.47771, 0.52133))
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    body_velocity = mdp.UniformBodyVelocityCommandCfg(
        asset_name="robot",
        body_name=BASE_NAME,
        resampling_time_range=(5, 15),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformBodyVelocityCommandCfg.Ranges(
            # lin_vel_x=(-0.12, 0.12), 
            # lin_vel_y=(-0.12, 0.12), 
            # ang_vel_z=(-math.pi/12, math.pi/12), 
            lin_vel_x=(0.12, 0.12),
            lin_vel_y=(-0.0, 0.0),
            ang_vel_z=(0, 0), 
            heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_vel_action = mdp.JointVelocityActionCfg(asset_name="robot", 
                                                  joint_names=["wheel11_left_joint",
                                                               "wheel11_right_joint",
                                                               "wheel12_left_joint",
                                                               "wheel12_right_joint"], 
                                                  scale=5.0)
    joint_pos_action = mdp.JointPositionActionCfg(asset_name="robot", 
                                                  joint_names=["leg1joint.*"], 
                                                  scale = 0.5,
                                                  use_default_offset=True)

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""

        # IMU in front wheel body
        # base_lin_vel = ObsTerm(
        #     func=mdp.base_lin_vel,
        #     noise = Unoise(n_min=-0.01, n_max=0.01),
        # )
        # base_ang_vel = ObsTerm(
        #     func=mdp.base_ang_vel,
        #     noise = Unoise(n_min=-0.01, n_max=0.01),
        # )
        # # IMU in rear wheel body
        # body_lin_vel = ObsTerm(
        #     func=mdp.body_lin_vel,
        #     noise = Unoise(n_min=-0.01, n_max=0.01),
        #     params = {"body_name": "leg1gripper2_base"}
        # )
        # body_ang_vel = ObsTerm(
        #     func=mdp.body_ang_vel,
        #     noise = Unoise(n_min=-0.01, n_max=0.01),
        #     params = {"body_name": "leg1gripper2_base"}
        # )
        body_lin_vel = ObsTerm(
            func=mdp.body_lin_vel,
            noise = Unoise(n_min=-0.01, n_max=0.01),
            params = {"body_name": BASE_NAME}
        )
        body_ang_vel = ObsTerm(
            func=mdp.body_ang_vel,
            noise = Unoise(n_min=-0.01, n_max=0.01),
            params = {"body_name": BASE_NAME}
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

        # height_scan = ObsTerm(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     clip=(-1.0, 1.0),
        # )


        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset_steering_joints,
        mode="reset",
        params={
            "position_range": (-math.pi/6, math.pi/6),
            # "position_range": (-0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="wheel.*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(10.0, 15.0),
    #     params={"velocity_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03)}},
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp_vehicle, weight=1.0, params={"command_name": "body_velocity",
                                                                   "body_name": BASE_NAME, 
                                                                   "std": math.sqrt(0.01)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp_vehicle, weight=1.0, params={"command_name": "body_velocity", 
                                                                  "body_name": BASE_NAME,
                                                                  "std": math.sqrt(0.01)}
    )
    # -- penalties
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400)
    # lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_body_l2, weight=-10.0, params = {"body_names": [BASE_NAME ]})
    # lin_acc_l2 = RewTerm(func=mdp.lin_acc_l2, weight=-1e-2)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_vehicle_l2, weight=-1.0e-4)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-2)
    
    energy = RewTerm(func=mdp.power_consumption, weight=-1.0e-1, params={"gear_ratio": {".*": 1.0}})
    upright_wheel_bodies = RewTerm(
        func=mdp.upright_wheel_bodies_angle,
        weight = -1.0,
        params = {"std": 0.25}
    )
    # body_above_wheels = RewTerm(
    #     func=mdp.body_height_above_wheels_l2,
    #     weight=-10.0,
    #     params={"body_name": "leg1link4"}
    # )
    # Penalty for not being in vehicle configuration 
    # joint_deviation_l1 = RewTerm(func=mdp.joint_deviation_vehicle_l1, weight = -0.2)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Terminate if the episode length is exceeded
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # (2) Terminate if any of the vehicle arm links collide with something
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="leg1link.*"), "threshold": 0.1},
    )
    # (3) Terminate if joint effort out of limits
    # effort_limit = DoneTerm(
    #     func=mdp.joint_vel_out_of_manual_limit_vehicle,
    #     params={"max_velocity": 0.05}
    # )
    # bad_orientation = DoneTerm(
    #     func=mdp.bad_orientation,
    #     params={"limit_angle": math.pi/4}
    # )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel, params = {"body_name": BASE_NAME})
    # action_rate = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "action_rate_l2", "weight": -0.005, "num_steps": 1000}
    # )

    # joint_vel = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -0.001, "num_steps": 1000}
    # )

@configclass
class VehicleEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for Moonshot Wheel Module environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=7.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    # curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.viewer.eye = (3.5,3.5,3.5) 
        # self.viewer.eye = (8.0, 8.0, 4.5) # use this for moon scene
        # self.viewer.resolution = (2540,1440) # uncomment when you want to export video
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        # if self.scene.height_scanner is not None:
        #     self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
