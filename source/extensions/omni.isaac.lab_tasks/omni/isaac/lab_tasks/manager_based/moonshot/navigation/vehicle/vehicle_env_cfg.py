# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns

import omni.isaac.lab_tasks.manager_based.moonshot.navigation.mdp as mdp

from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
##
# Pre-defined configs
##
from omni.isaac.lab_tasks.manager_based.moonshot.descriptions.config.moonbot_cfgs import VEHICLE_CFG  
from omni.isaac.lab_tasks.manager_based.moonshot.navigation.terrain.rough import ROUGH_TERRAINS_CFG

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with an Moonbot Vehicle robot."""

    # terrain (flat plane)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
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
    # robot
    robot = VEHICLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot") # type: ignore

    # sensors

    # height_scanner = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base_link",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    #     attach_yaw_only=True,
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    #     debug_vis=True,
    #     mesh_prim_paths=["/World/ground"],
    # )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_vel_action = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=["wheel11_left_joint",
                                                                                   "wheel11_right_joint",
                                                                                   "wheel12_left_joint",
                                                                                   "wheel12_right_joint"], scale=5.0)
    joint_pos_action = mdp.JointPositionActionCfg(asset_name="robot", joint_names=["leg1.*"], scale = 0.5, use_default_offset=True)

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""

        base_height = ObsTerm(func=mdp.base_pos_z)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        base_yaw_roll = ObsTerm(func=mdp.base_yaw_roll)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

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

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}},
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset_vehicle,
        mode="reset",
        params={
            "position_range": (-0.0, 0.0),
            "velocity_range": (-0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Reward for moving forward
    progress = RewTerm(func=mdp.progress_reward, weight=1.0, params={"target_pos": (1000.0, 0.0, 0.0)})
    # (4) Reward for moving in the right direction (rough grained)
    # move_to_target_rough = RewTerm(
    #     func=mdp.move_to_target_bonus, weight=1, params={"threshold": 0.8, "target_pos": (1000.0, 0.0, 0.0)}
    # )
    # (4) Reward for moving in the right direction (fine grained)
    move_to_target_fine = RewTerm(
        func=mdp.move_to_target_bonus, weight=0.05, params={"threshold": 0.95, "target_pos": (1000.0, 0.0, 0.0)}
    )
    desired_contacts_left = RewTerm(
        func=mdp.undesired_contacts,
        weight=1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_left"), "threshold": 0.05},
    )
    desired_contacts_right = RewTerm(
        func=mdp.undesired_contacts,
        weight=1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_right"), "threshold": 0.05},
    )

    # -- penalties
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_deviation = RewTerm(func=mdp.joint_deviation_vehicle_l1, weight = -2.0)
    energy = RewTerm(func=mdp.power_consumption, weight=-0.05, params={"gear_ratio": {".*": 1.0}})

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-4.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="leg1link.*"), "threshold": 0.1},
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Terminate if the episode length is exceeded
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # base_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="Leg1.*"), "threshold": 1.0},
    # )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class VehicleEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for Moonshot Vehicle environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    # curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 10.0
        # simulation settings
        self.viewer.eye = (3.5, 3.5, 3.5)
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
