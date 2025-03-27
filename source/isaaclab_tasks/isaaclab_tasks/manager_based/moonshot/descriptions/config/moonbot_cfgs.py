"""Configurations for different Moonbot types"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab_tasks.manager_based.moonshot.utils as moonshot_utils
import math

# Get full path, replace is for Windows paths
ISAAC_LAB_PATH = moonshot_utils.find_isaaclab_path().replace("\\","/") #  

##
# Configuration
##

'''
Unit notes: 
Joint velocity limits are in rad/s (revolute), m/s (prismatic)
Joint effort limits are in N*m (revolute), N (prismatic)

'''
WHEEL_MODULE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=ISAAC_LAB_PATH + "/source/isaaclab_tasks/isaaclab_tasks/manager_based/moonshot/descriptions/usd/hero_wheel_module.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
            # fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.30),
    joint_pos={"wm1_wheel_left_joint": 0.0,
               "wm1_wheel_right_joint": 0.0},
    ),
    actuators = {
        "wm1_wheel_left_actuator": ImplicitActuatorCfg(
            joint_names_expr=["wm1_wheel_left_joint"],
            effort_limit_sim=400.0,
            velocity_limit_sim=100.0,
            stiffness=0.0,
            damping=10.0,
        ),
        "wm1_wheel_right_actuator": ImplicitActuatorCfg(
            joint_names_expr=["wm1_wheel_right_joint"],
            effort_limit_sim=400.0,
            velocity_limit_sim=100.0,
            stiffness=0.0,
            damping=10.0,
        ),
    },
)

VEHICLE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=ISAAC_LAB_PATH + "/source/isaaclab_tasks/isaaclab_tasks/manager_based/moonshot/descriptions/usd/hero_vehicle/hero_vehicle.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
            # fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.30),
    joint_pos = {".*": 0.0}
    ),
    # joint_pos={"wheel.*": 0.0,
    #            "leg1joint[1-3]": 0.0,
    #            "leg1joint4": math.pi/2,
    #            "leg1joint[5-7]": 0.0},
    # ),
    # soft_joint_pos_limit_factor = 0.1,
    
    actuators = {
        "leg_joints": ImplicitActuatorCfg(
            joint_names_expr=["leg1joint1",
                              "leg1joint4",
                              "leg1joint7"],
            effort_limit_sim=136.11,
            velocity_limit_sim=0.145,
            stiffness=1e6,
            damping=100,
        ),
        "wheel_joints": ImplicitActuatorCfg(
            joint_names_expr=["wheel11_left_joint",
                              "wheel11_right_joint",
                              "wheel12_left_joint",
                              "wheel12_right_joint"],
            effort_limit_sim=136.11,
            velocity_limit_sim=0.5,
            stiffness=0,
            damping=10.0
        ),
        # TODO: Fill in suitable joint parameters
        # "gripper_joints": ImplicitActuatorCfg(
        #     joint_names_expr=["leg1gripper1_jaw_left_joint",
        #                       "leg1gripper1_jaw_right_joint",
        #                       "leg1gripper2_jaw_left_joint",
        #                       "leg1gripper2_jaw_right_joint"],
        #     effort_limit_sim=400,
        #     velocity_limit_sim=100,
        #     stiffness=0,
        #     damping=0
        # ),
    },
)

VEHICLE_ARTICULATED_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=ISAAC_LAB_PATH + "/source/isaaclab_tasks/isaaclab_tasks/manager_based/moonshot/descriptions/usd/robot/hero_vehicle_12467/hero_vehicle.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
            # fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.30),
    # pos=(-1.0, -5.0, 0.30),
    joint_pos = {".*": 0.0}
    ),
    # joint_pos={"wheel.*": 0.0,
    #            "leg1joint[1-3]": 0.0,
    #            "leg1joint4": math.pi/2,
    #            "leg1joint[5-7]": 0.0},
    # ),
    # soft_joint_pos_limit_factor = 0.1,
    
    actuators = {
        "leg_joints": ImplicitActuatorCfg(
            joint_names_expr=["leg1joint.*"],
            effort_limit_sim=36.11,
            velocity_limit_sim=0.145,
            stiffness=1e6,
            damping=100,
        ),
        "wheel_joints": ImplicitActuatorCfg(
            joint_names_expr=["wheel11_left_joint",
                              "wheel11_right_joint",
                              "wheel12_left_joint",
                              "wheel12_right_joint"],
            effort_limit_sim=136.11,
            velocity_limit_sim=0.5,
            stiffness=0,
            damping=10.0
        ),
    },
)

DRAGON_ARTICULATED_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=ISAAC_LAB_PATH + "/source/isaaclab_tasks/isaaclab_tasks/manager_based/moonshot/descriptions/usd/robot/hero_dragon_12467/hero_dragon.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
            # fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.30),
    # pos=(-1.0, -5.0, 0.30),
    joint_pos = {".*": 0.0}
    ),
    # joint_pos={"wheel.*": 0.0,
    #            "leg1joint[1-3]": 0.0,
    #            "leg1joint4": math.pi/2,
    #            "leg1joint[5-7]": 0.0},
    # ),
    # soft_joint_pos_limit_factor = 0.1,
    
    actuators = {
        "leg_joints": ImplicitActuatorCfg(
            joint_names_expr=["leg4joint.*"],
            effort_limit_sim=36.11,
            velocity_limit_sim=0.145, # 0.145
            stiffness=1e6,
            damping=100,
        ),
        "wheel_joints": ImplicitActuatorCfg(
            joint_names_expr=["wheel12_left_joint",
                              "wheel12_right_joint",
                              "wheel14_left_joint",
                              "wheel14_right_joint"],
            effort_limit_sim=136.11,
            velocity_limit_sim=0.5,
            stiffness=0,
            damping=10.0
        ),
    },
)

CARTER_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=ISAAC_LAB_PATH + "/source/isaaclab_tasks/isaaclab_tasks/manager_based/moonshot/descriptions/usd/carter.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=5.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
            # fix_root_link=True,
        ),
        # soft_joint_pos_limit_factor = 0.5,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.25), 
    joint_pos={"left_wheel": 0.0,
               "right_wheel": 0.0},
    ),
    actuators = {
        "leg3_wheel_left_actuator": ImplicitActuatorCfg(
            joint_names_expr=["left_wheel"],
            effort_limit_sim=400.0,
            velocity_limit_sim=100.0,
            stiffness=0.0,
            damping=10.0,
        ),
        "leg3_wheel_right_actuator": ImplicitActuatorCfg(
            joint_names_expr=["right_wheel"],
            effort_limit_sim=400.0,
            velocity_limit_sim=100.0,
            stiffness=0.0,
            damping=10.0,
        ),
    },
)