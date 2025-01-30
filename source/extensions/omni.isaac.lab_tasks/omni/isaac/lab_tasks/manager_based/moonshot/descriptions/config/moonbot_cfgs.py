"""Configurations for different Moonbot types"""

from __future__ import annotations

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
import omni.isaac.lab_tasks.manager_based.moonshot.utils as moonshot_utils


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
        usd_path=ISAAC_LAB_PATH + "/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/moonshot/descriptions/usd/hero_wheel_module.usd",
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
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=10.0,
        ),
        "wm1_wheel_right_actuator": ImplicitActuatorCfg(
            joint_names_expr=["wm1_wheel_right_joint"],
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=10.0,
        ),
    },
)

VEHICLE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=ISAAC_LAB_PATH + "/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/moonshot/descriptions/usd/hero_vehicle/hero_vehicle.usd",
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
    # rot=(0.92388,0,-0.38268,0), # comment out if EventTerm: reset_joints_by_offset_vehicle is enabled
    joint_pos={".*": 0.0},
    ),
    actuators = {
        "leg_joints": ImplicitActuatorCfg(
            joint_names_expr=["leg1joint[1-7]"],
            effort_limit=136.11,
            velocity_limit=0.145,
            stiffness=10000,
            damping=100,
        ),
        "wheel_joints": ImplicitActuatorCfg(
            joint_names_expr=["wheel11_left_joint",
                              "wheel11_right_joint",
                              "wheel12_left_joint",
                              "wheel12_right_joint"],
            effort_limit=136.11,
            velocity_limit=0.145,
            stiffness=0,
            damping=10
        ),
        # TODO: Fill in suitable joint parameters
        "gripper_joints": ImplicitActuatorCfg(
            joint_names_expr=["leg1gripper1_jaw_left_joint",
                              "leg1gripper1_jaw_right_joint",
                              "leg1gripper2_jaw_left_joint",
                              "leg1gripper2_jaw_right_joint"],
            effort_limit=400,
            velocity_limit=100,
            stiffness=0,
            damping=0
        ),
    },
)

CARTER_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=ISAAC_LAB_PATH + "/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/moonshot/descriptions/usd/carter.usd",
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
    ),
    init_state=ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.25), 
    joint_pos={"left_wheel": 0.0,
               "right_wheel": 0.0},
    ),
    actuators = {
        "leg3_wheel_left_actuator": ImplicitActuatorCfg(
            joint_names_expr=["left_wheel"],
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=10.0,
        ),
        "leg3_wheel_right_actuator": ImplicitActuatorCfg(
            joint_names_expr=["right_wheel"],
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=10.0,
        ),
    },
)