"""Configurations for different Moonbot types"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
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
scale2=100

DRAGON_ARTICULATED_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/hero_dragon",
    spawn=sim_utils.UsdFileCfg(
        usd_path=ISAAC_LAB_PATH + "/source/isaaclab_tasks/isaaclab_tasks/manager_based/moonshot/descriptions/usd/robot/Hero_dragon_grip/hero_dragon_gripPY.usd",
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
    pos=(0.0, 0.0, 0.3),
    joint_pos={"leg2grip1":-0.029, "leg2joint1":0.0, "leg2joint2":math.radians(135), "leg2joint3":0.0, "leg2joint4":math.radians(180), 
               "leg2joint5":0.0, "leg2joint6":math.radians(170), "leg2joint7":0.0, "leg2grip2":0.0},
    ),

    soft_joint_pos_limit_factor = 1.0,
    
    actuators = {

    # "leg2grip1": ImplicitActuatorCfg(
    #     joint_names_expr=["leg2grip1"],
    #     effort_limit_sim=30,
    #     velocity_limit_sim=5*0.15,
    #     stiffness = 3000,
    #     damping = 5,                 # per evitare vibrazioni
    #     ),


    # # "leg2grip1": ImplicitActuatorCfg(
    # #     joint_names_expr=["leg2grip1"],
    # #     effort_limit_sim=10,
    # #     velocity_limit_sim=scale*0.15,
    # #     stiffness=3000,
    # #     damping=160,
    # # ),

    # "leg2grip1bis": ImplicitActuatorCfg( #avtuator passivo che tanto e mimic
    #     joint_names_expr=["leg2grip1bis"],
    #     effort_limit_sim=30,
    #     velocity_limit_sim=5*0.15,
    #     stiffness = 3000,
    #     damping = 5, 
    # ), 


    "leg2joint1": ImplicitActuatorCfg(
        joint_names_expr=["leg2joint1"],
        effort_limit_sim=scale2*136.11,
        velocity_limit_sim=5*0.15,
        stiffness = scale2*0.966,
        damping = scale2*0.234,
    ),
    "leg2joint2": ImplicitActuatorCfg(
        joint_names_expr=["leg2joint2"],
        effort_limit_sim=scale2*136.11,
        velocity_limit_sim=5*0.15,
        stiffness = scale2*0.966,
        damping = scale2*0.234,
    ),
    "leg2joint3": ImplicitActuatorCfg(
        joint_names_expr=["leg2joint3"],
        effort_limit_sim=scale2*136.11,
        velocity_limit_sim=5*0.15,
        stiffness = scale2*0.966,
        damping =scale2* 0.234,
    ),
    "leg2joint4": ImplicitActuatorCfg(
        joint_names_expr=["leg2joint4"],
        effort_limit_sim=scale2*136.11,
        velocity_limit_sim=5*0.15,
        stiffness = scale2*0.966,
        damping = scale2*0.234,
    ),
    "leg2joint5": ImplicitActuatorCfg(
        joint_names_expr=["leg2joint5"],
        effort_limit_sim=scale2*136.11,
        velocity_limit_sim=5*0.15,
        stiffness = scale2*0.966,
        damping = scale2*0.234,
    ),
    "leg2joint6": ImplicitActuatorCfg(
        joint_names_expr=["leg2joint6"],
        effort_limit_sim=scale2*136.11,
        velocity_limit_sim=5*0.15,
        stiffness = scale2*0.966,
        damping = scale2*0.234,
    ),
    "leg2joint7": ImplicitActuatorCfg(
        joint_names_expr=["leg2joint7"],
        effort_limit_sim=scale2*136.11,
        velocity_limit_sim=20*0.15,
        stiffness =scale2*0.966,
        damping = scale2*0.234,
    ),
    # "leg2grip2": ImplicitActuatorCfg(
    #     joint_names_expr=["leg2grip2"],
    #     effort_limit_sim=30,
    #     velocity_limit_sim=5*0.15,
    #     stiffness =3000,
    #     damping =5,           # per evitare vibrazioni
    #     ),
    
    # # "leg2grip2": ImplicitActuatorCfg(
    # #     joint_names_expr=["leg2grip2"],
    # #     effort_limit_sim=10,
    # #     velocity_limit_sim=2,
    # #     stiffness=3000,
    # #     damping=160,
    # # ),
    # "leg2grip2bis": ImplicitActuatorCfg(
    #     joint_names_expr=["leg2grip2bis"],
    #     effort_limit_sim=30,
    #     velocity_limit_sim=5*0.15,
    #     stiffness =3000,
    #     damping = 5,
    # ),
    },
)

WHEEL_WITH_HANDLE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/hero_wheel",
    spawn=sim_utils.UsdFileCfg(
        usd_path=ISAAC_LAB_PATH + "/source/isaaclab_tasks/isaaclab_tasks/manager_based/moonshot/descriptions/usd/robot/HERO_wheel/hero_wheel.usd",
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(1.6, 0.0, 0.4),  # adatta alla scena
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={".*": 0.0}
    ),
    actuators={},  # Nessun controllo per ora
)
