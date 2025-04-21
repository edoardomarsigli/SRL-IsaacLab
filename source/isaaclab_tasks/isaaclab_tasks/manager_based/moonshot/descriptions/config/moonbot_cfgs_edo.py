"""Configurations for different Moonbot types"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
import isaaclab_tasks.manager_based.moonshot.utils as moonshot_utils

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

DRAGON_ARTICULATED_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/hero_dragon",
    spawn=sim_utils.UsdFileCfg(
        usd_path=ISAAC_LAB_PATH + "/source/isaaclab_tasks/isaaclab_tasks/manager_based/moonshot/descriptions/usd/robot/hero_dragon_12467/hero_dragon_edo.usd",
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
    pos=(0.0, 0.0, 0.35),
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
    "leg3_joint1": ImplicitActuatorCfg(
        joint_names_expr=["leg3joint1"],
        effort_limit_sim=1.87,
        velocity_limit_sim=0.5,
        stiffness=1e6,
        damping=300,
    ),
    "leg3_joint2": ImplicitActuatorCfg(
        joint_names_expr=["leg3joint2"],
        effort_limit_sim=3.78,
        velocity_limit_sim=0.5,
        stiffness=1e6,
        damping=300,
    ),
    "leg3_joint3": ImplicitActuatorCfg(
        joint_names_expr=["leg3joint3"],
        effort_limit_sim=4.52,
        velocity_limit_sim=0.5,
        stiffness=1e6,
        damping=300,
    ),
    "leg3_joint4": ImplicitActuatorCfg(
        joint_names_expr=["leg3joint4"],
        effort_limit_sim=10.92,
        velocity_limit_sim=0.5,
        stiffness=1e6,
        damping=300,
    ),
    "leg3_joint5": ImplicitActuatorCfg(
        joint_names_expr=["leg3joint5"],
        effort_limit_sim=1.81,
        velocity_limit_sim=0.5,
        stiffness=1e6,
        damping=300,
    ),
    "leg3_joint6": ImplicitActuatorCfg(
        joint_names_expr=["leg3joint6"],
        effort_limit_sim=2.53,
        velocity_limit_sim=0.5,
        stiffness=1e6,
        damping=300,
    ),
    "leg3_joint7": ImplicitActuatorCfg(
        joint_names_expr=["leg3joint7"],
        effort_limit_sim=0.82,
        velocity_limit_sim=0.5,
        stiffness=1e6,
        damping=300,
    ),
    }
)

WHEEL_WITH_HANDLE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/hero_wheel",
    spawn=sim_utils.UsdFileCfg(
        usd_path=ISAAC_LAB_PATH + "/source/isaaclab_tasks/isaaclab_tasks/manager_based/moonshot/descriptions/usd/robot/HERO_wheel/hero_wheel.usd",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.8, 0.0, 0.3),  # adatta alla scena
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={".*": 0.0}
    ),
    actuators={},  # Nessun controllo per ora
)
