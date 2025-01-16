# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn a cart-pole and interact with it.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/01_assets/run_articulation.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with an articulation.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg



##
# Configuration
##

"""Configuration for a HERO DRAGON"""
wheel12_body_pos = {"wheel12_left_joint": 0.0, "wheel12_right_joint": 0.0}
wheel14_body_pos = {"wheel14_left_joint": 0.0, "wheel14_right_joint": 0.0}

leg2gripper1_pos = {"leg2base_link_link2": 0.0}
leg4gripper1_pos = {"leg4base_link_link2": 0.0}

leg2link2_pos = {"leg2link2_link3": 0.0}
leg2link3_pos = {"leg2link3_link4": 0.0}
leg2link4_pos = {"leg2link4_link5": 0.0}
leg2link5_pos = {"leg2link5_link6": 0.0}
leg2link6_pos = {"leg2link6_link7": 0.0}
leg2link7_pos = {"leg2link7_link8": 0.0}

leg4link2_pos = {"leg4link2_link3": 0.0}
leg4link3_pos = {"leg4link3_link4": 0.0}
leg4link4_pos = {"leg4link4_link5": 0.0}
leg4link5_pos = {"leg4link5_link6": 0.0}
leg4link6_pos = {"leg4link6_link7": 0.0}
leg4link7_pos = {"leg4link7_link8": 0.0}

all_joint_pos = wheel12_body_pos | leg4gripper1_pos | leg4link2_pos|leg4link3_pos | leg4link4_pos  \
    | leg4link5_pos|leg4link6_pos |leg4link7_pos | leg2gripper1_pos| leg2link2_pos | leg2link3_pos \
    | leg2link4_pos | leg2link5_pos | leg2link6_pos | leg2link7_pos | wheel14_body_pos

HERO_DRAGON_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="C:/Users/marcu/Documents/SpaceRoboticsLab_Tohoku/ros2_ws/src/urdf_packer/usd/hero_dragon.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0), joint_pos=all_joint_pos
    ),
    actuators = {
        "wheel12_left_actuator": ImplicitActuatorCfg(
            joint_names_expr=["wheel12_left_joint"],
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=10.0,
        ),
        "wheel12_right_actuator": ImplicitActuatorCfg(
            joint_names_expr=["wheel12_right_joint"],
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=10.0,
        ),
        "wheel14_left_actuator": ImplicitActuatorCfg(
            joint_names_expr=["wheel14_left_joint"],
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=10.0,
        ),
        "wheel14_right_actuator": ImplicitActuatorCfg(
            joint_names_expr=["wheel14_right_joint"],
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=10.0,
        ),
        "leg2_base_actuator": ImplicitActuatorCfg(
            joint_names_expr=["leg2base_link_link2"],
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=10.0,
        ),
        "leg4_base_actuator": ImplicitActuatorCfg(
            joint_names_expr=["leg4base_link_link2"],
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=10.0,
        ),
        "leg2_link2_actuator": ImplicitActuatorCfg(
            joint_names_expr=["leg2link2_link3"],
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=10.0,
        ),
        "leg2_link3_actuator": ImplicitActuatorCfg(
            joint_names_expr=["leg2link3_link4"],
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=10.0,
        ),
        "leg2_link4_actuator": ImplicitActuatorCfg(
            joint_names_expr=["leg2link4_link5"],
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=10.0,
        ),
        "leg2_link5_actuator": ImplicitActuatorCfg(
            joint_names_expr=["leg2link5_link6"],
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=10.0,
        ),
        "leg2_link6_actuator": ImplicitActuatorCfg(
            joint_names_expr=["leg2link6_link7"],
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=10.0,
        ),
        "leg2_link7_actuator": ImplicitActuatorCfg(
            joint_names_expr=["leg2link7_link8"],
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=10.0,
        ),
        "leg4_link2_actuator": ImplicitActuatorCfg(
            joint_names_expr=["leg4link2_link3"],
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=10.0,
        ),
        "leg4_link3_actuator": ImplicitActuatorCfg(
            joint_names_expr=["leg4link3_link4"],
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=10.0,
        ),
        "leg4_link4_actuator": ImplicitActuatorCfg(
            joint_names_expr=["leg4link4_link5"],
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=10.0,
        ),
        "leg4_link5_actuator": ImplicitActuatorCfg(
            joint_names_expr=["leg4link5_link6"],
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=10.0,
        ),
        "leg4_link6_actuator": ImplicitActuatorCfg(
            joint_names_expr=["leg4link6_link7"],
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=10.0,
        ),
        "leg4_link7_actuator": ImplicitActuatorCfg(
            joint_names_expr=["leg4link7_link8"],
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=10.0,
        ),
    },
)
"""Configuration for a HERO DRAGON"""


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Create separate groups called "Origin1", "Origin2"
    # Each group will have a robot in it
    origins = [[0.0, 0.0, 0.5], 
               [-3.0, 0.0, 0.5],
               [0.0, -3.0, 0.5],
               [-3.0, -3.0, 0.5]
               ]
    # Origin 1
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
    # Origin 2
    prim_utils.create_prim("/World/Origin2", "Xform", translation=origins[1])
    prim_utils.create_prim("/World/Origin3", "Xform", translation=origins[2])
    prim_utils.create_prim("/World/Origin4", "Xform", translation=origins[3])

    # Articulation
    dragon_cfg = HERO_DRAGON_CFG.copy()
    dragon_cfg.prim_path = "/World/Origin.*/Robot"
    dragon = Articulation(cfg=dragon_cfg)

    # return the scene information
    scene_entities = {"dragon": dragon}
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    robot = entities["dragon"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0

# Get the list of all joint names in the robot
    joint_names = robot.data.joint_names  # List of all joint names in the articulation
    wheel_joint_names = [
        "wheel12_left_joint",
        "wheel12_right_joint",
        "wheel14_left_joint",
        "wheel14_right_joint",
    ]
    
    # Find the indices of the wheel joints within the joint_names list
    wheel_joint_indices = [joint_names.index(name) for name in wheel_joint_names if name in joint_names]

    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += origins
            robot.write_root_state_to_sim(root_state)
            # set joint positions with some noise
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            # joint_pos += torch.rand_like(joint_pos) * 0.01
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            robot.reset()
            print("[INFO]: Resetting robot state...")
        # Apply random action
        # Generate effort only for wheel joints
        efforts = torch.zeros_like(robot.data.joint_pos)  # Initialize efforts as zeros
        # efforts[:, 0:4] = torch.tensor([[5.0, 5.0, 5.0, 5.0]] * len(origins), dtype=torch.float32)  
        robot.set_joint_effort_target(efforts)
        # -- write data to sim
        robot.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        robot.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
