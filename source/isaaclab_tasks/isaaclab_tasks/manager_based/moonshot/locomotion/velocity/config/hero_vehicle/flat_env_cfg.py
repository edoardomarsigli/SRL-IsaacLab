# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rough_env_cfg import HeroVehicleRoughEnvCfg


@configclass
class HeroVehicleFlatEnvCfg(HeroVehicleRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


class HeroVehicleFlatEnvCfg_PLAY(HeroVehicleFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        self.commands.body_velocity.ranges.lin_vel_x = (0.12, 0.12)
        self.commands.body_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.body_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
