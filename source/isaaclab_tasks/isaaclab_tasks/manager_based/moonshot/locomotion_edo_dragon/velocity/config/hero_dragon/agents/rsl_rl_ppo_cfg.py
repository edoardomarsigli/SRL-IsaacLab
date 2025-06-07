# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from isaaclab.utils import configclass


@configclass
class HeroDragonRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 64
    max_iterations = 20000
    save_interval = 100
    experiment_name = "hero_dragon_nav"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[128,64],
        critic_hidden_dims=[128,64],
        # actor_hidden_dims=[400,200,100],
        # critic_hidden_dims=[400,200,100],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=5e-3,
        num_learning_epochs=5,
        num_mini_batches=64,
        learning_rate=1e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1,
    )


@configclass
class HeroDragonFlatPPORunnerCfg(HeroDragonRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 2000
        self.experiment_name = "hero_dragon_nav"
        self.policy.actor_hidden_dims = [128, 64]
        self.policy.critic_hidden_dims = [128, 64]
