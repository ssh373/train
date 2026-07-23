"""Recurrent PPO settings based on the repository's validated locomotion PPO."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticRecurrentCfg, RslRlPpoAlgorithmCfg,
)


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 200
    experiment_name = "k1_goto"
    empirical_normalization = True
    logger = "tensorboard"
    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=128,
        rnn_num_layers=2,
    )
    # PPO values are inherited from the existing Booster locomotion baseline.
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
        entropy_coef=0.005, num_learning_epochs=5, num_mini_batches=4,
        learning_rate=1.0e-3, schedule="adaptive", gamma=0.99, lam=0.95,
        desired_kl=0.01, max_grad_norm=1.0,
        # RSL-RL 2.3.1's symmetry update does not augment recurrent masks and
        # hidden states together with sequence observations.  Keep symmetry
        # fully disabled for the runnable LSTM baseline; symmetry.py retains
        # the verified K1 reflection for a future recurrent-aware PPO patch.
        symmetry_cfg=None,
    )


@configclass
class SmokePPORunnerCfg(PPORunnerCfg):
    max_iterations = 5
    save_interval = 5
    experiment_name = "k1_goto_smoke"
