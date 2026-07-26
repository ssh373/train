"""Exact recurrent PPO configuration from the best recorded K1 GoTo run."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticRecurrentCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class LegacyPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Historical LSTM runner, selected explicitly with ``--agent``."""

    seed = 42
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 200
    experiment_name = "k1_goto"
    empirical_normalization = True
    clip_actions = 1.0
    logger = "tensorboard"
    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=128,
        rnn_num_layers=2,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,
    )
