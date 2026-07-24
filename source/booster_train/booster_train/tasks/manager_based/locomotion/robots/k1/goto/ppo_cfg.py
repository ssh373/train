"""Recurrent PPO settings based on the repository's validated locomotion PPO."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticRecurrentCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg,
)

from .recurrent_symmetry import RecurrentSymmetryPPO
from .symmetry import data_augmentation_func

# OnPolicyRunner 2.3.1 resolves the algorithm from its module-level ``PPO``
# symbol.  Keep class_name="PPO" (required for RL training type detection) and
# replace that symbol only when this GoTo agent configuration is imported.
import rsl_rl.runners.on_policy_runner as _runner_module

_runner_module.PPO = RecurrentSymmetryPPO


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
        # The custom PPO consumes this config to obtain the wrapped environment;
        # stock update-time augmentation is replaced by paired recurrent
        # trajectories in RecurrentSymmetryPPO.
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=False,
            mirror_loss_coeff=0.0,
            data_augmentation_func=data_augmentation_func,
        ),
    )


@configclass
class SmokePPORunnerCfg(PPORunnerCfg):
    max_iterations = 5
    save_interval = 5
    experiment_name = "k1_goto_smoke"
