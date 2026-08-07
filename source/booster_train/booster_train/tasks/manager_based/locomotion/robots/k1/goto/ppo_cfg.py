"""Feed-forward PPO settings based on the repository's validated locomotion PPO."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg,
)

from .symmetry import data_augmentation_func


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 200
    experiment_name = "k1_goto"
    empirical_normalization = True
    clip_actions = 1.0
    logger = "tensorboard"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    # PPO values are inherited from the existing Booster locomotion baseline.
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
        entropy_coef=0.005, num_learning_epochs=5, num_mini_batches=4,
        learning_rate=1.0e-3, schedule="adaptive", gamma=0.99, lam=0.95,
        desired_kl=0.01, max_grad_norm=1.0,
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


@configclass
class FineTunePPORunnerCfg(PPORunnerCfg):
    max_iterations = 30000
    save_interval = 100
    # Keep the same log root so --load_run can resolve an existing GoTo run.
    # Use --run_name to distinguish the new fine-tuning output directory.
    experiment_name = "k1_goto"

    def __post_init__(self):
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.schedule = "adaptive"
        self.algorithm.desired_kl = 0.005
        self.algorithm.entropy_coef = 0.001


@configclass
class PhaseAPPORunnerCfg(FineTunePPORunnerCfg):
    max_iterations = 5000
    save_interval = 100
