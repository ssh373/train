"""Task-specific PPO configuration."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlSymmetryCfg

from booster_train.tasks.manager_based.adjust_kick.agents.rsl_rl_ppo_cfg import AdjustKickPPORunnerCfg
from .symmetry import data_augmentation_func


@configclass
class PPORunnerCfg(AdjustKickPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=False,
            mirror_loss_coeff=0.0,
            data_augmentation_func=data_augmentation_func,
        )


@configclass
class UnifiedPPORunnerCfg(AdjustKickPPORunnerCfg):
    """PPO configuration for the final single integrated actor."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "k1_adjust_kick_unified_001"
        self.max_iterations = 12000
        self.policy.init_noise_std = 0.5
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
        self.algorithm.learning_rate = 3.0e-5
        self.algorithm.entropy_coef = 0.003
