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
