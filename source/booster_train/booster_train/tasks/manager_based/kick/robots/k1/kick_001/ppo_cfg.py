from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlSymmetryCfg

from booster_train.tasks.manager_based.kick.agents.rsl_rl_ppo_cfg import KickPPORunnerCfg

from .symmetry import data_augmentation_func


@configclass
class PPORunnerCfg(KickPPORunnerCfg):
    experiment_name = "k1_kick_001"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=False,
            mirror_loss_coeff=0.0,
            data_augmentation_func=data_augmentation_func,
        )
