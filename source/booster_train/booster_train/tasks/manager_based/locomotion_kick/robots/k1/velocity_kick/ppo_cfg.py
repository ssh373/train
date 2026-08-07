from isaaclab.utils import configclass
from booster_train.tasks.manager_based.locomotion_kick.agents import LocomotionKickPPORunnerCfg


@configclass
class PPORunnerCfg(LocomotionKickPPORunnerCfg):
    experiment_name = "k1_velocity_kick_001"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.symmetry_cfg = None
