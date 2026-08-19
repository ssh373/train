"""Gym registration for the K1 end-to-end adjust-kick task."""

import gymnasium as gym


gym.register(
    id="Booster-K1-Adjust-Kick_001-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:K1AdjustKickEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Booster-K1-Adjust-Kick-Unified_001-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:K1AdjustKickUnifiedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:UnifiedPPORunnerCfg",
    },
)

gym.register(
    id="Booster-K1-Adjust-Kick-Unified_001-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:K1AdjustKickUnifiedPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:UnifiedPPORunnerCfg",
    },
)

gym.register(
    id="Booster-K1-Adjust-Kick_001-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:K1AdjustKickPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PPORunnerCfg",
    },
)
