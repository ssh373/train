"""Gym registrations for K1 short-range goal locomotion."""

import gymnasium as gym


def _register(task_id: str, env_cfg: str, agent_cfg: str = "PPORunnerCfg"):
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:{env_cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:{agent_cfg}",
            "legacy_rsl_rl_cfg_entry_point": f"{__name__}.legacy_ppo_cfg:LegacyPPORunnerCfg",
        },
    )


_register("Booster-K1-GoTo-v0", "K1GoToEnvCfg")
_register("Booster-K1-GoTo-Smoke-v0", "K1GoToSmokeEnvCfg", "SmokePPORunnerCfg")
_register("Booster-K1-GoTo-Sim2Real-v0", "K1GoToSim2RealEnvCfg")
_register("Booster-K1-GoTo-v0-Play", "K1GoToPlayEnvCfg")
_register("Booster-K1-GoTo-AStar-v0-Play", "K1GoToAStarPlayEnvCfg")
_register("Booster-K1-GoTo-Dynamic-v0", "K1GoToDynamicEnvCfg")
_register("Booster-K1-GoTo-Dynamic-v0-Play", "K1GoToDynamicPlayEnvCfg")
_register("Booster-K1-GoTo-PhaseA-v0", "K1GoToPhaseAEnvCfg", "PhaseAPPORunnerCfg")
_register("Booster-K1-GoTo-PhaseA-v0-Play", "K1GoToPhaseAPlayEnvCfg", "PhaseAPPORunnerCfg")
_register("Booster-K1-GoTo-FineTune-v0", "K1GoToFineTuneEnvCfg", "FineTunePPORunnerCfg")
_register("Booster-K1-GoTo-FineTune-v0-Play", "K1GoToFineTunePlayEnvCfg", "FineTunePPORunnerCfg")
