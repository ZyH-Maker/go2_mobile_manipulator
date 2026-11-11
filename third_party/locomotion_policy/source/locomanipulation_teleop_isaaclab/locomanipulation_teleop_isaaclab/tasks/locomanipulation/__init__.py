"""Standard PiPER tasks. Legacy Vision/Blind task IDs are retired."""
import gymnasium as gym
from .locomanipulation_env import LocomotionManipulationEnv
from .locomanipulation_env_cfg import Go2FlatEnvCfg, Go2RoughEnvCfg

for name, cfg, agent in (("Flat", Go2FlatEnvCfg, "FlatPPORunnerCfg"),
                         ("Rough", Go2RoughEnvCfg, "RoughPPORunnerCfg")):
    gym.register(id=f"Go2-Piper-Standard-{name}-v0", entry_point=LocomotionManipulationEnv,
        disable_env_checker=True, kwargs={"env_cfg_entry_point": cfg,
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:{agent}"})
