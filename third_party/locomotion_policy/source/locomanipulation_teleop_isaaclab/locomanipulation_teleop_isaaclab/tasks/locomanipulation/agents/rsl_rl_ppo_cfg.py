from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100
    experiment_name = "go2_piper_standard_flat"
    empirical_normalization = False
    logger = "tensorboard"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    policy = RslRlPpoActorCriticCfg(init_noise_std=.6, actor_hidden_dims=[256,128,128],
        critic_hidden_dims=[256,128,128], activation="elu",
        actor_obs_normalization=False, critic_obs_normalization=False)
    algorithm = RslRlPpoAlgorithmCfg(value_loss_coef=1., use_clipped_value_loss=True,
        clip_param=.2, entropy_coef=.005, num_learning_epochs=5, num_mini_batches=4,
        learning_rate=1.e-3, schedule="adaptive", gamma=.99, lam=.95, desired_kl=.01, max_grad_norm=1.)


@configclass
class RoughPPORunnerCfg(FlatPPORunnerCfg):
    max_iterations = 15000
    experiment_name = "go2_piper_standard_rough"
