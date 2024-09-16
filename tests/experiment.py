import gymnasium as gym
import numpy as np
import torch
from dynamics_models.half_cheetah import HalfCheetah
from dynamics_models.humanoid import Humanoid
from dynamics_models.neural_model import NeuralModel, RolloutDataset
from dynamics_models.pendulum import Pendulum
from dynamics_models.walker_2d import Walker2D
from pytorch_mppi import mppi
from gymnasium import logger as gym_log
import wandb

from experiment_launcher import single_experiment, run_experiment

torch.set_num_interop_threads(1)
torch.set_num_threads(1)

def no_train(new_data):
    pass
    
#@single_experiment
def experiment(env_name: str = "pendulum",
#def experiment(env_name: str = "half_cheetah",
#def experiment(env_name: str = "walker",
#def experiment(env_name: str = "humanoid",
#def experiment(env_name: str = "humanoid",
               neural_model: bool = False,
               dataset_path: str = None,
               #dataset_path: str = "humanoid_fcem_nc7_sig7_h30_ns100.pt",
               n_episodes_per_fit: int = -1,
               #neural_model: bool = True,
               #n_episodes_per_fit: int = 2,
               #n_warmup_episodes: int = 10,
               n_episodes: int = 2,
               #horizon: int = 15,
               horizon: int = 30,
               n_samples: int = 100,
               #n_samples: int = 10,
               noise_sigma: float = 7.0,
               lambda_: float = 1.0,
               downward_start: bool = True,
               device: str = "cpu",
               dtype: str = "double",
               noise_beta: float = None,
               #noise_cutoff_freq: float = None,
               #noise_beta: float = 0.2,
               noise_cutoff_freq: float = 7.0,
               render: bool = False,
               #render: bool = True,
               results_dir: str = "./results",
               #wandb_logging: bool = True,
               wandb_logging: bool = False,
               seed: int = 444,
               ) -> None:
    gym_log.set_level(gym_log.ERROR)

    d = device
    dtype = torch.double

    mode = "online" if wandb_logging else "disabled"
    wandb.init(project="mppi", entity="kicai", mode=mode)
    wandb.config.update(locals())

    noise_sigma = torch.tensor(noise_sigma, device=d, dtype=dtype)
    lambda_ = lambda_

    dataset = RolloutDataset()
    if dataset_path is not None:
        dataset.load_data(dataset_path)

    if env_name == "pendulum":
        model = Pendulum()
        env = gym.make("Pendulum-v1", render_mode="human" if render else None)
    elif env_name == "half_cheetah":
        model = HalfCheetah()
        noise_sigma = noise_sigma * torch.eye(model.env.action_space.shape[0], device=d, dtype=dtype)
        env = gym.make("HalfCheetah-v4", render_mode="human" if render else None, exclude_current_positions_from_observation=False)
    elif env_name == "walker":
        model = Walker2D()
        noise_sigma = noise_sigma * torch.eye(model.env.action_space.shape[0], device=d, dtype=dtype)
        env = gym.make("Walker2d-v4", render_mode="human" if render else None, exclude_current_positions_from_observation=False,
                       terminate_when_unhealthy=False)
    elif env_name == "humanoid":
        model = Humanoid()
        noise_sigma = noise_sigma * torch.eye(model.env.action_space.shape[0], device=d, dtype=dtype)
        env = gym.make("Humanoid-v4", render_mode="human" if render else None, exclude_current_positions_from_observation=False,
                       terminate_when_unhealthy=False)
    else:
        raise ValueError("Unknown environment")

    state_dim = model.state_dim
    action_dim = model.action_dim

    if neural_model:
        model = NeuralModel(model, state_dim, action_dim, device=d, dtype=dtype)

    dt = env.unwrapped.dt
    nx = env.observation_space.shape[0]
    mppi_gym = mppi.MPPI(model.dynamics, model.running_cost, nx, noise_sigma, num_samples=n_samples, horizon=horizon,
                        lambda_=lambda_, u_min=torch.tensor(env.unwrapped.action_space.low, device=d),
                        u_max=torch.tensor(env.unwrapped.action_space.high, device=d), device=d,
                        noise_beta=noise_beta, noise_cutoff_freq=noise_cutoff_freq, sampling_freq=1./dt)

    rewards = []
    trajectories = []
    for i in range(n_episodes):
        env.reset()
        if env_name == "pendulum" and downward_start:
            env.state = env.unwrapped.state = [np.pi, 0]
        total_reward, history = mppi.run_mppi(mppi_gym, env, model.train, iter=100, retrain_after_iter=100, render=render)
        print(f"Episode {i} Total reward", total_reward)
        states = history[..., :state_dim]
        actions = history[..., -action_dim:]
        dataset.append_data(states, actions)
        if n_episodes_per_fit > 0 and i % n_episodes_per_fit == 0:
        #if n_episodes_per_fit > 0 and i > n_warmup_episodes and i % n_episodes_per_fit == 0:
            model.train(dataset())
        wandb.log({"total_reward": total_reward}, step=i)
        rewards.append(total_reward)
        #trajectories.append(history)
    #trajectories = np.array(trajectories)
    #np.save(f"half_cheetah_mppi_trajectories_lp7_sig7.npy", trajectories)
    dataset.save_data("humanoid_fcem_nc7_sig7_h30_ns100.pt")
    return np.mean(rewards)

if __name__ == "__main__":
    run_experiment(single_experiment(experiment))


