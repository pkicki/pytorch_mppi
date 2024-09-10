import gymnasium as gym
import numpy as np
import torch
from dynamics_models.half_cheetah import HalfCheetah
from dynamics_models.pendulum import Pendulum
from pytorch_mppi import mppi
from gymnasium import logger as gym_log

from experiment_launcher import single_experiment, run_experiment

torch.set_num_interop_threads(1)
torch.set_num_threads(1)
    
@single_experiment
#def experiment(env: str = "pendulum",
def experiment(env: str = "half_cheetah",
               n_episodes: int = 100,    
               horizon: int = 15,
               #horizon: int = 50,
               #n_samples: int = 100,
               n_samples: int = 5,
               noise_sigma: float = 10.,
               lambda_: float = 1.0,
               downward_start: bool = True,
               device: str = "cpu",
               dtype: str = "double",
               noise_beta: float = None,
               #noise_cutoff_freq: float = None,
               #noise_beta: float = 2.,
               noise_cutoff_freq: float = 5.0,
               #render: bool = False,
               render: bool = True,
               results_dir: str = "./results",
               seed: int = 444,
               ) -> None:
    gym_log.set_level(gym_log.ERROR)

    d = device
    dtype = torch.double

    noise_sigma = torch.tensor(noise_sigma, device=d, dtype=dtype)
    lambda_ = lambda_

    if env == "pendulum":
        gym_env_name = "Pendulum-v1"
        model = Pendulum()
    elif env == "half_cheetah":
        gym_env_name = "HalfCheetah-v4"
        model = HalfCheetah()
        noise_sigma = noise_sigma * torch.eye(model.env.action_space.shape[0], device=d, dtype=dtype)
    else:
        raise ValueError("Unknown environment")


    env = gym.make(gym_env_name, render_mode="human" if render else None, exclude_current_positions_from_observation=False)
    trajectories = []
    for i in range(n_episodes):
        env.reset()
        if env == "pendulum" and downward_start:
            env.state = env.unwrapped.state = [np.pi, 1]

        dt = env.unwrapped.dt
        nx = env.observation_space.shape[0]
        mppi_gym = mppi.MPPI(model.dynamics, model.running_cost, nx, noise_sigma, num_samples=n_samples, horizon=horizon,
                            lambda_=lambda_, u_min=torch.tensor(model.action_low, device=d),
                            u_max=torch.tensor(model.action_high, device=d), device=d,
                            noise_beta=noise_beta, noise_cutoff_freq=noise_cutoff_freq, sampling_freq=1./dt)
        total_reward, history = mppi.run_mppi(mppi_gym, env, model.train, iter=100, retrain_after_iter=100, render=render)
        print(f"Episode {i} Total reward", total_reward)
        trajectories.append(history)
    trajectories = np.array(trajectories)
    np.save(f"mppi_trajectories_lp05.npy", trajectories)

if __name__ == "__main__":
    run_experiment(experiment)


