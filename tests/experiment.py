from copy import deepcopy
from time import perf_counter
import gymnasium as gym
import numpy as np
import torch
from dynamics_models.neural_model import NeuralModel, RolloutDataset
from pytorch_mppi import mppi
from pytorch_mppi import my_mppi
from gymnasium import logger as gym_log
import wandb

from dm_control import suite
import matplotlib.pyplot as plt

from experiment_launcher import single_experiment, run_experiment

from pytorch_mppi.utils import load_env_and_model

torch.set_num_interop_threads(1)
torch.set_num_threads(1)

def no_train(new_data):
    pass
    
#@single_experiment
#def experiment(env_name: str = "pendulum",
#def experiment(env_name: str = "acrobot",
#def experiment(env_name: str = "go1",
#def experiment(env_name: str = "walker",
#def experiment(env_name: str = "half_cheetah",
#def experiment(env_name: str = "hopper",
#def experiment(env_name: str = "swimmer",
#def experiment(env_name: str = "humanoid",
#def experiment(env_name: str = "humanoid_standup",
#def experiment(env_name: str = "quadrotor",
def experiment(env_name: str = "car",
               neural_model: bool = False,
               dataset_path: str = None,
               #dataset_path: str = "humanoid_fcem_nc7_sig7_h30_ns100.pt",
               n_episodes_per_fit: int = -1,
               #neural_model: bool = True,
               #n_episodes_per_fit: int = 2,
               #n_warmup_episodes: int = 10,
               n_episodes: int = 1,
               #n_steps: int = 10,
               #n_steps: int = 100,
               n_steps: int = 200,
               #horizon: int = 5,
               horizon: int = 30,
               #horizon: int = 100,
               #n_samples: int = 30,
               n_samples: int = 100,
               #noise_sigma: float = 10.0,
               noise_sigma: float = 5.0,
               #lambda_: float = 1.0,
               lambda_: float = 1e-2,
               downward_start: bool = True,
               device: str = "cpu",
               dtype: str = "double",
               #noise_beta: float = None,
               noise_cutoff_freq: float = None,
               noise_beta: float = 2.0,
               #noise_cutoff_freq: float = 5.0,
               #noise_cutoff_freq: float = 4.4,
               #noise_interpolate_nodes: int = 5,
               noise_interpolate_nodes: int = None,
               interpolation_type: str = None,
               #interpolation_type: str = "cubic_actions",
               #interpolation_type: str = "cubic_noise",
               #render: bool = False,
               render: bool = True,
               results_dir: str = "./results",
               #wandb_logging: bool = True,
               wandb_logging: bool = False,
               seed: int = 444,
               ) -> None:
    gym_log.set_level(gym_log.ERROR)

    d = device
    #dtype = torch.double
    dtype = torch.float

    mode = "online" if wandb_logging else "disabled"
    wandb.init(project="mppi", entity="kicai", mode=mode)
    wandb.config.update(locals())

    noise_sigma = torch.tensor(noise_sigma, device=d, dtype=dtype)
    lambda_ = lambda_

    dataset = RolloutDataset()
    if dataset_path is not None:
        dataset.load_data(dataset_path)

    env, model = load_env_and_model(env_name, simulator, n_envs=n_samples, render=render)

    #s = env.reset()
    #for i in range(10):
    #    t0 = perf_counter()
    #    env.step(np.zeros(model.action_dim))
    #    t1 = perf_counter()
    #    print(f"Step time", t1 - t0)
    #assert False

    noise_sigma = noise_sigma * torch.eye(model.action_dim, device=d, dtype=dtype)
    state_dim = model.state_dim
    action_dim = model.action_dim

    if neural_model:
        model = NeuralModel(model, state_dim, action_dim, device=d, dtype=dtype)

    dt = model.dt
    nx = state_dim
    action_lb = torch.tensor(model.action_low, device=d, dtype=dtype) 
    action_ub = torch.tensor(model.action_high, device=d, dtype=dtype)
    #nx = env.observation_space.shape[0]
    #mppi_gym = mppi.MPPI(model.dynamics, model.dynamics, model.running_cost, nx,
    #                     noise_sigma, num_samples=n_samples, horizon=horizon,
    #                     lambda_=lambda_, u_min=action_lb, u_max=action_ub, device=d,
    #                     noise_beta=noise_beta, noise_cutoff_freq=noise_cutoff_freq,
    #                     noise_interpolate_nodes=noise_interpolate_nodes,
    #                     interpolation_type=interpolation_type, sampling_freq=1./dt)

    mppi_gym = my_mppi.MPPI(dynamics=model.dynamics, horizon=horizon, num_samples=n_samples,
                            control_dim=action_dim, state_dim=state_dim, lambda_=lambda_,
                            noise_sigma=noise_sigma, noise_beta=noise_beta,
                            noise_cutoff_freq=noise_cutoff_freq, sampling_freq=1./dt)

    rewards = []
    trajectories = []
    for i in range(n_episodes):
        t0 = perf_counter()
        env.reset()
        if env_name == "pendulum" and downward_start:
            env.state = env.unwrapped.state = [np.pi, 0]
        if env_name == "acrobot" and downward_start:
            env.unwrapped.physics.named.data.qpos[['shoulder', 'elbow']] = [np.pi, 0.]
            env.unwrapped.physics.named.data.qvel[['shoulder', 'elbow']] = np.zeros(2)
        #total_reward, history = mppi.run_mppi(mppi_gym, env, model.train, iter=n_steps,
        #                                      shift_nominal_trajectory=(interpolation_type != "cubic_actions"),
        #                                      retrain_after_iter=n_steps, render=render)
        total_reward, history = my_mppi.run_mppi(mppi_gym, env, iter=n_steps, render=render)
        print(f"Episode {i} Total reward", total_reward)
        states = history[..., :state_dim]
        actions = history[..., -action_dim:]
        dataset.append_data(states, actions)
        if n_episodes_per_fit > 0 and i % n_episodes_per_fit == 0:
        #if n_episodes_per_fit > 0 and i > n_warmup_episodes and i % n_episodes_per_fit == 0:
            model.train(dataset())
        wandb.log({"total_reward": total_reward}, step=i)
        rewards.append(total_reward)
        #for i in range(states.shape[-1]):
        #    plt.subplot(4, 5, i+1)
        #    plt.plot(states[:, i])
        #plt.show()
        t1 = perf_counter()
        print(f"Episode {i} Time", t1 - t0)
        if render:
            env.save_rendering(f"humanoid_200steps_{i}.html")
        #trajectories.append(history)
    #trajectories = np.array(trajectories)
    #np.save(f"half_cheetah_mppi_trajectories_lp7_sig7.npy", trajectories)
    #dataset.save_data("humanoid_fcem_nc7_sig7_h30_ns100.pt")
    return np.mean(rewards)

if __name__ == "__main__":
    run_experiment(single_experiment(experiment))


