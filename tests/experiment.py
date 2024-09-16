from copy import deepcopy
from time import perf_counter
import gymnasium as gym
import numpy as np
import torch
from dynamics_models.acrobot import Acrobot
from dynamics_models.go1 import Go1
from dynamics_models.gym_model import GymModel
from dynamics_models.half_cheetah import HalfCheetah
from dynamics_models.hopper import Hopper
from dynamics_models.humanoid import Humanoid
from dynamics_models.inverted_pendulum import InvertedPendulum
from dynamics_models.neural_model import NeuralModel, RolloutDataset
from dynamics_models.pendulum import Pendulum
from dynamics_models.walker_2d import Walker2D
from pytorch_mppi import mppi
from gymnasium import logger as gym_log
import wandb

from dm_control import suite

from experiment_launcher import single_experiment, run_experiment

torch.set_num_interop_threads(1)
torch.set_num_threads(1)

def no_train(new_data):
    pass
    
#@single_experiment
#def experiment(env_name: str = "pendulum",
#def experiment(env_name: str = "acrobot",
#def experiment(env_name: str = "go1",
#def experiment(env_name: str = "walker",
def experiment(env_name: str = "hopper",
#def experiment(env_name: str = "humanoid",
               neural_model: bool = False,
               dataset_path: str = None,
               #dataset_path: str = "humanoid_fcem_nc7_sig7_h30_ns100.pt",
               n_episodes_per_fit: int = -1,
               #neural_model: bool = True,
               #n_episodes_per_fit: int = 2,
               #n_warmup_episodes: int = 10,
               n_episodes: int = 3,
               #n_steps: int = 100,
               n_steps: int = 200,
               #horizon: int = 15,
               horizon: int = 50,
               n_samples: int = 30,
               #n_samples: int = 10,
               #noise_sigma: float = 10.0,
               noise_sigma: float = 15.0,
               lambda_: float = 1.0,
               downward_start: bool = True,
               device: str = "cpu",
               dtype: str = "double",
               noise_beta: float = None,
               #noise_cutoff_freq: float = None,
               #noise_beta: float = 0.2,
               noise_cutoff_freq: float = 1.0,
               #render: bool = False,
               render: bool = True,
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
    elif env_name == "acrobot":
        model = Acrobot()
        env = gym.make("dm_control/acrobot-swingup-v0", render_mode="human" if render else None)
        from shimmy.utils.dm_env import dm_obs2gym_obs
        def get_obs(self):
            return np.concat([self.physics.data.qpos, self.physics.data.qvel])
        #env.unwrapped._get_obs = get_obs
        setattr(type(env.unwrapped), "_get_obs", get_obs)
    elif env_name == "go1":
        env = gym.make(
            'Ant-v5',
            xml_file='./mujoco_menagerie/unitree_go1/scene.xml',
            forward_reward_weight=1,  # kept the same as the 'Ant' environment
            ctrl_cost_weight=0.05,  # changed because of the stronger motors of `Go1`
            contact_cost_weight=5e-4,  # kept the same as the 'Ant' environment
            healthy_reward=1,  # kept the same as the 'Ant' environment
            main_body=1,  # represents the "trunk" of the `Go1` robot
            healthy_z_range=(0.195, 0.75),
            include_cfrc_ext_in_observation=False,
            exclude_current_positions_from_observation=False,
            reset_noise_scale=0.1,
            frame_skip=5,
            max_episode_steps=100,
            terminate_when_unhealthy=False,
            render_mode="human" if render else None,
        )
        model = Go1(deepcopy(env))
        noise_sigma = noise_sigma * torch.eye(model.env.action_space.shape[0], device=d, dtype=dtype)
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
        env = gym.make("Humanoid-v5", render_mode="human" if render else None, exclude_current_positions_from_observation=False,
                       #terminate_when_unhealthy=False)
                       terminate_when_unhealthy=False,
                       include_cinert_in_observation=False,
                       include_cvel_in_observation=False,
                       include_qfrc_actuator_in_observation=False,
                       include_cfrc_ext_in_observation=False)
        model = Humanoid(deepcopy(env))
        noise_sigma = noise_sigma * torch.eye(model.env.action_space.shape[0], device=d, dtype=dtype)
    elif env_name == "hopper":
        env = gym.make("Hopper-v5", render_mode="human" if render else None, exclude_current_positions_from_observation=False,
                         terminate_when_unhealthy=False)
        model = Hopper(deepcopy(env))
        noise_sigma = noise_sigma * torch.eye(model.env.action_space.shape[0], device=d, dtype=dtype)
    else:
        raise ValueError("Unknown environment")

    state_dim = model.state_dim
    action_dim = model.action_dim

    if neural_model:
        model = NeuralModel(model, state_dim, action_dim, device=d, dtype=dtype)

    dt = model.dt
    nx = state_dim
    action_lb = torch.tensor(model.action_low, device=d, dtype=dtype) 
    action_ub = torch.tensor(model.action_high, device=d, dtype=dtype)
    #nx = env.observation_space.shape[0]
    mppi_gym = mppi.MPPI(model.dynamics, model.running_cost, nx, noise_sigma, num_samples=n_samples, horizon=horizon,
                        lambda_=lambda_, u_min=action_lb,
                        u_max=torch.tensor(action_ub, device=d), device=d,
                        noise_beta=noise_beta, noise_cutoff_freq=noise_cutoff_freq, sampling_freq=1./dt)

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
        total_reward, history = mppi.run_mppi(mppi_gym, env, model.train, iter=n_steps, retrain_after_iter=n_steps, render=render)
        print(f"Episode {i} Total reward", total_reward)
        states = history[..., :state_dim]
        actions = history[..., -action_dim:]
        dataset.append_data(states, actions)
        if n_episodes_per_fit > 0 and i % n_episodes_per_fit == 0:
        #if n_episodes_per_fit > 0 and i > n_warmup_episodes and i % n_episodes_per_fit == 0:
            model.train(dataset())
        wandb.log({"total_reward": total_reward}, step=i)
        rewards.append(total_reward)
        t1 = perf_counter()
        print(f"Episode {i} Time", t1 - t0)
        #trajectories.append(history)
    #trajectories = np.array(trajectories)
    #np.save(f"half_cheetah_mppi_trajectories_lp7_sig7.npy", trajectories)
    #dataset.save_data("humanoid_fcem_nc7_sig7_h30_ns100.pt")
    return np.mean(rewards)

if __name__ == "__main__":
    run_experiment(single_experiment(experiment))


