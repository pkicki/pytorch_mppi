import torch
import numpy as np
from time import perf_counter
from experiment_launcher import single_experiment, run_experiment

from pytorch_mppi import my_mppi
from pytorch_mppi.utils import load_env_and_model

torch.set_num_interop_threads(1)
torch.set_num_threads(1)

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
               #simulator: str = "brax",
               simulator: str = "gym",
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
    torch.manual_seed(seed)
    np.random.seed(seed)

    d = device
    #dtype = torch.double
    dtype = torch.float

    noise_sigma = torch.tensor(noise_sigma, device=d, dtype=dtype)
    lambda_ = lambda_

    env, model = load_env_and_model(env_name, simulator, n_envs=n_samples, render=render, compile=True)

    noise_sigma = noise_sigma * torch.eye(model.action_dim, device=d, dtype=dtype)
    state_dim = model.state_dim
    action_dim = model.action_dim

    dt = model.dt

    mppi_gym = my_mppi.MPPI(env=model, horizon=horizon, num_samples=n_samples,
                            control_dim=action_dim, state_dim=state_dim, lambda_=lambda_,
                            noise_sigma=noise_sigma, noise_beta=noise_beta,
                            noise_cutoff_freq=noise_cutoff_freq,
                            noise_interpolate_nodes=noise_interpolate_nodes,
                            sampling_freq=1./dt)

    rewards = []
    trajectories = []
    actionss = []
    statess = []
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
        total_reward, history, info = my_mppi.run_mppi(mppi_gym, env, iter=n_steps, render=render)
        print(f"Episode {i} Total reward", total_reward)
        states = history[..., :state_dim]
        actions = history[..., -action_dim:]
        actionss.append(actions)
        statess.append(states)
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


