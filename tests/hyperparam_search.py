import optuna
from experiment import experiment as mppi_experiment
from experiment_launcher import single_experiment, run_experiment


@single_experiment
def experiment(
    #env_name: str = "half_cheetah",
    #env_name: str = "walker",
    #env_name: str = "pendulum",
    env_name: str = "hopper",
    n_episodes: int = 100,
    n_steps: int = 200,
    horizon: int = 5,
    n_samples: int = 100,
    alg: str = "fcem",
    #alg: str = "mppi",
    simulator: str = "gym",
    #simulator: str = "brax",
    results_dir: str = "./results",
    seed: int = 444,
    ):

    def objective(trial):
        #x = trial.suggest_float('x', -10, 10)
        #noise_sigma = trial.suggest_float('noise_sigma', 0.1, 10.)
        noise_sigma = trial.suggest_float('noise_sigma', 0.1, 20.)
        noise_beta = None
        noise_cutoff_freq = None
        noise_interpolate_nodes = None
        interpolation_type = None
        if alg == "icem":
            noise_beta = trial.suggest_float('noise_beta', 0.1, 10.)
        elif alg == "fcem":
            noise_cutoff_freq = trial.suggest_float('noise_cutoff_freq', 0.1, 10.)
        elif alg == "mppi":
            pass
        elif alg == "cubic_noise":
            noise_interpolate_nodes = trial.suggest_int('noise_interpolate_nodes', 2, horizon - 1)
            interpolation_type = "cubic_noise"
        elif alg == "cubic_actions":
            noise_interpolate_nodes = trial.suggest_int('noise_interpolate_nodes', 2, horizon - 1)
            interpolation_type = "cubic_actions"
        else:
            raise ValueError("Unknown algorithm")
        mean_reward = mppi_experiment(env_name=env_name,
                                      simulator=simulator,
                                      n_steps=n_steps,
                                      n_episodes=n_episodes,
                                      horizon=horizon,
                                      n_samples=n_samples,
                                      noise_sigma=noise_sigma,
                                      noise_beta=noise_beta,
                                      noise_cutoff_freq=noise_cutoff_freq,
                                      noise_interpolate_nodes=noise_interpolate_nodes,
                                      interpolation_type=interpolation_type,
                                      render=False,
                                      seed=seed,
                                      results_dir="./results")

        return mean_reward

    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(f"./newlp_hyperparam_search_{env_name}_local.log")
        #optuna.storages.journal.JournalFileBackend(f"./test.log")
    )
    study = optuna.create_study(study_name=f"{env_name}_h{horizon}_ns{n_samples}_{alg}",
                                storage=storage, load_if_exists=True,
                                direction="maximize")
    study.optimize(objective, n_trials=10)
    print(study.best_params)

if __name__ == "__main__":
    run_experiment(experiment)
