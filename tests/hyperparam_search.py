import optuna
from experiment import experiment as mppi_experiment
from experiment_launcher import single_experiment, run_experiment


@single_experiment
def experiment(
    #env_name: str = "half_cheetah",
    #env_name: str = "walker",
    env_name: str = "pendulum",
    #env_name: str = "hopper",
    n_episodes: int = 100,
    n_steps: int = 100,
    horizon: int = 50,
    n_samples: int = 10,
    alg: str = "icem",
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
        if alg == "icem":
            noise_beta = trial.suggest_float('noise_beta', 0.1, 10.)
            noise_cutoff_freq = None
        elif alg == "fcem":
            noise_beta = None
            noise_cutoff_freq = trial.suggest_float('noise_cutoff_freq', 0.1, 10.)
        elif alg == "mppi":
            noise_beta = None
            noise_cutoff_freq = None
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
                                      render=False,
                                      seed=seed,
                                      results_dir="./results")

        return mean_reward

    #db_name = "hyperparameter_search.db"
    db_name = "test.db"
    storage_name = f"sqlite:///{db_name}"
    study = optuna.create_study(study_name=f"{env_name}_h{horizon}_ns{n_samples}_{alg}",
                                storage=storage_name, load_if_exists=True,
                                direction="maximize")
    study.optimize(objective, n_trials=10)
    print(study.best_params)

if __name__ == "__main__":
    run_experiment(experiment)
