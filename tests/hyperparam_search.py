import optuna
from experiment import experiment
#import sqlite3

env_name = "half_cheetah"
horizon = 30
n_samples = 100

def objective(trial):
    #x = trial.suggest_float('x', -10, 10)
    noise_sigma = trial.suggest_float('noise_sigma', 0.1, 10.)
    noise_cutoff_freq = trial.suggest_float('noise_cutoff_freq', 0.1, 10.)
    mean_reward = experiment(env=env_name,
                             n_episodes=100,
                             horizon=horizon,
                             n_samples=n_samples,
                             noise_sigma=noise_sigma,
                             noise_beta=None,
                             noise_cutoff_freq=noise_cutoff_freq,
                             render=False,
                             seed=444,
                             results_dir="./results")

    return -mean_reward

db_name = "hyperparameter_search.db"
#con = sqlite3.connect(db_name)
storage_name = f"sqlite:///{db_name}"
study = optuna.create_study(study_name=f"{env_name}_h{horizon}_ns{n_samples}_fcem", storage=storage_name, load_if_exists=True)
#study = optuna.load_study(study_name="distributed-example", storage=storage_name)
study.optimize(objective, n_trials=10)
print(study.best_params)  # E.g. {'x': 2.002108042}