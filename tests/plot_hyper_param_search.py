from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import optuna
import os

file_paths = []
#file_paths.append("hyperparam_search_humanoid_standup_eagle.log")
#file_paths.append("hyperparam_search_humanoid_eagle.log")
#file_paths.append("hyperparam_search_walker_eagle.log")
#file_paths.append("hyperparam_search_swimmer_eagle.log")
file_paths.append("hyperparam_search_hopper_eagle.log")

# ignore these
##file_paths.append("hyperparam_search_pendulum_eagle.log")
##file_paths.append("hyperparam_search_go1_eagle.log")


def plot_hyper_param_search(file_path):
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(file_path),
    )

    results = defaultdict(dict) 

    studies = storage.get_all_studies()
    for s in studies:
        study_name = s.study_name
        horizon = int(study_name.split("_")[-3][1:])
        n_samples = int(study_name.split("_")[-2][2:])
        alg = study_name.split("_")[-1]
        study_id = s._study_id
        try:
            best_trial = storage.get_best_trial(study_id)
            all_trials = storage.get_all_trials(study_id)
            n_trials = len(all_trials)
            results[(horizon, n_samples)][alg] = {"n_trials": n_trials, 
                                                "best": best_trial.value,
                                                "all": [x.value for x in all_trials]}
        except ValueError:
            print(f"Study {study_name} failed")
            results[(horizon, n_samples)][alg] = {"n_trials": 0, 
                                                "best": -1.,
                                                "all": [-1.]}

    hs = sorted(list(set([x[0] for x in results.keys()])))
    nss = sorted(list(set([x[1] for x in results.keys()])))

    improvements = np.zeros((len(hs), len(nss)))
    bests_fcem = np.zeros((len(hs), len(nss)))
    bests_icem = np.zeros((len(hs), len(nss)))

    for (h, ns), v in results.items():
        best_fcem = v["fcem"]["best"]
        best_icem = v["icem"]["best"]
        improvements[hs.index(h), nss.index(ns)] = (best_fcem - best_icem) / best_icem
        bests_fcem[hs.index(h), nss.index(ns)] = best_fcem
        bests_icem[hs.index(h), nss.index(ns)] = best_icem
    plt.subplot(311)
    plt.matshow(improvements, fignum=False)
    plt.xticks(range(len(nss)), nss)
    plt.yticks(range(len(hs)), hs)
    plt.colorbar()
    plt.subplot(312)
    plt.matshow(bests_fcem, fignum=False)
    plt.xticks(range(len(nss)), nss)
    plt.yticks(range(len(hs)), hs)
    plt.colorbar()
    plt.subplot(313)
    plt.matshow(bests_icem, fignum=False)
    plt.xticks(range(len(nss)), nss)
    plt.yticks(range(len(hs)), hs)
    plt.colorbar()
    plt.title(os.path.basename(file_path))
    plt.show()

for file_path in file_paths:
    plot_hyper_param_search(file_path)