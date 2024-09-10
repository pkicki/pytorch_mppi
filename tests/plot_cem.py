from itertools import product
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def read_cem(env_name, name):
    r = []
    for f in os.listdir(f"exp_results/{env_name}/{name}"):
        if f.endswith(".res"):
            rewards = np.loadtxt(f"exp_results/{env_name}/{name}/{f}")
            r.append([*[float(x) for x in f[:-4].split("_")], np.mean(rewards)])
    r = np.array(r)
    return r
#
#results_icem = pickle.load(open('icem_n20.pkl', 'rb'))
#results_fcem = pickle.load(open('fcem_n20.pkl', 'rb'))
#
#icem = np.array([[k1, k2, v] for (k1, k2), v in results_icem['means'].items()])
#fcem = np.array([[k1, k2, v] for (k1, k2), v in results_fcem['means'].items()])
#
#plt.subplot(211)
#plt.scatter(icem[:, 0], icem[:, 1], c=icem[:, 2], label="icem")
#plt.subplot(212)
#plt.scatter(fcem[:, 0], fcem[:, 1], c=fcem[:, 2], label="fcem")
#plt.colorbar()
#plt.show()

def plot_results(n_samples, horizon, save=True):
    #env_name = "halfcheetah_running"
    #env_name = "humanoid_standup"
    env_name = "pendulum"

    fcem = read_cem(env_name, f"fcem_s{n_samples}_h{horizon}")
    icem = read_cem(env_name, f"icem_s{n_samples}_h{horizon}")
    fcem_max, fcem_min, fcem_med = np.max(fcem[:, 2]), np.min(fcem[:, 2]), np.median(fcem[:, 2])
    icem_max, icem_min, icem_med = np.max(icem[:, 2]), np.min(icem[:, 2]), np.median(icem[:, 2])
    print("FCEM MAX:", fcem_max, "FCEM MIN:", fcem_min, "FCEM MED:", fcem_med)
    print("ICEM MAX:", icem_max, "ICEM MIN:", icem_min, "ICEM MED:", icem_med)
    vmin = min(fcem_min, icem_min)
    vmax = max(fcem_max, icem_max)
    plt.subplot(121)
    plt.scatter(fcem[:, 0], fcem[:, 1], c=fcem[:, 2], label="fcem", vmin=vmin, vmax=vmax)
    plt.legend()
    plt.subplot(122)
    plt.scatter(icem[:, 0], icem[:, 1], c=icem[:, 2], label="icem", vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.legend()
    if save:
        plt.savefig(f"./imgs/{env_name}_ns{n_samples}_h{horizon}.png")
        plt.clf()
    else:
        plt.show()

#n_samples = 10
#horizon = 15
#n_samples = 5
#horizon = 50
n_samples = [3, 5, 10, 20, 30]
horizon = [5, 10, 15, 30, 50]
#algs = ["fcem"]
#n_samples = [3]
#horizon = [5]
args = list(product(n_samples, horizon))
[plot_results(x, y, save=True) for x, y in args]
#plot_results(n_samples, horizon, save=False)

a = 0