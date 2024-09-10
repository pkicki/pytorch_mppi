import asyncio
import pickle 
import os, sys
import subprocess
import numpy as np
from itertools import product
from tqdm.asyncio import tqdm_asyncio
from multiprocessing import Pool
from tqdm import tqdm


def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped


ENVBIN = sys.exec_prefix
PYTHON = os.path.join(ENVBIN, "bin", "python")

#N = 100
N = 1
#skip_existing = False
skip_existing = True
#env_name = "pendulum"
#env_name = "half_cheetah"
env_name = "walker"

def run_exp(name, sweep_names, horizon, n_samples, values):
    filename = f"exp_results/{env_name}/{name}/{'_'.join(str(v) for v in values)}.res"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if skip_existing and os.path.exists(filename):
        return True

    #cmd = f"taskset -c {idx%5} {PYTHON} main.py "
    cmd = f"{PYTHON} experiment.py "
    cmd += f'--n_episodes={N} --horizon={horizon} --n_samples={n_samples} '

    for i, v in enumerate(values):
        cmd += f'--{sweep_names[i]}={v} '

    result = subprocess.check_output(cmd, shell=True, text=True)
    rewards = [float(x.split(" ")[-1]) for x in result.split("\n")[:-1]]
    fh = open(filename, "a+")
    fh.write("\n".join(str(x) for x in rewards) + "\n")

def sweep(alg, n_samples, horizon):
    #init_stds = np.linspace(5., 15., 11).tolist()
    #init_stds = np.linspace(6., 14., 5).tolist()
    init_stds = np.linspace(1., 7., 4).tolist()
    if alg == "icem":
        name = f"icem_s{n_samples}_h{horizon}"
        noise_betas = np.linspace(0.5, 5.0, 10).tolist()
        sweep_values = [noise_betas, init_stds]
        sweep_names = ['noise_beta', 'noise_sigma']
    elif alg == "fcem":
        name = f"fcem_s{n_samples}_h{horizon}"
        cutoff_freqs = np.linspace(0.5, 5.0, 10).tolist()#[:2]
        sweep_values = [cutoff_freqs, init_stds]
        sweep_names = ['noise_cutoff_freq', 'noise_sigma']
    else:
        raise ValueError("Unknown algorithm")


    #loop = asyncio.get_event_loop()

    #looper = asyncio.gather(*[run_exp(i, values) for i, values in enumerate(product(*sweep_values))])
    #looper = tqdm_asyncio.gather(*[run_exp(i, values) for i, values in enumerate(product(*sweep_values))])
                                
    #a = loop.run_until_complete(looper)
    pool = Pool()
    arguments = list(product(*sweep_values))
    arguments = [(name, sweep_names, horizon, n_samples, x) for x in arguments]
    #pool.map(run_exp, arguments)
    pool.starmap(run_exp, arguments)
    #pool.join()

algs = ["icem", "fcem"]
#n_samples = [3, 5, 10, 20, 30]
n_samples = [3, 5, 10, 30, 100]
horizon = [5, 10, 15, 30]
#algs = ["fcem"]
#n_samples = [3]
#horizon = [5]
args = list(product(algs, n_samples, horizon))
for arg in args:
    sweep(*arg)