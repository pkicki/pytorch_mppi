import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
import colorednoise
from scipy.signal import lfilter as scipy_lfilter
from scipy.signal import butter#, lfilter

fs = 1. / 0.05
#trajectories = np.load("mppi_trajectories.npy")
trajectories = np.load("mppi_trajectories_lp05.npy")
actions = trajectories[..., -1]
#random_actions = 5.0 * np.random.randn(*actions.shape)
random_actions = 8.0 * np.random.randn(*actions.shape)
cn1 = 1.35 * colorednoise.powerlaw_psd_gaussian(1., size=actions.shape)
cn2 = 1.35 * colorednoise.powerlaw_psd_gaussian(2., size=actions.shape)
lp_filter_b, lp_filter_a = butter(2, 0.4, fs=fs, btype='low', analog=False)
filtered_actions = scipy_lfilter(lp_filter_b, lp_filter_a, random_actions, axis=-1)

f, psd = welch(actions, fs=fs)
f_random, psd_random = welch(random_actions, fs=fs)
f_lp, psd_lp = welch(filtered_actions, fs=fs)
f_cn1, psd_cn1 = welch(cn1, fs=fs)
f_cn2, psd_cn2 = welch(cn2, fs=fs)

plt.plot(f, np.mean(psd, axis=0), label="real_data")
plt.plot(f_random, np.mean(psd_random, axis=0), label="random")
plt.plot(f_lp, np.mean(psd_lp, axis=0), label="lp")
plt.plot(f_cn1, np.mean(psd_cn1, axis=0), label="cn1")
plt.plot(f_cn2, np.mean(psd_cn2, axis=0), label="cn2")
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()