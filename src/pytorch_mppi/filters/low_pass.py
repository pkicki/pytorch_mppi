from torch.distributions.multivariate_normal import MultivariateNormal, _batch_mv
from torch.distributions.utils import _standard_normal
from scipy.signal import butter, periodogram, lfiltic
from scipy.signal import lfilter as scipy_lfilter
import torch
#from torch_lfilter import lfilter
from torchaudio.functional import lfilter
import matplotlib.pyplot as plt


class LPFilter:
    def __init__(self, cutoff_freq, sampling_freq, order=1, dtype=torch.float64, device='cpu'):
        self.order = order
        self.cutoff_freq = cutoff_freq
        self.sampling_freq = sampling_freq
        lp_filter_b, lp_filter_a = butter(self.order, self.cutoff_freq, fs=self.sampling_freq,
                                          btype='low', analog=False)
        self.np_lp_filter_a = lp_filter_a
        self.np_lp_filter_b = lp_filter_b
        self.lp_filter_a = torch.tensor(lp_filter_a, dtype=dtype, device=device)
        self.lp_filter_b = torch.tensor(lp_filter_b, dtype=dtype, device=device)
    

    def apply(self, data, history):
        #import numpy as np
        #zi = lfiltic(self.np_lp_filter_b, self.np_lp_filter_a, np.ones((2,)))
        #y = scipy_lfilter(self.np_lp_filter_b, self.np_lp_filter_a, np.ones((5,)).astype(np.float64), axis=-1, zi=np.ones((1,), dtype=np.float64))


        #history = 2. * torch.ones_like(history)
        data = data.transpose(-1, -2)
        previous_outputs = history[..., :self.order, :].transpose(-1, -2)
        #data = torch.ones((3, 5), dtype=torch.float64)
        #data = torch.randn((3, 5), dtype=torch.float64) + 2.
        # TODO make it wotk not only with order = 1
        x1 = data[..., :1]
        w = 1./(self.lp_filter_b[0] * (1 - self.lp_filter_a[1])) * (previous_outputs - self.lp_filter_b[1] * x1)
        data_history = torch.cat([w, x1, data], dim=-1)
        y_padded = lfilter(data_history, self.lp_filter_a, self.lp_filter_b, clamp=False)
        y = y_padded[..., self.order+1:]

        #for i in range(3):
        #    plt.subplot(131+i)
        #    plt.plot(np.arange(1, data[i].shape[0]+1), data[i].cpu().numpy(), label="original")
        #    plt.plot(y[i, self.order:].cpu().numpy(), label="filtered")
        #    plt.legend()
        #plt.show()


        #previous_inputs = history.detach().numpy()[:self.order]
        #data = data.transpose(-1, -2)
        ##max = data.max(-1, keepdims=True)[0]
        ##min = data.min(-1, keepdims=True)[0]
        ##max_min_diff = max - min
        ##max_min_sum = max + min
        ##data_normalized = (2 * data - max_min_sum) / max_min_diff
        ##previous_inputs_normalized = (2 * previous_inputs - max_min_sum) / max_min_diff
        #zi = lfiltic(self.np_lp_filter_b[:, None], self.np_lp_filter_a[:, None], previous_inputs)
        #y = scipy_lfilter(data, self.np_lp_filter_a, self.np_lp_filter_b, axis=-1, zi=zi)
        ##y = lfilter(data_normalized, self.lp_filter_a, self.lp_filter_b, clamp=False)
        ##y = lfilter(data, self.lp_filter_a, self.lp_filter_b, clamp=False)
        ##y_unnormalized = (y * max_min_diff + max_min_sum) / 2.
        ##y_ = y_unnormalized.transpose(-1, -2)
        y_ = y.transpose(-1, -2)
        return y_
        
        #if len(data.shape) > 2:
        #    fin, magin = periodogram(data, fs=self.sampling_freq)
        #    fout, magout = periodogram(y, fs=self.sampling_freq)
        #    import numpy as np
        #    plt.plot(fin, np.mean(magin, axis=0)[0], label='in')
        #    plt.plot(fout, np.mean(magout, axis=0)[0], label='out')
        #    plt.xscale('log')
        #    plt.yscale('log')
        #    plt.legend()
        #    plt.show()

        ##y_ = scipy_lfilter(self.lp_filter_b.cpu().numpy(), self.lp_filter_a.cpu().numpy(), data.cpu().numpy(), axis=-1)
        ##plt.plot(data[0].cpu().numpy())
        ##plt.plot(y[:, 0].cpu().numpy())
        ##plt.plot(y_[0])
        ##plt.show()

        #for i in range(3):
        #    plt.subplot(131+i)
        #    plt.plot(data[i].cpu().numpy(), label="original")
        #    plt.plot(data_normalized[i].cpu().numpy(), label="original")
        #    plt.plot(y[i].cpu().numpy(), label="filtered")
        #    plt.plot(y_unnormalized[i].cpu().numpy(), label="filtered_unnormalized")
        #    plt.legend()
        #plt.show()