from torch.distributions.multivariate_normal import MultivariateNormal, _batch_mv
from torch.distributions.utils import _standard_normal
from scipy.signal import butter, periodogram
from scipy.signal import lfilter as scipy_lfilter
import torch
#from torch_lfilter import lfilter
from torchaudio.functional import lfilter
import matplotlib.pyplot as plt


class FilteredMultivariateNormal(MultivariateNormal):
    def __init__(self, loc, sampling_freq, cutoff_freq, order=1, covariance_matrix=None, precision_matrix=None,
                 scale_tril=None, validate_args=None):
        super(FilteredMultivariateNormal, self).__init__(loc, covariance_matrix, precision_matrix, scale_tril, validate_args)
        self.order = order
        self.cutoff_freq = cutoff_freq
        self.sampling_freq = sampling_freq
        lp_filter_b, lp_filter_a = butter(self.order, self.cutoff_freq, fs=self.sampling_freq,
                                          btype='low', analog=False)
        self.np_lp_filter_a = lp_filter_a
        self.np_lp_filter_b = lp_filter_b
        self.lp_filter_a = torch.tensor(lp_filter_a, dtype=self.loc.dtype, device=self.loc.device)
        self.lp_filter_b = torch.tensor(lp_filter_b, dtype=self.loc.dtype, device=self.loc.device)

    def lp_filter(self, data):
        #data = data.transpose(-1, -2)
        #data0 = data[..., :1]
        #data_zeroed = data - data0
        #max_deviation = data_zeroed.abs().max(-1, keepdims=True)[0]
        #data_zeroed_normalized = data_zeroed / max_deviation
        #y = lfilter(data_zeroed_normalized, self.lp_filter_a, self.lp_filter_b, clamp=False)
        #y_unnormalized = y * max_deviation + data0

        #for i in range(3):
        #    plt.subplot(131+i)
        #    plt.plot(data[i].cpu().numpy(), label="original")
        #    plt.plot(data_zeroed[i].cpu().numpy(), label="zeroed")
        #    plt.plot(y[i].cpu().numpy(), label="filtered")
        #    plt.plot(y_unnormalized[i].cpu().numpy(), label="filtered_unnormalized")
        #    plt.legend()
        #plt.show()
        #y = y_unnormalized.transpose(-1, -2)
        #assert False

        data = data.transpose(-1, -2)
        #max = data.max(-1, keepdims=True)[0]
        #min = data.min(-1, keepdims=True)[0]
        #max_min_diff = max - min
        #max_min_sum = max + min
        #data_normalized = (2 * data - max_min_sum) / max_min_diff
        #y = lfilter(data_normalized, self.lp_filter_a, self.lp_filter_b, clamp=False)
        #y_unnormalized = (y * max_min_diff + max_min_sum) / 2.
        #y_ = y_unnormalized.transpose(-1, -2)

        y_unnormalized_ref = lfilter(data, self.lp_filter_a, self.lp_filter_b, clamp=False)
        scale = y_unnormalized_ref.std(0).mean()
        y_unnormalized_ref = y_unnormalized_ref / scale
        #y_unnormalized_ref[:, 1] = data[:, 1] # do not filter the current
        y_ = y_unnormalized_ref.transpose(-1, -2)
        #y_unnormalized_ref_np = scipy_lfilter(self.np_lp_filter_b, self.np_lp_filter_a, data.detach().numpy())
        #for i in range(3):
        #    plt.subplot(131+i)
        #    plt.plot(data[i].cpu().numpy(), label="input")
        #    plt.plot(y_unnormalized_ref[i].cpu().numpy(), label="output_pytorch")
        #    plt.plot(y_unnormalized_ref_np[i] + 0.01, label="output_scipy")
        #    plt.legend()
        #plt.show()



        #for i in range(3):
        #    plt.subplot(131+i)
        #    plt.plot(data[i].cpu().numpy(), label="input")
        #    plt.plot(y_unnormalized_ref[i].cpu().numpy(), label="output")
        #    plt.plot(data_normalized[i].cpu().numpy(), label="input normalized")
        #    plt.plot(y_unnormalized[i].cpu().numpy(), label="output normalized")
        #    plt.legend()
        #plt.show()
        #y_ = y.transpose(-1, -2)

        
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
        return y_

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps_filtered = self.lp_filter(eps)
        eps_filtered_scaled = _batch_mv(self._unbroadcasted_scale_tril, eps_filtered)
        return self.loc + eps_filtered_scaled