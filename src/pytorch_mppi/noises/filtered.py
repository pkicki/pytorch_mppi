from torch.distributions.multivariate_normal import MultivariateNormal, _batch_mv
from torch.distributions.utils import _standard_normal
from scipy.signal import butter#, lfilter
from scipy.signal import lfilter as scipy_lfilter
import torch
#from torch_lfilter import lfilter
from torchaudio.functional import lfilter
import matplotlib.pyplot as plt


class FilteredMultivariateNormal(MultivariateNormal):
    def __init__(self, loc, sampling_freq, cutoff_freq, order=2, covariance_matrix=None, precision_matrix=None,
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
        data = data.transpose(-1, -2)
        max = data.max(-1, keepdims=True)[0]
        min = data.min(-1, keepdims=True)[0]
        max_min_diff = max - min
        max_min_sum = max + min
        data_normalized = (2 * data - max_min_sum) / max_min_diff
        y = lfilter(data_normalized, self.lp_filter_a, self.lp_filter_b, clamp=False)
        y_unnormalized = (y * max_min_diff + max_min_sum) / 2.
        y = y_unnormalized.transpose(-1, -2)
        #y_ = scipy_lfilter(self.lp_filter_b.cpu().numpy(), self.lp_filter_a.cpu().numpy(), data.cpu().numpy(), axis=-1)
        #plt.plot(data[0].cpu().numpy())
        #plt.plot(y[:, 0].cpu().numpy())
        #plt.plot(y_[0])
        #plt.show()
        return y

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps = self.lp_filter(eps)
        return self.loc + _batch_mv(self._unbroadcasted_scale_tril, eps)