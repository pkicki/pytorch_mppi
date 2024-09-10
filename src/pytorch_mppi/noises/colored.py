import torch
import numpy as np
import colorednoise
from torch.distributions.utils import _standard_normal
from torch.distributions.multivariate_normal import MultivariateNormal, _batch_mv


class ColoredMultivariateNormal(MultivariateNormal):
    def __init__(self, loc, noise_beta, covariance_matrix=None, precision_matrix=None,
                 scale_tril=None, validate_args=None):
        super(ColoredMultivariateNormal, self).__init__(loc, covariance_matrix, precision_matrix, scale_tril, validate_args)
        self.noise_beta = noise_beta

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        #eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps = colorednoise.powerlaw_psd_gaussian(self.noise_beta,
                    size=np.concatenate([shape[:-2], shape[-1:], shape[-2:-1]]).astype(np.int64))
        eps = torch.tensor(eps, dtype=self.loc.dtype, device=self.loc.device).transpose(-1, -2)
        return self.loc + _batch_mv(self._unbroadcasted_scale_tril, eps)