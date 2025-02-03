import torch
from torch.distributions.multivariate_normal import MultivariateNormal, _batch_mv
from torch.distributions.utils import _standard_normal


class LiftedMultivariateNormal(MultivariateNormal):
    def __init__(self, loc, sampling_freq, max_noise_dot, covariance_matrix=None, precision_matrix=None,
                 scale_tril=None, validate_args=None):
        super(LiftedMultivariateNormal, self).__init__(loc, covariance_matrix, precision_matrix, scale_tril, validate_args)
        self.max_noise_dot = max_noise_dot
        self.sampling_freq = sampling_freq

    def noise_dynamics(self, dnoise):
        dnoise = torch.clip(dnoise * self.max_noise_dot, -self.max_noise_dot, self.max_noise_dot)
        noise = torch.cumsum(dnoise / self.sampling_freq, dim=-2)
        #import matplotlib.pyplot as plt
        #for i in range(3):
        #    for k in range(3):
        #        plt.subplot(3, 3, i * 3 + k + 1)
        #        plt.plot(dnoise[i, :, k].cpu().numpy(), label="dnoise")
        #        plt.plot(noise[i, :, k].cpu().numpy(), label="noise")
        #        plt.legend()
        #plt.show()
        return noise

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        deps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        deps_scaled = _batch_mv(self._unbroadcasted_scale_tril, deps)
        eps = self.noise_dynamics(deps_scaled)
        return self.loc + eps