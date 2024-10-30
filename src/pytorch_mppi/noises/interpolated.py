from torch.distributions.multivariate_normal import MultivariateNormal, _batch_mv
from torch.distributions.utils import _standard_normal
import torch
import matplotlib.pyplot as plt
from torchcubicspline import(natural_cubic_spline_coeffs, 
                            NaturalCubicSpline)


class InterpolatedMultivariateNormal(MultivariateNormal):
    def __init__(self, loc, horizon, covariance_matrix=None, precision_matrix=None,
                 scale_tril=None, validate_args=None):
        super(InterpolatedMultivariateNormal, self).__init__(loc, covariance_matrix, precision_matrix, scale_tril, validate_args)
        self.horizon = horizon

    def rsample(self, sample_shape=torch.Size()):
        samples = super(InterpolatedMultivariateNormal, self).rsample(sample_shape)
        t = torch.linspace(0, 1, samples.shape[-2])
        coeffs = natural_cubic_spline_coeffs(t, samples)
        spline = NaturalCubicSpline(coeffs)
        interpolation_points = torch.linspace(0, 1, self.horizon)
        interpolated_samples = spline.evaluate(interpolation_points)
        #if len(interpolated_samples.shape) == 2:
        #    for i in range(3):
        #        plt.plot(interpolation_points, interpolated_samples[..., i])
        #        plt.plot(t, samples[..., i], 'o')
        #    plt.show()
        #elif len(interpolated_samples.shape) == 3:
        #    for i in range(3):
        #        plt.plot(interpolation_points, interpolated_samples[0, ..., i])
        #        plt.plot(t, samples[0, ..., i], 'o')
        #    plt.show()
        return interpolated_samples