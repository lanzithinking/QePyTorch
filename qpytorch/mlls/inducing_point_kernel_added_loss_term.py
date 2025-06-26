#!/usr/bin/env python3

from typing import Union
import torch

from ..distributions import MultivariateNormal, MultivariateQExponential
from ..likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood, QExponentialLikelihood, MultitaskQExponentialLikelihood
from gpytorch.mlls.added_loss_term import AddedLossTerm


class InducingPointKernelAddedLossTerm(AddedLossTerm):
    r"""
    An added loss term that computes the additional "regularization trace term" of the SGPR (SQEPR) objective function.

    .. math::
        Gaussian: -\frac{1}{2 \sigma^2} \text{Tr} \left( \mathbf K_{\mathbf X \mathbf X} - \mathbf Q \right)
    .. math::
        Q-Exponential: \frac{d}{2}\left(-\log\sigma^2 +\left(\frac{q}{2}-1\right)\log r\right) -\frac{1}{2}r^{\frac{q}{2}}, 
        r = \frac{1}{\sigma^2}\text{Tr} \left( \mathbf K_{\mathbf X \mathbf X} - \mathbf Q \right)


    where :math:`\mathbf Q = \mathbf K_{\mathbf X \mathbf Z} \mathbf K_{\mathbf Z \mathbf Z}^{-1}
    \mathbf K_{\mathbf Z \mathbf X}` is the Nystrom approximation of :math:`\mathbf K_{\mathbf X \mathbf X}`
    given by inducing points :math:`\mathbf Z`, :math:`\sigma^2` is the observational noise
    of the Gaussian (Q-Exponential) likelihood, and :math:`d` is the dimensions being summed over, 
    i.e. :math:`N` for likelihood or :math:`ND` for multi-task likelihood.

    See `Titsias, 2009`_, Eq. 9 for more more information.

    :param prior_dist: A multivariate normal :math:`\mathcal N ( \mathbf 0, \mathbf K_{\mathbf X \mathbf X} )`
        or q-exponential :math:`\mathcal Q ( \mathbf 0, \mathbf K_{\mathbf X \mathbf X} )`
        with covariance matrix :math:`\mathbf K_{\mathbf X \mathbf X}`.
    :param variational_dist: A multivariate normal :math:`\mathcal N ( \mathbf 0, \mathbf Q)`
        or or q-exponential :math:`\mathcal Q ( \mathbf 0, \mathbf Q)`
        with covariance matrix :math:`\mathbf Q = \mathbf K_{\mathbf X \mathbf Z}
        \mathbf K_{\mathbf Z \mathbf Z}^{-1} \mathbf K_{\mathbf Z \mathbf X}`.
    :param likelihood: The Gaussian (QExponential) likelihood with observational noise :math:`\sigma^2`.

    .. _Titsias, 2009:
        https://proceedings.mlr.press/v9/titsias10a/titsias10a.pdf
    """

    def __init__(
        self, prior_dist: Union[MultivariateNormal, MultivariateQExponential], 
        variational_dist: Union[MultivariateNormal, MultivariateQExponential], 
        likelihood: Union[GaussianLikelihood, QExponentialLikelihood],
    ):
        self.prior_dist = prior_dist
        self.variational_dist = variational_dist
        self.likelihood = likelihood

    def loss(self, *params) -> torch.Tensor:
        prior_covar = self.prior_dist.lazy_covariance_matrix
        variational_covar = self.variational_dist.lazy_covariance_matrix
        diag = prior_covar.diagonal(dim1=-1, dim2=-2) - variational_covar.diagonal(dim1=-1, dim2=-2)
        shape = prior_covar.shape[:-1]
        if isinstance(self.likelihood, (MultitaskGaussianLikelihood, MultitaskQExponentialLikelihood)):
            shape = torch.Size([*shape, 1])
            diag = diag.unsqueeze(-1)
        noise_diag = self.likelihood._shaped_noise_covar(shape, *params).diagonal(dim1=-1, dim2=-2)
        if isinstance(self.likelihood, (MultitaskGaussianLikelihood, MultitaskQExponentialLikelihood)):
            noise_diag = noise_diag.reshape(*shape[:-1], -1)
            r = (diag / noise_diag).sum(dim=[-1, -2])
        else:
            r = (diag / noise_diag).sum(dim=-1)
        res = -0.5 * r**(self.likelihood.power/2. if hasattr(self.likelihood,'power') else 1)
        if 'QExponential' in self.likelihood.__class__.__name__:
            if self.likelihood.power!=2: res += -0.5 * noise_diag.log().sum() + torch.tensor(noise_diag.shape[-2:]).prod()/2. * (self.likelihood.power/2.-1) * r.log()
        return res
