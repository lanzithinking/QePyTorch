#!/usr/bin/env python3
import math
from typing import Union
import torch
from torch import Tensor

from ..distributions import MultivariateNormal, MultivariateQExponential
from .exact_marginal_log_likelihood import ExactMarginalLogLikelihood


class LeaveOneOutPseudoLikelihood(ExactMarginalLogLikelihood):
    r"""
    The leave one out cross-validation (LOO-CV) likelihood from RW 5.4.2 for an exact Gaussian (Q-Exponential) process with a
    Gaussian (Q-Exponential) likelihood. This offers an alternative to the exact marginal log likelihood where we
    instead maximize the sum of the leave one out log probabilities :math:`\log p(y_i | X, y_{-i}, \theta)`.

    Naively, this will be O(n^4) with Cholesky as we need to compute `n` Cholesky factorizations. Fortunately,
    given the Cholesky factorization of the full kernel matrix (without any points removed), we can compute
    both the mean and variance of each removed point via a bordered system formulation making the total
    complexity O(n^3).

    The LOO-CV approach can be more robust against model mis-specification as it gives an estimate for the
    (log) predictive probability, whether or not the assumptions of the model is fulfilled.

    .. note::
        This module will not work with anything other than a :obj:`~qpytorch.likelihoods.GaussianLikelihood`
        (:obj:`~qpytorch.likelihoods.QExponentialLikelihood`) and a :obj:`~gpytorch.models.ExactGP` (:obj:`~qpytorch.models.ExactQEP`). 
        It also cannot be used in conjunction with stochastic optimization.

    :param ~qpytorch.likelihoods.GaussianLikelihood (~qpytorch.likelihoods.QExponentialLikelihood) likelihood: The Gaussian (Q-Exponential) likelihood for the model
    :param ~gpytorch.models.ExactGP (~qpytorch.models.ExactQEP) model: The exact GP (QEP) model

    Example:
        >>> # model is a qpytorch.models.ExactGP or qpytorch.models.ExactQEP
        >>> # likelihood is a qpytorch.likelihoods.Likelihood
        >>> loocv = qpytorch.mlls.LeaveOneOutPseudoLikelihood(likelihood, model)
        >>>
        >>> output = model(train_x)
        >>> loss = -loocv(output, train_y)
        >>> loss.backward()
    """

    def __init__(self, likelihood, model):
        super().__init__(likelihood=likelihood, model=model)
        self.likelihood = likelihood
        self.model = model

    def forward(self, function_dist: Union[MultivariateNormal, MultivariateQExponential], target: Tensor, *params) -> Tensor:
        r"""
        Computes the leave one out likelihood given :math:`p(\mathbf f)` and :math:`\mathbf y`

        :param ~gpytorch.distributions.MultivariateNormal (~qpytorch.distributions.MultivariateQExponential) 
            output: the outputs of the latent function (the :obj:`~gpytorch.models.GP` or :obj:`~qpytorch.models.QEP`)
        :param torch.Tensor target: :math:`\mathbf y` The target values
        :param dict kwargs: Additional arguments to pass to the likelihood's forward function.
        """
        output = self.likelihood(function_dist, *params)
        m, L = output.mean, output.lazy_covariance_matrix.cholesky(upper=False)
        m = m.reshape(*target.shape)
        identity = torch.eye(*L.shape[-2:], dtype=m.dtype, device=m.device)
        sigma2 = 1.0 / L._cholesky_solve(identity, upper=False).diagonal(dim1=-1, dim2=-2)  # 1 / diag(inv(K))
        mu = target - L._cholesky_solve((target - m).unsqueeze(-1), upper=False).squeeze(-1) * sigma2
        term1 = -0.5 * sigma2.log()
        power = getattr(self.likelihood, 'power', torch.tensor(2.0))
        term2 = -0.5 * (target - mu).abs()**power / sigma2**(power/2.)
        res = (term1 + term2).sum(dim=-1)
        if power!=2: res += (power/2.-1) * ((target - mu).abs().log() + term1).sum(dim=-1)

        res = self._add_other_terms(res, params)

        # Scale by the amount of data we have and then add on the scaled constant
        num_data = target.size(-1)
        return res.div_(num_data) - 0.5 * math.log(2 * math.pi)
