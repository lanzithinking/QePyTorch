#!/usr/bin/env python3

from typing import Union

import torch
from linear_operator.operators import CholLinearOperator, TriangularLinearOperator

from ..distributions import MultivariateNormal, MultivariateQExponential
from ._variational_distribution import _VariationalDistribution


class CholeskyVariationalDistribution(_VariationalDistribution):
    """
    A :obj:`~qpytorch.variational._VariationalDistribution` that is defined to be a multivariate normal (q-exponential) distribution
    with a full covariance matrix.

    The most common way this distribution is defined is to parameterize it in terms of a mean vector and a covariance
    matrix. In order to ensure that the covariance matrix remains positive definite, we only consider the lower
    triangle.

    :param num_inducing_points: Size of the variational distribution. This implies that the variational mean
        should be this size, and the variational covariance matrix should have this many rows and columns.
    :param batch_shape: Specifies an optional batch size
        for the variational parameters. This is useful for example when doing additive variational inference.
    :param mean_init_std: (Default: 1e-3) Standard deviation of gaussian (q-exponential) noise to add to the mean initialization.
    """

    def __init__(
        self,
        num_inducing_points: int,
        batch_shape: torch.Size = torch.Size([]),
        mean_init_std: float = 1e-3,
        **kwargs,
    ):
        super().__init__(num_inducing_points=num_inducing_points, batch_shape=batch_shape, mean_init_std=mean_init_std)
        mean_init = torch.zeros(num_inducing_points)
        covar_init = torch.eye(num_inducing_points, num_inducing_points)
        mean_init = mean_init.repeat(*batch_shape, 1)
        covar_init = covar_init.repeat(*batch_shape, 1, 1)

        self.register_parameter(name="variational_mean", parameter=torch.nn.Parameter(mean_init))
        self.register_parameter(name="chol_variational_covar", parameter=torch.nn.Parameter(covar_init))
        
        if 'power' in kwargs: self.power = kwargs.pop('power')

    def forward(self) -> Union[MultivariateNormal, MultivariateQExponential]:
        chol_variational_covar = self.chol_variational_covar
        dtype = chol_variational_covar.dtype
        device = chol_variational_covar.device

        # First make the cholesky factor is upper triangular
        lower_mask = torch.ones(self.chol_variational_covar.shape[-2:], dtype=dtype, device=device).tril(0)
        chol_variational_covar = TriangularLinearOperator(chol_variational_covar.mul(lower_mask))

        # Now construct the actual matrix
        variational_covar = CholLinearOperator(chol_variational_covar)
        if not hasattr(self, 'power'):
            return MultivariateNormal(self.variational_mean, variational_covar)
        else:
            return MultivariateQExponential(self.variational_mean, variational_covar, power=self.power)

    def initialize_variational_distribution(self, prior_dist: Union[MultivariateNormal, MultivariateQExponential]) -> None:
        self.variational_mean.data.copy_(prior_dist.mean)
        self.variational_mean.data.add_(torch.randn_like(prior_dist.mean), alpha=self.mean_init_std)
        self.chol_variational_covar.data.copy_(prior_dist.lazy_covariance_matrix.cholesky().to_dense())
