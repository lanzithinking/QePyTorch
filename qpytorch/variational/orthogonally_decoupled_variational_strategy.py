#!/usr/bin/env python3

from typing import Optional, Union

import torch
from linear_operator.operators import LinearOperator
from torch import Tensor

from ..distributions import MultivariateNormal, MultivariateQExponential
from gpytorch.utils.memoize import add_to_cache, cached
from ._variational_distribution import _VariationalDistribution
from ._variational_strategy import _VariationalStrategy
from .delta_variational_distribution import DeltaVariationalDistribution


class OrthogonallyDecoupledVariationalStrategy(_VariationalStrategy):
    r"""
    Implements orthogonally decoupled VGPs as defined in `Salimbeni et al. (2018)`_.
    This variational strategy uses a different set of inducing points for the mean and covariance functions.
    The idea is to use more inducing points for the (computationally efficient) mean and fewer inducing points for the
    (computationally expensive) covaraince.

    This variational strategy defines the inducing points/:obj:`~qpytorch.variational._VariationalDistribution`
    for the mean function.
    It then wraps a different :obj:`~qpytorch.variational._VariationalStrategy` which
    defines the covariance inducing points.

    :param covar_variational_strategy:
        The variational strategy for the covariance term.
    :param inducing_points: Tensor containing a set of inducing
        points to use for variational inference.
    :param variational_distribution: A
        VariationalDistribution object that represents the form of the variational distribution :math:`q(\mathbf u)`
    :param jitter_val: Amount of diagonal jitter to add for Cholesky factorization numerical stability

    Example:
        >>> mean_inducing_points = torch.randn(1000, train_x.size(-1), dtype=train_x.dtype, device=train_x.device)
        >>> covar_inducing_points = torch.randn(100, train_x.size(-1), dtype=train_x.dtype, device=train_x.device)
        >>>
        >>> covar_variational_strategy = qpytorch.variational.VariationalStrategy(
        >>>     model, covar_inducing_points,
        >>>     qpytorch.variational.CholeskyVariationalDistribution(covar_inducing_points.size(-2)),
        >>>     learn_inducing_locations=True
        >>> )
        >>>
        >>> variational_strategy = qpytorch.variational.OrthogonallyDecoupledVariationalStrategy(
        >>>     covar_variational_strategy, mean_inducing_points,
        >>>     qpytorch.variational.DeltaVariationalDistribution(mean_inducing_points.size(-2)),
        >>> )

    .. _Salimbeni et al. (2018):
        https://arxiv.org/abs/1809.08820
    """

    def __init__(
        self,
        covar_variational_strategy: _VariationalStrategy,
        inducing_points: Tensor,
        variational_distribution: _VariationalDistribution,
        jitter_val: Optional[float] = None,
    ):
        if not isinstance(variational_distribution, DeltaVariationalDistribution):
            raise NotImplementedError(
                "OrthogonallyDecoupledVariationalStrategy currently works with DeltaVariationalDistribution"
            )

        super().__init__(
            covar_variational_strategy,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
            jitter_val=jitter_val,
        )
        self.base_variational_strategy = covar_variational_strategy

    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self) -> Union[MultivariateNormal, MultivariateQExponential]:
        out = self.model(self.inducing_points)
        if isinstance(out, MultivariateNormal):
            res = MultivariateNormal(out.mean, out.lazy_covariance_matrix.add_jitter(self.jitter_val))
        elif isinstance(out, MultivariateQExponential):
            res = MultivariateQExponential(out.mean, out.lazy_covariance_matrix.add_jitter(self.jitter_val), power=out.power)
        return res

    def forward(
        self,
        x: Tensor,
        inducing_points: Tensor,
        inducing_values: Tensor,
        variational_inducing_covar: Optional[LinearOperator] = None,
        **kwargs,
    ) -> Union[MultivariateNormal, MultivariateQExponential]:
        if variational_inducing_covar is not None:
            raise NotImplementedError(
                "OrthogonallyDecoupledVariationalStrategy currently works with DeltaVariationalDistribution"
            )

        num_data = x.size(-2)
        full_output = self.model(torch.cat([x, inducing_points], dim=-2), **kwargs)
        full_mean = full_output.mean
        full_covar = full_output.lazy_covariance_matrix

        if self.training:
            induc_mean = full_mean[..., num_data:]
            induc_induc_covar = full_covar[..., num_data:, num_data:]
            if isinstance(full_output, MultivariateNormal):
                prior_dist = MultivariateNormal(induc_mean, induc_induc_covar)
            if isinstance(full_output, MultivariateQExponential):
                prior_dist = MultivariateQExponential(induc_mean, induc_induc_covar, power=full_output.power)
            add_to_cache(self, "prior_distribution_memo", prior_dist)

        test_mean = full_mean[..., :num_data]
        data_induc_covar = full_covar[..., :num_data, num_data:]
        predictive_mean = (data_induc_covar @ inducing_values.unsqueeze(-1)).squeeze(-1).add(test_mean)
        predictive_covar = full_covar[..., :num_data, :num_data]

        # Return the distribution
        if isinstance(full_output, MultivariateNormal):
            return MultivariateNormal(predictive_mean, predictive_covar)
        elif isinstance(full_output, MultivariateQExponential):
            return MultivariateQExponential(predictive_mean, predictive_covar, power=full_output.power)

    def kl_divergence(self) -> Tensor:
        mean = self.variational_distribution.mean
        induc_induc_covar = self.prior_distribution.lazy_covariance_matrix
        kl = self.model.kl_divergence() + ((induc_induc_covar @ mean.unsqueeze(-1)).squeeze(-1) * mean).sum(-1).mul(0.5)
        return kl
