#!/usr/bin/env python3

import math
from typing import Optional, Tuple, Union

import torch
from linear_operator import to_dense
from linear_operator.operators import (
    CholLinearOperator,
    DiagLinearOperator,
    LinearOperator,
    PsdSumLinearOperator,
    RootLinearOperator,
    TriangularLinearOperator,
    ZeroLinearOperator,
)
from linear_operator.utils.cholesky import psd_safe_cholesky
from linear_operator.utils.errors import NotPSDError
from torch import Tensor

from .. import settings
from ..distributions import MultivariateNormal, MultivariateQExponential
from gpytorch.utils.memoize import add_to_cache, cached
from ._variational_strategy import _VariationalStrategy
from .cholesky_variational_distribution import CholeskyVariationalDistribution


class UnwhitenedVariationalStrategy(_VariationalStrategy):
    r"""
    Similar to :obj:`~qpytorch.variational.VariationalStrategy`, but does not perform the
    whitening operation. In almost all cases :obj:`~qpytorch.variational.VariationalStrategy`
    is preferable, with a few exceptions:

    - When the inducing points are exactly equal to the training points (i.e. :math:`\mathbf Z = \mathbf X`).
      Unwhitened models are faster in this case.

    - When the number of inducing points is very large (e.g. >2000). Unwhitened models can use CG for faster
      computation.

    :param ~model: Model this strategy is applied to.
        Typically passed in when the VariationalStrategy is created in the
        __init__ method of the user defined model.
        It should contain power if Q-Exponential distribution is involved in.
    :param inducing_points: Tensor containing a set of inducing
        points to use for variational inference.
    :param variational_distribution: A
        VariationalDistribution object that represents the form of the variational distribution :math:`q(\mathbf u)`
    :param learn_inducing_locations: (default True): Whether or not
        the inducing point locations :math:`\mathbf Z` should be learned (i.e. are they
        parameters of the model).
    :param jitter_val: Amount of diagonal jitter to add for Cholesky factorization numerical stability
    """
    has_fantasy_strategy = True

    @cached(name="cholesky_factor", ignore_args=True)
    def _cholesky_factor(self, induc_induc_covar: LinearOperator) -> TriangularLinearOperator:
        # Maybe used - if we're not using CG
        L = psd_safe_cholesky(to_dense(induc_induc_covar))
        return TriangularLinearOperator(L)

    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self) -> Union[MultivariateNormal, MultivariateQExponential]:
        out = self.model.forward(self.inducing_points)
        if hasattr(self.model, 'power'):
            res = MultivariateQExponential(out.mean, out.lazy_covariance_matrix.add_jitter(), power=self.model.power)
        else:
            res = MultivariateNormal(out.mean, out.lazy_covariance_matrix.add_jitter())
        return res

    @property
    @cached(name="pseudo_points_memo")
    def pseudo_points(self) -> Tuple[Tensor, Tensor]:
        # TODO: implement for other distributions
        # retrieve the variational mean, m and covariance matrix, S.
        if not isinstance(self._variational_distribution, CholeskyVariationalDistribution):
            raise NotImplementedError(
                "Only CholeskyVariationalDistribution has pseudo-point support currently, ",
                "but your _variational_distribution is a ",
                self._variational_distribution.__name__,
            )

        # retrieve the variational mean, m and covariance matrix, S.
        var_cov_root = TriangularLinearOperator(self._variational_distribution.chol_variational_covar)
        var_cov = CholLinearOperator(var_cov_root)
        var_mean = self.variational_distribution.mean  # .unsqueeze(-1)
        if var_mean.shape[-1] != 1:
            var_mean = var_mean.unsqueeze(-1)

        # R = K - S
        Kmm = self.model.covar_module(self.inducing_points)
        res = Kmm - var_cov

        cov_diff = res

        # D_a = (S^{-1} - K^{-1})^{-1} = S + S R^{-1} S
        # note that in the whitened case R = I - S, unwhitened R = K - S
        # we compute (R R^{T})^{-1} R^T S for stability reasons as R is probably not PSD.
        eval_lhs = var_cov.to_dense()
        eval_rhs = cov_diff.transpose(-1, -2).matmul(eval_lhs)
        inner_term = cov_diff.matmul(cov_diff.transpose(-1, -2))
        # TODO: flag the jitter here
        inner_solve = inner_term.add_jitter(self.jitter_val).solve(eval_rhs, eval_lhs.transpose(-1, -2))
        inducing_covar = var_cov + inner_solve

        # mean term: D_a S^{-1} m
        # unwhitened: (S - S R^{-1} S) S^{-1} m = (I - S R^{-1}) m
        rhs = cov_diff.transpose(-1, -2).matmul(var_mean)
        inner_rhs_mean_solve = inner_term.add_jitter(self.jitter_val).solve(rhs)
        pseudo_target_mean = var_mean + var_cov.matmul(inner_rhs_mean_solve)

        # ensure inducing covar is psd
        try:
            pseudo_target_covar = CholLinearOperator(inducing_covar.add_jitter(self.jitter_val).cholesky()).to_dense()
        except NotPSDError:
            from linear_operator.operators import DiagLinearOperator

            evals, evecs = torch.linalg.eigh(inducing_covar)
            pseudo_target_covar = (
                evecs.matmul(DiagLinearOperator(evals + self.jitter_val)).matmul(evecs.transpose(-1, -2)).to_dense()
            )

        return pseudo_target_covar, pseudo_target_mean

    def forward(
        self,
        x: Tensor,
        inducing_points: Tensor,
        inducing_values: Tensor,
        variational_inducing_covar: Optional[LinearOperator] = None,
        **kwargs,
    ) -> Union[MultivariateNormal, MultivariateQExponential]:
        # If our points equal the inducing points, we're done
        if torch.equal(x, inducing_points):
            if variational_inducing_covar is None:
                raise RuntimeError
            else:
                if hasattr(self.model, 'power'):
                    return MultivariateQExponential(inducing_values, variational_inducing_covar, power=self.model.power)
                else:
                    return MultivariateNormal(inducing_values, variational_inducing_covar)

        # Otherwise, we have to marginalize
        num_induc = inducing_points.size(-2)
        full_inputs = torch.cat([inducing_points, x], dim=-2)
        full_output = self.model.forward(full_inputs)
        full_mean, full_covar = full_output.mean, full_output.lazy_covariance_matrix

        # Mean terms
        test_mean = full_mean[..., num_induc:]
        induc_mean = full_mean[..., :num_induc]
        mean_diff = (inducing_values - induc_mean).unsqueeze(-1)

        # Covariance terms
        induc_induc_covar = full_covar[..., :num_induc, :num_induc].add_jitter(self.jitter_val)
        induc_data_covar = full_covar[..., :num_induc, num_induc:].to_dense()
        data_data_covar = full_covar[..., num_induc:, num_induc:]

        # Compute Cholesky factorization of inducing covariance matrix
        if settings.fast_computations.log_prob.off() or (num_induc <= settings.max_cholesky_size.value()):
            induc_induc_covar = CholLinearOperator(self._cholesky_factor(induc_induc_covar))

        # If we are making predictions and don't need variances, we can do things very quickly.
        if not self.training and settings.skip_posterior_variances.on():
            self._mean_cache = induc_induc_covar.solve(mean_diff).detach()
            predictive_mean = torch.add(
                test_mean, induc_data_covar.transpose(-2, -1).matmul(self._mean_cache).squeeze(-1)
            )
            predictive_covar = ZeroLinearOperator(test_mean.size(-1), test_mean.size(-1))
            if hasattr(self.model, 'power'):
                return MultivariateQExponential(predictive_mean, predictive_covar, power=self.model.power)
            else:
                return MultivariateNormal(predictive_mean, predictive_covar)

        # Expand everything to the right size
        shapes = [mean_diff.shape[:-1], induc_data_covar.shape[:-1], induc_induc_covar.shape[:-1]]
        root_variational_covar = None
        if variational_inducing_covar is not None:
            root_variational_covar = variational_inducing_covar.root_decomposition().root.to_dense()
            shapes.append(root_variational_covar.shape[:-1])
        shape = torch.broadcast_shapes(*shapes)
        mean_diff = mean_diff.expand(*shape, mean_diff.size(-1))
        induc_data_covar = induc_data_covar.expand(*shape, induc_data_covar.size(-1))
        induc_induc_covar = induc_induc_covar.expand(*shape, induc_induc_covar.size(-1))
        if variational_inducing_covar is not None:
            root_variational_covar = root_variational_covar.expand(*shape, root_variational_covar.size(-1))

        # Cache the kernel matrix with the cached CG calls
        if self.training:
            if hasattr(self.model, 'power'):
                prior_dist = MultivariateQExponential(induc_mean, induc_induc_covar, power=self.model.power)
            else:
                prior_dist = MultivariateNormal(induc_mean, induc_induc_covar)
            add_to_cache(self, "prior_distribution_memo", prior_dist)

        # Compute predictive mean
        if variational_inducing_covar is None:
            left_tensors = mean_diff
        else:
            left_tensors = torch.cat([mean_diff, root_variational_covar], -1)
        inv_products = induc_induc_covar.solve(induc_data_covar, left_tensors.transpose(-1, -2))
        predictive_mean = torch.add(test_mean, inv_products[..., 0, :])

        # Compute covariance
        if self.training:
            interp_data_data_var, _ = induc_induc_covar.inv_quad_logdet(
                induc_data_covar, logdet=False, reduce_inv_quad=False
            )
            data_covariance = DiagLinearOperator(
                (data_data_covar.diagonal(dim1=-1, dim2=-2) - interp_data_data_var).clamp(0, math.inf)
            )
        else:
            neg_induc_data_data_covar = torch.matmul(
                induc_data_covar.transpose(-1, -2).mul(-1), induc_induc_covar.solve(induc_data_covar)
            )
            data_covariance = data_data_covar + neg_induc_data_data_covar
        predictive_covar = PsdSumLinearOperator(
            RootLinearOperator(inv_products[..., 1:, :].transpose(-1, -2)), data_covariance
        )

        # Done!
        if hasattr(self.model, 'power'):
            return MultivariateQExponential(predictive_mean, predictive_covar, power=self.model.power)
        else:
            return MultivariateNormal(predictive_mean, predictive_covar)
