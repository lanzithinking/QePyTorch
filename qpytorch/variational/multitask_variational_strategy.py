#!/usr/bin/env python3

import warnings
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import torch
from linear_operator import to_dense
from linear_operator.operators import (
    CholLinearOperator,
    DiagLinearOperator,
    LinearOperator,
    MatmulLinearOperator,
    RootLinearOperator,
    SumLinearOperator,
    TriangularLinearOperator,
    BlockDiagLinearOperator,
    KroneckerProductLinearOperator
)
from linear_operator.utils.cholesky import psd_safe_cholesky
from linear_operator.utils.errors import NotPSDError
from torch import Tensor

from ._variational_strategy import _VariationalStrategy
from .cholesky_variational_distribution import CholeskyVariationalDistribution

from ..distributions import MultivariateNormal, MultivariateQExponential, MultitaskMultivariateNormal, MultitaskMultivariateQExponential
from ..models import ApproximateGP, ApproximateQEP
from gpytorch.settings import _linalg_dtype_cholesky, trace_mode
from gpytorch.utils.errors import CachingError
from gpytorch.utils.memoize import cached, clear_cache_hook, pop_from_cache_ignore_args
from ..utils.warnings import OldVersionWarning
from . import _VariationalDistribution


def _ensure_updated_strategy_flag_set(
    state_dict: Dict[str, Tensor],
    prefix: str,
    local_metadata: Dict[str, Any],
    strict: bool,
    missing_keys: Iterable[str],
    unexpected_keys: Iterable[str],
    error_msgs: Iterable[str],
):
    device = state_dict[list(state_dict.keys())[0]].device
    if prefix + "updated_strategy" not in state_dict:
        state_dict[prefix + "updated_strategy"] = torch.tensor(False, device=device)
        warnings.warn(
            "You have loaded a variational GP (QEP) model (using `VariationalStrategy`) from a previous version of "
            "QPyTorch. We have updated the parameters of your model to work with the new version of "
            "`VariationalStrategy` that uses whitened parameters.\nYour model will work as expected, but we "
            "recommend that you re-save your model.",
            OldVersionWarning,
        )


class MultitaskVariationalStrategy(_VariationalStrategy):
    r"""
    The modified variational strategy, as defined by `Hensman et al. (2015)`_.
    This strategy takes a set of :math:`m \ll n` inducing points :math:`\mathbf Z`
    and applies an approximate distribution :math:`q( \mathbf u)` over their function values.
    (Here, we use the common notation :math:`\mathbf u = f(\mathbf Z)`.
    The approximate function distribution for any abitrary input :math:`\mathbf X` is given by:

    .. math::

        q( f(\mathbf X) ) = \int p( f(\mathbf X) \mid \mathbf u) q(\mathbf u) \: d\mathbf u

    This variational strategy uses "whitening" to accelerate the optimization of the variational
    parameters. See `Matthews (2017)`_ for more info.

    :param model: Model this strategy is applied to.
        Typically passed in when the VariationalStrategy is created in the
        __init__ method of the user defined model.
        It should contain power if Q-Exponential distribution is involved in.
        It contain forward that outputs a MultitaskMultivariateNormal (MultitaskMultivariateQExponential) distribution.
    :param inducing_points: Tensor containing a set of inducing
        points to use for variational inference.
    :param variational_distribution: A
        VariationalDistribution object that represents the form of the variational distribution :math:`q(\mathbf u)`
    :param learn_inducing_locations: (Default True): Whether or not
        the inducing point locations :math:`\mathbf Z` should be learned (i.e. are they
        parameters of the model).
    :param jitter_val: Amount of diagonal jitter to add for Cholesky factorization numerical stability

    .. _Hensman et al. (2015):
        http://proceedings.mlr.press/v38/hensman15.pdf
    .. _Matthews (2017):
        https://www.repository.cam.ac.uk/handle/1810/278022
    """

    def __init__(
        self,
        model: Union[ApproximateGP, ApproximateQEP],
        inducing_points: Tensor,
        variational_distribution: _VariationalDistribution,
        learn_inducing_locations: bool = True,
        jitter_val: Optional[float] = None,
    ):
        super().__init__(
            model, inducing_points, variational_distribution, learn_inducing_locations, jitter_val=jitter_val
        )
        self.register_buffer("updated_strategy", torch.tensor(True))
        self._register_load_state_dict_pre_hook(_ensure_updated_strategy_flag_set)
        self.has_fantasy_strategy = True

    @cached(name="cholesky_factor", ignore_args=True)
    def _cholesky_factor(self, induc_induc_covar: LinearOperator) -> TriangularLinearOperator:
        L = psd_safe_cholesky(to_dense(induc_induc_covar).type(_linalg_dtype_cholesky.value()))
        return TriangularLinearOperator(L)

    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self) -> Union[MultivariateNormal, MultivariateQExponential]:
        zeros = torch.zeros(
            self._variational_distribution.shape(),
            dtype=self._variational_distribution.dtype,
            device=self._variational_distribution.device,
        )
        ones = torch.ones_like(zeros)
        if hasattr(self.model, 'power'):
            res = MultivariateQExponential(zeros, DiagLinearOperator(ones), power=self.model.power)
        else:
            res = MultivariateNormal(zeros, DiagLinearOperator(ones))
        return res

    @property
    @cached(name="pseudo_points_memo")
    def pseudo_points(self) -> Tuple[Tensor, Tensor]:
        # TODO: have var_mean, var_cov come from a method of _variational_distribution
        # while having Kmm_root be a root decomposition to enable CIQVariationalDistribution support.

        # retrieve the variational mean, m and covariance matrix, S.
        if not isinstance(self._variational_distribution, CholeskyVariationalDistribution):
            raise NotImplementedError(
                "Only CholeskyVariationalDistribution has pseudo-point support currently, ",
                "but your _variational_distribution is a ",
                self._variational_distribution.__name__,
            )

        var_cov_root = TriangularLinearOperator(self._variational_distribution.chol_variational_covar)
        var_cov = CholLinearOperator(var_cov_root)
        var_mean = self.variational_distribution.mean
        if var_mean.shape[-1] != 1:
            var_mean = var_mean.unsqueeze(-1)

        # compute R = I - S
        cov_diff = var_cov.add_jitter(-1.0)
        cov_diff = -1.0 * cov_diff

        # K^{1/2}
        Kmm = self.model.covar_module(self.inducing_points)
        Kmm_root = Kmm.cholesky()

        # D_a = (S^{-1} - K^{-1})^{-1} = S + S R^{-1} S
        # note that in the whitened case R = I - S, unwhitened R = K - S
        # we compute (R R^{T})^{-1} R^T S for stability reasons as R is probably not PSD.
        eval_var_cov = var_cov.to_dense()
        eval_rhs = cov_diff.transpose(-1, -2).matmul(eval_var_cov)
        inner_term = cov_diff.matmul(cov_diff.transpose(-1, -2))
        # TODO: flag the jitter here
        inner_solve = inner_term.add_jitter(self.jitter_val).solve(eval_rhs, eval_var_cov.transpose(-1, -2))
        inducing_covar = var_cov + inner_solve

        inducing_covar = Kmm_root.matmul(inducing_covar).matmul(Kmm_root.transpose(-1, -2))

        # mean term: D_a S^{-1} m
        # unwhitened: (S - S R^{-1} S) S^{-1} m = (I - S R^{-1}) m
        rhs = cov_diff.transpose(-1, -2).matmul(var_mean)
        # TODO: this jitter too
        inner_rhs_mean_solve = inner_term.add_jitter(self.jitter_val).solve(rhs)
        pseudo_target_mean = Kmm_root.matmul(inner_rhs_mean_solve)

        # ensure inducing covar is psd
        # TODO: make this be an explicit root decomposition
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
    ) -> Union[MultitaskMultivariateNormal, MultitaskMultivariateQExponential]:
        # Compute full prior distribution
        full_inputs = torch.cat([inducing_points, x], dim=-2)
        full_output = self.model.forward(full_inputs, **kwargs) # MultitaskMultivariateNormal or MultitaskMultivariateQExponential
        if not type(full_output) in (MultitaskMultivariateNormal, MultitaskMultivariateQExponential):
            raise TypeError(
                "The type of model forward p(f(X)) is ",
                full_output.__class__.__name__,
                ", not multitask. Please use regular VariationalStrategy instead.")
        full_covar = full_output.lazy_covariance_matrix

        num_tasks = full_output.num_tasks#.event_shape[-1]
        _interleaved = full_output._interleaved
        # Covariance terms
        num_induc = kwargs.pop('num_induc', inducing_values.shape[1:].numel())#inducing_points.size(-2))
        test_mean = full_output.mean[..., num_induc:, :]
        if _interleaved:
            induc_induc_covar = full_covar[..., :(num_induc*num_tasks), :(num_induc*num_tasks)].add_jitter(self.jitter_val) # interleaved
            induc_data_covar = full_covar[..., :(num_induc*num_tasks), (num_induc*num_tasks):].to_dense()
            data_data_covar = full_covar[..., (num_induc*num_tasks):, (num_induc*num_tasks):]
        else:
            induc_idx = (torch.arange(num_induc, device=full_covar.device)+torch.arange(num_tasks, device=full_covar.device)[:,None]*full_output.event_shape[0]).flatten()
            data_idx = (torch.arange(num_induc, full_output.event_shape[0], device=full_covar.device)+torch.arange(num_tasks, device=full_covar.device)[:,None]*full_output.event_shape[0]).flatten()
            induc_induc_covar = full_covar[..., induc_idx, :][..., induc_idx].add_jitter(self.jitter_val) # not interleaved
            induc_data_covar = full_covar[..., induc_idx, :][..., data_idx].to_dense()
            data_data_covar = full_covar[..., data_idx, :][..., data_idx]

        # Compute interpolation terms
        # K_ZZ^{-1/2} K_ZX
        # K_ZZ^{-1/2} \mu_Z
        L = self._cholesky_factor(induc_induc_covar)
        if L.shape != induc_induc_covar.shape:
            # Aggressive caching can cause nasty shape incompatibilies when evaluating with different batch shapes
            # TODO: Use a hook fo this
            try:
                pop_from_cache_ignore_args(self, "cholesky_factor")
            except CachingError:
                pass
            L = self._cholesky_factor(induc_induc_covar)
        interp_term = L.solve(induc_data_covar.type(_linalg_dtype_cholesky.value())).to(full_inputs.dtype)

        # Compute the mean of q(f)
        # k_XZ K_ZZ^{-1/2} (m - K_ZZ^{-1/2} \mu_Z) + \mu_X
        if variational_inducing_covar.batch_dim > 0:
            for _ in range(variational_inducing_covar.batch_dim):
                if _interleaved: inducing_values = inducing_values.transpose(-1, -2)
                inducing_values = inducing_values.reshape(*inducing_values.shape[:-2], -1)
        else:
            inducing_values = inducing_values.repeat_interleave(num_tasks,-1) if _interleaved else inducing_values.tile(num_tasks)
        predictive_mean = (interp_term.transpose(-1, -2) @ inducing_values.unsqueeze(-1)).squeeze(-1)
        if _interleaved:
            predictive_mean = predictive_mean.reshape_as(test_mean) + test_mean
        else:
            new_shape = test_mean.shape[:-2] + test_mean.shape[:-3:-1]
            predictive_mean = predictive_mean.view(new_shape).transpose(-1, -2).contiguous() + test_mean

        # Compute the covariance of q(f)
        # K_XX + k_XZ K_ZZ^{-1/2} (S - I) K_ZZ^{-1/2} k_ZX
        middle_term = self.prior_distribution.lazy_covariance_matrix.mul(-1)
        if variational_inducing_covar is not None:
            middle_term = SumLinearOperator(variational_inducing_covar, middle_term)
        if variational_inducing_covar.batch_dim > 0:
            for _ in range(variational_inducing_covar.batch_dim):
                b,n=middle_term.shape[-3:-1]
                middle_term = BlockDiagLinearOperator(middle_term)
                if _interleaved:
                    pi = torch.arange(b * n, device=middle_term.device).view(b, n).t().reshape((b * n))
                    middle_term = middle_term[..., pi, :][..., :, pi]
        else:
            if _interleaved:
                middle_term = KroneckerProductLinearOperator(middle_term, DiagLinearOperator(torch.ones(num_tasks, device=middle_term.device)))
            else:
                middle_term = KroneckerProductLinearOperator(DiagLinearOperator(torch.ones(num_tasks, device=middle_term.device)), middle_term)

        if trace_mode.on():
            predictive_covar = (
                data_data_covar.add_jitter(self.jitter_val).to_dense()
                + interp_term.transpose(-1, -2) @ middle_term.to_dense() @ interp_term
            )
        else:
            predictive_covar = SumLinearOperator(
                data_data_covar.add_jitter(self.jitter_val),
                MatmulLinearOperator(interp_term.transpose(-1, -2), middle_term @ interp_term),
            )

        # Return the distribution
        if hasattr(self.model, 'power'):
            return MultitaskMultivariateQExponential(predictive_mean, predictive_covar, power=self.model.power, interleaved=_interleaved)
        else:
            return MultitaskMultivariateNormal(predictive_mean, predictive_covar, interleaved=_interleaved)

    def __call__(self, x: Tensor, prior: bool = False, **kwargs) -> Union[MultivariateNormal, MultivariateQExponential]:
        if not self.updated_strategy.item() and not prior:
            with torch.no_grad():
                # Get unwhitened p(u)
                prior_function_dist = self(self.inducing_points, prior=True)
                prior_mean = prior_function_dist.loc
                L = self._cholesky_factor(prior_function_dist.lazy_covariance_matrix.add_jitter(self.jitter_val))

                # Temporarily turn off noise that's added to the mean
                orig_mean_init_std = self._variational_distribution.mean_init_std
                self._variational_distribution.mean_init_std = 0.0

                # Change the variational parameters to be whitened
                variational_dist = self.variational_distribution
                if isinstance(variational_dist, (MultivariateNormal, MultivariateQExponential)):
                    mean_diff = (variational_dist.loc - prior_mean).unsqueeze(-1).type(_linalg_dtype_cholesky.value())
                    whitened_mean = L.solve(mean_diff).squeeze(-1).to(variational_dist.loc.dtype)
                    covar_root = variational_dist.lazy_covariance_matrix.root_decomposition().root.to_dense()
                    covar_root = covar_root.type(_linalg_dtype_cholesky.value())
                    whitened_covar = RootLinearOperator(L.solve(covar_root).to(variational_dist.loc.dtype))
                    whitened_variational_distribution = variational_dist.__class__(whitened_mean, whitened_covar)
                    if isinstance(variational_dist, MultivariateQExponential): whitened_variational_distribution.power = variational_dist.power
                    self._variational_distribution.initialize_variational_distribution(
                        whitened_variational_distribution
                    )

                # Reset the random noise parameter of the model
                self._variational_distribution.mean_init_std = orig_mean_init_std

                # Reset the cache
                clear_cache_hook(self)

                # Mark that we have updated the variational strategy
                self.updated_strategy.fill_(True)

        return super().__call__(x, prior=prior, **kwargs)
