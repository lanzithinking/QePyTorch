#!/usr/bin/env python3

from typing import Any, Optional, Tuple, Union

import torch
from linear_operator import to_linear_operator
from linear_operator.operators import (
    ConstantDiagLinearOperator,
    DiagLinearOperator,
    KroneckerProductDiagLinearOperator,
    KroneckerProductLinearOperator,
    LinearOperator,
    RootLinearOperator,
    BlockDiagLinearOperator,
    ZeroLinearOperator
)
from torch import Tensor

from ..constraints import GreaterThan, Interval
from ..distributions import base_distributions, QExponential, MultitaskMultivariateQExponential, Distribution
from ..lazy import LazyEvaluatedKernelTensor
from ..likelihoods import _QExponentialLikelihoodBase, Likelihood
from ..priors import Prior
from .noise_models import FixedNoise, MultitaskHomoskedasticNoise, Noise


class _MultitaskQExponentialLikelihoodBase(_QExponentialLikelihoodBase):
    r"""
    Base class for multi-task QExponential Likelihoods, supporting general heteroskedastic noise models.

    :param num_tasks: Number of tasks.
    :param noise_covar: A model for the noise covariance. This can be a simple homoskedastic noise model, or a QEP
        that is to be fitted on the observed measurement errors.
    :param rank: The rank of the task noise covariance matrix to fit. If `rank`
        is set to 0, then a diagonal covariance matrix is fit.
    :param task_correlation_prior: Prior to use over the task noise correlation
        matrix. Only used when :math:`\text{rank} > 0`.
    :param batch_shape: Number of batches.
    """

    def __init__(
        self,
        num_tasks: int,
        noise_covar: Union[Noise, FixedNoise],
        rank: int = 0,
        task_correlation_prior: Optional[Prior] = None,
        batch_shape: torch.Size = torch.Size(),
        **kwargs: Any,
    ) -> None:
        super().__init__(noise_covar=noise_covar, **kwargs)
        if rank != 0:
            if rank > num_tasks:
                raise ValueError(f"Cannot have rank ({rank}) greater than num_tasks ({num_tasks})")
            tidcs = torch.tril_indices(num_tasks, rank, dtype=torch.long)
            self.tidcs: Tensor = tidcs[:, 1:]  # (1, 1) must be 1.0, no need to parameterize this
            task_noise_corr = torch.randn(*batch_shape, self.tidcs.size(-1))
            self.register_parameter("task_noise_corr", torch.nn.Parameter(task_noise_corr))
            if task_correlation_prior is not None:
                self.register_prior(
                    "MultitaskErrorCorrelationPrior", task_correlation_prior, lambda m: m._eval_corr_matrix
                )
        elif task_correlation_prior is not None:
            raise ValueError("Can only specify task_correlation_prior if rank>0")
        self.num_tasks = num_tasks
        self.rank = rank

    def _eval_corr_matrix(self) -> Tensor:
        tnc = self.task_noise_corr
        fac_diag = torch.ones(*tnc.shape[:-1], self.num_tasks, device=tnc.device, dtype=tnc.dtype)
        Cfac = torch.diag_embed(fac_diag)
        Cfac[..., self.tidcs[0], self.tidcs[1]] = self.task_noise_corr
        # squared rows must sum to one for this to be a correlation matrix
        C = Cfac / Cfac.pow(2).sum(dim=-1, keepdim=True).sqrt()
        return C @ C.transpose(-1, -2)

    def marginal(
        self, function_dist: MultitaskMultivariateQExponential, *params: Any, **kwargs: Any
    ) -> MultitaskMultivariateQExponential:  # pyre-ignore[14]
        r"""
        If :math:`\text{rank} = 0`, adds the task noises to the diagonal of the
        covariance matrix of the supplied
        :obj:`~qpytorch.distributions.MultivariateQExponential` or
        :obj:`~qpytorch.distributions.MultitaskMultivariateQExponential`.  Otherwise,
        adds a rank `rank` covariance matrix to it.

        To accomplish this, we form a new
        :obj:`~linear_operator.operators.KroneckerProductLinearOperator`
        between :math:`I_{n}`, an identity matrix with size equal to the data
        and a (not necessarily diagonal) matrix containing the task noises
        :math:`D_{t}`.

        We also incorporate a shared `noise` parameter from the base
        :class:`qpytorch.likelihoods.QExponentialLikelihood` that we extend.

        The final covariance matrix after this method is then
        :math:`\mathbf K + \mathbf D_{t} \otimes \mathbf I_{n} + \sigma^{2} \mathbf I_{nt}`.

        :param function_dist: Random variable whose covariance
            matrix is a :obj:`~linear_operator.operators.LinearOperator` we intend to augment.
        :rtype: `qpytorch.distributions.MultitaskMultivariateQExponential`:
        :return: A new random variable whose covariance matrix is a
            :obj:`~linear_operator.operators.LinearOperator` with
            :math:`\mathbf D_{t} \otimes \mathbf I_{n}` and :math:`\sigma^{2} \mathbf I_{nt}` added.
        """
        mean, covar, power = function_dist.mean, function_dist.lazy_covariance_matrix, function_dist.power

        # ensure that sumKroneckerLT is actually called
        if isinstance(covar, LazyEvaluatedKernelTensor):
            covar = covar.evaluate_kernel()

        covar_kron_lt = self._shaped_noise_covar(
            mean.shape, add_noise=self.has_global_noise, interleaved=function_dist._interleaved
        )
        covar = covar + covar_kron_lt

        return function_dist.__class__(mean, covar, power, interleaved=function_dist._interleaved)

    def _shaped_noise_covar(
        self, shape: torch.Size, add_noise: Optional[bool] = True, interleaved: bool = True, *params: Any, **kwargs: Any
    ) -> LinearOperator:
        if not self.has_task_noise:
            noise = ConstantDiagLinearOperator(self.noise, diag_shape=shape[-2] * self.num_tasks)
            return noise

        if self.rank == 0:
            task_noises = self.raw_task_noises_constraint.transform(self.raw_task_noises)
            task_var_lt = DiagLinearOperator(task_noises)
            dtype, device = task_noises.dtype, task_noises.device
            ckl_init = KroneckerProductDiagLinearOperator
        else:
            task_noise_covar_factor = self.task_noise_covar_factor
            task_var_lt = RootLinearOperator(task_noise_covar_factor)
            dtype, device = task_noise_covar_factor.dtype, task_noise_covar_factor.device
            ckl_init = KroneckerProductLinearOperator

        eye_lt = ConstantDiagLinearOperator(
            torch.ones(*shape[:-2], 1, dtype=dtype, device=device), diag_shape=shape[-2]
        )
        task_var_lt = task_var_lt.expand(*shape[:-2], *task_var_lt.matrix_shape)  # pyre-ignore[6]

        # to add the latent noise we exploit the fact that
        # I \kron D_T + \sigma^2 I_{NT} = I \kron (D_T + \sigma^2 I)
        # which allows us to move the latent noise inside the task dependent noise
        # thereby allowing exploitation of Kronecker structure in this likelihood.
        if add_noise and self.has_global_noise:
            noise = ConstantDiagLinearOperator(self.noise, diag_shape=task_var_lt.shape[-1])
            task_var_lt = task_var_lt + noise

        if interleaved:
            covar_kron_lt = ckl_init(eye_lt, task_var_lt)
        else:
            covar_kron_lt = ckl_init(task_var_lt, eye_lt)

        return covar_kron_lt

    def forward(self, function_samples: Tensor, *params: Any, **kwargs: Any) -> QExponential:
        noise = self._shaped_noise_covar(function_samples.shape, *params, **kwargs).diagonal(dim1=-1, dim2=-2)
        noise = noise.reshape(*noise.shape[:-1], *function_samples.shape[-2:])
        return base_distributions.Independent(QExponential(function_samples, noise.sqrt(), self.power), 1)


class MultitaskQExponentialLikelihood(_MultitaskQExponentialLikelihoodBase):
    r"""
    A convenient extension of the :class:`~qpytorch.likelihoods.QExponentialLikelihood` to the multitask setting that allows
    for a full cross-task covariance structure for the noise. The fitted covariance matrix has rank `rank`.
    If a strictly diagonal task noise covariance matrix is desired, then rank=0 should be set. (This option still
    allows for a different `noise` parameter for each task.)

    Like the Q-Exponential likelihood, this object can be used with exact inference.

    .. note::
        At least one of :attr:`has_global_noise` or :attr:`has_task_noise` should be specified.

    .. note::
        MultitaskQExponentialLikelihood has an analytic marginal distribution.

    :param num_tasks: Number of tasks.
    :param noise_covar: A model for the noise covariance. This can be a simple homoskedastic noise model, or a QEP
        that is to be fitted on the observed measurement errors.
    :param rank: The rank of the task noise covariance matrix to fit. If `rank`
        is set to 0, then a diagonal covariance matrix is fit.
    :param task_prior: Prior to use over the task noise correlation
        matrix. Only used when :math:`\text{rank} > 0`.
    :param batch_shape: Number of batches.
    :param has_global_noise: Whether to include a :math:`\sigma^2 \mathbf I_{nt}` term in the noise model.
    :param has_task_noise: Whether to include task-specific noise terms, which add
        :math:`\mathbf I_n \otimes \mathbf D_T` into the noise model.
    :param kwargs: power (default: 2.0), miu (default: False).

    :ivar torch.Tensor task_noise_covar: The inter-task noise covariance matrix
    :ivar torch.Tensor task_noises: (Optional) task specific noise variances (added onto the `task_noise_covar`)
    :ivar torch.Tensor noise: (Optional) global noise variance (added onto the `task_noise_covar`)
    """

    def __init__(
        self,
        num_tasks: int,
        rank: int = 0,
        batch_shape: torch.Size = torch.Size(),
        task_prior: Optional[Prior] = None,
        noise_prior: Optional[Prior] = None,
        noise_constraint: Optional[Interval] = None,
        has_global_noise: bool = True,
        has_task_noise: bool = True,
        **kwargs: Any,
    ) -> None:
        super(Likelihood, self).__init__()  # pyre-ignore[20]
        self.power = kwargs.pop('power', torch.tensor(2.0))
        self.miu = kwargs.pop('miu', False) # marginally identical but uncorrelated
        if noise_constraint is None:
            noise_constraint = GreaterThan(1e-4)

        if not has_task_noise and not has_global_noise:
            raise ValueError(
                "At least one of has_task_noise or has_global_noise must be specified. "
                "Attempting to specify a likelihood that has no noise terms."
            )

        if has_task_noise:
            if rank == 0:
                self.register_parameter(
                    name="raw_task_noises", parameter=torch.nn.Parameter(torch.zeros(*batch_shape, num_tasks))
                )
                self.register_constraint("raw_task_noises", noise_constraint)
                if noise_prior is not None:
                    self.register_prior("raw_task_noises_prior", noise_prior, lambda m: m.task_noises)
                if task_prior is not None:
                    raise RuntimeError("Cannot set a `task_prior` if rank=0")
            else:
                self.register_parameter(
                    name="task_noise_covar_factor",
                    parameter=torch.nn.Parameter(torch.randn(*batch_shape, num_tasks, rank)),
                )
                if task_prior is not None:
                    self.register_prior("MultitaskErrorCovariancePrior", task_prior, lambda m: m._eval_covar_matrix)
        self.num_tasks = num_tasks
        self.rank = rank

        if has_global_noise:
            self.register_parameter(name="raw_noise", parameter=torch.nn.Parameter(torch.zeros(*batch_shape, 1)))
            self.register_constraint("raw_noise", noise_constraint)
            if noise_prior is not None:
                self.register_prior("raw_noise_prior", noise_prior, lambda m: m.noise)

        self.has_global_noise = has_global_noise
        self.has_task_noise = has_task_noise

    @property
    def noise(self) -> Optional[Tensor]:
        return self.raw_noise_constraint.transform(self.raw_noise)

    @noise.setter
    def noise(self, value: Union[float, Tensor]) -> None:
        self._set_noise(value)

    @property
    def task_noises(self) -> Optional[Tensor]:
        if self.rank == 0:
            return self.raw_task_noises_constraint.transform(self.raw_task_noises)
        else:
            raise AttributeError("Cannot set diagonal task noises when covariance has ", self.rank, ">0")

    @task_noises.setter
    def task_noises(self, value: Union[float, Tensor]) -> None:
        if self.rank == 0:
            self._set_task_noises(value)
        else:
            raise AttributeError("Cannot set diagonal task noises when covariance has ", self.rank, ">0")

    def _set_noise(self, value: Union[float, Tensor]) -> None:
        self.initialize(raw_noise=self.raw_noise_constraint.inverse_transform(value))

    def _set_task_noises(self, value: Union[float, Tensor]) -> None:
        self.initialize(raw_task_noises=self.raw_task_noises_constraint.inverse_transform(value))

    @property
    def task_noise_covar(self) -> Tensor:
        if self.rank > 0:
            return self.task_noise_covar_factor.matmul(self.task_noise_covar_factor.transpose(-1, -2))
        else:
            raise AttributeError("Cannot retrieve task noises when covariance is diagonal.")

    @task_noise_covar.setter
    def task_noise_covar(self, value: Tensor) -> None:
        # internally uses a pivoted cholesky decomposition to construct a low rank
        # approximation of the covariance
        if self.rank > 0:
            with torch.no_grad():
                self.task_noise_covar_factor.data = to_linear_operator(value).pivoted_cholesky(rank=self.rank)
        else:
            raise AttributeError("Cannot set non-diagonal task noises when covariance is diagonal.")

    def _eval_covar_matrix(self) -> Tensor:
        covar_factor = self.task_noise_covar_factor
        noise = self.noise
        D = noise * torch.eye(self.num_tasks, dtype=noise.dtype, device=noise.device)  # pyre-fixme[16]
        return covar_factor.matmul(covar_factor.transpose(-1, -2)) + D

    def marginal(
        self, function_dist: MultitaskMultivariateQExponential, *args: Any, **kwargs: Any
    ) -> MultitaskMultivariateQExponential:
        r"""
        :return: Analytic marginal :math:`p(\mathbf y)`.
        """
        return super().marginal(function_dist, *args, **kwargs)


class MultitaskFixedNoiseQExponentialLikelihood(_MultitaskQExponentialLikelihoodBase):
    r"""
    A convenient extension of the :class:`~qpytorch.likelihoods.FixedNoiseQExponentialLikelihood` to the multitask setting
    that assumes fixed heteroscedastic noise. This is useful when you have fixed, known observation
    noise for each training example.

    Note that this likelihood takes an additional argument when you call it, `noise`, that adds a specified amount
    of noise to the passed MultivariateQExponential. This allows for adding known observational noise to test data.

    .. note::
        This likelihood can be used for exact or approximate inference.

    :param num_tasks: Number of tasks.
    :param noise: Known observation noise (variance) for each training example.
    :type noise: torch.Tensor (... x N)
    :param rank: The rank of the task noise covariance matrix to fit. If `rank`
        is set to 0, then a diagonal covariance matrix is fit.
    :param learn_additional_noise: Set to true if you additionally want to
        learn added diagonal noise, similar to QExponentialLikelihood.
    :type learn_additional_noise: bool, optional
    :param batch_shape: The batch shape of the learned noise parameter (default
        []) if :obj:`learn_additional_noise=True`.
    :type batch_shape: torch.Size, optional
    :param kwargs: power (default: 2.0), miu (default: False).

    :var torch.Tensor noise: :math:`\sigma^2` parameter (noise)

    .. note::
        MultitaskFixedNoiseQExponentialLikelihood has an analytic marginal distribution.

    Example:
        >>> num_tasks = 2
        >>> train_x = torch.randn(55, 2)
        >>> noises = torch.ones(55) * 0.01
        >>> likelihood = MultitaskFixedNoiseQExponentialLikelihood(num_tasks=num_tasks, noise=noises, learn_additional_noise=True)
        >>> pred_y = likelihood(qep_model(train_x))
        >>>
        >>> test_x = torch.randn(21, 2)
        >>> test_noises = torch.ones(21) * 0.02
        >>> pred_y = likelihood(qep_model(test_x), noise=test_noises)
    """

    def __init__(
        self,
        num_tasks: int,
        noise: Tensor,
        rank: int = 0,
        learn_additional_noise: Optional[bool] = False,
        batch_shape: Optional[torch.Size] = torch.Size(),
        **kwargs: Any,
    ) -> None:
        super().__init__(num_tasks=num_tasks, noise_covar=FixedNoise(noise=noise), rank=rank, batch_shape=batch_shape, **kwargs)

        self.second_noise_covar: Optional[MultitaskHomoskedasticNoise] = None
        if learn_additional_noise:
            noise_prior = kwargs.get("noise_prior", None)
            noise_constraint = kwargs.get("noise_constraint", None)
            self.second_noise_covar = MultitaskHomoskedasticNoise(
                num_tasks=1, noise_prior=noise_prior, noise_constraint=noise_constraint, batch_shape=batch_shape
            )

    @property
    def noise(self) -> Tensor:
        return self.noise_covar.noise + self.second_noise

    @noise.setter
    def noise(self, value: Tensor) -> None:
        self.noise_covar.initialize(noise=value)

    @property
    def second_noise(self) -> Union[float, Tensor]:
        if self.second_noise_covar is None:
            return 0.0
        else:
            return self.second_noise_covar.noise

    @second_noise.setter
    def second_noise(self, value: Tensor) -> None:
        if self.second_noise_covar is None:
            raise RuntimeError(
                "Attempting to set secondary learned noise for MultitaskFixedNoiseQExponentialLikelihood, "
                "but learn_additional_noise must have been False!"
            )
        self.second_noise_covar.initialize(noise=value)

    def get_fantasy_likelihood(self, **kwargs: Any) -> "MultitaskFixedNoiseQExponentialLikelihood":
        if "noise" not in kwargs:
            raise RuntimeError("MultitaskFixedNoiseQExponentialLikelihood.fantasize requires a `noise` kwarg")
        old_noise_covar = self.noise_covar
        self.noise_covar = None  # pyre-fixme[8]
        fantasy_liklihood = deepcopy(self)
        self.noise_covar = old_noise_covar

        old_noise = old_noise_covar.noise
        new_noise = kwargs.get("noise")
        if old_noise.dim() != new_noise.dim():
            old_noise = old_noise.expand(*new_noise.shape[:-1], old_noise.shape[-1])
        fantasy_liklihood.noise_covar = FixedNoise(noise=torch.cat([old_noise, new_noise], -1))
        return fantasy_liklihood

    def _shaped_noise_covar(self, base_shape: torch.Size, *params: Any, **kwargs: Any) -> Union[Tensor, LinearOperator]:
        if len(params) > 0:
            # we can infer the shape from the params
            shape = None
        else:
            # here shape[:-1] is the batch shape requested, and shape[-1] is `n`, the number of points
            shape = base_shape[:-2]+base_shape[-2:][::-1]

        res = self.noise_covar(*params, shape=shape, **kwargs)

        if self.second_noise_covar is not None:
            res = res + self.second_noise_covar(*params, shape=shape, **kwargs)
        elif isinstance(res, ZeroLinearOperator):
            warnings.warn(
                "You have passed data through a FixedNoiseQExponentialLikelihood that did not match the size "
                "of the fixed noise, *and* you did not specify noise. This is treated as a no-op.",
                QEPInputWarning,
            )

        return BlockDiagLinearOperator(res)

    def marginal(self, function_dist: MultitaskMultivariateQExponential, *args: Any, **kwargs: Any) -> MultitaskMultivariateQExponential:
        r"""
        :return: Analytic marginal :math:`p(\mathbf y)`.
        """
        return super().marginal(function_dist, *args, **kwargs)


class MultitaskQExponentialDirichletClassificationLikelihood(MultitaskFixedNoiseQExponentialLikelihood):
    r"""
    A multi-classification likelihood that treats the labels as regression targets with fixed heteroscedastic noise.
    From Milios et al, NeurIPS, 2018 [https://arxiv.org/abs/1805.10915].

    .. note::
        This multitask likelihood can be used for exact or approximate inference and in deep models.

    :param targets: (... x N) Classification labels.
    :param alpha_epsilon: Tuning parameter for the scaling of the likeihood targets. We'd suggest 0.01 or setting
        via cross-validation.
    :param learn_additional_noise: Set to true if you additionally want to
        learn added diagonal noise, similar to QExponentialLikelihood.
    :param batch_shape: The batch shape of the learned noise parameter (default
        []) if :obj:`learn_additional_noise=True`.
    :param kwargs: power (default: 2.0), miu (default: False).

    :ivar torch.Tensor noise: :math:`\sigma^2` parameter (noise)

    .. note::
        MultitaskDirichletClassificationLikelihood has an analytic marginal distribution.

    Example:
        >>> train_x = torch.randn(55, 1)
        >>> labels = torch.round(train_x).long()
        >>> likelihood = MultitaskDirichletClassificationLikelihood(targets=labels, learn_additional_noise=True)
        >>> pred_y = likelihood(qep_model(train_x))
        >>>
        >>> test_x = torch.randn(21, 1)
        >>> test_labels = torch.round(test_x).long()
        >>> pred_y = likelihood(qep_model(test_x), targets=labels)
    """

    def _prepare_targets(
        self, targets: Tensor, num_classes: Optional = None, alpha_epsilon: float = 0.01, dtype: torch.dtype = torch.float
    ) -> Tuple[Tensor, Tensor, int]:
        if num_classes is None: num_classes = int(targets.max() + 1)
        # set alpha = \alpha_\epsilon
        alpha = alpha_epsilon * torch.ones(targets.shape[-1], num_classes, device=targets.device, dtype=dtype)

        # alpha[class_labels] = 1 + \alpha_\epsilon
        alpha[torch.arange(len(targets)), targets] = alpha[torch.arange(len(targets)), targets] + 1.0

        # sigma^2 = log(1 / alpha + 1)
        sigma2_i = torch.log(alpha.reciprocal() + 1.0)

        # y = log(alpha) - 0.5 * sigma^2
        transformed_targets = alpha.log() - 0.5 * sigma2_i

        return sigma2_i.transpose(-2, -1).type(dtype), transformed_targets.type(dtype), num_classes

    def __init__(
        self,
        targets: Tensor,
        alpha_epsilon: float = 0.01,
        learn_additional_noise: Optional[bool] = False,
        batch_shape: torch.Size = torch.Size(),
        dtype: torch.dtype = torch.float,
        **kwargs: Any,
    ) -> None:
        sigma2_labels, transformed_targets, num_classes = self._prepare_targets(
            targets, alpha_epsilon=alpha_epsilon, dtype=dtype
        )
        super().__init__(
            num_tasks=num_classes,
            noise=sigma2_labels,
            learn_additional_noise=learn_additional_noise,
            batch_shape=torch.Size((num_classes,)),
            **kwargs,
        )
        self.transformed_targets: Tensor = transformed_targets.transpose(-2, -1)
        self.num_classes: int = num_classes
        self.targets: Tensor = targets
        self.alpha_epsilon: float = alpha_epsilon

    def get_fantasy_likelihood(self, **kwargs: Any) -> "MultitaskDirichletClassificationLikelihood":
        # we assume that the number of classes does not change.

        if "targets" not in kwargs:
            raise RuntimeError("FixedNoiseQExponentialLikelihood.fantasize requires a `targets` kwarg")

        old_noise_covar = self.noise_covar
        self.noise_covar = None  # pyre-fixme[8]
        fantasy_liklihood = deepcopy(self)
        self.noise_covar = old_noise_covar

        old_noise = old_noise_covar.noise
        new_targets = kwargs.get("noise")
        new_noise, new_targets, _ = fantasy_liklihood._prepare_targets(new_targets, self.alpha_epsilon)
        fantasy_liklihood.targets = torch.cat([fantasy_liklihood.targets, new_targets], -1)

        if old_noise.dim() != new_noise.dim():
            old_noise = old_noise.expand(*new_noise.shape[:-1], old_noise.shape[-1])

        fantasy_liklihood.noise_covar = FixedNoise(noise=torch.cat([old_noise, new_noise], -1))
        return fantasy_liklihood

    def marginal(self, function_dist: MultitaskMultivariateQExponential, *args: Any, **kwargs: Any) -> MultitaskMultivariateQExponential:
        r"""
        :return: Analytic marginal :math:`p(\mathbf y)`.
        """
        return super().marginal(function_dist, *args, **kwargs)

    def __call__(self, input: Union[Tensor, MultitaskMultivariateQExponential], *args: Any, **kwargs: Any) -> Distribution:
        if "targets" in kwargs:
            targets = kwargs.pop("targets")
            dtype = self.transformed_targets.dtype
            new_noise, _, _ = self._prepare_targets(targets, dtype=dtype)
            kwargs["noise"] = new_noise
        return super().__call__(input, *args, **kwargs)
