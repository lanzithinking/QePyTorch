#!/usr/bin/env python3

from __future__ import annotations

import math
import warnings
from numbers import Number
from typing import Optional, Tuple, Union

import torch
from linear_operator import to_dense, to_linear_operator
from linear_operator.operators import DiagLinearOperator, LinearOperator, RootLinearOperator
from torch import Tensor
from torch.distributions import MultivariateNormal as TMultivariateNormal, Chi2
from torch.distributions.kl import register_kl
from torch.distributions.utils import _standard_normal, lazy_property

from .. import settings
from ..utils.warnings import NumericalWarning
from gpytorch.distributions.distribution import Distribution


class MultivariateQExponential(TMultivariateNormal, Distribution):
    """
    Constructs a multivariate q-exponential random variable, based on mean and covariance, whose density is
    
    .. math::
    
        p(x; \\mu, C) = \\frac{q}{2} (2\\pi)^{-\\frac{N}{2}} |C|^{-\\frac{1}{2}} 
        r^{\\left(\\frac{q}{2}-1\\right)\\frac{N}{2}} \\exp\\left\\{ -0.5 * r^{\\frac{q}{2}} \\right\\}, \\quad
        r(x) = (x - \\mu)^T C^{-1} (x - \\mu).
    
    The result can be multivariate, or a batch of multivariate q-exponentials.
    Passing a vector mean corresponds to a multivariate q-exponential.
    Passing a matrix mean corresponds to a batch of multivariate q-exponentials.

    :param mean: `... x N` mean of qep distribution.
    :param covariance_matrix: `... x N X N` covariance matrix of qep distribution.
    :param power: (scalar) power of qep distribution. (Default: 2.)
    :param validate_args: If True, validate `mean` and `covariance_matrix` arguments. (Default: False.)

    :ivar torch.Size base_sample_shape: The shape of a base sample (without
        batching) that is used to generate a single sample.
    :ivar torch.Tensor covariance_matrix: The covariance matrix, represented as a dense :class:`torch.Tensor`
    :ivar ~linear_operator.LinearOperator lazy_covariance_matrix: The covariance matrix, represented
        as a :class:`~linear_operator.LinearOperator`.
    :ivar torch.Tensor mean: The mean.
    :ivar torch.Tensor stddev: The standard deviation.
    :ivar torch.Tensor variance: The variance.
    """

    def __init__(self, mean: Tensor, covariance_matrix: Union[Tensor, LinearOperator], power: Tensor = torch.tensor(2.0), validate_args: bool = False):
        self._islazy = isinstance(mean, LinearOperator) or isinstance(covariance_matrix, LinearOperator)
        if self._islazy:
            if validate_args:
                ms = mean.size(-1)
                cs1 = covariance_matrix.size(-1)
                cs2 = covariance_matrix.size(-2)
                if not (ms == cs1 and ms == cs2):
                    raise ValueError(f"Wrong shapes in {self._repr_sizes(mean, covariance_matrix)}")
            self.loc = mean
            self._covar = covariance_matrix
            self.__unbroadcasted_scale_tril = None
            self._validate_args = validate_args
            batch_shape = torch.broadcast_shapes(self.loc.shape[:-1], covariance_matrix.shape[:-2])

            event_shape = self.loc.shape[-1:]

            # TODO: Integrate argument validation for LinearOperators into torch.distribution validation logic
            super(TMultivariateNormal, self).__init__(batch_shape, event_shape, validate_args=False)
        else:
            super().__init__(loc=mean, covariance_matrix=covariance_matrix, validate_args=validate_args)
        self.power = power

    def _extended_shape(self, sample_shape: torch.Size = torch.Size()) -> torch.Size:
        """
        Returns the size of the sample returned by the distribution, given
        a `sample_shape`. Note, that the batch and event shapes of a distribution
        instance are fixed at the time of construction. If this is empty, the
        returned shape is upcast to (1,).

        :param sample_shape: the size of the sample to be drawn.
        """
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        return sample_shape + self._batch_shape + self.base_sample_shape

    @staticmethod
    def _repr_sizes(mean: Tensor, covariance_matrix: Union[Tensor, LinearOperator], power: Tensor = torch.tensor(2.0)) -> str:
        return f"MultivariateQExponential(loc: {mean.size()}, scale: {covariance_matrix.size()}, pow: {power.size()})"

    @property
    def _unbroadcasted_scale_tril(self) -> Tensor:
        if self.islazy and self.__unbroadcasted_scale_tril is None:
            # cache root decoposition
            ust = to_dense(self.lazy_covariance_matrix.cholesky())
            self.__unbroadcasted_scale_tril = ust
        return self.__unbroadcasted_scale_tril

    @_unbroadcasted_scale_tril.setter
    def _unbroadcasted_scale_tril(self, ust: Tensor):
        if self.islazy:
            raise NotImplementedError("Cannot set _unbroadcasted_scale_tril for lazy QEP distributions")
        else:
            self.__unbroadcasted_scale_tril = ust

    def add_jitter(self, noise: float = 1e-4) -> MultivariateQExponential:
        r"""
        Adds a small constant diagonal to the QEP covariance matrix for numerical stability.

        :param noise: The size of the constant diagonal.
        """
        return self.__class__(self.mean, self.lazy_covariance_matrix.add_jitter(noise), self.power)

    @property
    def base_sample_shape(self) -> torch.Size:
        base_sample_shape = self.event_shape
        if isinstance(self.lazy_covariance_matrix, RootLinearOperator):
            base_sample_shape = self.lazy_covariance_matrix.root.shape[-1:]

        return base_sample_shape

    @lazy_property
    def covariance_matrix(self) -> Tensor:
        if self.islazy:
            return self._covar.to_dense()
        else:
            return super().covariance_matrix

    @property
    def rescalor(self) -> Tensor:
        n = self.event_shape[0]
        return torch.exp((2./self.power*math.log(2) - math.log(n) + torch.lgamma(n/2.+2./self.power) - math.lgamma(n/2.))/2.)

    def confidence_region(self, rescale=False) -> Tuple[Tensor, Tensor]:
        """
        Returns 2 standard deviations above and below the mean.

        :return: Pair of tensors of size `... x N`, where N is the
            dimensionality of the random variable. The first (second) Tensor is the
            lower (upper) end of the confidence region.
        """
        std2 = self.stddev.mul(2).mul(self.rescalor if rescale else 1)
        mean = self.mean
        return mean.sub(std2), mean.add(std2)

    def expand(self, batch_size: torch.Size) -> MultivariateQExponential:
        r"""
        See :py:meth:`torch.distributions.Distribution.expand
        <torch.distributions.distribution.Distribution.expand>`.
        """
        # NOTE: Pyro may call this method with list[int] instead of torch.Size.
        batch_size = torch.Size(batch_size)
        new_loc = self.loc.expand(batch_size + self.loc.shape[-1:])
        if self.islazy:
            new_covar = self._covar.expand(batch_size + self._covar.shape[-2:])
            new = self.__class__(mean=new_loc, covariance_matrix=new_covar, power=self.power)
            if self.__unbroadcasted_scale_tril is not None:
                # Reuse the scale tril if available.
                new.__unbroadcasted_scale_tril = self.__unbroadcasted_scale_tril.expand(
                    batch_size + self.__unbroadcasted_scale_tril.shape[-2:]
                )
        else:
            # Non-lazy QEP is represented using scale_tril in PyTorch.
            # Constructing it from scale_tril will avoid unnecessary computation.
            # Initialize using  __new__, so that we can skip __init__ and use scale_tril.
            new = self.__new__(type(self))
            new._islazy = False
            new_scale_tril = self.__unbroadcasted_scale_tril.expand(
                batch_size + self.__unbroadcasted_scale_tril.shape[-2:]
            )
            super(MultivariateQExponential, new).__init__(loc=new_loc, scale_tril=new_scale_tril)
            new.power = self.power
            # Set the covar matrix, since it is always available for QPyTorch QEP.
            new.covariance_matrix = self.covariance_matrix.expand(batch_size + self.covariance_matrix.shape[-2:])
        return new

    def unsqueeze(self, dim: int) -> MultivariateQExponential:
        r"""
        Constructs a new MultivariateQExponential with the batch shape unsqueezed
        by the given dimension.
        For example, if `self.batch_shape = torch.Size([2, 3])` and `dim = 0`, then
        the returned MultivariateQExponential will have `batch_shape = torch.Size([1, 2, 3])`.
        If `dim = -1`, then the returned MultivariateQExponential will have
        `batch_shape = torch.Size([2, 3, 1])`.
        """
        if dim > len(self.batch_shape) or dim < -len(self.batch_shape) - 1:
            raise IndexError(
                "Dimension out of range (expected to be in range of "
                f"[{-len(self.batch_shape) - 1}, {len(self.batch_shape)}], but got {dim})."
            )
        if dim < 0:
            # If dim is negative, get the positive equivalent.
            dim = len(self.batch_shape) + dim + 1

        new_loc = self.loc.unsqueeze(dim)
        if self.islazy:
            new_covar = self._covar.unsqueeze(dim)
            new = self.__class__(mean=new_loc, covariance_matrix=new_covar, power=self.power)
            if self.__unbroadcasted_scale_tril is not None:
                # Reuse the scale tril if available.
                new.__unbroadcasted_scale_tril = self.__unbroadcasted_scale_tril.unsqueeze(dim)
        else:
            # Non-lazy QEP is represented using scale_tril in PyTorch.
            # Constructing it from scale_tril will avoid unnecessary computation.
            # Initialize using  __new__, so that we can skip __init__ and use scale_tril.
            new = self.__new__(type(self))
            new._islazy = False
            new_scale_tril = self.__unbroadcasted_scale_tril.unsqueeze(dim)
            super(MultivariateQExponential, new).__init__(loc=new_loc, scale_tril=new_scale_tril)
            new.power = self.power
            # Set the covar matrix, since it is always available for QPyTorch QEP.
            new.covariance_matrix = self.covariance_matrix.unsqueeze(dim)
        return new

    def get_base_samples(self, sample_shape: torch.Size = torch.Size(), rescale = False) -> Tensor:
        r"""
        Returns marginally identical but uncorrelated (m.i.u.) standard Q-Exponential samples to be used with
        :py:meth:`MultivariateQExponential.rsample(base_samples=base_samples)
        <qpytorch.distributions.MultivariateQExponential.rsample>`.

        :param sample_shape: The number of samples to generate. (Default: `torch.Size([])`.)
        :return: A `*sample_shape x *batch_shape x N` tensor of m.i.u. standard Q-Exponential samples.
        """
        with torch.no_grad():
            shape = self._extended_shape(sample_shape)
            base_samples = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
            if self.power!=2: base_samples = torch.nn.functional.normalize(base_samples, dim=-1)*Chi2(shape[-1]).sample(shape[:-1]+torch.Size([1])).to(self.loc.device)**(1./self.power)
            if rescale: base_samples /= self.rescalor
        return base_samples

    @lazy_property
    def lazy_covariance_matrix(self) -> LinearOperator:
        if self.islazy:
            return self._covar
        else:
            return to_linear_operator(super().covariance_matrix)

    def log_prob(self, value: Tensor) -> Tensor:
        r"""
        See :py:meth:`torch.distributions.Distribution.log_prob
        <torch.distributions.distribution.Distribution.log_prob>`.
        """
        if settings.fast_computations.log_prob.off():
            return super().log_prob(value)

        if self._validate_args:
            self._validate_sample(value)

        mean, covar, power = self.loc, self.lazy_covariance_matrix, self.power
        diff = value - mean

        # Repeat the covar to match the batch shape of diff
        if diff.shape[:-1] != covar.batch_shape:
            if len(diff.shape[:-1]) < len(covar.batch_shape):
                diff = diff.expand(covar.shape[:-1])
            else:
                padded_batch_shape = (*(1 for _ in range(diff.dim() + 1 - covar.dim())), *covar.batch_shape)
                covar = covar.repeat(
                    *(diff_size // covar_size for diff_size, covar_size in zip(diff.shape[:-1], padded_batch_shape)),
                    1,
                    1,
                )

        # Get log determininant and first part of quadratic form
        covar = covar.evaluate_kernel()
        inv_quad, logdet = covar.inv_quad_logdet(inv_quad_rhs=diff.unsqueeze(-1), logdet=True)

        res = -0.5 * sum([inv_quad**(power/2.), logdet, diff.size(-1) * math.log(2 * math.pi)])
        if power!=2: res += sum([0.5 * diff.size(-1) * (power/2.-1) * torch.log(inv_quad), torch.log(power/2.)])
        return res

    def entropy(self, exact: bool = False) -> Tensor:
        r"""
        See :py:meth:`torch.distributions.Distribution.entropy
        <torch.distributions.distribution.Distribution.entropy>`.
        """
        d = self._event_shape[0] #self.loc.shape[-1]
        if self.islazy:
            logdet = self.lazy_covariance_matrix.logdet()
            res = 0.5 * sum([d*math.log(2*math.pi), logdet, d**(1 if exact else self.power/2.)])
        else:
            res = super().entropy()
            if not exact: res += 0.5*(-d + d**(self.power/2.))
        if self.power!=2:
            res += sum([d/2.*(self.power/2.-1) *(2./self.power* Chi2(d).entropy() if exact else -math.log(d)), -torch.log(self.power/2.)])
        return res

    def zero_mean_qep_samples(self, op: LinearOperator, num_samples: int, **kwargs) -> Tensor:
        r"""
        Assumes that the LinearOpeator :math:`\mathbf A` is a covariance
        matrix, or a batch of covariance matrices.
        Returns samples from a zero-mean QEP, defined by :math:`\mathcal Q( \mathbf 0, \mathbf A)`.
    
        :param num_samples: Number of samples to draw.
        :return: Samples from QEP :math:`\mathcal Q( \mathbf 0, \mathbf A)`.
        """
        from linear_operator.utils.contour_integral_quad import contour_integral_quad
    
        if settings.ciq_samples.on():
            base_samples = self.get_base_samples(torch.Size([num_samples]), **kwargs)
            if len(self.event_shape)==2: # multitask case
                if not self._interleaved: base_samples = base_samples.transpose(-1,-2)
                base_samples = base_samples.reshape(base_samples.shape[:-2] + op.shape[-1:])
            # base_samples = base_samples.permute(-1, *range(op.dim() - 1)).contiguous()
            base_samples = base_samples.unsqueeze(-1)
            solves, weights, _, _ = contour_integral_quad(
                op.evaluate_kernel(),
                base_samples,
                inverse=False,
                num_contour_quadrature=settings.num_contour_quadrature.value(),
            )
    
            return (solves * weights).sum(0).squeeze(-1)
    
        else:
            if op.size()[-2:] == torch.Size([1, 1]):
                covar_root = op.to_dense().sqrt()
            else:
                covar_root = op.root_decomposition().root
    
            base_samples = self.get_base_samples(torch.Size([num_samples]), **kwargs)
            if len(self.event_shape)==2: # multitask case
                if not self._interleaved: base_samples = base_samples.transpose(-1,-2)
                base_samples = base_samples.reshape(base_samples.shape[:-2] + op.shape[-1:])
            base_samples = base_samples.permute(*range(1, base_samples.dim() ), 0)
            if covar_root.shape < op.shape: base_samples = base_samples[...,:covar_root.size(-1),:]
            samples = covar_root.matmul(base_samples).permute(-1, *range(base_samples.dim() - 1)).contiguous()
    
        return samples

    def rsample(self, sample_shape: torch.Size = torch.Size(), base_samples: Optional[Tensor] = None, **kwargs) -> Tensor:
        r"""
        Generates a `sample_shape` shaped reparameterized sample or `sample_shape`
        shaped batch of reparameterized samples if the distribution parameters
        are batched.

        For the MultivariateQExponential distribution, this is accomplished through:

        .. math::
            \boldsymbol \mu + \mathbf L \boldsymbol \epsilon

        where :math:`\boldsymbol \mu \in \mathcal R^N` is the QEP mean,
        :math:`\mathbf L \in \mathcal R^{N \times N}` is a "root" of the
        covariance matrix :math:`\mathbf K` (i.e. :math:`\mathbf L \mathbf
        L^\top = \mathbf K`), and :math:`\boldsymbol \epsilon \in \mathcal R^N` is a
        vector of (approximately) m.i.u. standard Q-Exponential random variables.

        :param sample_shape: The number of samples to generate. (Default: `torch.Size([])`.)
        :param base_samples: The `*sample_shape x *batch_shape x N` tensor of
            m.i.u. (or approximately m.i.u.) standard Q-Exponential samples to
            reparameterize. (Default: None.)
        :return: A `*sample_shape x *batch_shape x N` tensor of m.i.u. reparameterized samples.
        """
        covar = self.lazy_covariance_matrix
        if base_samples is None:
            # Create some samples
            num_samples = sample_shape.numel() or 1 # s

            # covar_base = covar#.base_linear_op if hasattr(covar, 'base_linear_op') else covar
            # if covar_base.size()[-2:] == torch.Size([1, 1]):
            #     covar_root = covar_base.to_dense().sqrt()
            # else:
            #     covar_root = covar_base.root_decomposition().root # [b] x e x e
            #
            # base_samples = self.get_base_samples(torch.Size([num_samples]), **kwargs) # s x b x e or s x n x t
            # if len(self.event_shape)==2: # multitask case
            #     if not self._interleaved: base_samples = base_samples.transpose(-1,-2) # s x t x n
            #     base_samples = base_samples.reshape(base_samples.shape[:-2] + covar_base.shape[-1:]) # s x e, e = nt
            # base_samples = base_samples.permute(*range(1, covar_base.dim() ), 0) # [b] x e x s
            # if covar_root.shape < covar_base.shape: base_samples = base_samples[...,:covar_root.size(-1),:]
            #
            # # Get samples
            # res = covar_root.matmul(base_samples).permute(-1, *range(covar_base.dim()-1)).contiguous() # s x [b] x e
            # # if hasattr(covar, '_remove_batch_dim'): res = covar._remove_batch_dim(res.unsqueeze(-1)).squeeze(-1)
            # res = res + self.loc.unsqueeze(0)
            res = self.zero_mean_qep_samples(covar, num_samples, **kwargs) + self.loc.unsqueeze(0)
            res = res.view(sample_shape + self.loc.shape)

        else:
            covar_root = covar.root_decomposition().root

            # Make sure that the base samples agree with the distribution
            if (
                self.loc.shape != base_samples.shape[-self.loc.dim() :]
                and covar_root.shape[-1] < base_samples.shape[-1]
            ):
                raise RuntimeError(
                    "The size of base_samples (minus sample shape dimensions) should agree with the size "
                    "of self.loc. Expected ...{} but got {}".format(self.loc.shape, base_samples.shape)
                )

            # Determine what the appropriate sample_shape parameter is
            sample_shape = base_samples.shape[: base_samples.dim() - self.loc.dim()]

            # Reshape samples to be batch_size x num_dim x num_samples
            # or num_bim x num_samples
            base_samples = base_samples.view(-1, *self.loc.shape[:-1], covar_root.shape[-1])
            base_samples = base_samples.permute(*range(1, self.loc.dim() + 1), 0)

            # Now reparameterize those base samples
            # If necessary, adjust base_samples for rank of root decomposition
            if covar_root.shape[-1] < base_samples.shape[-2]:
                base_samples = base_samples[..., : covar_root.shape[-1], :]
            elif covar_root.shape[-1] > base_samples.shape[-2]:
                # raise RuntimeError("Incompatible dimension of `base_samples`")
                covar_root = covar_root.transpose(-2, -1)
            res = covar_root.matmul(base_samples) + self.loc.unsqueeze(-1)

            # Permute and reshape new samples to be original size
            res = res.permute(-1, *range(self.loc.dim())).contiguous()
            res = res.view(sample_shape + self.loc.shape)

        return res

    def sample(self, sample_shape: torch.Size = torch.Size(), base_samples: Optional[Tensor] = None, **kwargs) -> Tensor:
        r"""
        Generates a `sample_shape` shaped sample or `sample_shape`
        shaped batch of samples if the distribution parameters
        are batched.

        Note that these samples are not reparameterized and therefore cannot be backpropagated through.

        :param sample_shape: The number of samples to generate. (Default: `torch.Size([])`.)
        :param base_samples: The `*sample_shape x *batch_shape x N` tensor of
            m.i.u. (or approximately m.i.u.) standard Q-Exponential samples to
            reparameterize. (Default: None.)
        :return: A `*sample_shape x *batch_shape x N` tensor of m.i.u. samples.
        """
        with torch.no_grad():
            return self.rsample(sample_shape=sample_shape, base_samples=base_samples, **kwargs)

    @property
    def stddev(self) -> Tensor:
        # self.variance is guaranteed to be positive, because we do clamping.
        return self.variance.sqrt()

    def to_data_uncorrelated_dist(self) -> MultivariateQExponential:
        """
        Convert a `... x N` QEP distribution into a batch of uncorrelated Q-Exponential distributions.
        Essentially, this throws away all covariance information
        and treats all dimensions as batch dimensions.

        :returns: A (data-uncorrelated) Q-Exponential distribution with batch shape `*batch_shape x N`.
        """
        # Create batch distribution where all data are uncorrelated, but the tasks are dependent
        # try:
        #     # If pyro is installed, use that set of base distributions
        #     import pyro.distributions as base_distributions
        # except ImportError:
        #     # Otherwise, use PyTorch
        #     import torch.distributions as base_distributions
        # return base_distributions.Normal(self.mean, self.stddev)
        new_cov = DiagLinearOperator(
            self.lazy_covariance_matrix.diagonal(dim1=-1, dim2=-2)
                )
        return self.__class__(mean=self.mean, covariance_matrix=new_cov, power=self.power)
    
    to_data_independent_dist = to_data_uncorrelated_dist # alias to the same function with a more appropriate name

    @property
    def variance(self) -> Tensor:
        if self.islazy:
            # overwrite this since torch uses unbroadcasted_scale_tril for this
            diag = self.lazy_covariance_matrix.diagonal(dim1=-1, dim2=-2)
            diag = diag.view(diag.shape[:-1] + self._event_shape)
            variance = diag.expand(self._batch_shape + self._event_shape)
        else:
            variance = super().variance

        # Check to make sure that variance isn't lower than minimum allowed value (default 1e-6).
        # This ensures that all variances are positive
        min_variance = settings.min_variance.value(variance.dtype)
        if variance.lt(min_variance).any():
            warnings.warn(
                f"Negative variance values detected. "
                "This is likely due to numerical instabilities. "
                f"Rounding negative variances up to {min_variance}.",
                NumericalWarning,
            )
            variance = variance.clamp_min(min_variance)
        return variance

    def __add__(self, other: MultivariateQExponential) -> MultivariateQExponential:
        if isinstance(other, MultivariateQExponential):
            return self.__class__(
                mean=self.mean + other.mean,
                covariance_matrix=(self.lazy_covariance_matrix + other.lazy_covariance_matrix),
                power=self.power
            )
        elif isinstance(other, int) or isinstance(other, float):
            return self.__class__(self.mean + other, self.lazy_covariance_matrix, self.power)
        else:
            raise RuntimeError("Unsupported type {} for addition w/ MultivariateQExponential".format(type(other)))

    def __getitem__(self, idx) -> MultivariateQExponential:
        r"""
        Constructs a new MultivariateQExponential that represents a random variable
        modified by an indexing operation.

        The mean and covariance matrix arguments are indexed accordingly.

        :param idx: Index to apply to the mean. The covariance matrix is indexed accordingly.
        """

        if not isinstance(idx, tuple):
            idx = (idx,)
        if len(idx) > self.mean.dim() and Ellipsis in idx:
            idx = tuple(i for i in idx if i != Ellipsis)
            if len(idx) < self.mean.dim():
                raise IndexError("Multiple ambiguous ellipsis in index!")

        rest_idx = idx[:-1]
        last_idx = idx[-1]
        new_mean = self.mean[idx]

        if len(idx) <= self.mean.dim() - 1 and (Ellipsis not in rest_idx):
            # We are only indexing the batch dimensions in this case
            new_cov = self.lazy_covariance_matrix[idx]
        elif len(idx) > self.mean.dim():
            raise IndexError(f"Index {idx} has too many dimensions")
        else:
            # In this case we know last_idx corresponds to the last dimension
            # of mean and the last two dimensions of lazy_covariance_matrix
            if isinstance(last_idx, int):
                new_cov = DiagLinearOperator(
                    self.lazy_covariance_matrix.diagonal(dim1=-1, dim2=-2)[(*rest_idx, last_idx)]
                )
            elif isinstance(last_idx, slice):
                new_cov = self.lazy_covariance_matrix[(*rest_idx, last_idx, last_idx)]
            elif last_idx is (...):
                new_cov = self.lazy_covariance_matrix[rest_idx]
            else:
                new_cov = self.lazy_covariance_matrix[(*rest_idx, last_idx, slice(None, None, None))][..., last_idx]
        return self.__class__(mean=new_mean, covariance_matrix=new_cov, power=self.power)

    def __mul__(self, other: Number) -> MultivariateQExponential:
        if not (isinstance(other, int) or isinstance(other, float)):
            raise RuntimeError("Can only multiply by scalars")
        if other == 1:
            return self
        return self.__class__(mean=self.mean * other, covariance_matrix=self.lazy_covariance_matrix * (other**2), power=self.power)

    def __radd__(self, other: MultivariateQExponential) -> MultivariateQExponential:
        if other == 0:
            return self
        return self.__add__(other)

    def __truediv__(self, other: Number) -> MultivariateQExponential:
        return self.__mul__(1.0 / other)


@register_kl(MultivariateQExponential, MultivariateQExponential)
def kl_qep_qep(p_dist: MultivariateQExponential, q_dist: MultivariateQExponential, exact: bool = False) -> Tensor:
    output_shape = torch.broadcast_shapes(p_dist.batch_shape, q_dist.batch_shape)
    if output_shape != p_dist.batch_shape:
        p_dist = p_dist.expand(output_shape)
    if output_shape != q_dist.batch_shape:
        q_dist = q_dist.expand(output_shape)

    q_mean = q_dist.loc
    q_covar = q_dist.lazy_covariance_matrix

    p_mean = p_dist.loc
    p_covar = p_dist.lazy_covariance_matrix
    root_p_covar = p_covar.root_decomposition().root.to_dense()

    mean_diffs = p_mean - q_mean
    dim = float(mean_diffs.size(-1))
    if isinstance(root_p_covar, LinearOperator):
        # right now this just catches if root_p_covar is a DiagLinearOperator,
        # but we may want to be smarter about this in the future
        root_p_covar = root_p_covar.to_dense()
    inv_quad_rhs = torch.cat([mean_diffs.unsqueeze(-1), root_p_covar], -1)
    logdet_p_covar = p_covar.logdet()
    trace_plus_inv_quad_form, logdet_q_covar = q_covar.inv_quad_logdet(inv_quad_rhs=inv_quad_rhs, logdet=True)

    # Compute the KL Divergence.
    res = 0.5 * sum([logdet_q_covar, logdet_p_covar.mul(-1), trace_plus_inv_quad_form**(q_dist.power/2.), -dim**(1 if exact else p_dist.power/2.)])
    if q_dist.power!=2: res +=  dim/2. * sum([-(q_dist.power/2.-1)*torch.log(trace_plus_inv_quad_form), -(p_dist.power/2.-1)*(2./p_dist.power*Chi2(dim).entropy() if exact else -math.log(dim))]) # exact value is intractable; an approximation is provided instead.
    return res
