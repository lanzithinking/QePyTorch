#!/usr/bin/env python3

import math
from numbers import Number, Real

import torch
from torch.distributions import constraints, Chi2
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.kl import register_kl
from torch.distributions.utils import _standard_normal, broadcast_all

from gpytorch.distributions.distribution import Distribution

__all__ = ["QExponential"]


class QExponential(ExponentialFamily, Distribution):
    r"""
    Creates a q-exponential distribution parameterized by
    :attr:`loc`, :attr:`scale` and :attr:`power`, with the following density
    
    .. math::
    
        p(x; \mu, \sigma^2) = \frac{q}{2}(2\pi\sigma^2)^{-\frac{1}{2}}
        \left|\frac{x-\mu}{\sigma}\right|^{\frac{q}{2}-1} 
        \exp\left\{-\frac{1}{2}\left|\frac{x-\mu}{\sigma}\right|^q\right\}

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = QExponential(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # q-exponentially distributed with loc=0, scale=1 and power=2
        tensor([ 0.1046])

    Args:
        loc (float or Tensor): mean of the distribution (often referred to as mu)
        scale (float or Tensor): standard deviation of the distribution
            (often referred to as sigma)
        power (float or Tensor): power of the distribution
    """
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive, "power": constraints.positive}
    support = constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.loc

    @property
    def mode(self):
        return self.loc

    @property
    def stddev(self):
        return self.scale

    @property
    def variance(self):
        return self.stddev.pow(2)

    @property
    def rescalor(self):
        return torch.exp((2./self.power*math.log(2) + torch.lgamma(0.5+2./self.power) - math.log(math.pi)/2.)/2.)

    def __init__(self, loc, scale, power=torch.tensor(2.0), validate_args=None):
        self.loc, self.scale = broadcast_all(loc, scale)
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        self.power = power
        super().__init__(batch_shape, validate_args=validate_args)

    def confidence(self, alpha=0.05):
        lower = self.icdf(torch.tensor(alpha/2))
        upper = self.icdf(torch.tensor(1-alpha/2))
        return lower, upper

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(QExponential, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.power = self.power
        super(QExponential, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size(), rescale=False):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            eps = Chi2(1).sample(shape).to(self.loc.device)**(1./self.power) * _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device).sign()
            if rescale: eps /= self.rescalor
            return self.loc.expand(shape) + eps * self.scale.expand(shape) 

    def rsample(self, sample_shape=torch.Size(), rescale=False):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        if self.power!=2: eps = eps.abs()**(2./self.power-1) * eps
        if rescale: eps /= self.rescalor
        return self.loc + eps * self.scale

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        log_scale = (
            math.log(self.scale) if isinstance(self.scale, Real) else self.scale.log()
        )
        scaled_diff = ((value - self.loc) / self.scale).abs()
        res = -.5* ( scaled_diff**self.power + math.log(2 * math.pi) ) - log_scale
        if self.power!=2: res += (self.power/2.-1)*scaled_diff.log() + torch.log(self.power/2.)
        return res

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        scaled_diff = (value - self.loc) * self.scale.reciprocal()
        if self.power!=2: scaled_diff *= scaled_diff.abs()**(self.power/2.-1)
        return 0.5 * (
            1 + torch.erf(scaled_diff / math.sqrt(2))
        )

    def icdf(self, value):
        erfinv = torch.erfinv(2 * value - 1) * math.sqrt(2)
        if self.power!=2: erfinv *= erfinv.abs()**(2./self.power-1)
        return self.loc + self.scale * erfinv

    def entropy(self, exact=False):
        res = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(self.scale)
        if self.power!=2: res += 0.5*(self.power/2.-1) *(2./self.power* Chi2(1).entropy() if exact else 0) - torch.log(self.power/2.)
        return res

    @property
    def _natural_params(self):
        if self.power!=2:
            raise ValueError(f"Q-Exponential distribution with power {self.power} does not belong to exponential family!")
        else:
            return (self.loc / self.scale.pow(2), -0.5 * self.scale.pow(2).reciprocal())

    def _log_normalizer(self, x, y):
        if self.power!=2:
            raise ValueError(f"Q-Exponential distribution with power {self.power} does not belong to exponential family!")
        else:
            return -0.25 * x.pow(2) / y + 0.5 * torch.log(-math.pi / y)


@register_kl(QExponential, QExponential)
def _kl_qexponential_qexponential(p, q, exact=False):
    var_ratio = (p.scale / q.scale).pow(2)
    t1 = ((p.loc - q.loc) / q.scale).pow(2)
    res = 0.5 * ((var_ratio + t1).pow(q.power/2.) - 1 - var_ratio.log())
    if q.power!=2: res += 0.5 * ( -(q.power/2.-1)*torch.log(var_ratio + t1) + (p.power/2.-1) * (-2./p.power*Chi2(1).entropy() if exact else 0) )
    return res