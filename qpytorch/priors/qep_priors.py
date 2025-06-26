#!/usr/bin/env python3

import torch
from torch.nn import Module as TModule
from linear_operator import to_linear_operator

from ..distributions import QExponential, MultivariateQExponential
from gpytorch.priors.prior import Prior
from gpytorch.priors.utils import _bufferize_attributes, _del_attributes

QEP_LAZY_PROPERTIES = ("covariance_matrix",)


class QExponentialPrior(Prior, QExponential):
    """
    QExponential Prior

    pdf(x) = q/2 * (2 * pi * sigma^2)^-0.5 * |(x - mu)/sigma|^(q/2-1) * exp(-0.5*|(x - mu)/sigma|^q)

    where mu is the mean and sigma^2 is the variance.
    """

    def __init__(self, loc, scale, power=torch.tensor(1.0), validate_args=False, transform=None):
        TModule.__init__(self)
        QExponential.__init__(self, loc=loc, scale=scale, power=power, validate_args=validate_args)
        _bufferize_attributes(self, ("loc", "scale"))
        self._transform = transform

    def expand(self, batch_shape):
        batch_shape = torch.Size(batch_shape)
        return QExponentialPrior(self.loc.expand(batch_shape), self.scale.expand(batch_shape), self.power)


class MultivariateQExponentialPrior(Prior, MultivariateQExponential):
    """Multivariate Q-Exponential prior

    pdf(x) = q/2 * det(2 * pi * Sigma)^-0.5 * r^((q/2-1)*d/2) * exp(-0.5 * r^(q/2)), r = (x - mu)' Sigma^-1 (x - mu)

    where mu is the mean and Sigma > 0 is the covariance matrix.
    """

    def __init__(
        self, mean, covariance_matrix, power=torch.tensor(1.0), validate_args=False, transform=None
    ):
        TModule.__init__(self)
        MultivariateQExponential.__init__(
            self,
            mean=mean,
            covariance_matrix=covariance_matrix,
            power=power,
            validate_args=validate_args,
        )
        _bufferize_attributes(self, ("loc",))
        self._transform = transform

    def cuda(self, device=None):
        """Applies module-level cuda() call and resets all lazy properties"""
        module = self._apply(lambda t: t.cuda(device))
        _del_attributes(module, QEP_LAZY_PROPERTIES)
        return module

    def cpu(self):
        """Applies module-level cpu() call and resets all lazy properties"""
        module = self._apply(lambda t: t.cpu())
        _del_attributes(module, QEP_LAZY_PROPERTIES)
        return module

    @property
    def lazy_covariance_matrix(self):
        if self.islazy:
            return self._covar
        else:
            return to_linear_operator(super().covariance_matrix)

    def expand(self, batch_shape):
        batch_shape = torch.Size(batch_shape)
        cov_shape = batch_shape + self.event_shape
        new_loc = self.loc.expand(batch_shape)
        new_covar = self._covar.expand(cov_shape)

        return MultivariateQExponentialPrior(mean=new_loc, covariance_matrix=new_covar, power=self.power)