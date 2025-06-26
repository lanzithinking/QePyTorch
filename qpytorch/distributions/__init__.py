#!/usr/bin/env python3

from .delta import Delta
from gpytorch.distributions.distribution import Distribution
from .qexponential import QExponential
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.distributions.multitask_multivariate_normal import MultitaskMultivariateNormal
from .multivariate_qexponential import MultivariateQExponential
from .multitask_multivariate_qexponential import MultitaskMultivariateQExponential
from .power import Power

# Get the set of distributions from either PyTorch or Pyro
try:
    # If pyro is installed, use that set of base distributions
    import pyro.distributions as base_distributions
except ImportError:
    # Otherwise, use PyTorch
    import torch.distributions as base_distributions


__all__ = ["Delta", "QExponential", "Distribution", "MultivariateNormal", "MultitaskMultivariateNormal", "MultivariateQExponential", "MultitaskMultivariateQExponential", "Power", "base_distributions"]
