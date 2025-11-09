#!/usr/bin/env python3

import warnings

from gpytorch.models import deep_gps, gplvm
from . import deep_qeps, exact_prediction_strategies, qeplvm, pyro
from gpytorch.models.approximate_gp import ApproximateGP
from gpytorch.models.exact_gp import ExactGP
from gpytorch.models.gp import GP
from .approximate_qep import ApproximateQEP
from .exact_qep import ExactQEP
from .qep import QEP
from .model_list import AbstractModelList, IndependentModelList, UncorrelatedModelList
from .pyro import PyroGP, PyroQEP
from .pde_solver import PDESolver

# Alternative name for ApproximateGP, ApproximateQEP
VariationalGP = ApproximateGP
VariationalQEP = ApproximateQEP


# Deprecated for 0.4 release
class AbstractVariationalGP(ApproximateGP):
    # Remove after 1.0
    def __init__(self, *args, **kwargs):
        warnings.warn("AbstractVariationalGP has been renamed to ApproximateGP.", DeprecationWarning)
        super().__init__(*args, **kwargs)


# Deprecated for 0.4 release
class PyroVariationalGP(ApproximateGP):
    # Remove after 1.0
    def __init__(self, *args, **kwargs):
        warnings.warn("PyroVariationalGP has been renamed to PyroGP.", DeprecationWarning)
        super().__init__(*args, **kwargs)


# Deprecated for 0.4 release
class AbstractVariationalQEP(ApproximateQEP):
    # Remove after 1.0
    def __init__(self, *args, **kwargs):
        warnings.warn("AbstractVariationalQEP has been renamed to ApproximateQEP.", DeprecationWarning)
        super().__init__(*args, **kwargs)


# Deprecated for 0.4 release
class PyroVariationalQEP(ApproximateQEP):
    # Remove after 1.0
    def __init__(self, *args, **kwargs):
        warnings.warn("PyroVariationalQEP has been renamed to PyroQEP.", DeprecationWarning)
        super().__init__(*args, **kwargs)

__all__ = [
    "AbstractModelList",
    "ApproximateGP",
    "ApproximateQEP",
    "ExactGP",
    "ExactQEP",
    "GP",
    "QEP",
    "IndependentModelList",
    "PyroGP",
    "PyroQEP",
    "UncorrelatedModelList",
    "VariationalGP",
    "VariationalQEP",
    "deep_gps",
    "deep_qeps",
    "gplvm",
    "qeplvm",
    "exact_prediction_strategies",
    "pyro",
    "PDESolver",
]
