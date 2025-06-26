#!/usr/bin/env python3

try:
    from ._pyro_mixin import _PyroMixin
    from gpytorch.models.pyro.pyro_gp import PyroGP
    from .pyro_qep import PyroQEP
except ImportError:

    class PyroGP(object):
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Cannot use a PyroGP because you dont have Pyro installed.")

    class PyroQEP(object):
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Cannot use a PyroQEP because you dont have Pyro installed.")

    class _PyroMixin(object):
        def pyro_factors(self, *args, **kwargs):
            raise RuntimeError("Cannot call `pyro_factors` because you dont have Pyro installed.")

        def pyro_guide(self, *args, **kwargs):
            raise RuntimeError("Cannot call `pyro_sample` because you dont have Pyro installed.")

        def pyro_model(self, *args, **kwargs):
            raise RuntimeError("Cannot call `pyro_sample` because you dont have Pyro installed.")


__all__ = ["PyroGP", "_PyroMixin", "PyroQEP"]
