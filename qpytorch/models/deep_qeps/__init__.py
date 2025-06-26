#!/usr/bin/env python3

import warnings

from .deep_qep import DeepQEP, DeepQEPLayer, DeepLikelihood


# Deprecated for 1.0 release
class AbstractDeepQEP(DeepQEP):
    def __init__(self, *args, **kwargs):
        warnings.warn("AbstractDeepQEP has been renamed to DeepQEP.", DeprecationWarning)
        super().__init__(*args, **kwargs)


# Deprecated for 1.0 release
class AbstractDeepQEPLayer(DeepQEPLayer):
    def __init__(self, *args, **kwargs):
        warnings.warn("AbstractDeepQEPLayer has been renamed to DeepQEPLayer.", DeprecationWarning)
        super().__init__(*args, **kwargs)


__all__ = ["DeepQEPLayer", "DeepQEP", "AbstractDeepQEPLayer", "AbstractDeepQEP", "DeepLikelihood"]
