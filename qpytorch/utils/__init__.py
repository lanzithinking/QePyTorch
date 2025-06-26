#!/usr/bin/env python3

from __future__ import annotations

import warnings as _warnings
from typing import Any

import linear_operator

from gpytorch.utils import deprecation, errors, generic, grid, interpolation, quadrature, transforms
from . import warnings
from gpytorch.utils.memoize import cached
from gpytorch.utils.nearest_neighbors import NNUtil
from gpytorch.utils.sum_interaction_terms import sum_interaction_terms

__all__ = [
    "cached",
    "deprecation",
    "errors",
    "generic",
    "grid",
    "interpolation",
    "quadrature",
    "sum_interaction_terms",
    "transforms",
    "warnings",
    "NNUtil",
]


def __getattr__(name: str) -> Any:
    if hasattr(linear_operator.utils, name):
        _warnings.warn(
            f"gpytorch.utils.{name} is deprecated. Use linear_operator.utils.{name} instead.",
            DeprecationWarning,
        )
        return getattr(linear_operator.utils, name)
    raise AttributeError(f"module gpytorch.utils has no attribute {name}")
