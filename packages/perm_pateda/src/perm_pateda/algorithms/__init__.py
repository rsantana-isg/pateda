"""
Plug-and-play permutation EDA wrappers::

    from perm_pateda import EHMEDA, MallowsKendallEDA, ...
"""

from perm_pateda.algorithms.permutation import (
    EHMEDA,
    NHMEDA,
    MallowsKendallEDA,
    MallowsCayleyEDA,
    GMallowsKendallEDA,
    GMallowsCayleyEDA,
)

__all__ = [
    "EHMEDA",
    "NHMEDA",
    "MallowsKendallEDA",
    "MallowsCayleyEDA",
    "GMallowsKendallEDA",
    "GMallowsCayleyEDA",
]
