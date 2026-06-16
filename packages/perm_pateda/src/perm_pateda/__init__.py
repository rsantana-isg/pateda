"""
perm_pateda - Estimation of Distribution Algorithms for permutation problems.

An independent, permutation-focused companion to :mod:`pateda`.  It contributes
the learning and sampling methods specific to permutation spaces (histogram
models and distance-based Mallows / Generalized Mallows models) and reuses
``pateda`` for everything common to all EDAs (the core EDA engine, selection,
replacement, statistics and visualization utilities).

A Python port of the algorithms in ``perm_mateda`` (Irurozki, Ceberio,
Santamaria & Mendiburu, 2018, "Algorithm 989: perm_mateda").

Quick start::

    from perm_pateda import MallowsKendallEDA
    from perm_pateda.functions import create_random_lop
    import numpy as np

    lop = create_random_lop(15, seed=0)
    alg = MallowsKendallEDA(n_vars=15, fitness_func=lop.evaluate,
                            pop_size=100, n_gen=50, random_seed=0)
    stats, _ = alg.run()
    print("Best:", stats.best_fitness_overall)
"""

__version__ = "0.1.0"
__author__ = "Roberto Santana (roberto.santana@ehu.eus)"

# Plug-and-play permutation EDA wrappers
from perm_pateda.algorithms.permutation import (
    EHMEDA,
    NHMEDA,
    MallowsKendallEDA,
    MallowsCayleyEDA,
    GMallowsKendallEDA,
    GMallowsCayleyEDA,
)

# Permutation distance metrics and consensus (central-permutation) estimators
from perm_pateda.distances import (
    kendall_distance,
    cayley_distance,
    ulam_distance,
)
from perm_pateda.consensus import (
    find_consensus_borda,
    find_consensus_median,
)

__all__ = [
    # Algorithms
    "EHMEDA",
    "NHMEDA",
    "MallowsKendallEDA",
    "MallowsCayleyEDA",
    "GMallowsKendallEDA",
    "GMallowsCayleyEDA",
    # Distances
    "kendall_distance",
    "cayley_distance",
    "ulam_distance",
    # Consensus
    "find_consensus_borda",
    "find_consensus_median",
]
