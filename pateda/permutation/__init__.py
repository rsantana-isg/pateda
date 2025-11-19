"""Permutation-based probability models for EDAs

This module implements permutation-based probability distributions for
Estimation of Distribution Algorithms (EDAs), including:

- Distance metrics (Kendall, Cayley, Ulam)
- Mallows models with different distance metrics (Kendall, Cayley)
- Generalized Mallows models (in development)
- Histogram models (Edge Histogram Model, Node Histogram Model)
"""

from pateda.permutation.distances import (
    kendall_distance,
    cayley_distance,
    ulam_distance,
)

from pateda.permutation.consensus import (
    find_consensus_borda,
    find_consensus_median,
)

__all__ = [
    "kendall_distance",
    "cayley_distance",
    "ulam_distance",
    "find_consensus_borda",
    "find_consensus_median",
]
