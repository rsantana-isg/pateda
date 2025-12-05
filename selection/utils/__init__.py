"""
Selection utilities

This module provides utility functions for selection methods,
particularly for multi-objective optimization.
"""

from pateda.selection.utils.pareto import (
    find_pareto_set,
    pareto_dominates,
    pareto_ranking,
    fitness_ranking,
)

__all__ = [
    "find_pareto_set",
    "pareto_dominates",
    "pareto_ranking",
    "fitness_ranking",
]
