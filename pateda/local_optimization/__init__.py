"""Local optimization methods for EDAs"""

from pateda.local_optimization.scipy_local_search import ScipyLocalSearch
from pateda.local_optimization.greedy_search import GreedySearch

__all__ = [
    "ScipyLocalSearch",
    "GreedySearch",
]
