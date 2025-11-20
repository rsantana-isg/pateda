"""Local optimization methods for EDAs"""

from pateda.local_optimization.scipy_local_search import ScipyLocalSearch
from pateda.local_optimization.greedy_search import GreedySearch
from pateda.local_optimization.contiguous_block_opt import ContiguousBlockOptimizer

__all__ = [
    "ScipyLocalSearch",
    "GreedySearch",
    "ContiguousBlockOptimizer",
]
