"""Local optimization methods for EDAs"""

from pateda.local_optimization.scipy_local_search import ScipyLocalSearch
from pateda.local_optimization.greedy_search import GreedySearch
from pateda.local_optimization.contiguous_block_opt import ContiguousBlockOptimizer
from pateda.local_optimization.discrete_greedy_search import DiscreteGreedySearch
from pateda.local_optimization.discrete_simulated_annealing import (
    DiscreteSimulatedAnnealing,
    DiscreteSimulatedAnnealingLinear,
)

# Budget/subset-aware memetic local searches (shared interface).
from pateda.local_optimization.budgeted_search import BudgetedLocalSearch
from pateda.local_optimization.hill_climbing import (
    DeterministicHillClimber,
    FirstImprovementHillClimber,
    StochasticHillClimber,
)
from pateda.local_optimization.simulated_annealing import SimulatedAnnealing
from pateda.local_optimization.variable_neighborhood_search import (
    VariableNeighborhoodSearch,
    ReducedVariableNeighborhoodSearch,
)
from pateda.local_optimization.substructural_search import SubstructuralLocalSearch

__all__ = [
    "ScipyLocalSearch",
    "GreedySearch",
    "ContiguousBlockOptimizer",
    "DiscreteGreedySearch",
    "DiscreteSimulatedAnnealing",
    "DiscreteSimulatedAnnealingLinear",
    # Budget/subset-aware memetic local searches
    "BudgetedLocalSearch",
    "DeterministicHillClimber",
    "FirstImprovementHillClimber",
    "StochasticHillClimber",
    "SimulatedAnnealing",
    "VariableNeighborhoodSearch",
    "ReducedVariableNeighborhoodSearch",
    "SubstructuralLocalSearch",
]
