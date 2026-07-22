"""Stopping conditions"""

from pateda.stop_conditions.max_generations import MaxGenerations
from pateda.stop_conditions.max_generations_or_optimum import MaxGenerationsOrOptimum
from pateda.stop_conditions.no_improvement import NoImprovement
from pateda.stop_conditions.convergence import PopulationConvergence
from pateda.stop_conditions.composite import CompositeStop

__all__ = [
    "MaxGenerations",
    "MaxGenerationsOrOptimum",
    "NoImprovement",
    "PopulationConvergence",
    "CompositeStop",
]
