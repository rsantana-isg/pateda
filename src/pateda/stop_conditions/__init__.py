"""Stopping conditions"""

from pateda.stop_conditions.max_generations import MaxGenerations
from pateda.stop_conditions.max_generations_or_optimum import MaxGenerationsOrOptimum

__all__ = ["MaxGenerations", "MaxGenerationsOrOptimum"]
