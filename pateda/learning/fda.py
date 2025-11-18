"""
Factorized Distribution Algorithm (FDA) learning

Equivalent to MATEDA's LearnFDA.m
"""

from typing import Any
import numpy as np

from pateda.core.components import LearningMethod
from pateda.core.models import FactorizedModel
from pateda.learning.utils.marginal_prob import learn_fda_parameters


class LearnFDA(LearningMethod):
    """
    Learn a Factorized Distribution Algorithm (FDA) model

    FDA represents the probability distribution as a product of factors (cliques).
    For UMDA (univariate case), each variable is independent (single-variable cliques).
    """

    def __init__(self, cliques: np.ndarray = None):
        """
        Initialize FDA learning

        Args:
            cliques: Clique structure matrix. If None, creates univariate structure
                    Each row: [n_overlap, n_new, overlap_indices..., new_indices...]
                    For UMDA: Each row is [0, 1, -, var_index] (no overlaps)
        """
        self.cliques = cliques

    def learn(
        self,
        generation: int,
        n_vars: int,
        cardinality: np.ndarray,
        population: np.ndarray,
        fitness: np.ndarray,
        **params: Any,
    ) -> FactorizedModel:
        """
        Learn FDA model from population

        Args:
            generation: Current generation number
            n_vars: Number of variables
            cardinality: Variable cardinalities
            population: Selected population to learn from
            fitness: Fitness values (not used for FDA learning)
            **params: Additional parameters

        Returns:
            Learned FactorizedModel
        """
        # Get or create clique structure
        if self.cliques is not None:
            cliques = self.cliques
        else:
            # Create univariate structure (UMDA)
            # Each variable is independent: [0, 1, var_index]
            cliques = np.zeros((n_vars, 3))
            cliques[:, 0] = 0  # No overlapping variables
            cliques[:, 1] = 1  # One new variable per clique
            cliques[:, 2] = np.arange(n_vars)  # Variable index

        # Learn probability tables for each clique
        tables = learn_fda_parameters(cliques, population, n_vars, cardinality)

        # Create and return model
        model = FactorizedModel(
            structure=cliques,
            parameters=tables,
            metadata={"generation": generation, "model_type": "FDA"},
        )

        return model
