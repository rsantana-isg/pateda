"""
Univariate Marginal Distribution Algorithm (UMDA) learning

Equivalent to MATEDA's LearnUMDA.m
UMDA is a special case of FDA with univariate (independent) variables.
"""

from typing import Any
import numpy as np

from pateda.core.components import LearningMethod
from pateda.core.models import FactorizedModel


class LearnUMDA(LearningMethod):
    """
    Learn a UMDA (Univariate Marginal Distribution Algorithm) model

    UMDA represents the probability distribution as a product of independent
    univariate marginal distributions. This is the simplest EDA, assuming
    complete independence between variables.
    """

    def __init__(self, alpha: float = 0.0):
        """
        Initialize UMDA learning

        Args:
            alpha: Smoothing parameter (Laplace smoothing) to avoid zero probabilities
                  Default 0.0 means no smoothing
        """
        self.alpha = alpha

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
        Learn UMDA model from population

        Args:
            generation: Current generation number
            n_vars: Number of variables
            cardinality: Variable cardinalities
            population: Selected population to learn from
            fitness: Fitness values (not used for UMDA learning)
            **params: Additional parameters
                     - alpha: Override smoothing parameter

        Returns:
            Learned FactorizedModel with univariate structure
        """
        alpha = params.get("alpha", self.alpha)
        pop_size = population.shape[0]

        # Create univariate structure (each variable is independent)
        # Format: [n_overlap, n_new, overlap_indices..., new_indices...]
        # For univariate: [0, 1, var_index]
        cliques = np.zeros((n_vars, 3))
        cliques[:, 0] = 0  # No overlapping variables
        cliques[:, 1] = 1  # One new variable per clique
        cliques[:, 2] = np.arange(n_vars)  # Variable index

        # Learn marginal probabilities for each variable
        tables = []
        for var_idx in range(n_vars):
            k = int(cardinality[var_idx])  # Number of values this variable can take

            # Count occurrences of each value
            counts = np.zeros(k)
            for val in range(k):
                counts[val] = np.sum(population[:, var_idx] == val)

            # Apply Laplace smoothing if requested
            if alpha > 0:
                counts += alpha

            # Normalize to get probabilities
            probabilities = counts / np.sum(counts)

            tables.append(probabilities)

        # Create and return model
        model = FactorizedModel(
            structure=cliques,
            parameters=tables,
            metadata={
                "generation": generation,
                "model_type": "UMDA",
                "alpha": alpha,
            },
        )

        return model
