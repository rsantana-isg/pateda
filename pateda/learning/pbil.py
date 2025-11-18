"""
Population-Based Incremental Learning (PBIL)

PBIL is an EDA that maintains a probability vector and updates it incrementally
based on the best solutions found. The probability vector is updated using a
learning rate (alpha) that controls how much the new population influences the
probability distribution.

References:
    - Baluja, S. (1994). Population-based incremental learning: A method for
      integrating genetic search based function optimization and competitive learning.
      CMU-CS-94-163, Carnegie Mellon University.
    - Baluja, S., & Caruana, R. (1995). Removing the genetics from the standard
      genetic algorithm. In ICML (Vol. 95, pp. 38-46).
"""

from typing import Any, Optional
import numpy as np

from pateda.core.components import LearningMethod
from pateda.core.models import FactorizedModel


class LearnPBIL(LearningMethod):
    """
    Learn a PBIL (Population-Based Incremental Learning) model

    PBIL maintains a probability vector that is incrementally updated based on
    selected individuals. Unlike UMDA which completely replaces the probability
    distribution, PBIL uses a learning rate to smoothly update probabilities:

    P_new(X_i = k) = (1 - alpha) * P_old(X_i = k) + alpha * freq(X_i = k)

    This incremental approach provides more stability and can help maintain
    diversity in the population.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        initial_prob: Optional[np.ndarray] = None,
        mutation_prob: float = 0.0,
        mutation_shift: float = 0.05,
    ):
        """
        Initialize PBIL learning

        Args:
            alpha: Learning rate (default: 0.1). Controls how much the new
                  population influences the probability vector. Higher values
                  lead to faster adaptation but less stability.
            initial_prob: Initial probability vector. If None, starts with
                         uniform distribution (0.5 for binary variables).
            mutation_prob: Probability of mutating each probability value (default: 0.0)
            mutation_shift: Amount to shift probabilities during mutation (default: 0.05)
        """
        self.alpha = alpha
        self.initial_prob = initial_prob
        self.mutation_prob = mutation_prob
        self.mutation_shift = mutation_shift
        self._probability_vector = None

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
        Learn PBIL model from population

        Args:
            generation: Current generation number
            n_vars: Number of variables
            cardinality: Variable cardinalities
            population: Selected population to learn from
            fitness: Fitness values (not used for standard PBIL learning)
            **params: Additional parameters
                     - alpha: Override learning rate
                     - mutation_prob: Override mutation probability
                     - mutation_shift: Override mutation shift amount
                     - reset: If True, reset the probability vector (default: False)

        Returns:
            Learned FactorizedModel with univariate structure
        """
        alpha = params.get("alpha", self.alpha)
        mutation_prob = params.get("mutation_prob", self.mutation_prob)
        mutation_shift = params.get("mutation_shift", self.mutation_shift)
        reset = params.get("reset", False)

        pop_size = population.shape[0]

        # Initialize probability vector on first generation or if reset is requested
        if self._probability_vector is None or reset or generation == 0:
            if self.initial_prob is not None:
                self._probability_vector = [
                    self.initial_prob[i].copy() for i in range(n_vars)
                ]
            else:
                # Initialize with uniform distribution
                self._probability_vector = [
                    np.ones(int(cardinality[i])) / int(cardinality[i])
                    for i in range(n_vars)
                ]

        # Create univariate structure (each variable is independent)
        cliques = np.zeros((n_vars, 3))
        cliques[:, 0] = 0  # No overlapping variables
        cliques[:, 1] = 1  # One new variable per clique
        cliques[:, 2] = np.arange(n_vars)  # Variable index

        # Learn/update marginal probabilities for each variable
        tables = []
        for var_idx in range(n_vars):
            k = int(cardinality[var_idx])

            # Count occurrences of each value in selected population
            counts = np.zeros(k)
            for val in range(k):
                counts[val] = np.sum(population[:, var_idx] == val)

            # Calculate frequency
            freq = counts / pop_size

            # Update probability vector using PBIL rule
            # P_new = (1 - alpha) * P_old + alpha * freq
            updated_prob = (1 - alpha) * self._probability_vector[var_idx] + alpha * freq

            # Apply mutation if specified
            if mutation_prob > 0:
                for val in range(k):
                    if np.random.random() < mutation_prob:
                        # Mutate this probability
                        direction = 1 if np.random.random() < 0.5 else -1
                        updated_prob[val] += direction * mutation_shift
                        # Clip to valid range
                        updated_prob[val] = np.clip(updated_prob[val], 0.0, 1.0)

            # Normalize to ensure it's a valid probability distribution
            updated_prob = updated_prob / np.sum(updated_prob)

            # Store updated probability vector
            self._probability_vector[var_idx] = updated_prob.copy()

            tables.append(updated_prob)

        # Create and return model
        model = FactorizedModel(
            structure=cliques,
            parameters=tables,
            metadata={
                "generation": generation,
                "model_type": "PBIL",
                "alpha": alpha,
                "mutation_prob": mutation_prob,
                "mutation_shift": mutation_shift,
            },
        )

        return model

    def reset(self):
        """Reset the internal probability vector"""
        self._probability_vector = None
