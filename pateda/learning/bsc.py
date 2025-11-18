"""
Bisection (BSC) Learning

BSC is an EDA variant that uses fitness values to weight probability estimates
rather than using simple frequency counts. Each value's probability is proportional
to the sum of fitness values of individuals having that value.

This approach biases the probability distribution towards values that appear in
high-fitness individuals, potentially leading to faster convergence.

References:
    - Based on the implementation in MATEDA-1.0 (EDA.cpp, BayesianNetwork.cpp)
    - Inza, I., Larranaga, P., Etxeberria, R., & Sierra, B. (2000).
      Feature subset selection by Bayesian network-based optimization.
      Artificial Intelligence, 123(1-2), 157-184.
"""

from typing import Any
import numpy as np

from pateda.core.components import LearningMethod
from pateda.core.models import FactorizedModel


class LearnBSC(LearningMethod):
    """
    Learn a BSC (Bisection) model

    BSC estimates probabilities based on fitness values rather than frequencies.
    For each variable value, the probability is calculated as:

    P(X_i = k) = sum(fitness of individuals with X_i = k) / sum(all fitness)

    This makes the probability distribution biased towards values that appear
    in high-fitness individuals, which can lead to faster convergence but may
    reduce diversity.
    """

    def __init__(self, alpha: float = 0.0, normalize_fitness: bool = True):
        """
        Initialize BSC learning

        Args:
            alpha: Smoothing parameter (Laplace smoothing) to avoid zero probabilities
                  Default 0.0 means no smoothing
            normalize_fitness: If True, normalize fitness values to [0, 1] range
                              This can help with numerical stability.
        """
        self.alpha = alpha
        self.normalize_fitness = normalize_fitness

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
        Learn BSC model from population

        Args:
            generation: Current generation number
            n_vars: Number of variables
            cardinality: Variable cardinalities
            population: Selected population to learn from
            fitness: Fitness values used to weight probabilities
            **params: Additional parameters
                     - alpha: Override smoothing parameter
                     - normalize_fitness: Override fitness normalization flag

        Returns:
            Learned FactorizedModel with univariate structure

        Raises:
            ValueError: If fitness array is not provided or has wrong shape
        """
        if fitness is None:
            raise ValueError("BSC requires fitness values to be provided")

        if len(fitness.shape) != 1 or fitness.shape[0] != population.shape[0]:
            raise ValueError(
                f"Fitness array must be 1D with length equal to population size. "
                f"Got shape {fitness.shape}, expected ({population.shape[0]},)"
            )

        alpha = params.get("alpha", self.alpha)
        normalize_fitness = params.get("normalize_fitness", self.normalize_fitness)

        pop_size = population.shape[0]

        # Normalize fitness values if requested
        if normalize_fitness:
            # Handle the case where all fitness values are the same
            fitness_range = np.max(fitness) - np.min(fitness)
            if fitness_range > 1e-10:
                fitness_normalized = (fitness - np.min(fitness)) / fitness_range
            else:
                # If all fitness values are the same, use uniform weights
                fitness_normalized = np.ones_like(fitness) / pop_size
        else:
            fitness_normalized = fitness.copy()

        # Ensure all fitness values are non-negative
        if np.any(fitness_normalized < 0):
            # Shift to make all values non-negative
            fitness_normalized = fitness_normalized - np.min(fitness_normalized)

        # Calculate total fitness
        total_fitness = np.sum(fitness_normalized)

        # Avoid division by zero
        if total_fitness < 1e-10:
            # Fall back to uniform probabilities
            fitness_normalized = np.ones_like(fitness) / pop_size
            total_fitness = 1.0

        # Create univariate structure (each variable is independent)
        cliques = np.zeros((n_vars, 3))
        cliques[:, 0] = 0  # No overlapping variables
        cliques[:, 1] = 1  # One new variable per clique
        cliques[:, 2] = np.arange(n_vars)  # Variable index

        # Learn fitness-weighted probabilities for each variable
        tables = []
        for var_idx in range(n_vars):
            k = int(cardinality[var_idx])

            # Calculate fitness-weighted values for each possible value
            values = np.zeros(k)
            for val in range(k):
                # Sum fitness of all individuals with this value
                mask = population[:, var_idx] == val
                values[val] = np.sum(fitness_normalized[mask])

            # Apply Laplace smoothing if requested
            if alpha > 0:
                values += alpha

            # Normalize to get probabilities
            probabilities = values / np.sum(values)

            tables.append(probabilities)

        # Create and return model
        model = FactorizedModel(
            structure=cliques,
            parameters=tables,
            metadata={
                "generation": generation,
                "model_type": "BSC",
                "alpha": alpha,
                "normalize_fitness": normalize_fitness,
            },
        )

        return model
