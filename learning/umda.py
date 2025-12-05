"""
Univariate Marginal Distribution Algorithm (UMDA) learning

UMDA is one of the first and simplest Estimation of Distribution Algorithms (EDAs),
introduced by Mühlenbein and Paass (1996). It models the probability distribution
of promising solutions as a product of independent univariate marginal distributions.

Mathematical representation:
    p(x) = ∏ᵢ₌₁ⁿ p(xᵢ)

where p(xᵢ) is the marginal probability distribution for variable i, estimated
from the selected population by counting the frequency of each value.

Key characteristics:
- Assumes complete independence between all variables
- Simplest factorization: each variable forms its own factor (clique)
- Each marginal distribution p(xᵢ) is a simple frequency table
- Suitable for problems with weak or no variable interactions
- Serves as a baseline for more sophisticated EDAs

The independence assumption makes UMDA computationally efficient but limits its
ability to model problems with strong variable dependencies. For such problems,
more advanced factorizations (Bayesian networks, Markov networks) are needed.

References:
- Mühlenbein, H., & Paass, G. (1996). "From recombination of genes to the
  estimation of distributions I. Binary parameters." Parallel Problem Solving
  from Nature, PPSN IV, 178-187.
- Larrañaga, P., & Lozano, J. A. (Eds.). (2002). "Estimation of Distribution
  Algorithms: A New Tool for Evolutionary Computation." Kluwer Academic Publishers.
- MATEDA-2.0 User Guide, Section 4.1: "Factorized distributions"

Equivalent to MATEDA's LearnUMDA.m
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

    Algorithm:
    1. For each variable i in {1, ..., n}:
       a. Count the frequency of each value k in the selected population
       b. Estimate p(xᵢ = k) = count(xᵢ = k) / N
       c. Optionally apply Laplace smoothing to avoid zero probabilities

    In MATEDA-2.0's factorization framework, UMDA creates a clique structure
    where each variable forms its own independent factor:
        Cliques[i] = [0, 1, i]  for i = 0, ..., n-1

    This indicates:
    - 0 overlapping variables (no dependencies)
    - 1 new variable (the variable itself)
    - Variable index i

    Laplace smoothing (alpha parameter):
    To avoid zero probabilities which can cause issues during sampling,
    a small count (alpha) can be added to each value:
        p(xᵢ = k) = (count(xᵢ = k) + alpha) / (N + alpha * |Vᵢ|)
    where |Vᵢ| is the cardinality (number of possible values) of variable i.
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
