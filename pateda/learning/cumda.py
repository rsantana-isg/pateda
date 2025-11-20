"""
Constraint Univariate Marginal Distribution Algorithm (CUMDA) learning

CUMDA is a modification of UMDA designed to deal with binary constraint problems
where the number of ones (unitation value) must be within specific bounds:
    1 ≤ a < u(x) ≤ b < n

where u(x) is the unitation function (sum of ones in the solution).

This algorithm was introduced by Santana and Ochoa in "A Constraint Univariate
Marginal Distribution Algorithm" to handle constraint optimization problems while
searching only in the space defined by feasible solutions.

Key Differences from UMDA:
- UMDA samples each variable independently from marginal probabilities
- CUMDA samples a fixed number r of variables to set to 1 (rest set to 0)
- Sampling uses normalized probabilities without replacement
- Uses Stochastic Universal Sampling (SUS) for low variance

Mathematical representation:
    Learning: p^s_i(x_i=1,t) - marginal frequency of x_i=1 in selected set
    Normalization: q_i = p^s_i(x_i=1,t) / S, where S = Σ p^s_i(x_i=1,t)
    Sampling: Select r variables with probabilities q_i (without replacement)

Algorithm Steps:
1. Select k ≤ N points according to selection method
2. Compute marginal frequencies p^s_i(x_i=1,t) of variables set to 1
3. Normalize: q_i = p^s_i(x_i=1,t) / sum(p^s_j)
4. For each new individual: sample r variables without replacement using q_i
5. Set sampled variables to 1, rest to 0

The Stochastic Universal Sampling (SUS) approach:
- Has zero bias: expected q_i matches algorithm sampling frequency
- Has minimal spread: range of possible values is {⌊q_i⌋, ⌈q_i⌉}
- Much smaller variance than roulette wheel selection

References:
- Santana, R., & Ochoa, A. "A Constraint Univariate Marginal Distribution Algorithm."
  Institute of Cybernetics, Mathematics and Physics.
- Baker, J. E. (1987). "Reducing bias and inefficiency in the selection algorithm."
  Proceedings of the Second International Conference on Genetic Algorithms, pp. 14-21.
"""

from typing import Any
import numpy as np

from pateda.core.components import LearningMethod
from pateda.core.models import FactorizedModel


class LearnCUMDA(LearningMethod):
    """
    Learn a CUMDA (Constraint Univariate Marginal Distribution Algorithm) model

    CUMDA learns marginal probabilities like UMDA, but stores them in a way suitable
    for constraint-aware sampling. The model stores the probability that each variable
    should be set to 1, which will later be normalized and used for sampling without
    replacement.

    The learning produces a univariate model structure (like UMDA) but with metadata
    indicating it's for constraint handling.

    Attributes:
        alpha: Smoothing parameter for Laplace smoothing (default 0.0)
    """

    def __init__(self, alpha: float = 0.0):
        """
        Initialize CUMDA learning

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
        Learn CUMDA model from population

        Computes the marginal frequencies p^s_i(x_i=1,t) for each variable,
        which represent how often each variable is set to 1 in the selected population.

        Args:
            generation: Current generation number
            n_vars: Number of variables
            cardinality: Variable cardinalities (should be all 2 for binary)
            population: Selected population to learn from (pop_size, n_vars)
            fitness: Fitness values (not used for CUMDA learning)
            **params: Additional parameters
                     - alpha: Override smoothing parameter

        Returns:
            Learned FactorizedModel with univariate structure and marginal probabilities

        Note:
            For binary variables, we only need to store p(x_i=1). The probability
            p(x_i=0) = 1 - p(x_i=1). However, for consistency with standard FDA
            representation, we store both probabilities.
        """
        alpha = params.get("alpha", self.alpha)
        pop_size = population.shape[0]

        # Verify binary variables
        if not np.all(cardinality == 2):
            raise ValueError("CUMDA only works with binary variables (cardinality=2)")

        # Create univariate structure (each variable is independent in structure)
        # Format: [n_overlap, n_new, overlap_indices..., new_indices...]
        # For univariate: [0, 1, var_index]
        cliques = np.zeros((n_vars, 3))
        cliques[:, 0] = 0  # No overlapping variables
        cliques[:, 1] = 1  # One new variable per clique
        cliques[:, 2] = np.arange(n_vars)  # Variable index

        # Learn marginal probabilities for each variable
        # For CUMDA, we need P(x_i = 1) to use in sampling
        tables = []
        for var_idx in range(n_vars):
            # Count occurrences of each value (0 and 1)
            count_0 = np.sum(population[:, var_idx] == 0)
            count_1 = np.sum(population[:, var_idx] == 1)

            # Apply Laplace smoothing if requested
            if alpha > 0:
                count_0 += alpha
                count_1 += alpha

            # Normalize to get probabilities
            total = count_0 + count_1
            p_0 = count_0 / total
            p_1 = count_1 / total

            # Store as [p(x_i=0), p(x_i=1)]
            probabilities = np.array([p_0, p_1])
            tables.append(probabilities)

        # Create and return model
        model = FactorizedModel(
            structure=cliques,
            parameters=tables,
            metadata={
                "generation": generation,
                "model_type": "CUMDA",
                "alpha": alpha,
                "constraint_type": "unitation",
            },
        )

        return model
