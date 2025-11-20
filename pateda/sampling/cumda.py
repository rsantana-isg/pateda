"""
Constraint Univariate Marginal Distribution Algorithm (CUMDA) sampling

Implements sampling for CUMDA models with unitation constraints.
Unlike standard UMDA which samples each variable independently, CUMDA samples
a fixed number of variables to set to 1, ensuring solutions satisfy the constraint.

Sampling Algorithm:
1. Extract marginal probabilities p^s_i(x_i=1,t) from the model
2. Normalize: q_i = p^s_i(x_i=1,t) / Σ_j p^s_j(x_j=1,t)
3. For each individual:
   a. Sample r variables without replacement using probabilities q_i
   b. Set sampled variables to 1, rest to 0

The key innovation is using Stochastic Universal Sampling (SUS) for selecting
which variables to set to 1. SUS has several advantages:
- Zero bias: expected sampling frequency matches q_i exactly
- Minimal spread: number of times variable i is set is in {⌊q_i⌋, ⌈q_i⌉}
- Lower variance than roulette wheel selection
- Ensures the sample closely matches the probability distribution

References:
- Santana, R., & Ochoa, A. "A Constraint Univariate Marginal Distribution Algorithm."
- Baker, J. E. (1987). "Reducing bias and inefficiency in the selection algorithm."
"""

from typing import Any, Optional
import numpy as np

from pateda.core.components import SamplingMethod
from pateda.core.models import Model, FactorizedModel
from pateda.sampling.utils import stochastic_universal_sampling


class SampleCUMDA(SamplingMethod):
    """
    Sample population from a CUMDA model with unitation constraint

    Generates binary solutions where exactly r variables are set to 1,
    using Stochastic Universal Sampling for low variance.

    Attributes:
        n_samples: Number of individuals to sample
        n_ones: Number of ones each solution should have (r in the paper)
    """

    def __init__(self, n_samples: int, n_ones: int):
        """
        Initialize CUMDA sampling

        Args:
            n_samples: Number of individuals to sample
            n_ones: Number of variables to set to 1 in each solution (unitation value)
        """
        self.n_samples = n_samples
        self.n_ones = n_ones

    def sample(
        self,
        n_vars: int,
        model: Model,
        cardinality: np.ndarray,
        aux_pop: Optional[np.ndarray] = None,
        aux_fitness: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
        **params: Any,
    ) -> np.ndarray:
        """
        Sample new population from CUMDA model with unitation constraint

        Args:
            n_vars: Number of variables
            model: FactorizedModel learned by CUMDA
            cardinality: Variable cardinalities (should be all 2)
            aux_pop: Auxiliary population (not used)
            aux_fitness: Auxiliary fitness (not used)
            rng: Random number generator (optional)
            **params: Additional parameters
                     - n_samples: Override instance n_samples
                     - n_ones: Override instance n_ones

        Returns:
            Sampled population (n_samples, n_vars) where each row has exactly n_ones ones

        Raises:
            TypeError: If model is not a FactorizedModel
            ValueError: If cardinalities are not binary or n_ones is invalid
        """
        if rng is None:
            rng = np.random.default_rng()

        if not isinstance(model, FactorizedModel):
            raise TypeError(f"Expected FactorizedModel, got {type(model)}")

        # Verify binary variables
        if not np.all(cardinality == 2):
            raise ValueError("CUMDA only works with binary variables (cardinality=2)")

        # Get parameters
        n_samples = params.get("n_samples", self.n_samples)
        n_ones = params.get("n_ones", self.n_ones)

        # Validate n_ones
        if n_ones < 0 or n_ones > n_vars:
            raise ValueError(f"n_ones ({n_ones}) must be between 0 and {n_vars}")

        # Extract marginal probabilities p(x_i = 1) from the model
        tables = model.parameters
        p_ones = np.array([table[1] for table in tables])  # p(x_i=1) for each variable

        # Normalize to get selection probabilities
        # q_i = p^s_i(x_i=1,t) / S where S = sum of all p^s_j(x_j=1,t)
        S = np.sum(p_ones)

        if S == 0:
            # Edge case: if all probabilities are zero, use uniform
            q_probs = np.ones(n_vars) / n_vars
        else:
            q_probs = p_ones / S

        # Create cumulative probabilities for SUS
        cum_probs = np.cumsum(q_probs)

        # Initialize population (all zeros)
        new_pop = np.zeros((n_samples, n_vars), dtype=int)

        # Sample each individual
        for i in range(n_samples):
            # Use Stochastic Universal Sampling to select n_ones variables
            # SUS returns indices of selected variables
            selected_indices = stochastic_universal_sampling(n_ones, cum_probs, rng)

            # Set selected variables to 1
            new_pop[i, selected_indices] = 1

        return new_pop


class SampleCUMDARange(SamplingMethod):
    """
    Sample population from a CUMDA model with unitation range constraint

    Generates binary solutions where the number of ones is within [min_ones, max_ones].
    For each individual, randomly chooses a target number of ones within the range,
    then samples that many variables.

    Attributes:
        n_samples: Number of individuals to sample
        min_ones: Minimum number of ones (inclusive)
        max_ones: Maximum number of ones (inclusive)
    """

    def __init__(self, n_samples: int, min_ones: int, max_ones: int):
        """
        Initialize CUMDA range sampling

        Args:
            n_samples: Number of individuals to sample
            min_ones: Minimum number of ones (inclusive)
            max_ones: Maximum number of ones (inclusive)
        """
        self.n_samples = n_samples
        self.min_ones = min_ones
        self.max_ones = max_ones

    def sample(
        self,
        n_vars: int,
        model: Model,
        cardinality: np.ndarray,
        aux_pop: Optional[np.ndarray] = None,
        aux_fitness: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
        **params: Any,
    ) -> np.ndarray:
        """
        Sample new population from CUMDA model with unitation range constraint

        Args:
            n_vars: Number of variables
            model: FactorizedModel learned by CUMDA
            cardinality: Variable cardinalities (should be all 2)
            aux_pop: Auxiliary population (not used)
            aux_fitness: Auxiliary fitness (not used)
            rng: Random number generator (optional)
            **params: Additional parameters
                     - n_samples: Override instance n_samples
                     - min_ones: Override instance min_ones
                     - max_ones: Override instance max_ones

        Returns:
            Sampled population where each row has between min_ones and max_ones ones

        Raises:
            TypeError: If model is not a FactorizedModel
            ValueError: If parameters are invalid
        """
        if rng is None:
            rng = np.random.default_rng()

        if not isinstance(model, FactorizedModel):
            raise TypeError(f"Expected FactorizedModel, got {type(model)}")

        # Verify binary variables
        if not np.all(cardinality == 2):
            raise ValueError("CUMDA only works with binary variables (cardinality=2)")

        # Get parameters
        n_samples = params.get("n_samples", self.n_samples)
        min_ones = params.get("min_ones", self.min_ones)
        max_ones = params.get("max_ones", self.max_ones)

        # Validate parameters
        if min_ones < 0 or max_ones > n_vars:
            raise ValueError(
                f"min_ones ({min_ones}) and max_ones ({max_ones}) must be in [0, {n_vars}]"
            )
        if min_ones > max_ones:
            raise ValueError(f"min_ones ({min_ones}) must be <= max_ones ({max_ones})")

        # Extract marginal probabilities p(x_i = 1) from the model
        tables = model.parameters
        p_ones = np.array([table[1] for table in tables])

        # Normalize to get selection probabilities
        S = np.sum(p_ones)
        if S == 0:
            q_probs = np.ones(n_vars) / n_vars
        else:
            q_probs = p_ones / S

        # Create cumulative probabilities for SUS
        cum_probs = np.cumsum(q_probs)

        # Initialize population (all zeros)
        new_pop = np.zeros((n_samples, n_vars), dtype=int)

        # Sample each individual
        for i in range(n_samples):
            # Randomly choose number of ones within range
            n_ones_this = rng.integers(min_ones, max_ones + 1)

            # Use SUS to select variables
            if n_ones_this > 0:
                selected_indices = stochastic_universal_sampling(n_ones_this, cum_probs, rng)
                new_pop[i, selected_indices] = 1

        return new_pop
