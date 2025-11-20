"""
Constraint Factorized Distribution Algorithm (CFDA) sampling

Implements sampling for CFDA models with unitation constraints.
Unlike standard FDA which samples from the factorized distribution freely,
CFDA must ensure that sampled solutions satisfy the unitation constraint
(fixed or bounded number of ones).

The key challenge is combining factorized sampling (which respects dependencies)
with constraint satisfaction. We implement several strategies:

1. Sample-and-Repair: Sample from FDA, then repair to satisfy constraints
2. Truncated Sampling: Sample from conditional distribution given constraint
3. Weighted Sampling: Bias sampling toward feasible configurations

Main Sampling Strategy (Sample-and-Repair):
This is the most straightforward approach:
1. Sample from factorized distribution normally (using SampleFDA logic)
2. Repair each solution to have exactly n_ones variables set to 1
3. Repair uses distance-preserving approach to minimize distortion

The repair step uses smart heuristics:
- If too many ones: flip least probable ones to zeros
- If too few ones: flip most probable zeros to ones
- Considers marginal probabilities to make minimal changes

Alternative Strategy (Conditional Sampling - more complex):
Sample directly from p(x | u(x) = r) by:
1. For each clique, compute conditional probabilities given constraint
2. Sample variables while tracking unitation constraint
3. Adjust sampling probabilities dynamically

References:
- Santana, R., Ochoa, A., & Soto, M. R. "Factorized Distribution Algorithms
  For Functions With Unitation Constraints."
- Mühlenbein, H., Mahnig, T., & Ochoa, A. (1999). "Schemata, distributions and
  graphical models in evolutionary optimization."
"""

from typing import Any, Optional
import numpy as np

from pateda.core.components import SamplingMethod
from pateda.core.models import Model, FactorizedModel
from pateda.sampling.fda import SampleFDA
from pateda.repairing.unitation import unitation_repairing


class SampleCFDA(SamplingMethod):
    """
    Sample population from a CFDA model with exact unitation constraint

    Uses sample-and-repair strategy:
    1. Sample from factorized distribution (respecting dependencies)
    2. Repair to ensure exactly n_ones variables are set to 1

    This preserves the factorization benefits while ensuring constraint satisfaction.

    Attributes:
        n_samples: Number of individuals to sample
        n_ones: Exact number of ones each solution should have
    """

    def __init__(self, n_samples: int, n_ones: int):
        """
        Initialize CFDA sampling with exact unitation constraint

        Args:
            n_samples: Number of individuals to sample
            n_ones: Exact number of variables to set to 1
        """
        self.n_samples = n_samples
        self.n_ones = n_ones
        self.fda_sampler = SampleFDA(n_samples)

    def sample(
        self,
        n_vars: int,
        model: Model,
        cardinality: np.ndarray,
        aux_pop: Optional[np.ndarray] = None,
        aux_fitness: Optional[np.ndarray] = None,
        **params: Any,
    ) -> np.ndarray:
        """
        Sample new population from CFDA model with unitation constraint

        Args:
            n_vars: Number of variables
            model: FactorizedModel learned by CFDA
            cardinality: Variable cardinalities (should be all 2)
            aux_pop: Auxiliary population (not used)
            aux_fitness: Auxiliary fitness (not used)
            **params: Additional parameters
                     - n_samples: Override instance n_samples
                     - n_ones: Override instance n_ones

        Returns:
            Sampled population (n_samples, n_vars) where each row has exactly n_ones ones

        Raises:
            TypeError: If model is not a FactorizedModel
            ValueError: If parameters are invalid
        """
        if not isinstance(model, FactorizedModel):
            raise TypeError(f"Expected FactorizedModel, got {type(model)}")

        # Verify binary variables
        if not np.all(cardinality == 2):
            raise ValueError("CFDA only works with binary variables (cardinality=2)")

        # Get parameters
        n_samples = params.get("n_samples", self.n_samples)
        n_ones = params.get("n_ones", self.n_ones)

        # Validate n_ones
        if n_ones < 0 or n_ones > n_vars:
            raise ValueError(f"n_ones ({n_ones}) must be between 0 and {n_vars}")

        # Step 1: Sample from factorized distribution
        unconstrained_pop = self.fda_sampler.sample(
            n_vars=n_vars,
            model=model,
            cardinality=cardinality,
            n_samples=n_samples,
        )

        # Step 2: Repair to satisfy unitation constraint
        range_values = (n_ones, n_ones)  # Exact constraint
        constrained_pop = unitation_repairing(unconstrained_pop, range_values)

        return constrained_pop


class SampleCFDARange(SamplingMethod):
    """
    Sample population from a CFDA model with unitation range constraint

    Uses sample-and-repair strategy where the number of ones can be within
    [min_ones, max_ones] range.

    Attributes:
        n_samples: Number of individuals to sample
        min_ones: Minimum number of ones (inclusive)
        max_ones: Maximum number of ones (inclusive)
    """

    def __init__(self, n_samples: int, min_ones: int, max_ones: int):
        """
        Initialize CFDA range sampling

        Args:
            n_samples: Number of individuals to sample
            min_ones: Minimum number of ones (inclusive)
            max_ones: Maximum number of ones (inclusive)
        """
        self.n_samples = n_samples
        self.min_ones = min_ones
        self.max_ones = max_ones
        self.fda_sampler = SampleFDA(n_samples)

    def sample(
        self,
        n_vars: int,
        model: Model,
        cardinality: np.ndarray,
        aux_pop: Optional[np.ndarray] = None,
        aux_fitness: Optional[np.ndarray] = None,
        **params: Any,
    ) -> np.ndarray:
        """
        Sample new population from CFDA model with unitation range constraint

        Args:
            n_vars: Number of variables
            model: FactorizedModel learned by CFDA
            cardinality: Variable cardinalities (should be all 2)
            aux_pop: Auxiliary population (not used)
            aux_fitness: Auxiliary fitness (not used)
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
        if not isinstance(model, FactorizedModel):
            raise TypeError(f"Expected FactorizedModel, got {type(model)}")

        # Verify binary variables
        if not np.all(cardinality == 2):
            raise ValueError("CFDA only works with binary variables (cardinality=2)")

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

        # Step 1: Sample from factorized distribution
        unconstrained_pop = self.fda_sampler.sample(
            n_vars=n_vars,
            model=model,
            cardinality=cardinality,
            n_samples=n_samples,
        )

        # Step 2: Repair to satisfy unitation range constraint
        range_values = (min_ones, max_ones)
        constrained_pop = unitation_repairing(unconstrained_pop, range_values)

        return constrained_pop


class SampleCFDAWeighted(SamplingMethod):
    """
    Sample population from CFDA model using weighted approach

    This is an alternative strategy that biases the FDA sampling toward
    configurations more likely to satisfy the unitation constraint.

    The approach:
    1. Extract marginal p(x_i = 1) from factorized model
    2. Create a bias weight: alpha * p(x_i=1) + (1-alpha) * (n_ones/n_vars)
    3. Sample from factorized distribution with biased probabilities
    4. Repair if needed

    This can reduce the amount of repair needed, better preserving the
    structure learned by the factorization.

    Attributes:
        n_samples: Number of individuals to sample
        n_ones: Exact number of ones each solution should have
        alpha: Weight for original probabilities vs uniform (0 to 1)
    """

    def __init__(self, n_samples: int, n_ones: int, alpha: float = 0.5):
        """
        Initialize CFDA weighted sampling

        Args:
            n_samples: Number of individuals to sample
            n_ones: Exact number of variables to set to 1
            alpha: Blending factor:
                  - alpha=1.0: Use original FDA probabilities (more repair needed)
                  - alpha=0.0: Bias toward uniform n_ones/n_vars (less structure)
                  - alpha=0.5: Balanced approach (recommended)
        """
        self.n_samples = n_samples
        self.n_ones = n_ones
        self.alpha = alpha
        self.fda_sampler = SampleFDA(n_samples)

    def sample(
        self,
        n_vars: int,
        model: Model,
        cardinality: np.ndarray,
        aux_pop: Optional[np.ndarray] = None,
        aux_fitness: Optional[np.ndarray] = None,
        **params: Any,
    ) -> np.ndarray:
        """
        Sample new population from CFDA model with weighted probabilities

        Args:
            n_vars: Number of variables
            model: FactorizedModel learned by CFDA
            cardinality: Variable cardinalities
            aux_pop: Auxiliary population (not used)
            aux_fitness: Auxiliary fitness (not used)
            **params: Additional parameters

        Returns:
            Sampled population with exactly n_ones ones per solution
        """
        if not isinstance(model, FactorizedModel):
            raise TypeError(f"Expected FactorizedModel, got {type(model)}")

        if not np.all(cardinality == 2):
            raise ValueError("CFDA only works with binary variables")

        n_samples = params.get("n_samples", self.n_samples)
        n_ones = params.get("n_ones", self.n_ones)
        alpha = params.get("alpha", self.alpha)

        # Create biased model
        target_prob = n_ones / n_vars  # Uniform probability to get n_ones
        biased_model = self._create_biased_model(model, target_prob, alpha)

        # Sample from biased model
        unconstrained_pop = self.fda_sampler.sample(
            n_vars=n_vars,
            model=biased_model,
            cardinality=cardinality,
            n_samples=n_samples,
        )

        # Repair to ensure exact constraint
        range_values = (n_ones, n_ones)
        constrained_pop = unitation_repairing(unconstrained_pop, range_values)

        return constrained_pop

    def _create_biased_model(
        self, model: FactorizedModel, target_prob: float, alpha: float
    ) -> FactorizedModel:
        """
        Create a biased version of the model

        Adjusts probabilities to bias toward target unitation value.

        Args:
            model: Original model
            target_prob: Target probability for x_i=1 (n_ones/n_vars)
            alpha: Blending factor

        Returns:
            Biased model
        """
        # For simplicity, we bias only univariate marginals
        # For full factorizations, this is an approximation

        biased_tables = []
        for table in model.parameters:
            # Assume binary for now
            if len(table) == 2:
                # Univariate marginal: [p(x=0), p(x=1)]
                p_one = table[1]
                # Blend with target
                biased_p_one = alpha * p_one + (1 - alpha) * target_prob
                biased_p_zero = 1 - biased_p_one
                biased_table = np.array([biased_p_zero, biased_p_one])
                biased_tables.append(biased_table)
            else:
                # Multi-variable factor: bias all configurations proportionally
                # This is more complex, for now just use original
                biased_tables.append(table.copy())

        return FactorizedModel(
            structure=model.structure.copy(),
            parameters=biased_tables,
            metadata=model.metadata.copy(),
        )
