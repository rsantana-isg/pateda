"""
Constraint Factorized Distribution Algorithm (CFDA) learning

CFDA extends FDA (Factorized Distribution Algorithms) to handle binary constraint
problems with unitation constraints. Like FDA, it uses factorized distributions to
capture dependencies between variables, but it's designed specifically for problems
where the number of ones must be within certain bounds.

Key Features:
- Uses factorized distributions to capture variable dependencies
- Designed for problems with unitation constraints: 1 ≤ a < u(x) ≤ b < n
- Can leverage problem structure knowledge (gray-box optimization)
- More sophisticated than CUMDA for problems with variable interactions

Mathematical Representation:
    p(x) = ∏ᵢ pᵢ(xsᵢ)

where:
- xsᵢ are subvectors (definition sets) representing factors
- pᵢ are marginal probability distributions over each factor
- Each factor can include multiple interacting variables

The main difference between CFDA and standard FDA is in the sampling phase,
where CFDA enforces the unitation constraint. The learning phase is similar to FDA,
but the model is intended to be used with constraint-aware sampling.

Algorithm:
1. Define or learn factorization structure (cliques)
2. For each clique:
   a. Identify variables in the factor
   b. Learn marginal probability distribution from selected population
   c. Store probability table for all configurations

Factorization Classes:
- Univariate: Each variable independent (equivalent to CUMDA)
- Pairwise: Adjacent variable pairs interact
- Block: Disjoint groups of variables
- Chain: Markov chain dependencies
- Tree: General tree-structured dependencies

References:
- Santana, R., Ochoa, A., & Soto, M. R. "Factorized Distribution Algorithms
  For Functions With Unitation Constraints."
- Mühlenbein, H., Mahnig, T., & Ochoa, A. (1999). "Schemata, distributions and
  graphical models in evolutionary optimization." Journal of Heuristics, 5(2):213-247.
"""

from typing import Any, Optional
import numpy as np

from pateda.core.components import LearningMethod
from pateda.core.models import FactorizedModel
from pateda.learning.utils.marginal_prob import learn_fda_parameters


class LearnCFDA(LearningMethod):
    """
    Learn a CFDA (Constraint Factorized Distribution Algorithm) model

    CFDA learns a factorized distribution that respects variable dependencies
    while being suitable for constraint-aware sampling. The factorization can be:
    - Provided explicitly (gray-box optimization)
    - Learned from problem structure
    - Predefined (e.g., chain, pairwise dependencies)

    The learning phase is similar to standard FDA, but the resulting model
    is intended for use with constraint sampling methods.

    Attributes:
        cliques: Factorization structure (clique matrix)
                If None, creates univariate structure (equivalent to CUMDA)
    """

    def __init__(self, cliques: Optional[np.ndarray] = None):
        """
        Initialize CFDA learning

        Args:
            cliques: Clique structure matrix. If None, creates univariate structure.
                    Each row: [n_overlap, n_new, overlap_indices..., new_indices...]

        Examples:
            >>> # Univariate (like CUMDA)
            >>> cfda = LearnCFDA(cliques=None)
            >>>
            >>> # Pairwise adjacent dependencies: (x0,x1), (x1,x2), (x2,x3), ...
            >>> n = 10
            >>> cliques = np.zeros((n-1, 4))
            >>> cliques[:, 0] = 1  # 1 overlap variable
            >>> cliques[:, 1] = 1  # 1 new variable
            >>> cliques[:, 2] = np.arange(n-1)  # overlap: 0,1,2,...,n-2
            >>> cliques[:, 3] = np.arange(1, n)  # new: 1,2,3,...,n-1
            >>> cfda = LearnCFDA(cliques=cliques)
            >>>
            >>> # Block structure: (x0,x1,x2), (x3,x4,x5), ...
            >>> # Each block has 3 variables, no overlap
            >>> block_size = 3
            >>> n_blocks = n // block_size
            >>> cliques = np.zeros((n_blocks, 2 + block_size))
            >>> cliques[:, 0] = 0  # No overlap
            >>> cliques[:, 1] = block_size  # block_size new variables
            >>> for i in range(n_blocks):
            >>>     cliques[i, 2:2+block_size] = np.arange(i*block_size, (i+1)*block_size)
            >>> cfda = LearnCFDA(cliques=cliques)
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
        Learn CFDA model from population

        Args:
            generation: Current generation number
            n_vars: Number of variables
            cardinality: Variable cardinalities (should be all 2 for binary)
            population: Selected population to learn from (pop_size, n_vars)
            fitness: Fitness values (not used for CFDA learning)
            **params: Additional parameters
                     - cliques: Override instance cliques structure

        Returns:
            Learned FactorizedModel with factorized structure

        Raises:
            ValueError: If cardinalities are not binary
        """
        # Verify binary variables
        if not np.all(cardinality == 2):
            raise ValueError("CFDA only works with binary variables (cardinality=2)")

        # Get or create clique structure
        cliques = params.get("cliques", self.cliques)

        if cliques is None:
            # Create univariate structure (like CUMDA)
            cliques = np.zeros((n_vars, 3))
            cliques[:, 0] = 0  # No overlapping variables
            cliques[:, 1] = 1  # One new variable per clique
            cliques[:, 2] = np.arange(n_vars)  # Variable index

        # Validate clique structure
        self._validate_cliques(cliques, n_vars)

        # Learn probability tables for each clique using FDA learning
        tables = learn_fda_parameters(cliques, population, n_vars, cardinality)

        # Create and return model
        model = FactorizedModel(
            structure=cliques,
            parameters=tables,
            metadata={
                "generation": generation,
                "model_type": "CFDA",
                "constraint_type": "unitation",
                "n_cliques": cliques.shape[0],
            },
        )

        return model

    def _validate_cliques(self, cliques: np.ndarray, n_vars: int) -> None:
        """
        Validate clique structure

        Args:
            cliques: Clique structure matrix
            n_vars: Number of variables

        Raises:
            ValueError: If clique structure is invalid
        """
        if cliques.shape[0] == 0:
            raise ValueError("Cliques matrix cannot be empty")

        if cliques.shape[1] < 3:
            raise ValueError(
                "Each clique row must have at least 3 elements: [n_overlap, n_new, ...]"
            )

        # Check that all variables are covered
        all_vars = set()
        for i in range(cliques.shape[0]):
            n_overlap = int(cliques[i, 0])
            n_new = int(cliques[i, 1])

            if n_overlap + n_new == 0:
                raise ValueError(f"Clique {i} has no variables")

            # Extract variable indices
            var_indices = cliques[i, 2 : 2 + n_overlap + n_new].astype(int)
            all_vars.update(var_indices)

        if len(all_vars) != n_vars:
            missing = set(range(n_vars)) - all_vars
            raise ValueError(
                f"Cliques do not cover all variables. Missing: {missing}"
            )


def create_pairwise_chain_cliques(n_vars: int) -> np.ndarray:
    """
    Create clique structure for pairwise chain: (x0,x1), (x1,x2), (x2,x3), ...

    This factorization assumes Markov chain dependencies where each variable
    depends on its predecessor.

    Args:
        n_vars: Number of variables

    Returns:
        Clique structure matrix for pairwise chain

    Examples:
        >>> cliques = create_pairwise_chain_cliques(5)
        >>> # Creates: (x0,x1), (x1,x2), (x2,x3), (x3,x4)
    """
    if n_vars < 2:
        raise ValueError("Need at least 2 variables for pairwise chain")

    # First clique: (x0, x1) - no overlap, 2 new variables
    # Subsequent cliques: (xi, xi+1) - 1 overlap (xi), 1 new (xi+1)

    cliques = np.zeros((n_vars - 1, 4))

    # First clique: [0, 2, -, 0, 1]
    cliques[0, 0] = 0  # no overlap
    cliques[0, 1] = 2  # 2 new variables
    cliques[0, 2] = 0  # x0
    cliques[0, 3] = 1  # x1

    # Subsequent cliques: [1, 1, xi, xi+1]
    for i in range(1, n_vars - 1):
        cliques[i, 0] = 1  # 1 overlap
        cliques[i, 1] = 1  # 1 new
        cliques[i, 2] = i  # overlap: xi
        cliques[i, 3] = i + 1  # new: xi+1

    return cliques


def create_block_cliques(n_vars: int, block_size: int) -> np.ndarray:
    """
    Create clique structure for disjoint blocks of variables

    Creates non-overlapping groups of variables, each forming an independent factor.
    Useful for problems with known block structure.

    Args:
        n_vars: Number of variables
        block_size: Size of each block

    Returns:
        Clique structure matrix for block factorization

    Raises:
        ValueError: If n_vars is not divisible by block_size

    Examples:
        >>> cliques = create_block_cliques(9, 3)
        >>> # Creates: (x0,x1,x2), (x3,x4,x5), (x6,x7,x8)
    """
    if n_vars % block_size != 0:
        raise ValueError(f"n_vars ({n_vars}) must be divisible by block_size ({block_size})")

    n_blocks = n_vars // block_size
    cliques = np.zeros((n_blocks, 2 + block_size))

    for i in range(n_blocks):
        cliques[i, 0] = 0  # No overlap
        cliques[i, 1] = block_size  # block_size new variables
        start_idx = i * block_size
        cliques[i, 2 : 2 + block_size] = np.arange(start_idx, start_idx + block_size)

    return cliques


def create_overlapping_windows_cliques(
    n_vars: int, window_size: int, stride: int = 1
) -> np.ndarray:
    """
    Create clique structure for overlapping windows

    Creates sliding windows of variables with specified size and stride.
    Useful for problems where local interactions are important.

    Args:
        n_vars: Number of variables
        window_size: Size of each window
        stride: Step size between windows (default 1)

    Returns:
        Clique structure matrix for overlapping windows

    Examples:
        >>> cliques = create_overlapping_windows_cliques(10, 3, 2)
        >>> # Creates: (x0,x1,x2), (x2,x3,x4), (x4,x5,x6), (x6,x7,x8)
    """
    if window_size > n_vars:
        raise ValueError(f"window_size ({window_size}) cannot exceed n_vars ({n_vars})")

    windows = []
    pos = 0
    while pos + window_size <= n_vars:
        windows.append(list(range(pos, pos + window_size)))
        pos += stride

    n_windows = len(windows)
    max_row_size = 2 + window_size + window_size  # worst case: all overlap + all new

    cliques = np.zeros((n_windows, max_row_size))

    for i, window_vars in enumerate(windows):
        if i == 0:
            # First window: no overlap
            cliques[i, 0] = 0
            cliques[i, 1] = window_size
            cliques[i, 2 : 2 + window_size] = window_vars
        else:
            # Find overlap with previous windows
            prev_vars = set()
            for j in range(i):
                prev_vars.update(windows[j])

            overlap_vars = [v for v in window_vars if v in prev_vars]
            new_vars = [v for v in window_vars if v not in prev_vars]

            n_overlap = len(overlap_vars)
            n_new = len(new_vars)

            cliques[i, 0] = n_overlap
            cliques[i, 1] = n_new

            if n_overlap > 0:
                cliques[i, 2 : 2 + n_overlap] = overlap_vars
            if n_new > 0:
                cliques[i, 2 + n_overlap : 2 + n_overlap + n_new] = new_vars

    # Trim unused columns
    max_used_cols = int(np.max(cliques[:, 0] + cliques[:, 1])) + 2
    cliques = cliques[:, :max_used_cols]

    return cliques
