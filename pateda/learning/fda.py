"""
Factorized Distribution Algorithm (FDA) learning

==============================================================================
OVERVIEW
==============================================================================

Factorized Distribution Algorithms (FDAs) are a class of Estimation of Distribution
Algorithms (EDAs) where the factorization is directly derived from the problem structure,
as opposed to learning it from data. FDAs were among the first EDAs to exploit
probabilistic graphical models (PGMs) and remain highly relevant for gray-box optimization
where problem structure is known a priori.

As discussed in "Gray-box optimization and factorized distribution algorithms: where
two worlds collide" (Santana, 2017), FDAs that use a fixed, a priori known structure
were foundational in showing how exploiting problem structure produces important gains
in EA efficiency.

==============================================================================
MATHEMATICAL FORMULATION
==============================================================================

A factorized distribution represents a probability distribution as a product of marginal
probability distributions, each called a factor:

    p(x) = ∏ᵢ pᵢ(xsᵢ)

where:
- xsᵢ are subvectors of x called the definition sets of the function
- pᵢ are the marginal probability distributions over each factor
- Each factor captures a subset of interacting variables

==============================================================================
TWO MAIN COMPONENTS
==============================================================================

1. **Structure (Factorization)**:
   - Contains information about which variables belong to each factor
   - Defines relationships between factors
   - Represents the graphical model structure
   - Can be derived from problem structure (gray-box) or learned from data (black-box)

2. **Parameters (Marginal Probabilities)**:
   - Probability values for each configuration of variables in a factor
   - Learned from the selected population via frequency counting
   - Stored as probability tables for each factor

==============================================================================
RELATIONSHIP TO GRAY-BOX OPTIMIZATION
==============================================================================

FDAs are particularly relevant for gray-box optimization problems where:
- The problem structure is known (e.g., additively decomposable functions - ADFs)
- Variables interactions follow a specific pattern (e.g., k-bounded dependencies)
- The structure can be represented as a graphical model

For k-order separable decomposable functions, FDAs have complexity exponential in k,
making them very efficient when k is small. As noted in the gray-box optimization
literature, k-order separable ADFs are trivial for FDAs when the correct factorization
is provided.

==============================================================================
FACTORIZATION CLASSES
==============================================================================

FDAs can represent various factorization classes:

1. **Univariate** (UMDA): Complete independence, each variable is its own factor
   - Factorization: p(x) = ∏ᵢ p(xᵢ)
   - Complexity: O(n)
   - Best for separable problems

2. **Marginal Product Factorizations**: Disjoint groups of variables
   - Factorization: p(x) = ∏ᵢ pᵢ(xsᵢ) where sᵢ ∩ sⱼ = ∅
   - Complexity depends on factor sizes
   - Best for block-structured problems

3. **Markov Chain Factorizations**: Sequential dependencies
   - See pateda.learning.markov.LearnMarkovChain
   - Complexity: O(n)
   - Best for chain-structured problems

4. **Junction Tree Factorizations**: General tree-structured dependencies
   - Can handle arbitrary problem structures via triangulation
   - Complexity related to tree-width
   - Used to solve deceptive functions (Mühlenbein et al., 1999)

==============================================================================
RELATION TO THEORETICAL CONCEPTS
==============================================================================

**Factors vs. Hyperplanes**:
As discussed in Santana (2017), factors play a role in EDAs similar to hyperplanes
in GA schema theory, but with important differences:
- Factors explicitly represent probability distributions over variable configurations
- Factors can be combined into factorizations using graph-theoretic methods
- The defining length of schemata becomes less relevant for FDAs
- Factors naturally handle overlapping variable sets via graphical representations

**Valid vs. Invalid Factorizations**:
- Valid factorizations allow exact sampling and efficient computation
- Invalid factorizations may introduce approximation errors
- See Mühlenbein et al. (1999) for formal definitions

==============================================================================
IMPLEMENTATION NOTES
==============================================================================

In MATEDA-2.0 (and this implementation), factorizations use two components:
1. **Cliques**: Matrix representing factor structure
2. **Tables**: Probability tables for each factor

Clique Structure Format:
Each row represents one factor with format:
    [n_overlap, n_new, overlap_var_indices..., new_var_indices...]

Where:
- n_overlap: Number of variables shared with previous factors
- n_new: Number of new variables in this factor
- overlap_var_indices: Indices of shared variables
- new_var_indices: Indices of new variables

This format efficiently represents junction tree factorizations and enables
exact sampling via the product-of-marginals decomposition.

==============================================================================
COMPLEXITY AND PROBLEM DIFFICULTY
==============================================================================

From Santana (2017):
- k-order separable ADFs have tree-width (k-1) and are trivial for FDAs: O(n)
- Random ADFs of order k have tree-width O(n), making them exponentially complex
- Real-world problems typically have intermediate complexity
- Tree-width of the interaction graph is a key indicator of problem difficulty

==============================================================================
REFERENCES
==============================================================================

- Mühlenbein, H., Mahnig, T., & Ochoa, A. (1999). "Schemata, distributions and
  graphical models in evolutionary optimization." Journal of Heuristics, 5(2):213-247.
  [Foundational FDA paper introducing factorizations and junction trees]

- Santana, R. (2017). "Gray-box optimization and factorized distribution algorithms:
  where two worlds collide." arXiv:1707.03093.
  [Comprehensive analysis of FDAs and gray-box optimization]

- Höns, R. (2006). "Estimation of Distribution Algorithms and Minimum Relative Entropy."
  PhD thesis, University of Bonn.
  [Theoretical foundations of factorizations]

- MATEDA-2.0 User Guide, Section 4.1: "Factorized distributions"
  [Implementation details for MATLAB version]

==============================================================================
SEE ALSO
==============================================================================

Related implementations in pateda:
- pateda.learning.markov: Markov chain factorizations (special case of FDA)
- pateda.learning.boa: Bayesian network EDAs (learn structure from data)
- pateda.learning.ebna: Tree-based factorizations
- pateda.learning.gaussian.learn_gmrf_eda: Gaussian Markov Random Fields (continuous case)
- pateda.sampling.fda: Sampling from FDA models
- pateda.sampling.map_sampling: MAP-based sampling for Markov networks

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

    In MATEDA-2.0, factorizations are represented using two components:
    1. Cliques: Represent the variables of each factor, specifying whether they are
       also included in previous factors or have not appeared before.
    2. Tables: Contain a probability table for each of the factors.

    Clique structure format:
    Each row of Cliques is a clique with format:
        [n_overlap, n_new, overlap_var_indices..., new_var_indices...]

    Where:
    - n_overlap: Number of overlapping variables with respect to previous cliques
    - n_new: Number of new variables in this clique
    - overlap_var_indices: Indices of overlapping variables
    - new_var_indices: Indices of new variables

    This format can represent various types of factorizations:
    - Univariate (UMDA): Each row is [0, 1, var_idx]
    - Marginal product factorizations
    - Markov chain factorizations
    - Factorizations from junction trees

    References:
    - MATEDA-2.0 User Guide, Section 4.1, Example 2
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
