"""
k-order Markov Chain EDA learning

Implements k-order Markov chain models for EDAs. Markov chains are a special type
of factorized distribution where variables have sequential dependencies following
a predefined ordering.

Markov Chain Factorization:
The k-order Markov model factorizes as:
    p_MK(x) = p(x₁, ..., x_{k+1}) * ∏ᵢ₌ₖ₊₂ⁿ p(xᵢ | xᵢ₋₁, ..., xᵢ₋ₖ)

This means:
- The first k+1 variables have a joint distribution (initial distribution)
- Each subsequent variable depends on the k previous variables
- The dependency structure forms a chain following the variable ordering

Markov Property:
A sequence has the Markov property if the conditional distribution of future
states depends only on the present state and k previous states, not on the
entire history. Formally:
    p(xᵢ | x₁, ..., xᵢ₋₁) = p(xᵢ | xᵢ₋ₖ, ..., xᵢ₋₁)

Special Cases:
- k=1 (First-order Markov): Each variable depends on only the previous variable
  p(x) = p(x₁) * ∏ᵢ₌₂ⁿ p(xᵢ | xᵢ₋₁)
- k=0 (Zero-order): Reduces to UMDA (all variables independent)
- k=n-1: Full joint distribution (no independence assumptions)

Advantages:
- More expressive than UMDA (can model sequential dependencies)
- Less complex than full Bayesian networks
- Natural for problems with inherent sequential structure
- Parameter estimation is straightforward (frequency counting)
- Linear number of cliques O(n) vs. potentially exponential for general BNs

Limitations:
- Assumes a specific variable ordering (ordering matters!)
- Can only model chain-structured dependencies
- Cannot represent arbitrary dependency patterns
- May not fit problems without natural sequential structure

Relationship to Markov Random Fields (MRFs):
Markov chains are directed models (Bayesian networks with chain structure).
Markov Random Fields are undirected graphical models where:
- Nodes represent variables
- Edges represent direct probabilistic dependencies
- Factorization based on cliques (fully connected subsets)
- Can represent more complex dependency patterns than chains

When to use Markov Chains:
- Problem has natural sequential ordering (e.g., scheduling, sequencing)
- Variables exhibit temporal or spatial dependencies
- Want middle ground between UMDA and full Bayesian networks
- Need efficient learning and sampling

Computational Complexity:
- Learning: O(n * m * k^(k+1)) where n=variables, m=samples, k=chain order, k=max cardinality
- More efficient than general Bayesian networks for long chains
- Memory: O(n * k^(k+1)) for storing conditional probability tables

Based on MATEDA-2.0 factorized distributions and Markov chain models.

References:
- Mühlenbein, H. (1997). "The equation for response to selection and its use for
  prediction." Evolutionary Computation, 5(3):303-346.
- Höns, R. (2005). "Estimation of Distribution Algorithms and Minimum Relative
  Entropy." PhD Thesis, University of Bonn.
- MATEDA-2.0 User Guide, Section 4.1: "Factorized distributions" and
  Section 4.3: "Markov network based factorizations"
"""

from typing import Any
import numpy as np

from pateda.core.components import LearningMethod
from pateda.core.models import FactorizedModel


class LearnMarkovChain(LearningMethod):
    """
    Learn a k-order Markov Chain model (MK-EDA)

    The k-order Markov model assumes that each variable depends on the k previous
    variables in a sequential ordering. This creates a chain structure where
    conditional dependencies follow the variable ordering.

    For k=1 (first-order Markov), each variable depends only on the immediately
    previous variable. For k>1, dependencies extend to k previous variables.
    """

    def __init__(self, k: int = 1, alpha: float = 0.0):
        """
        Initialize k-order Markov Chain learning

        Args:
            k: Order of the Markov chain (number of previous variables each depends on)
            alpha: Smoothing parameter for probability estimation (Laplace smoothing)
        """
        if k < 1:
            raise ValueError("k must be at least 1 for Markov chains")
        self.k = k
        self.alpha = alpha

    def _learn_joint_initial(
        self,
        population: np.ndarray,
        cardinality: np.ndarray,
        n_samples: int
    ) -> np.ndarray:
        """
        Learn joint distribution for first k+1 variables

        Args:
            population: Population data
            cardinality: Variable cardinalities
            n_samples: Number of samples

        Returns:
            Probability table for joint distribution of first k+1 variables
        """
        n_init_vars = self.k + 1
        init_cards = [int(cardinality[i]) for i in range(n_init_vars)]

        # Calculate number of possible configurations
        n_configs = int(np.prod(init_cards))

        # Count occurrences of each configuration
        counts = np.zeros(n_configs)

        for sample_idx in range(n_samples):
            # Calculate configuration index using mixed-radix numbering
            config_idx = 0
            mult = 1
            for var_idx in range(n_init_vars):
                config_idx += int(population[sample_idx, var_idx]) * mult
                mult *= init_cards[var_idx]

            counts[config_idx] += 1

        # Apply smoothing
        if self.alpha > 0:
            counts += self.alpha

        # Normalize to probabilities
        probabilities = counts / np.sum(counts)

        # Reshape to multidimensional array for easier indexing
        prob_table = probabilities.reshape(init_cards)

        return prob_table

    def _learn_conditional(
        self,
        var: int,
        population: np.ndarray,
        cardinality: np.ndarray,
        n_samples: int
    ) -> np.ndarray:
        """
        Learn conditional distribution p(x_var | x_{var-k}, ..., x_{var-1})

        Args:
            var: Variable index
            population: Population data
            cardinality: Variable cardinalities
            n_samples: Number of samples

        Returns:
            Conditional probability table
        """
        # Variables that this variable depends on (k previous variables)
        parent_vars = list(range(var - self.k, var))
        parent_cards = [int(cardinality[i]) for i in parent_vars]
        var_card = int(cardinality[var])

        # Number of parent configurations
        n_parent_configs = int(np.prod(parent_cards))

        # Calculate parent configuration indices for all samples
        parent_configs = np.zeros(n_samples, dtype=int)
        for sample_idx in range(n_samples):
            config_idx = 0
            mult = 1
            for i, parent_var in enumerate(parent_vars):
                config_idx += int(population[sample_idx, parent_var]) * mult
                mult *= parent_cards[i]
            parent_configs[sample_idx] = config_idx

        # Count conditional occurrences
        # cpd[parent_config, var_value] = count
        cpd = np.zeros((n_parent_configs, var_card))

        for sample_idx in range(n_samples):
            parent_config = parent_configs[sample_idx]
            var_value = int(population[sample_idx, var])
            cpd[parent_config, var_value] += 1

        # Apply smoothing and normalize
        if self.alpha > 0:
            cpd += self.alpha

        # Normalize each row to get conditional probabilities
        row_sums = cpd.sum(axis=1, keepdims=True)
        cpd = cpd / row_sums

        return cpd

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
        Learn k-order Markov Chain model from population

        Args:
            generation: Current generation number
            n_vars: Number of variables
            cardinality: Variable cardinalities
            population: Selected population to learn from
            fitness: Fitness values (not used for Markov learning)
            **params: Additional parameters
                     - alpha: Override smoothing parameter

        Returns:
            Learned FactorizedModel with Markov chain structure

        Note:
            The structure is represented using cliques format:
            - First clique: variables 0 to k (k+1 variables total)
            - Subsequent cliques: for each variable i from k+1 to n-1,
              clique contains variables (i-k, ..., i-1, i)
        """
        alpha = params.get("alpha", self.alpha)
        n_samples = population.shape[0]

        # Cliques structure for k-order Markov chain
        # Format: [n_overlap, n_new, overlap_indices..., new_indices...]

        cliques = []
        tables = []

        # First clique: joint distribution over first k+1 variables
        first_clique = np.zeros(2 + self.k + 1, dtype=int)
        first_clique[0] = 0  # No overlapping variables
        first_clique[1] = self.k + 1  # k+1 new variables
        for i in range(self.k + 1):
            first_clique[2 + i] = i  # Variable indices

        cliques.append(first_clique)
        tables.append(self._learn_joint_initial(population, cardinality, n_samples))

        # Subsequent cliques: conditional distributions
        # Each variable from k+1 onwards depends on k previous variables
        for var in range(self.k + 1, n_vars):
            # This clique has k overlap variables (the parents) and 1 new variable
            clique = np.zeros(2 + self.k + 1, dtype=int)
            clique[0] = self.k  # k overlapping variables (parents)
            clique[1] = 1  # 1 new variable

            # Overlap variables (parents): var-k to var-1
            for i in range(self.k):
                clique[2 + i] = var - self.k + i

            # New variable
            clique[2 + self.k] = var

            cliques.append(clique)
            tables.append(self._learn_conditional(var, population, cardinality, n_samples))

        # Convert cliques list to numpy array with padding
        max_clique_size = max(len(c) for c in cliques)
        cliques_array = np.zeros((len(cliques), max_clique_size), dtype=int)
        for i, clique in enumerate(cliques):
            cliques_array[i, :len(clique)] = clique

        # Create and return model
        model = FactorizedModel(
            structure=cliques_array,
            parameters=tables,
            metadata={
                "generation": generation,
                "model_type": "Markov Chain",
                "order_k": self.k,
                "alpha": alpha,
            },
        )

        return model
