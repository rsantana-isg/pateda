"""
Factorized Distribution Algorithm (FDA) learning

Factorized distributions represent probability distributions as a product of marginal
probability distributions, each called a factor. This provides a condensed representation
of otherwise very difficult to store probability distributions.

A factorized distribution can be represented as:
    p(x) = ∏ᵢ pᵢ(xsᵢ)

where xsᵢ are subvectors of x called the definition sets of the function, and pᵢ are
the marginal probability distributions.

Two main components of factorization:
1. Structure: Contains information about which variables belong to each factor and
   relationships with other factors
2. Parameters: The probability values of each factor configuration

References:
- Mühlenbein et al. (1999). "Schemata, distributions and graphical models in
  evolutionary optimization." Journal of Heuristics, 5(2):213-247.
- MATEDA-2.0 User Guide, Section 4.1: "Factorized distributions"

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
