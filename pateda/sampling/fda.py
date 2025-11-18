"""
Sampling from Factorized Distribution Algorithm (FDA) models

Equivalent to MATEDA's SampleFDA.m
"""

from typing import Any, Optional
import numpy as np

from pateda.core.components import SamplingMethod
from pateda.core.models import Model, FactorizedModel
from pateda.learning.utils.conversions import (
    find_acc_card,
    index_convert_card,
    num_convert_card,
)
from pateda.sampling.utils import stochastic_universal_sampling


class SampleFDA(SamplingMethod):
    """
    Sample population from a Factorized Distribution Algorithm (FDA) model

    Samples individuals by iterating through cliques in order, sampling
    variables conditioned on previously sampled overlapping variables.
    """

    def __init__(self, n_samples: int):
        """
        Initialize FDA sampling

        Args:
            n_samples: Number of individuals to sample
        """
        self.n_samples = n_samples

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
        Sample new population from FDA model

        Args:
            n_vars: Number of variables
            model: FactorizedModel to sample from
            cardinality: Variable cardinalities
            aux_pop: Auxiliary population (not used for basic FDA sampling)
            aux_fitness: Auxiliary fitness (not used)
            **params: Additional parameters
                     - n_samples: Override instance n_samples

        Returns:
            Sampled population (n_samples, n_vars)
        """
        if not isinstance(model, FactorizedModel):
            raise TypeError(f"Expected FactorizedModel, got {type(model)}")

        # Get parameters
        n_samples = params.get("n_samples", self.n_samples)
        cliques = model.structure
        tables = model.parameters

        n_cliques = cliques.shape[0]

        # Initialize population with -1 (unassigned)
        new_pop = -np.ones((n_samples, n_vars), dtype=int)

        # Sample each clique in order
        for c in range(n_cliques):
            n_overlap = int(cliques[c, 0])
            n_new = int(cliques[c, 1])

            # Get variable indices
            if n_overlap > 0:
                overlap_vars = cliques[c, 2 : 2 + n_overlap].astype(int)
            else:
                overlap_vars = np.array([], dtype=int)

            new_vars = cliques[c, 2 + n_overlap : 2 + n_overlap + n_new].astype(int)

            # Get probability table
            table = tables[c]

            # Calculate cardinalities and accumulated cardinalities
            new_card = cardinality[new_vars]
            new_acc_card = find_acc_card(n_new, new_card)
            n_new_configs = int(np.prod(new_card))

            if n_overlap == 0:
                # Root node: sample directly from marginal distribution
                cum_probs = np.cumsum(table)
                indices = stochastic_universal_sampling(n_samples, cum_probs)

                # Convert indices to variable values
                for j in range(n_samples):
                    var_values = index_convert_card(indices[j], n_new, new_acc_card)
                    new_pop[j, new_vars] = var_values

            else:
                # Non-root: sample conditioned on overlap variables
                overlap_card = cardinality[overlap_vars]
                overlap_acc_card = find_acc_card(n_overlap, overlap_card)
                n_overlap_configs = int(np.prod(overlap_card))

                # For each possible overlap configuration
                for k in range(n_overlap_configs):
                    # Get overlap variable values for this configuration
                    overlap_vals = index_convert_card(k, n_overlap, overlap_acc_card)

                    # Find individuals with this overlap configuration
                    if n_overlap == 1:
                        mask = new_pop[:, overlap_vars[0]] == overlap_vals[0]
                    else:
                        mask = np.all(
                            new_pop[:, overlap_vars] == overlap_vals, axis=1
                        )

                    which_indices = np.where(mask)[0]
                    n_matching = len(which_indices)

                    if n_matching > 0:
                        # Sample new variables for these individuals
                        # Get conditional probabilities P(new | overlap=k)
                        cond_probs = table[k, :]

                        # Normalize (should already be normalized, but just in case)
                        cond_probs = cond_probs / np.sum(cond_probs)

                        cum_probs = np.cumsum(cond_probs)
                        indices = stochastic_universal_sampling(n_matching, cum_probs)

                        # Assign values
                        for j, ind_idx in enumerate(which_indices):
                            var_values = index_convert_card(
                                indices[j], n_new, new_acc_card
                            )
                            new_pop[ind_idx, new_vars] = var_values

        # Verify all variables were assigned
        if np.any(new_pop == -1):
            unassigned = np.where(new_pop == -1)
            raise RuntimeError(
                f"Some variables were not assigned during sampling: {unassigned}"
            )

        return new_pop
