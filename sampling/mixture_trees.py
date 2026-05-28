"""
Sampling from Mixture of Trees models

Samples from mixture of tree models by:
1. Selecting a component according to mixture weights
2. Sampling from the selected tree component
"""

from typing import Any, Optional
import numpy as np

from pateda.core.components import SamplingMethod
from pateda.core.models import Model, MixtureModel
from pateda.sampling.fda import SampleFDA


class SampleMixtureTrees(SamplingMethod):
    """
    Sample population from a Mixture of Trees model

    The sampling process:
    1. For each individual, select a tree component according to mixture weights
    2. Sample the individual from the selected component's distribution

    This creates diversity by sampling from different tree structures.
    """

    def __init__(self, n_samples: int):
        """
        Initialize Mixture of Trees sampling

        Args:
            n_samples: Number of individuals to sample
        """
        self.n_samples = n_samples
        self._fda_sampler = SampleFDA(1)  # Sample one at a time

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
        Sample new population from Mixture of Trees model

        Args:
            n_vars: Number of variables
            model: MixtureModel with tree components
            cardinality: Variable cardinalities
            aux_pop: Auxiliary population (not used)
            aux_fitness: Auxiliary fitness (not used)
            rng: Random number generator (optional)
            **params: Additional parameters
                     - n_samples: Override instance n_samples

        Returns:
            Sampled population (n_samples, n_vars)

        Note:
            Each individual is sampled by:
            1. Selecting a component j with probability λ_j (mixture weight)
            2. Sampling from tree component f_j(x)
        """
        if rng is None:
            rng = np.random.default_rng()

        if not isinstance(model, MixtureModel):
            raise TypeError(f"Expected MixtureModel, got {type(model)}")

        n_samples = params.get("n_samples", self.n_samples)

        # Get mixture parameters
        weights = model.parameters["weights"]
        component_structures = model.structure
        component_parameters = model.parameters["components"]
        n_components = len(weights)

        # Normalize weights to ensure they sum to exactly 1.0
        # (avoids numerical precision issues)
        weights = np.array(weights)
        weights = weights / weights.sum()

        # Initialize population
        new_pop = np.zeros((n_samples, n_vars), dtype=int)

        # Sample each individual
        for i in range(n_samples):
            # Select component according to mixture weights
            component_idx = rng.choice(n_components, p=weights)

            # Create temporary FactorizedModel for this component
            from pateda.core.models import FactorizedModel

            # Get component structure and parameters
            # Note: component_parameters[component_idx] contains the tables from BMDA
            # We need to reconstruct the cliques structure

            # For now, use a simplified approach: convert tree to factorized format
            # The component was originally learned as a FactorizedModel (BMDA)
            # We can reconstruct it or use the tree directly

            # Sample from the component
            # Since trees can be sampled using FDA (they're a special factorization)
            # we convert back to FactorizedModel format

            component_model = self._tree_to_factorized(
                component_structures[component_idx],
                component_parameters[component_idx],
                n_vars,
                cardinality
            )

            # Sample one individual from this component
            individual = self._fda_sampler.sample(
                n_vars, component_model, cardinality, rng, n_samples=1
            )

            new_pop[i, :] = individual[0, :]

        return new_pop

    def _tree_to_factorized(
        self,
        tree_structure: np.ndarray,
        tree_parameters: list,
        n_vars: int,
        cardinality: np.ndarray
    ):
        """
        Convert tree structure back to FactorizedModel for sampling.

        The TreeModel produced by LearnMixtureTrees._factorized_to_tree already
        stores the structure in FDA clique format:
            [n_overlap, n_new, overlap_vars..., new_vars..., (zero-padding)]
        with the matching probability tables.  We reconstruct a FactorizedModel
        directly - no re-parsing or re-building of cliques needed.
        """
        from pateda.core.models import FactorizedModel

        return FactorizedModel(
            structure=tree_structure,
            parameters=tree_parameters,
            metadata={"model_type": "Tree (from mixture)"},
        )


class SampleMixtureTreesDirect(SamplingMethod):
    """
    Direct sampling from Mixture of Trees (without FDA conversion)

    Uses direct tree sampling for each component. More efficient than converting
    to FactorizedModel, but requires implementing tree sampling from scratch.
    """

    def __init__(self, n_samples: int):
        """
        Initialize direct mixture of trees sampling

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
        rng: Optional[np.random.Generator] = None,
        **params: Any,
    ) -> np.ndarray:
        """
        Sample using direct tree sampling

        Args:
            n_vars: Number of variables
            model: MixtureModel with tree components
            cardinality: Variable cardinalities
            aux_pop: Auxiliary population (not used)
            aux_fitness: Auxiliary fitness (not used)
            rng: Random number generator (optional)
            **params: Additional parameters

        Returns:
            Sampled population (n_samples, n_vars)
        """
        if rng is None:
            rng = np.random.default_rng()

        if not isinstance(model, MixtureModel):
            raise TypeError(f"Expected MixtureModel, got {type(model)}")

        n_samples = params.get("n_samples", self.n_samples)

        # Get mixture parameters
        weights = model.parameters["weights"]
        component_structures = model.structure
        component_parameters = model.parameters["components"]
        n_components = len(weights)

        # Initialize population
        new_pop = np.zeros((n_samples, n_vars), dtype=int)

        # Sample each individual
        for i in range(n_samples):
            # Select component
            component_idx = rng.choice(n_components, p=weights)

            # Sample from tree (would require implementing tree sampling)
            # For now, use random sampling as placeholder
            # TODO: Implement proper tree sampling
            new_pop[i, :] = rng.integers(0, cardinality, size=n_vars)

        return new_pop
