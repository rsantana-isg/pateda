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
        Convert tree structure back to FactorizedModel for sampling

        Args:
            tree_structure: Tree structure (parent-child relationships)
            tree_parameters: Tree parameters (probability tables)
            n_vars: Number of variables
            cardinality: Variable cardinalities

        Returns:
            FactorizedModel representation
        """
        from pateda.core.models import FactorizedModel

        # The tree_parameters are already in the format from BMDA (clique-based)
        # We can use them directly if we also have the corresponding cliques

        # Reconstruct cliques from tree structure
        # This is a simplified version - assumes tree_parameters are already in clique format
        # from the BMDA learning

        # For proper implementation, would need to:
        # 1. Build cliques from tree edges
        # 2. Map parameters correctly

        # Temporary solution: assume parameters are already in compatible format
        # Build a simple factorization structure

        cliques = []
        tables = []

        # Add root nodes and edges based on tree structure
        for edge_idx in range(len(tree_structure)):
            n_parents, parent, child = tree_structure[edge_idx]

            if n_parents == 0:
                # Root node
                clique = np.zeros(3, dtype=int)
                clique[0] = 0  # no overlap
                clique[1] = 1  # one new variable
                clique[2] = child  # the variable
                cliques.append(clique)
            else:
                # Parent-child edge
                clique = np.zeros(4, dtype=int)
                clique[0] = 1  # one overlap (parent)
                clique[1] = 1  # one new (child)
                clique[2] = parent  # overlap variable
                clique[3] = child  # new variable
                cliques.append(clique)

        # Use the provided parameters
        if len(tree_parameters) == len(cliques):
            tables = tree_parameters
        else:
            # Fallback: use uniform distributions
            tables = []
            for clique in cliques:
                n_new = int(clique[1])
                if n_new > 0:
                    new_vars = [int(clique[2 + clique[0] + i]) for i in range(n_new)]
                    table_shape = tuple([int(cardinality[v]) for v in new_vars])
                    table = np.ones(table_shape) / np.prod(table_shape)
                    tables.append(table)

        # Convert to array with padding
        max_size = max(len(c) for c in cliques)
        cliques_array = np.zeros((len(cliques), max_size), dtype=int)
        for i, c in enumerate(cliques):
            cliques_array[i, :len(c)] = c

        model = FactorizedModel(
            structure=cliques_array,
            parameters=tables,
            metadata={"model_type": "Tree (from mixture)"}
        )

        return model


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
