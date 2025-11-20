"""
Sampling from Bayesian Network models

Equivalent to MATEDA's sampling methods for EBNA/BOA
"""

from typing import Any, Optional
import numpy as np

from pateda.core.components import SamplingMethod
from pateda.core.models import Model, BayesianNetworkModel


class SampleBayesianNetwork(SamplingMethod):
    """
    Sample population from a Bayesian Network model

    Uses ancestral sampling: sample variables in topological order,
    conditioning on previously sampled parent values.
    """

    def __init__(self, n_samples: int):
        """
        Initialize Bayesian network sampling

        Args:
            n_samples: Number of individuals to sample
        """
        self.n_samples = n_samples

    def _topological_sort(self, adjacency: np.ndarray) -> np.ndarray:
        """
        Perform topological sort on DAG

        Args:
            adjacency: Adjacency matrix (parent -> child)

        Returns:
            Array of variable indices in topological order
        """
        n_vars = adjacency.shape[0]
        in_degree = np.sum(adjacency, axis=0)  # Number of parents for each variable

        # Find variables with no parents
        queue = list(np.where(in_degree == 0)[0])
        order = []

        while queue:
            # Get variable with no remaining parents
            var = queue.pop(0)
            order.append(var)

            # Remove edges from this variable
            for child in range(n_vars):
                if adjacency[var, child] > 0:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)

        if len(order) != n_vars:
            raise ValueError("Graph contains cycles - not a valid DAG")

        return np.array(order)

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
        Sample new population from Bayesian network model

        Args:
            n_vars: Number of variables
            model: BayesianNetworkModel to sample from
            cardinality: Variable cardinalities
            aux_pop: Auxiliary population (not used)
            aux_fitness: Auxiliary fitness (not used)
            rng: Random number generator (optional)
            **params: Additional parameters
                     - n_samples: Override instance n_samples

        Returns:
            Sampled population (n_samples, n_vars)
        """
        if rng is None:
            rng = np.random.default_rng()

        if not isinstance(model, BayesianNetworkModel):
            raise TypeError(f"Expected BayesianNetworkModel, got {type(model)}")

        # Get parameters
        n_samples = params.get("n_samples", self.n_samples)
        adjacency = model.structure
        cpds = model.parameters

        # Get topological ordering
        order = self._topological_sort(adjacency)

        # Initialize population
        new_pop = np.zeros((n_samples, n_vars), dtype=int)

        # Sample in topological order
        for var in order:
            var_info = cpds[var]
            parents = var_info["parents"]
            cpd = var_info["cpd"]
            k = int(cardinality[var])

            if len(parents) == 0:
                # No parents: sample from marginal distribution
                # cpd is 1D array of probabilities
                for i in range(n_samples):
                    new_pop[i, var] = rng.choice(k, p=cpd)

            else:
                # Has parents: sample from conditional distribution
                parent_card = [int(cardinality[p]) for p in parents]

                for i in range(n_samples):
                    # Get parent configuration for this sample
                    config = 0
                    mult = 1
                    for j, p in enumerate(parents):
                        config += int(new_pop[i, p]) * mult
                        mult *= parent_card[j]

                    # Sample from P(var | parent_config)
                    probs = cpd[config, :]
                    new_pop[i, var] = rng.choice(k, p=probs)

        return new_pop
