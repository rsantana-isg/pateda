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


class SampleLocalStructureBN(SampleBayesianNetwork):
    """Sample a Bayesian network whose CPDs may carry a *local structure*
    (decision tree / decision graph), as produced by ``bayes_nets`` when a model
    is learned with ``fit(..., local_structure="dt"|"dg")`` or
    ``learn_local_structure(...)``.

    For each variable this sampler routes the (already sampled) parent
    configurations directly through the compact ``LocalStructureCPD`` stored in
    ``model.parameters[var]["local"]`` (via its vectorised ``sample_rows``),
    so the decision graph is exploited *at sampling time* and the dense
    conditional table is never materialised -- the point of hBOA-style local
    structure.  Variables without a ``"local"`` entry (e.g. a plain EBNA model)
    fall back to the tabular CPD, so this sampler is a drop-in superset of
    :class:`SampleBayesianNetwork`.
    """

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
        if rng is None:
            rng = np.random.default_rng()
        if not isinstance(model, BayesianNetworkModel):
            raise TypeError(f"Expected BayesianNetworkModel, got {type(model)}")

        n_samples = params.get("n_samples", self.n_samples)
        cardinality = np.asarray(cardinality, dtype=int)
        cpds = model.parameters
        order = self._topological_sort(model.structure)
        new_pop = np.zeros((n_samples, n_vars), dtype=int)

        for var in order:
            info = cpds[var]
            parents = list(info["parents"])
            k = int(cardinality[var])
            local = info.get("local") if isinstance(info, dict) else None

            if not parents:
                # Marginal distribution (1-D CPD).
                p = np.asarray(info["cpd"], dtype=float).ravel()
                new_pop[:, var] = rng.choice(k, size=n_samples, p=p)
            elif local is not None:
                # Route parent configurations through the decision tree/graph.
                parent_values = new_pop[:, parents]           # (n_samples, |Pa|)
                new_pop[:, var] = np.asarray(
                    local.sample_rows(parent_values, rng), dtype=int)
            else:
                # Tabular fallback (vectorised inverse-CDF sampling).
                cpd = np.asarray(info["cpd"], dtype=float)
                parent_card = [int(cardinality[p]) for p in parents]
                mult = np.cumprod([1] + parent_card[:-1])
                config = (new_pop[:, parents] * mult).sum(axis=1).astype(int)
                probs = cpd[config]                            # (n_samples, k)
                cdf = np.cumsum(probs, axis=1)
                u = rng.random(n_samples)[:, None]
                new_pop[:, var] = (u < cdf).argmax(axis=1)

        return new_pop
