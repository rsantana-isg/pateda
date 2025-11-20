"""
Gibbs Sampling for Markov Network models

Implements Gibbs sampling (MCMC) for generating solutions from Markov networks.
Primary sampling method for MOA, can also be used with MN-FDA.

Algorithm:
1. Initialize: Random configuration or from previous population
2. For each Gibbs iteration:
   - For each variable (in random or fixed order):
     - Sample Xi ~ P(Xi | X_neighbors) using conditional probability table
3. Return final configurations

References:
- Santana, R. (2013). "Message Passing Methods for EDAs Based on Markov Networks"
- Algorithm 3 (MOA), step 7: "Generate M new points sampling from Markov network"
- C++ implementation: cpp_EDAs/FDA.cpp:2446-2492 (GenIndividualMOA)
"""

from typing import Any, Optional
import numpy as np

from pateda.core.components import SamplingMethod
from pateda.core.models import Model, MarkovNetworkModel
from pateda.learning.utils.conversions import (
    find_acc_card,
    num_convert_card,
)


class SampleGibbs(SamplingMethod):
    """
    Gibbs sampling for Markov network models

    Uses Markov Chain Monte Carlo (MCMC) to sample from the learned
    Markov network distribution.

    For MOA models:
    - Uses P(Xi | neighbors) directly from tables
    - Iterates n * ln(n) * IT times (IT=4 recommended)

    For general Markov networks:
    - Requires conditional probability tables for each variable
    """

    def __init__(
        self,
        n_samples: int,
        n_iterations: Optional[int] = None,
        IT: int = 4,
        temperature: float = 1.0,
        random_order: bool = True,
        burnin: int = 0,
    ):
        """
        Initialize Gibbs sampler

        Args:
            n_samples: Number of individuals to sample
            n_iterations: Total Gibbs iterations per sample
                        If None, compute as: IT * n_vars * ln(n_vars)
            IT: Iteration factor for auto-computing n_iterations (default 4)
                From paper: r = n * ln(n) * IT with IT=4
            temperature: Temperature for Boltzmann sampling (default 1.0)
                       T > 1.0: more exploration
                       T < 1.0: more exploitation
                       T = 1.0: standard Gibbs
            random_order: Whether to randomize variable order each iteration
            burnin: Number of initial iterations to discard (default 0)
        """
        self.n_samples = n_samples
        self.n_iterations = n_iterations
        self.IT = IT
        self.temperature = temperature
        self.random_order = random_order
        self.burnin = burnin

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
        Sample new population using Gibbs sampling

        Args:
            n_vars: Number of variables
            model: MarkovNetworkModel with cliques and conditional tables
            cardinality: Variable cardinalities
            aux_pop: Auxiliary population for initialization (optional)
            aux_fitness: Auxiliary fitness (not used)
            rng: Random number generator (optional)
            **params: Additional parameters (n_samples override)

        Returns:
            Sampled population (n_samples, n_vars)
        """
        if rng is None:
            rng = np.random.default_rng()

        if not isinstance(model, MarkovNetworkModel):
            raise TypeError(
                f"SampleGibbs requires MarkovNetworkModel, got {type(model)}"
            )

        n_samples = params.get("n_samples", self.n_samples)

        # Determine number of iterations
        if self.n_iterations is None:
            # Auto-compute: IT * n * ln(n)
            n_iters = int(self.IT * n_vars * np.log(n_vars))
        else:
            n_iters = self.n_iterations

        # Initialize population
        new_pop = self._initialize_population(n_samples, n_vars, cardinality, aux_pop, rng)

        # Extract model structure and tables
        cliques = model.structure
        tables = model.parameters
        metadata = model.metadata

        # Perform Gibbs sampling for each individual
        for sample_idx in range(n_samples):
            new_pop[sample_idx, :] = self._gibbs_sample_one(
                new_pop[sample_idx, :],
                n_vars,
                n_iters,
                cliques,
                tables,
                cardinality,
                metadata,
                rng,
            )

        return new_pop

    def _initialize_population(
        self,
        n_samples: int,
        n_vars: int,
        cardinality: np.ndarray,
        aux_pop: Optional[np.ndarray],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Initialize population for Gibbs sampling"""
        if aux_pop is not None and len(aux_pop) >= n_samples:
            # Initialize from auxiliary population
            return aux_pop[:n_samples].copy()
        else:
            # Random initialization
            new_pop = np.zeros((n_samples, n_vars), dtype=int)
            for var in range(n_vars):
                new_pop[:, var] = rng.integers(
                    0, int(cardinality[var]), size=n_samples
                )
            return new_pop

    def _gibbs_sample_one(
        self,
        configuration: np.ndarray,
        n_vars: int,
        n_iterations: int,
        cliques: np.ndarray,
        tables: list,
        cardinality: np.ndarray,
        metadata: dict,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Perform Gibbs sampling for one individual

        Reference: C++ FDA.cpp:2446-2492 (GenIndividualMOA)

        Args:
            configuration: Initial configuration (n_vars,)
            n_vars: Number of variables
            n_iterations: Number of Gibbs iterations
            cliques: Model structure (list of cliques)
            tables: Conditional probability tables
            cardinality: Variable cardinalities
            metadata: Model metadata

        Returns:
            Final configuration after Gibbs sampling
        """
        config = configuration.copy()

        # Determine if this is MOA model (one clique per variable)
        is_moa = metadata.get("model_type") == "MOA"

        # Gibbs iterations (with burnin)
        total_iters = n_iterations + self.burnin

        for iter_idx in range(total_iters):
            # Determine variable order
            if self.random_order:
                var_order = rng.permutation(n_vars)
            else:
                var_order = np.arange(n_vars)

            # Sample each variable
            for var in var_order:
                if is_moa:
                    # MOA: variable corresponds directly to clique index
                    config[var] = self._sample_moa_variable(
                        var, config, cliques[var], tables[var], cardinality, rng
                    )
                else:
                    # General Markov network: find cliques containing this variable
                    config[var] = self._sample_variable_general(
                        var, config, cliques, tables, cardinality, rng
                    )

        return config

    def _sample_moa_variable(
        self,
        var: int,
        config: np.ndarray,
        clique: np.ndarray,
        table: np.ndarray,
        cardinality: np.ndarray,
        rng: np.random.Generator,
    ) -> int:
        """
        Sample one variable in MOA model

        MOA clique structure: [var, neighbor_1, neighbor_2, ...]
        Table: P(var | neighbors)

        Reference: C++ FDA.cpp:2466-2481
        """
        var_card = int(cardinality[var])

        # Get neighbors (all variables in clique except var itself)
        neighbors = clique[clique != var]

        if len(neighbors) == 0:
            # No neighbors: sample from marginal P(var)
            probs = table  # Should be 1D array
        else:
            # Sample from conditional P(var | neighbors)
            # Get neighbor configuration
            neighbor_config = config[neighbors].astype(int)
            neighbor_cards = cardinality[neighbors].astype(int)

            # Convert to table index
            n_neighbors = len(neighbors)
            neighbor_acc = find_acc_card(n_neighbors, neighbor_cards)
            neighbor_idx = num_convert_card(neighbor_config, n_neighbors, neighbor_acc)

            # Get conditional probabilities
            if table.ndim == 1:
                # Marginal distribution
                probs = table
            else:
                # Conditional distribution: table[neighbor_config, :]
                probs = table[neighbor_idx, :]

        # Apply temperature (Boltzmann sampling)
        if self.temperature != 1.0:
            # Boltzmann: p_i' = exp(log(p_i) / T) / Z
            log_probs = np.log(probs + 1e-10)  # Avoid log(0)
            log_probs_temp = log_probs / self.temperature
            probs_temp = np.exp(log_probs_temp)
            probs = probs_temp / np.sum(probs_temp)

        # Sample value
        value = rng.choice(var_card, p=probs)

        return value

    def _sample_variable_general(
        self,
        var: int,
        config: np.ndarray,
        cliques: np.ndarray,
        tables: list,
        cardinality: np.ndarray,
        rng: np.random.Generator,
    ) -> int:
        """
        Sample one variable in general Markov network

        For non-MOA models, find the clique containing this variable
        and sample from the appropriate conditional distribution.

        This is a simplified version - may need refinement for complex models.
        """
        var_card = int(cardinality[var])

        # Find cliques containing this variable
        var_cliques = []
        for clique_idx, clique in enumerate(cliques):
            if var in clique:
                var_cliques.append(clique_idx)

        if len(var_cliques) == 0:
            # Shouldn't happen, but fall back to uniform
            return rng.integers(0, var_card)

        # Use first clique containing this variable
        # (For better accuracy, should combine information from all cliques)
        clique_idx = var_cliques[0]
        clique = cliques[clique_idx]
        table = tables[clique_idx]

        # Simple approach: treat first variable as target, rest as conditioning
        # This assumes clique[0] is the target variable
        if clique[0] == var:
            # This clique has var as target
            neighbors = clique[1:]

            if len(neighbors) == 0:
                probs = table
            else:
                neighbor_config = config[neighbors].astype(int)
                neighbor_cards = cardinality[neighbors].astype(int)
                n_neighbors = len(neighbors)
                neighbor_acc = find_acc_card(n_neighbors, neighbor_cards)
                neighbor_idx = num_convert_card(
                    neighbor_config, n_neighbors, neighbor_acc
                )

                if table.ndim == 1:
                    probs = table
                else:
                    probs = table[neighbor_idx, :]

            # Sample
            if self.temperature != 1.0:
                log_probs = np.log(probs + 1e-10)
                log_probs_temp = log_probs / self.temperature
                probs_temp = np.exp(log_probs_temp)
                probs = probs_temp / np.sum(probs_temp)

            value = rng.choice(var_card, p=probs)
            return value
        else:
            # Var is not the target of this clique - use uniform for now
            # (Better approach: derive conditional from joint)
            return rng.integers(0, var_card)
