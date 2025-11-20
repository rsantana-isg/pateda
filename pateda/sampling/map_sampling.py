"""
MAP-based Sampling for Markov Network EDAs

Implements sampling strategies based on Maximum A Posteriori (MAP) configurations:
- Insert-MAP (S1): Directly insert MAP solution into population
- Template-MAP (S2): Use MAP as template for crossover
- Insert-MAP + Template-MAP (S3): Hybrid strategy

Based on:
Santana, R. (2013). "Message Passing Methods for Estimation of Distribution
Algorithms Based on Markov Networks". Chapter in "Estimation of Distribution
Algorithms: A New Tool for Evolutionary Computation", Springer.

Key findings from the paper:
- Insert-MAP (S1) generally outperforms other strategies
- Performance advantage increases with variable cardinality
- Exact and approximate inference methods (BP, Dec-BP) show similar performance
- Hybrid S3 combines benefits of direct insertion and template-based exploration

Algorithm 4 (Insert-MAP): Generate M samples using PLS, replace worst with MAP
Algorithm 5 (Template-MAP): Use MAP as template, sample other variables from model
"""

from typing import Any, Optional, List
import numpy as np

from pateda.core.components import SamplingMethod
from pateda.core.models import Model, MarkovNetworkModel, FactorizedModel
from pateda.inference.map_inference import MAPInference, MAPMethod


class SampleInsertMAP(SamplingMethod):
    """
    Insert-MAP Sampling (Strategy S1 from Santana 2013)

    Algorithm:
    1. Sample M individuals using Probabilistic Logic Sampling (PLS)
    2. Compute MAP configuration using message passing
    3. Replace worst individual with MAP configuration

    This strategy directly ensures the most probable configuration is in
    the population, which can significantly improve convergence especially
    for problems with high variable cardinality.

    Parameters:
        n_samples: Number of individuals to sample
        map_method: Method for MAP computation ("exact", "bp", "decimation")
        n_map_inserts: Number of MAP solutions to insert (default 1)
        k_map: If > 1, use k-MAP to insert k best configurations
        replace_worst: If True, replace worst individuals; if False, replace random
    """

    def __init__(
        self,
        n_samples: int,
        map_method: str = "bp",
        n_map_inserts: int = 1,
        k_map: int = 1,
        replace_worst: bool = True,
    ):
        """
        Initialize Insert-MAP sampler

        Args:
            n_samples: Number of individuals to sample
            map_method: MAP inference method ("exact", "bp", "decimation")
            n_map_inserts: How many MAP solutions to insert (default 1)
            k_map: Use k-MAP to find k best configurations (default 1)
            replace_worst: Replace worst individuals (True) or random (False)
        """
        self.n_samples = n_samples
        self.map_method = map_method
        self.n_map_inserts = n_map_inserts
        self.k_map = k_map
        self.replace_worst = replace_worst

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
        Sample using Insert-MAP strategy

        Args:
            n_vars: Number of variables
            model: MarkovNetworkModel or FactorizedModel
            cardinality: Variable cardinalities
            aux_pop: Auxiliary population (optional, for replacement)
            aux_fitness: Auxiliary fitness (for identifying worst individuals)
            **params: Additional parameters

        Returns:
            Sampled population (n_samples, n_vars) with MAP inserted
        """
        # Override n_samples if provided in params
        n_samples = params.get('n_samples', self.n_samples)

        # Step 1: Sample M individuals using PLS
        population = self._sample_pls(n_vars, model, cardinality, n_samples)

        # Step 2: Compute MAP configuration(s)
        if self.k_map > 1:
            map_configs = self._compute_k_map(model, cardinality, self.k_map)
        else:
            map_config = self._compute_map(model, cardinality)
            map_configs = [map_config]

        # Step 3: Insert MAP into population
        n_insert = min(self.n_map_inserts, len(map_configs), n_samples)

        if self.replace_worst and aux_fitness is not None:
            # Replace worst individuals based on fitness
            worst_indices = np.argsort(aux_fitness)[:n_insert]
            for i, idx in enumerate(worst_indices):
                if i < len(map_configs):
                    population[idx] = map_configs[i]
        else:
            # Replace first n_insert individuals (or random)
            for i in range(n_insert):
                population[i] = map_configs[i]

        return population

    def _sample_pls(
        self,
        n_vars: int,
        model: Model,
        cardinality: np.ndarray,
        n_samples: int
    ) -> np.ndarray:
        """
        Probabilistic Logic Sampling (PLS) for factorized models

        Samples by forward sampling through ordered cliques.
        """
        population = np.zeros((n_samples, n_vars), dtype=int)

        # Get cliques and tables from model
        if isinstance(model, MarkovNetworkModel):
            cliques = model.structure  # List of cliques
            tables = model.parameters  # List of probability tables
        elif isinstance(model, FactorizedModel):
            # Convert factorized structure to cliques
            cliques = [model.structure[i] for i in range(len(model.structure))]
            tables = model.parameters
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")

        # Order cliques for sampling (topological-like order)
        ordered_cliques, ordered_tables = self._order_cliques(
            cliques, tables, n_vars
        )

        # Sample each individual
        for i in range(n_samples):
            sampled = np.full(n_vars, -1, dtype=int)

            for clique, table in zip(ordered_cliques, ordered_tables):
                # Determine which variables are already sampled
                overlap = [v for v in clique if sampled[v] >= 0]
                new_vars = [v for v in clique if sampled[v] < 0]

                if not new_vars:
                    continue

                # Get probability distribution for new variables
                if overlap:
                    # Conditional sampling
                    prob_dist = self._get_conditional_distribution(
                        clique, table, sampled, cardinality
                    )
                else:
                    # Marginal sampling
                    prob_dist = table.ravel()

                # Sample configuration for new variables
                if len(prob_dist) > 0 and np.sum(prob_dist) > 0:
                    prob_dist = prob_dist / np.sum(prob_dist)
                    idx = np.random.choice(len(prob_dist), p=prob_dist)

                    # Convert index to configuration
                    config = self._index_to_config(
                        idx, [cardinality[v] for v in new_vars]
                    )

                    for j, v in enumerate(new_vars):
                        sampled[v] = config[j]

            # Fill any remaining unsampled variables randomly
            for v in range(n_vars):
                if sampled[v] < 0:
                    sampled[v] = np.random.randint(0, cardinality[v])

            population[i] = sampled

        return population

    def _order_cliques(
        self,
        cliques: List[np.ndarray],
        tables: List[np.ndarray],
        n_vars: int
    ) -> tuple:
        """Order cliques for forward sampling"""
        # Simple ordering: by first variable in clique
        clique_list = list(cliques) if not isinstance(cliques, list) else cliques
        table_list = list(tables) if not isinstance(tables, list) else tables

        # Create pairs and sort
        pairs = list(zip(clique_list, table_list))
        pairs.sort(key=lambda x: x[0][0] if len(x[0]) > 0 else 0)

        ordered_cliques = [p[0] for p in pairs]
        ordered_tables = [p[1] for p in pairs]

        return ordered_cliques, ordered_tables

    def _get_conditional_distribution(
        self,
        clique: np.ndarray,
        table: np.ndarray,
        sampled: np.ndarray,
        cardinality: np.ndarray
    ) -> np.ndarray:
        """Get conditional probability distribution given sampled variables"""
        # Ensure table has correct shape for the clique
        expected_shape = tuple(cardinality[v] for v in clique)
        if table.shape != expected_shape:
            table = table.reshape(expected_shape)

        # Build marginal distribution conditioned on already sampled variables
        overlap = [i for i, v in enumerate(clique) if sampled[v] >= 0]
        new_vars = [i for i, v in enumerate(clique) if sampled[v] < 0]

        if not new_vars:
            return np.array([1.0])

        # Extract slice from table corresponding to sampled values
        slices = [slice(None)] * len(clique)
        for i in overlap:
            slices[i] = sampled[clique[i]]

        prob_dist = table[tuple(slices)].ravel()
        return prob_dist

    def _index_to_config(self, index: int, cardinalities: List[int]) -> np.ndarray:
        """Convert flat index to configuration (mixed-radix)"""
        config = np.zeros(len(cardinalities), dtype=int)
        for i in range(len(cardinalities) - 1, -1, -1):
            config[i] = index % cardinalities[i]
            index //= cardinalities[i]
        return config

    def _compute_map(self, model: Model, cardinality: np.ndarray) -> np.ndarray:
        """Compute MAP configuration using inference"""
        if isinstance(model, MarkovNetworkModel):
            cliques = model.structure
            tables = model.parameters
        elif isinstance(model, FactorizedModel):
            cliques = [model.structure[i] for i in range(len(model.structure))]
            tables = model.parameters
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")

        # Use MAPInference to compute MAP
        inference = MAPInference(
            cliques=cliques,
            tables=tables,
            cardinalities=cardinality,
            method=self.map_method
        )

        result = inference.compute_map()
        return result.configuration

    def _compute_k_map(
        self,
        model: Model,
        cardinality: np.ndarray,
        k: int
    ) -> List[np.ndarray]:
        """Compute k-MAP configurations"""
        if isinstance(model, MarkovNetworkModel):
            cliques = model.structure
            tables = model.parameters
        elif isinstance(model, FactorizedModel):
            cliques = [model.structure[i] for i in range(len(model.structure))]
            tables = model.parameters
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")

        inference = MAPInference(
            cliques=cliques,
            tables=tables,
            cardinalities=cardinality,
            method=self.map_method
        )

        result = inference.compute_k_map(k)
        return [result.configurations[i] for i in range(len(result.configurations))]


class SampleTemplateMAP(SamplingMethod):
    """
    Template-MAP Sampling (Strategy S2 from Santana 2013)

    Algorithm:
    1. Compute MAP configuration using message passing
    2. Use MAP as a template:
       - For each new individual:
         - Select subset of variables to inherit from MAP
         - Sample remaining variables from learned model
    3. This creates variation around the high-quality MAP solution

    Parameters:
        n_samples: Number of individuals to sample
        map_method: Method for MAP computation
        template_prob: Probability of inheriting each variable from MAP (default 0.5)
        min_template_vars: Minimum variables to inherit from template
    """

    def __init__(
        self,
        n_samples: int,
        map_method: str = "bp",
        template_prob: float = 0.5,
        min_template_vars: int = 1,
    ):
        """
        Initialize Template-MAP sampler

        Args:
            n_samples: Number of individuals to sample
            map_method: MAP inference method
            template_prob: Probability of inheriting variable from MAP template
            min_template_vars: Minimum variables to keep from template
        """
        self.n_samples = n_samples
        self.map_method = map_method
        self.template_prob = template_prob
        self.min_template_vars = min_template_vars

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
        Sample using Template-MAP strategy

        Args:
            n_vars: Number of variables
            model: MarkovNetworkModel or FactorizedModel
            cardinality: Variable cardinalities
            aux_pop: Auxiliary population (optional)
            aux_fitness: Auxiliary fitness (not used)
            **params: Additional parameters

        Returns:
            Sampled population using MAP as template
        """
        n_samples = params.get('n_samples', self.n_samples)

        # Step 1: Compute MAP template
        map_config = self._compute_map(model, cardinality)

        # Step 2: Generate population using MAP as template
        population = np.zeros((n_samples, n_vars), dtype=int)

        # Get cliques and tables
        if isinstance(model, MarkovNetworkModel):
            cliques = model.structure
            tables = model.parameters
        elif isinstance(model, FactorizedModel):
            cliques = [model.structure[i] for i in range(len(model.structure))]
            tables = model.parameters
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")

        # Sample each individual
        for i in range(n_samples):
            # Determine which variables to inherit from template
            if i == 0:
                # First individual is pure MAP
                population[i] = map_config.copy()
            else:
                # Create variation around MAP
                individual = map_config.copy()

                # Randomly select variables to resample
                resample_mask = np.random.random(n_vars) > self.template_prob

                # Ensure minimum template variables
                n_template = np.sum(~resample_mask)
                if n_template < self.min_template_vars:
                    # Keep some template variables
                    keep_indices = np.random.choice(
                        n_vars,
                        size=self.min_template_vars,
                        replace=False
                    )
                    resample_mask[keep_indices] = False

                # Resample selected variables from model
                for var in np.where(resample_mask)[0]:
                    individual[var] = self._sample_variable(
                        var, individual, cliques, tables, cardinality
                    )

                population[i] = individual

        return population

    def _sample_variable(
        self,
        var: int,
        current_config: np.ndarray,
        cliques: List[np.ndarray],
        tables: List[np.ndarray],
        cardinality: np.ndarray
    ) -> int:
        """Sample a single variable conditioned on current configuration"""
        # Find cliques containing this variable
        marginal = np.ones(cardinality[var])

        for clique, table in zip(cliques, tables):
            if var in clique:
                # Ensure table has correct shape for the clique
                expected_shape = tuple(cardinality[v] for v in clique)
                if table.shape != expected_shape:
                    table = table.reshape(expected_shape)

                # Get conditional probability P(var | other vars in clique)
                var_idx = np.where(clique == var)[0][0]

                # Build slice for current configuration
                slices = [current_config[v] for v in clique]

                # Marginalize over this variable
                for val in range(cardinality[var]):
                    slices[var_idx] = val
                    prob = table[tuple(slices)]
                    marginal[val] *= max(prob, 1e-300)

        # Normalize and sample
        if np.sum(marginal) > 0:
            marginal /= np.sum(marginal)
            return np.random.choice(cardinality[var], p=marginal)
        else:
            return np.random.randint(0, cardinality[var])

    def _compute_map(self, model: Model, cardinality: np.ndarray) -> np.ndarray:
        """Compute MAP configuration"""
        if isinstance(model, MarkovNetworkModel):
            cliques = model.structure
            tables = model.parameters
        elif isinstance(model, FactorizedModel):
            cliques = [model.structure[i] for i in range(len(model.structure))]
            tables = model.parameters
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")

        inference = MAPInference(
            cliques=cliques,
            tables=tables,
            cardinalities=cardinality,
            method=self.map_method
        )

        result = inference.compute_map()
        return result.configuration


class SampleHybridMAP(SamplingMethod):
    """
    Hybrid Insert-MAP + Template-MAP (Strategy S3 from Santana 2013)

    Combines both strategies:
    1. Use Template-MAP to generate most of population
    2. Use Insert-MAP to ensure pure MAP is included

    This balances exploration (template variation) with exploitation
    (guaranteed MAP inclusion).

    Parameters:
        n_samples: Number of individuals to sample
        map_method: Method for MAP computation
        template_prob: Probability of inheriting from MAP in template
        n_map_inserts: Number of pure MAP configurations to insert
    """

    def __init__(
        self,
        n_samples: int,
        map_method: str = "bp",
        template_prob: float = 0.5,
        n_map_inserts: int = 1,
    ):
        """
        Initialize Hybrid MAP sampler

        Args:
            n_samples: Number of individuals to sample
            map_method: MAP inference method
            template_prob: Template inheritance probability
            n_map_inserts: Number of pure MAP solutions to insert
        """
        self.n_samples = n_samples
        self.template_sampler = SampleTemplateMAP(
            n_samples=n_samples,
            map_method=map_method,
            template_prob=template_prob
        )
        self.n_map_inserts = n_map_inserts

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
        Sample using Hybrid MAP strategy

        Args:
            n_vars: Number of variables
            model: MarkovNetworkModel or FactorizedModel
            cardinality: Variable cardinalities
            aux_pop: Auxiliary population
            aux_fitness: Auxiliary fitness
            **params: Additional parameters

        Returns:
            Sampled population combining template and insert strategies
        """
        n_samples = params.get('n_samples', self.n_samples)

        # Use Template-MAP for sampling
        population = self.template_sampler.sample(
            n_vars, model, cardinality, aux_pop, aux_fitness, **params
        )

        # The first individual from Template-MAP is already pure MAP,
        # so hybrid strategy is naturally satisfied when n_map_inserts=1

        # If we want multiple MAP inserts, duplicate the first individual
        if self.n_map_inserts > 1:
            for i in range(1, min(self.n_map_inserts, n_samples)):
                population[i] = population[0].copy()

        return population
