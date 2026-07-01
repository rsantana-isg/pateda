"""
MAP-based Sampling for Markov Network EDAs

Implements sampling strategies based on Maximum A Posteriori (MAP) configurations:
- Insert-MAP (S1): Directly insert MAP solution into population
- Template-MAP (S2): Use MAP as template for crossover
- Insert-MAP + Template-MAP (S3): Hybrid strategy
- Template-Best (S6): Use best elite individual as template (addresses S2 failure mode)

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

NOTE on S2 failure mode
-----------------------
In practice, the MAP configuration computed from the learned model has LOWER
fitness than the best population individual in ~94% of generations (verified on
bqp50 with TreeEDA over 50 generations).  This is because the model MAP
maximises model probability, not actual fitness, and the model is inaccurate
especially in early generations.

With template_prob=0.5, S2 fixes 50% of every offspring's bits to this
suboptimal MAP, severely constraining diversity ("MAP lock-in").  The result is
a catastrophic performance collapse: mean ≈700 vs PLS mean ≈1051 on bqp50.

S6-TemplateBest (SampleTemplateBest) addresses this by replacing the MAP
template with the best verified individual from the selected population.
"""

from typing import Any, Optional, List
import numpy as np

from pateda.core.components import SamplingMethod
from pateda.core.models import Model, MarkovNetworkModel, FactorizedModel
from pateda.inference.map_inference import MAPInference, MAPMethod
from pateda.sampling.fda import SampleFDA
from pateda.sampling.partial import SamplePartialFDA


def _extract_cliques_and_tables(model: Model):
    """Extract (cliques, tables) where each clique is a variable-index array.

    FactorizedModel rows have layout [n_overlap, n_new, overlap_vars..., new_vars...].
    We strip the header fields and return only the variable index arrays.
    MarkovNetworkModel stores cliques directly as variable-index lists.
    """
    if isinstance(model, MarkovNetworkModel):
        return list(model.structure), list(model.parameters)

    if isinstance(model, FactorizedModel):
        cliques = []
        structure = model.structure
        for i in range(len(structure)):
            n_overlap = int(structure[i, 0])
            n_new = int(structure[i, 1])
            vars_i = structure[i, 2:2 + n_overlap + n_new].astype(int)
            cliques.append(vars_i)
        return cliques, list(model.parameters)

    raise ValueError(f"Unsupported model type: {type(model)}")


def _gibbs_sample_variable(
    var: int,
    current_config: np.ndarray,
    cliques: List,
    tables: List,
    cardinality: np.ndarray,
    rng: np.random.Generator,
) -> int:
    """Sample one variable conditioned on current configuration (Gibbs step)."""
    marginal = np.ones(cardinality[var])
    for clique, table in zip(cliques, tables):
        if var in clique:
            expected_shape = tuple(cardinality[v] for v in clique)
            if table.shape != expected_shape:
                table = table.reshape(expected_shape)
            var_idx = int(np.where(clique == var)[0][0])
            slices = list(current_config[v] for v in clique)
            for val in range(cardinality[var]):
                slices[var_idx] = val
                marginal[val] *= max(float(table[tuple(slices)]), 1e-300)
    total = np.sum(marginal)
    if total > 0:
        marginal /= total
        return int(rng.choice(cardinality[var], p=marginal))
    return int(rng.integers(0, cardinality[var]))


def _is_non_overlapping(model: FactorizedModel) -> bool:
    """Return True if all cliques have n_overlap=0 (e.g., UMDA)."""
    return bool(np.all(model.structure[:, 0] == 0))


def _map_non_overlapping(model: FactorizedModel, cardinality: np.ndarray) -> np.ndarray:
    """Fast MAP for fully independent cliques: argmax per table.

    Valid only when _is_non_overlapping(model) is True.  Each table encodes
    an unconditional marginal P(var), so MAP[var] = argmax P(var).
    """
    structure = model.structure
    tables = model.parameters
    n_vars = len(cardinality)
    map_config = np.zeros(n_vars, dtype=int)
    for i in range(len(structure)):
        n_new = int(structure[i, 1])
        new_vars = structure[i, 2 : 2 + n_new].astype(int)
        table = tables[i]
        if n_new == 1:
            map_config[new_vars[0]] = int(np.argmax(table.ravel()))
        else:
            idx = int(np.argmax(table.ravel()))
            shape = tuple(int(cardinality[v]) for v in new_vars)
            for j, val in enumerate(np.unravel_index(idx, shape)):
                map_config[new_vars[j]] = val
    return map_config


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
        rng: Optional[np.random.Generator] = None,
        **params: Any,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()

        n_samples = params.get('n_samples', self.n_samples)

        # Step 1: Sample M individuals using PLS
        population = self._sample_pls(n_vars, model, cardinality, n_samples, rng)

        # Step 2: Compute MAP configuration(s)
        if self.k_map > 1:
            map_configs = self._compute_k_map(model, cardinality, self.k_map)
        else:
            map_config = self._compute_map(model, cardinality)
            map_configs = [map_config]

        # Step 3: Insert MAP into population
        n_insert = min(self.n_map_inserts, len(map_configs), n_samples)

        if self.replace_worst and aux_fitness is not None:
            worst_indices = np.argsort(aux_fitness)[:n_insert]
            for i, idx in enumerate(worst_indices):
                if i < len(map_configs):
                    population[idx] = map_configs[i]
        else:
            for i in range(n_insert):
                population[i] = map_configs[i]

        return population

    def _sample_pls(
        self,
        n_vars: int,
        model: Model,
        cardinality: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Probabilistic Logic Sampling via SampleFDA for FactorizedModel."""
        if isinstance(model, FactorizedModel):
            sampler = SampleFDA(n_samples=n_samples)
            return sampler.sample(
                n_vars=n_vars,
                model=model,
                cardinality=cardinality,
                rng=rng,
            )

        if not isinstance(model, MarkovNetworkModel):
            raise ValueError(f"Unsupported model type: {type(model)}")

        population = np.zeros((n_samples, n_vars), dtype=int)
        cliques = model.structure
        tables = model.parameters

        ordered_cliques, ordered_tables = self._order_cliques(
            list(cliques), list(tables), n_vars
        )

        for i in range(n_samples):
            sampled = np.full(n_vars, -1, dtype=int)

            for clique, table in zip(ordered_cliques, ordered_tables):
                new_vars = [v for v in clique if sampled[v] < 0]

                if not new_vars:
                    continue

                overlap = [v for v in clique if sampled[v] >= 0]
                if overlap:
                    prob_dist = self._get_conditional_distribution(
                        clique, table, sampled, cardinality
                    )
                else:
                    prob_dist = table.ravel()

                if len(prob_dist) > 0 and np.sum(prob_dist) > 0:
                    prob_dist = prob_dist / np.sum(prob_dist)
                    idx = rng.choice(len(prob_dist), p=prob_dist)
                    config = self._index_to_config(
                        idx, [cardinality[v] for v in new_vars]
                    )
                    for j, v in enumerate(new_vars):
                        sampled[v] = config[j]

            for v in range(n_vars):
                if sampled[v] < 0:
                    sampled[v] = rng.integers(0, cardinality[v])

            population[i] = sampled

        return population

    def _order_cliques(
        self,
        cliques: List[np.ndarray],
        tables: List[np.ndarray],
        n_vars: int
    ) -> tuple:
        pairs = list(zip(cliques, tables))
        pairs.sort(key=lambda x: x[0][0] if len(x[0]) > 0 else 0)
        return [p[0] for p in pairs], [p[1] for p in pairs]

    def _get_conditional_distribution(
        self,
        clique: np.ndarray,
        table: np.ndarray,
        sampled: np.ndarray,
        cardinality: np.ndarray
    ) -> np.ndarray:
        expected_shape = tuple(cardinality[v] for v in clique)
        if table.shape != expected_shape:
            table = table.reshape(expected_shape)

        overlap = [i for i, v in enumerate(clique) if sampled[v] >= 0]
        new_vars = [i for i, v in enumerate(clique) if sampled[v] < 0]

        if not new_vars:
            return np.array([1.0])

        slices = [slice(None)] * len(clique)
        for i in overlap:
            slices[i] = sampled[clique[i]]

        return table[tuple(slices)].ravel()

    def _index_to_config(self, index: int, cardinalities: List[int]) -> np.ndarray:
        config = np.zeros(len(cardinalities), dtype=int)
        for i in range(len(cardinalities) - 1, -1, -1):
            config[i] = index % cardinalities[i]
            index //= cardinalities[i]
        return config

    def _compute_map(self, model: Model, cardinality: np.ndarray) -> np.ndarray:
        """Compute MAP; fast path for non-overlapping (UMDA) models."""
        if isinstance(model, FactorizedModel) and _is_non_overlapping(model):
            return _map_non_overlapping(model, cardinality)
        cliques, tables = _extract_cliques_and_tables(model)
        inference = MAPInference(
            cliques=cliques,
            tables=tables,
            cardinalities=cardinality,
            method=self.map_method,
        )
        return inference.compute_map().configuration

    def _compute_k_map(
        self,
        model: Model,
        cardinality: np.ndarray,
        k: int,
    ) -> List[np.ndarray]:
        cliques, tables = _extract_cliques_and_tables(model)
        inference = MAPInference(
            cliques=cliques,
            tables=tables,
            cardinalities=cardinality,
            method=self.map_method,
        )
        result = inference.compute_k_map(k)
        return [result.configurations[i] for i in range(len(result.configurations))]


class SampleTemplateMAP(SamplingMethod):
    """
    Template-MAP Sampling (Strategy S2 from Santana 2013)

    Algorithm:
    1. Compute MAP configuration using message passing
    2. For each new individual:
       - Randomly select variables to inherit from MAP (template_prob per variable)
       - Sample remaining variables from the learned model conditioned on inherited values
    3. Individual 0 is the pure MAP; individuals 1..n_samples-1 are MAP crossovers

    For FactorizedModel, resampled variables are drawn via SamplePartialFDA, which
    correctly propagates parent values through the tree/factor structure.
    For MarkovNetworkModel, a Gibbs-style per-variable fallback is used.

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
        rng: Optional[np.random.Generator] = None,
        **params: Any,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()

        n_samples = params.get('n_samples', self.n_samples)

        # Compute MAP template once
        map_config = self._compute_map(model, cardinality)

        population = np.zeros((n_samples, n_vars), dtype=int)
        population[0] = map_config  # First individual is the pure MAP

        if n_samples == 1:
            return population

        if isinstance(model, FactorizedModel):
            # Build (n_samples-1) templates: MAP values at kept positions, NaN elsewhere.
            # Then batch-sample all at once via SamplePartialFDA.
            templates = np.tile(map_config.astype(float), (n_samples - 1, 1))
            for i in range(n_samples - 1):
                resample_mask = rng.random(n_vars) > self.template_prob
                n_kept = int(np.sum(~resample_mask))
                if n_kept < self.min_template_vars:
                    keep_idx = rng.choice(n_vars, size=self.min_template_vars, replace=False)
                    resample_mask[keep_idx] = False
                templates[i, resample_mask] = np.nan

            partial_sampler = SamplePartialFDA(n_samples=n_samples - 1)
            population[1:] = partial_sampler.sample(
                n_vars=n_vars,
                model=model,
                cardinality=cardinality,
                aux_pop=templates,
                rng=rng,
                n_samples=n_samples - 1,
            )
        else:
            # Fallback for MarkovNetworkModel: per-variable Gibbs-style sampling
            cliques, tables = _extract_cliques_and_tables(model)
            for i in range(1, n_samples):
                individual = map_config.copy()
                resample_mask = rng.random(n_vars) > self.template_prob
                n_kept = int(np.sum(~resample_mask))
                if n_kept < self.min_template_vars:
                    keep_idx = rng.choice(n_vars, size=self.min_template_vars, replace=False)
                    resample_mask[keep_idx] = False
                for var in np.where(resample_mask)[0]:
                    individual[var] = self._sample_variable(
                        var, individual, cliques, tables, cardinality, rng
                    )
                population[i] = individual

        return population

    def _sample_variable(
        self,
        var: int,
        current_config: np.ndarray,
        cliques: List[np.ndarray],
        tables: List[np.ndarray],
        cardinality: np.ndarray,
        rng: np.random.Generator,
    ) -> int:
        """Gibbs-style single-variable sampler for MarkovNetworkModel fallback."""
        return _gibbs_sample_variable(var, current_config, cliques, tables, cardinality, rng)

    def _compute_map(self, model: Model, cardinality: np.ndarray) -> np.ndarray:
        """Compute MAP; fast path for non-overlapping (UMDA) models."""
        if isinstance(model, FactorizedModel) and _is_non_overlapping(model):
            return _map_non_overlapping(model, cardinality)
        cliques, tables = _extract_cliques_and_tables(model)
        inference = MAPInference(
            cliques=cliques,
            tables=tables,
            cardinalities=cardinality,
            method=self.map_method,
        )
        return inference.compute_map().configuration


class SampleHybridMAP(SamplingMethod):
    """
    Hybrid Insert-MAP + Template-MAP (Strategy S3 from Santana 2013)

    Combines both S1 and S2:
    - First half of population: pure PLS samples (S1-style diversity)
    - Second half: MAP-template crossovers via SamplePartialFDA (S2-style exploitation)
    - MAP configuration inserted at the worst individual slot

    This balances wide exploration (PLS) with focused exploitation (template
    crossover and direct MAP insertion).

    Parameters:
        n_samples: Number of individuals to sample
        map_method: Method for MAP computation
        template_prob: Probability of inheriting from MAP in template crossover
        n_map_inserts: Number of pure MAP configurations to insert
    """

    def __init__(
        self,
        n_samples: int,
        map_method: str = "bp",
        template_prob: float = 0.5,
        n_map_inserts: int = 1,
    ):
        self.n_samples = n_samples
        self.map_method = map_method
        self.template_prob = template_prob
        self.n_map_inserts = n_map_inserts

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

        n_samples = params.get('n_samples', self.n_samples)

        # Split: first half from PLS, second half from Template-MAP crossover
        n_pls = n_samples // 2
        n_template = n_samples - n_pls

        # PLS component (broad exploration, like S1)
        pls_pop = SampleFDA(n_samples=n_pls).sample(
            n_vars=n_vars, model=model, cardinality=cardinality, rng=rng
        )

        # Template-MAP component (focused crossover around MAP, like S2)
        map_config = self._compute_map(model, cardinality)
        template_pop = self._sample_template(
            n_vars, model, cardinality, map_config, n_template, rng
        )

        population = np.vstack([pls_pop, template_pop])

        # Insert MAP at worst individual(s)
        n_insert = min(self.n_map_inserts, n_samples)
        if aux_fitness is not None and len(aux_fitness) == n_samples:
            worst_indices = np.argsort(aux_fitness)[:n_insert]
        else:
            worst_indices = np.arange(n_samples - n_insert, n_samples)
        for i in range(n_insert):
            population[worst_indices[i]] = map_config

        return population

    def _sample_template(
        self,
        n_vars: int,
        model: Model,
        cardinality: np.ndarray,
        map_config: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate n_samples individuals using map_config as crossover template."""
        population = np.zeros((n_samples, n_vars), dtype=int)
        population[0] = map_config

        if n_samples == 1:
            return population

        if isinstance(model, FactorizedModel):
            templates = np.tile(map_config.astype(float), (n_samples - 1, 1))
            for i in range(n_samples - 1):
                resample_mask = rng.random(n_vars) > self.template_prob
                if int(np.sum(~resample_mask)) < 1:
                    resample_mask[rng.integers(n_vars)] = False
                templates[i, resample_mask] = np.nan

            partial_sampler = SamplePartialFDA(n_samples=n_samples - 1)
            population[1:] = partial_sampler.sample(
                n_vars=n_vars,
                model=model,
                cardinality=cardinality,
                aux_pop=templates,
                rng=rng,
                n_samples=n_samples - 1,
            )
        else:
            cliques, tables = _extract_cliques_and_tables(model)
            for i in range(1, n_samples):
                individual = map_config.copy()
                resample_mask = rng.random(n_vars) > self.template_prob
                if int(np.sum(~resample_mask)) < 1:
                    resample_mask[rng.integers(n_vars)] = False
                for var in np.where(resample_mask)[0]:
                    individual[var] = _gibbs_sample_variable(
                        var, individual, cliques, tables, cardinality, rng
                    )
                population[i] = individual

        return population

    def _compute_map(self, model: Model, cardinality: np.ndarray) -> np.ndarray:
        """Compute MAP; fast path for non-overlapping (UMDA) models."""
        if isinstance(model, FactorizedModel) and _is_non_overlapping(model):
            return _map_non_overlapping(model, cardinality)
        cliques, tables = _extract_cliques_and_tables(model)
        inference = MAPInference(
            cliques=cliques,
            tables=tables,
            cardinalities=cardinality,
            method=self.map_method,
        )
        return inference.compute_map().configuration


class SampleTemplateBest(SamplingMethod):
    """
    Template-Best Sampling (Strategy S6): crossover with best elite individual.

    Addresses the key failure mode of SampleTemplateMAP (S2): the MAP
    configuration computed from the learned model is worse than the best
    population individual in ~94% of generations (see module docstring).
    Fixing 50% of all offspring's bits to that suboptimal MAP causes a
    catastrophic fitness collapse.

    This class replaces the MAP crossover template with the best verified
    individual from the selected population (aux_pop).  That individual is
    always a high-fitness solution—guaranteed by the selection step—so the
    template quality is reliable at every generation.

    The MAP is still computed and inserted at population[0] (when insert_map
    is True), preserving the direct exploitation benefit of S1-InsertMAP.
    When the MAP genuinely outperforms the current best (rare but possible in
    late generations), it will be selected and become the new best_elite in
    subsequent generations, making the scheme self-correcting.

    Algorithm
    ---------
    1.  Find best individual in aux_pop (the selected/elite population).
    2.  population[0] ← MAP  (direct model-maximum exploitation, like S1).
    3.  For each remaining slot i = 1 … n_samples-1:
        a.  Build a float template from best_elite.
        b.  Set ~(1-template_prob) of the positions to NaN (to be resampled).
        c.  Call SamplePartialFDA to sample the NaN positions from the learned
            model conditioned on the fixed (best_elite) values.
    4.  Falls back to pure PLS when aux_pop is not provided (first generation).

    Parameters
    ----------
    n_samples : int
        Number of individuals to generate.
    map_method : str
        Method for MAP computation (used for population[0] only).
    template_prob : float
        Per-variable probability of inheriting each bit from best_elite.
        Default 0.5 (same as S2); safe here because best_elite is good.
    min_template_vars : int
        Minimum number of variables inherited from the template per individual.
    insert_map : bool
        If True (default), insert MAP at population[0].  If False, all slots
        are best_elite-crossover individuals (no MAP insertion).
    """

    def __init__(
        self,
        n_samples: int,
        map_method: str = "bp",
        template_prob: float = 0.5,
        min_template_vars: int = 1,
        insert_map: bool = True,
    ):
        self.n_samples = n_samples
        self.map_method = map_method
        self.template_prob = template_prob
        self.min_template_vars = min_template_vars
        self.insert_map = insert_map

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
        Generate a new population using best-elite-template crossover.

        Args:
            n_vars: Number of variables.
            model: Learned probabilistic model (FactorizedModel preferred).
            cardinality: Shape (n_vars,) cardinality vector.
            aux_pop: Selected/elite population; shape (n_elite, n_vars).
                     Required for template crossover.  When None (first
                     generation), falls back to pure PLS.
            aux_fitness: Fitness values of aux_pop, shape (n_elite,) or
                         (n_elite, 1).  Used to identify the best individual.
            rng: Optional random-number generator.
            **params: n_samples (int) overrides self.n_samples.

        Returns:
            Population array of shape (n_samples, n_vars).
        """
        if rng is None:
            rng = np.random.default_rng()

        n_samples = params.get("n_samples", self.n_samples)

        # No verified elite available: pure PLS (first generation)
        if aux_pop is None or aux_fitness is None:
            return SampleFDA(n_samples=n_samples).sample(
                n_vars=n_vars, model=model, cardinality=cardinality, rng=rng
            )

        # Best verified individual from the selected population
        fit_flat = np.asarray(aux_fitness).ravel()
        best_idx = int(np.argmax(fit_flat))
        best_elite = np.asarray(aux_pop)[best_idx].copy()

        population = np.zeros((n_samples, n_vars), dtype=int)

        # Position 0: MAP configuration (direct exploitation, like S1)
        if self.insert_map:
            map_config = self._compute_map(model, cardinality)
            population[0] = map_config
            start = 1
        else:
            start = 0

        n_template = n_samples - start
        if n_template == 0:
            return population

        if isinstance(model, FactorizedModel):
            # Efficient batch crossover via SamplePartialFDA.
            # NaN marks positions to resample from the model.
            templates = np.tile(best_elite.astype(float), (n_template, 1))
            for i in range(n_template):
                resample_mask = rng.random(n_vars) > self.template_prob
                n_kept = int(np.sum(~resample_mask))
                if n_kept < self.min_template_vars:
                    keep_idx = rng.choice(n_vars, size=self.min_template_vars, replace=False)
                    resample_mask[keep_idx] = False
                templates[i, resample_mask] = np.nan

            partial = SamplePartialFDA(n_samples=n_template)
            population[start:] = partial.sample(
                n_vars=n_vars,
                model=model,
                cardinality=cardinality,
                aux_pop=templates,
                rng=rng,
                n_samples=n_template,
            )
        else:
            # Gibbs-style fallback for MarkovNetworkModel
            cliques, tables = _extract_cliques_and_tables(model)
            for i in range(n_template):
                individual = best_elite.copy()
                resample_mask = rng.random(n_vars) > self.template_prob
                n_kept = int(np.sum(~resample_mask))
                if n_kept < self.min_template_vars:
                    keep_idx = rng.choice(n_vars, size=self.min_template_vars, replace=False)
                    resample_mask[keep_idx] = False
                for var in np.where(resample_mask)[0]:
                    individual[var] = _gibbs_sample_variable(
                        var, individual, cliques, tables, cardinality, rng
                    )
                population[start + i] = individual

        return population

    def _compute_map(self, model: Model, cardinality: np.ndarray) -> np.ndarray:
        if isinstance(model, FactorizedModel) and _is_non_overlapping(model):
            return _map_non_overlapping(model, cardinality)
        cliques, tables = _extract_cliques_and_tables(model)
        inference = MAPInference(
            cliques=cliques,
            tables=tables,
            cardinalities=cardinality,
            method=self.map_method,
        )
        return inference.compute_map().configuration
