"""
K-MAP Sampling Methods for EDAs

Implements two sampling strategies that incorporate the K most probable
configurations (K-MPCs) into the generated population:

SampleInsertKMAP (S4 / Insert-K-MAP)
    1. Sample M individuals using Probabilistic Logic Sampling (PLS).
    2. Compute the K most probable configurations using KMPC.
    3. Replace the K worst individuals with the K-MPCs.

    Extends the single-MAP Insert-MAP (S1) from Santana (2013) to K
    configurations, providing richer exploitation of the probabilistic model.

SampleTemplateKMAP (S5 / Template-K-MAP)
    1. Compute K most probable configurations.
    2. Divide the population into K approximately equal groups.
    3. For each group, use the corresponding K-MPC as a template:
       each individual is created by inheriting variables from the template
       with probability *template_prob* and resampling the rest from the model.

    Extends the Template-MAP (S2) from Santana (2013) to K templates,
    creating diversity around K high-quality solutions simultaneously.

References
----------
Nilsson, D. (1998). An efficient algorithm for finding the M most probable
  configurations in probabilistic expert systems. Statistics and Computing, 8.
Echegoyen, C. et al. (2010). Analyzing the k Most Probable Solutions in EDAs
  Based on Bayesian Networks.
Santana, R. (2013). Message Passing Methods for Estimation of Distribution
  Algorithms Based on Markov Networks.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from pateda.core.components import SamplingMethod
from pateda.core.models import FactorizedModel, MarkovNetworkModel, Model
from pateda.inference.kmpc import KMPCResult, compute_kmpc
from pateda.sampling.fda import SampleFDA


# ---------------------------------------------------------------------------
# Shared helper: PLS sampling for factorized models
# ---------------------------------------------------------------------------

def _sample_pls(
    n_vars: int,
    model: Model,
    cardinality: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample *n_samples* individuals via Probabilistic Logic Sampling (PLS).

    Delegates to the efficient ``SampleFDA`` implementation when the model is
    a ``FactorizedModel``.  For ``MarkovNetworkModel``, falls back to a simple
    clique-based forward sampler.
    """
    if isinstance(model, FactorizedModel):
        sampler = SampleFDA(n_samples=n_samples)
        return sampler.sample(
            n_vars=n_vars,
            model=model,
            cardinality=cardinality,
            rng=rng,
        )

    # Fallback for Markov networks: sample each variable from its marginal
    population = np.zeros((n_samples, n_vars), dtype=int)
    if isinstance(model, MarkovNetworkModel):
        cliques = model.structure
        tables = model.parameters
        for s in range(n_samples):
            for clique, table in zip(cliques, tables):
                flat = table.ravel()
                flat = np.maximum(flat, 0)
                total = flat.sum()
                if total <= 0:
                    continue
                flat = flat / total
                idx = rng.choice(len(flat), p=flat)
                # Decode index into variable values
                card_clique = [int(cardinality[v]) for v in clique]
                vals = []
                for c in reversed(card_clique):
                    vals.append(idx % c)
                    idx //= c
                vals = list(reversed(vals))
                for v, val in zip(clique, vals):
                    population[s, v] = val
    else:
        # Unknown model: random initialization
        for i in range(n_vars):
            population[:, i] = rng.integers(0, int(cardinality[i]), size=n_samples)

    return population


def _sample_variable_from_model(
    var: int,
    current_config: np.ndarray,
    model: Model,
    cardinality: np.ndarray,
    rng: np.random.Generator,
) -> int:
    """Sample variable *var* conditioned on *current_config* using the model."""
    card_v = int(cardinality[var])
    log_marginal = np.zeros(card_v)

    if isinstance(model, (FactorizedModel, MarkovNetworkModel)):
        if isinstance(model, FactorizedModel):
            structure = model.structure
            tables = model.parameters
            n_cliques = structure.shape[0]
            for c in range(n_cliques):
                n_overlap = int(structure[c, 0])
                n_new = int(structure[c, 1])
                overlap_vars = structure[c, 2:2 + n_overlap].astype(int)
                new_vars = structure[c, 2 + n_overlap:2 + n_overlap + n_new].astype(int)
                all_vars = list(overlap_vars) + list(new_vars)
                if var not in all_vars:
                    continue
                table = tables[c]
                var_pos = all_vars.index(var)
                for val in range(card_v):
                    idx = [int(current_config[v]) for v in all_vars]
                    idx[var_pos] = val
                    try:
                        prob = float(table[tuple(idx)])
                    except (IndexError, TypeError):
                        prob = 1e-300
                    log_marginal[val] += np.log(max(prob, 1e-300))
        else:
            cliques = model.structure
            tables = model.parameters
            for clique, table in zip(cliques, tables):
                cv = list(clique)
                if var not in cv:
                    continue
                var_pos = cv.index(var)
                for val in range(card_v):
                    idx = [int(current_config[v]) for v in cv]
                    idx[var_pos] = val
                    try:
                        prob = float(table[tuple(idx)])
                    except (IndexError, TypeError):
                        prob = 1e-300
                    log_marginal[val] += np.log(max(prob, 1e-300))

    # Normalize and sample
    log_marginal -= log_marginal.max()  # numerical stability
    marginal = np.exp(log_marginal)
    total = marginal.sum()
    if total <= 0:
        return int(rng.integers(0, card_v))
    marginal /= total
    return int(rng.choice(card_v, p=marginal))


# ---------------------------------------------------------------------------
# S4: Insert-K-MAP
# ---------------------------------------------------------------------------

class SampleInsertKMAP(SamplingMethod):
    """Insert-K-MAP Sampling (Strategy S4).

    Extends the single-MAP Insert-MAP (S1) to the K most probable
    configurations.  The algorithm is:

    1.  Generate *n_samples* individuals using PLS.
    2.  Compute the K most probable configurations (K-MPCs) via KMPC.
    3.  Insert the K-MPCs into the population, replacing the *k* worst
        individuals (or a random selection when fitness is unavailable).

    This strategy guarantees that the K-MPCs are always present in the
    population, increasing exploitation while the PLS individuals maintain
    diversity.

    Parameters
    ----------
    n_samples : int
        Population size to generate.
    k : int
        Number of most probable configurations to insert (default 5).
    replace_worst : bool
        If True, replace worst individuals by fitness (requires aux_fitness).
        If False, replace the first *k* individuals.
    """

    def __init__(
        self,
        n_samples: int,
        k: int = 5,
        replace_worst: bool = True,
    ) -> None:
        self.n_samples = n_samples
        self.k = k
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
        """Sample population using Insert-K-MAP strategy.

        Parameters
        ----------
        n_vars : int
        model : FactorizedModel or MarkovNetworkModel
        cardinality : (n_vars,) array
        aux_pop : optional current population (not used)
        aux_fitness : optional current fitness (used when replace_worst=True)
        rng : random generator
        **params :
            n_samples : override instance n_samples
            k : override instance k

        Returns
        -------
        population : ndarray of shape (n_samples, n_vars)
        """
        if rng is None:
            rng = np.random.default_rng()

        n_samples = int(params.get("n_samples", self.n_samples))
        k = int(params.get("k", self.k))
        k = min(k, n_samples)

        # Step 1: PLS sampling
        population = _sample_pls(n_vars, model, cardinality, n_samples, rng)

        # Step 2: Compute K most probable configurations
        try:
            result: KMPCResult = compute_kmpc(model, cardinality, k)
            kmpc_configs = result.configurations[: result.k_found]
        except Exception:
            # Graceful fallback: no insertion
            return population

        n_insert = min(k, len(kmpc_configs), n_samples)

        # Step 3: Insert K-MPCs into population
        if self.replace_worst and aux_fitness is not None and len(aux_fitness) == n_samples:
            worst_indices = np.argsort(aux_fitness)[:n_insert]
            for i, idx in enumerate(worst_indices):
                if i < len(kmpc_configs):
                    population[idx] = kmpc_configs[i]
        else:
            for i in range(n_insert):
                population[i] = kmpc_configs[i]

        return population


# ---------------------------------------------------------------------------
# S5: Template-K-MAP
# ---------------------------------------------------------------------------

class SampleTemplateKMAP(SamplingMethod):
    """Template-K-MAP Sampling (Strategy S5).

    Extends the Template-MAP (S2) to K templates simultaneously:

    1.  Compute the K most probable configurations (K-MPCs) via KMPC.
    2.  Divide the *n_samples* population into K approximately equal groups.
    3.  For each group, the corresponding K-MPC is used as a template:
        each individual in the group inherits each variable from the template
        with probability *template_prob* and resamples the remainder from the
        learned probabilistic model.
    4.  The first individual of each group is the pure K-MPC (no resampling).

    This strategy simultaneously explores neighbourhoods around K high-quality
    solutions, balancing exploitation of the K-MPCs with model-guided
    exploration.

    Parameters
    ----------
    n_samples : int
        Population size to generate.
    k : int
        Number of most probable configuration templates to use (default 5).
    template_prob : float
        Per-variable probability of inheriting from the template (default 0.5).
    min_template_vars : int
        Minimum variables to inherit from the template (default 1).
    fallback_pls : bool
        If KMPC fails, fall back to PLS sampling (default True).
    """

    def __init__(
        self,
        n_samples: int,
        k: int = 5,
        template_prob: float = 0.5,
        min_template_vars: int = 1,
        fallback_pls: bool = True,
    ) -> None:
        self.n_samples = n_samples
        self.k = k
        self.template_prob = template_prob
        self.min_template_vars = min_template_vars
        self.fallback_pls = fallback_pls

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
        """Sample population using Template-K-MAP strategy.

        Parameters
        ----------
        n_vars : int
        model : FactorizedModel or MarkovNetworkModel
        cardinality : (n_vars,) array
        aux_pop : optional (not used)
        aux_fitness : optional (not used)
        rng : random generator
        **params :
            n_samples : override instance n_samples
            k : override instance k
            template_prob : override instance template_prob

        Returns
        -------
        population : ndarray of shape (n_samples, n_vars)
        """
        if rng is None:
            rng = np.random.default_rng()

        n_samples = int(params.get("n_samples", self.n_samples))
        k = int(params.get("k", self.k))
        template_prob = float(params.get("template_prob", self.template_prob))

        # Step 1: Compute K most probable configurations
        try:
            result: KMPCResult = compute_kmpc(model, cardinality, k)
            templates = result.configurations[: result.k_found]
            k_actual = len(templates)
        except Exception:
            if self.fallback_pls:
                return _sample_pls(n_vars, model, cardinality, n_samples, rng)
            raise

        population = np.zeros((n_samples, n_vars), dtype=int)

        # Step 2: Divide population into k_actual groups
        group_sizes = self._compute_group_sizes(n_samples, k_actual)

        sample_idx = 0
        for g, (template, group_size) in enumerate(
            zip(templates, group_sizes)
        ):
            if group_size == 0:
                continue

            # First individual in each group = pure template
            population[sample_idx] = template.copy()
            sample_idx += 1

            # Step 3: Generate remaining group members around template
            for _ in range(group_size - 1):
                individual = template.copy()

                # Select variables to resample
                resample_mask = rng.random(n_vars) > template_prob

                # Enforce minimum template variables
                n_inherited = int(np.sum(~resample_mask))
                if n_inherited < self.min_template_vars:
                    keep = rng.choice(
                        n_vars, size=self.min_template_vars, replace=False
                    )
                    resample_mask[keep] = False

                # Resample selected variables from model
                for var in np.where(resample_mask)[0]:
                    individual[var] = _sample_variable_from_model(
                        int(var), individual, model, cardinality, rng
                    )

                if sample_idx < n_samples:
                    population[sample_idx] = individual
                    sample_idx += 1

        # Fill any remaining slots with PLS samples
        if sample_idx < n_samples:
            extra = _sample_pls(
                n_vars, model, cardinality, n_samples - sample_idx, rng
            )
            population[sample_idx:] = extra

        return population

    @staticmethod
    def _compute_group_sizes(n_samples: int, k: int) -> List[int]:
        """Divide n_samples into k approximately equal groups."""
        base = n_samples // k
        remainder = n_samples % k
        sizes = [base + (1 if i < remainder else 0) for i in range(k)]
        return sizes
