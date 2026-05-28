"""
Compare discrete EDA sampling variants on packaged UBQP real-world instances.

Based on ``scripts/compare_discrete_edas_rw.py`` but restricted to:
- Algorithms: UMDA, TreeEDA, TreeEDA-r, MNFDA, MNFDAR, FDA
- Problem: UBQP
- Instances: bqp50.txt, bqp100.txt
- Sampling variants:
  - Probabilistic Logic Sampling (PLS / baseline)
  - Partial sampling
  - Insert-MAP (S1)
  - Template-MAP (S2)
  - Insert-MAP + Template-MAP (S3)
  - Insert-K-MAP (S4)
  - Template-K-MAP (S5)

Usage:
    python scripts/compare_sampling_variants_rw.py
"""

from __future__ import annotations

import time
import traceback
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from pateda import UMDA, TreeEDA, TreeEDAR, MNFDA, MNFDAR, FDA
from pateda.core.components import SamplingMethod
from pateda.functions.discrete.ubqp import (
    build_ubqp_interaction_matrix,
    evaluate_ubqp,
    load_ubqp_benchmark_instance,
)
from pateda.sampling.fda import SampleFDA
from pateda.sampling.kmap_sampling import SampleInsertKMAP, SampleTemplateKMAP
from pateda.sampling.map_sampling import (
    SampleHybridMAP,
    SampleInsertMAP,
    SampleTemplateMAP,
    SampleTemplateBest,
)
from pateda.sampling.partial import SamplePartialFDA


ALGORITHMS = [
    ("UMDA", UMDA),
    ("TreeEDA", TreeEDA),
    ("TreeEDA-r", TreeEDAR),
    ("MNFDA", MNFDA),
    ("MNFDAR", MNFDAR),
    ("FDA", FDA),
]

PROBLEMS = [
    ("UBQP", "bqp50.txt"),
    ("UBQP", "bqp100.txt"),
]

RESTRICTED_ALGORITHMS = {"TreeEDA-r", "MNFDAR"}
POP_SIZE = 200
N_GEN = 50
SEL_RATIO = 0.5
N_RUNS = 3
SEEDS = list(range(1, N_RUNS + 1))
UBQP_THRESHOLD_RATIO = 0.5  # Keep strongest UBQP interactions when building priors


class _FlattenFitnessSampler(SamplingMethod):
    """Adapter that flattens (N,1) fitness arrays before delegating."""

    def __init__(self, sampler: SamplingMethod):
        self._sampler = sampler

    def sample(
        self,
        n_vars: int,
        model: Any,
        cardinality: np.ndarray,
        aux_pop: Optional[np.ndarray] = None,
        aux_fitness: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
        **params: Any,
    ) -> np.ndarray:
        fit = aux_fitness
        if fit is not None:
            fit = np.asarray(fit)
            if fit.ndim > 1:
                fit = fit.reshape(-1)

        return self._sampler.sample(
            n_vars=n_vars,
            model=model,
            cardinality=cardinality,
            aux_pop=aux_pop,
            aux_fitness=fit,
            rng=rng,
            **params,
        )


class _RandomMaskPartialSampler(SamplingMethod):
    """Use current population as template and set random positions to NaN."""

    def __init__(self, n_samples: int, inherit_prob: float = 0.5):
        self.inherit_prob = inherit_prob
        self._inner = SamplePartialFDA(n_samples=n_samples)

    def sample(
        self,
        n_vars: int,
        model: Any,
        cardinality: np.ndarray,
        aux_pop: Optional[np.ndarray] = None,
        aux_fitness: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
        **params: Any,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()

        templates = None
        if aux_pop is not None:
            templates = np.asarray(aux_pop, dtype=float).copy()
            keep_mask = rng.random(templates.shape) < self.inherit_prob
            templates[~keep_mask] = np.nan

        return self._inner.sample(
            n_vars=n_vars,
            model=model,
            cardinality=cardinality,
            aux_pop=templates,
            aux_fitness=aux_fitness,
            rng=rng,
            **params,
        )


def _single_objective(values):
    """Convert scalar-like outputs to scalar/1-D single-objective outputs."""
    arr = np.asarray(values)
    if arr.ndim == 0:
        return float(arr)
    if arr.ndim == 1:
        return float(arr[0]) if arr.size == 1 else arr
    return arr[:, 0]


def load_problem(problem_type: str, instance_name: str):
    """Load UBQP benchmark and interaction matrix."""
    if problem_type.upper() != "UBQP":
        raise ValueError(f"Unsupported problem type: {problem_type}")

    ubqp_instance, optimal = load_ubqp_benchmark_instance(instance_name)
    interaction_matrix = build_ubqp_interaction_matrix(
        ubqp_instance,
        threshold_ratio=UBQP_THRESHOLD_RATIO,
    )

    def fitness_func(solution):
        return _single_objective(evaluate_ubqp(np.asarray(solution), ubqp_instance))

    return fitness_func, ubqp_instance.n_vars, 2, interaction_matrix, optimal


def _build_sampler(sampling_name: str) -> SamplingMethod:
    if sampling_name == "PLS":
        return SampleFDA(n_samples=POP_SIZE)
    if sampling_name == "Partial":
        return _RandomMaskPartialSampler(n_samples=POP_SIZE, inherit_prob=0.5)
    if sampling_name == "S1-InsertMAP":
        return _FlattenFitnessSampler(
            SampleInsertMAP(
                n_samples=POP_SIZE,
                map_method="bp",
                n_map_inserts=1,
                k_map=1,
                replace_worst=True,
            )
        )
    if sampling_name == "S2-TemplateMAP":
        # WARNING: template_prob=0.5 locks 50% of each offspring's bits to the
        # model MAP, which is worse than the best population individual in ~94%
        # of generations (verified on bqp50/TreeEDA).  Results are catastrophic
        # (mean ~700 vs PLS ~1051).  See S6-TemplateBest for a working fix.
        return SampleTemplateMAP(
            n_samples=POP_SIZE,
            map_method="bp",
            template_prob=0.5,
            min_template_vars=1,
        )
    if sampling_name == "S2-TemplateMAP-p01":
        # S2 with a much smaller template probability (0.1 instead of 0.5):
        # only 10% of bits are fixed to MAP per offspring.  Greatly reduces
        # MAP lock-in damage while retaining some MAP guidance.
        return SampleTemplateMAP(
            n_samples=POP_SIZE,
            map_method="bp",
            template_prob=0.1,
            min_template_vars=1,
        )
    if sampling_name == "S3-HybridMAP":
        return SampleHybridMAP(
            n_samples=POP_SIZE,
            map_method="bp",
            template_prob=0.5,
            n_map_inserts=1,
        )
    if sampling_name == "S6-TemplateBest":
        # Fixed alternative to S2: uses the best verified elite individual
        # (from aux_pop) as the crossover template instead of the unverified
        # model MAP.  MAP is still inserted at position 0 (like S1).
        # Template quality is guaranteed good at every generation.
        return _FlattenFitnessSampler(
            SampleTemplateBest(
                n_samples=POP_SIZE,
                map_method="bp",
                template_prob=0.5,
                min_template_vars=1,
                insert_map=True,
            )
        )
    if sampling_name == "S6-TemplateBest-p01":
        # S6 with template_prob=0.1: only 10% of bits fixed to best_elite.
        # Lower inheritance fraction preserves early-generation diversity
        # while still guiding search toward the current best known solution.
        return _FlattenFitnessSampler(
            SampleTemplateBest(
                n_samples=POP_SIZE,
                map_method="bp",
                template_prob=0.1,
                min_template_vars=1,
                insert_map=True,
            )
        )
    if sampling_name == "S4-InsertKMAP":
        return _FlattenFitnessSampler(
            SampleInsertKMAP(
                n_samples=POP_SIZE,
                k=5,
                replace_worst=True,
            )
        )
    if sampling_name == "S5-TemplateKMAP":
        return SampleTemplateKMAP(
            n_samples=POP_SIZE,
            k=5,
            template_prob=0.5,
            min_template_vars=1,
            fallback_pls=True,
        )

    raise ValueError(f"Unknown sampling variant: {sampling_name}")


SAMPLING_VARIANTS: List[str] = [
    "PLS",
    "Partial",
    "S1-InsertMAP",
    "S2-TemplateMAP",           # broken: MAP lock-in, mean ~700 vs PLS ~1051
    "S2-TemplateMAP-p01",       # partial fix: lower template_prob reduces MAP lock-in
    "S3-HybridMAP",
    "S4-InsertKMAP",
    "S5-TemplateKMAP",
    "S6-TemplateBest",          # fix (tp=0.5): best elite template + MAP at slot 0
    "S6-TemplateBest-p01",      # fix (tp=0.1): best elite template, minimal inheritance
]


def run_single_seed(
    alg_name: str,
    alg_cls: Callable,
    sampling_name: str,
    n_vars: int,
    cardinality: int,
    fitness_func: Callable,
    interaction_matrix: np.ndarray,
    seed: int,
) -> Tuple[float, float]:
    """Run one (algorithm, sampling variant, instance, seed)."""
    kwargs = {}
    if alg_name in RESTRICTED_ALGORITHMS:
        kwargs["interaction_matrix"] = interaction_matrix

    alg = alg_cls(
        n_vars=n_vars,
        cardinality=cardinality,
        fitness_func=fitness_func,
        pop_size=POP_SIZE,
        n_gen=N_GEN,
        selection_ratio=SEL_RATIO,
        random_seed=seed,
        **kwargs,
    )

    alg._eda.components.sampling = _build_sampler(sampling_name)  # noqa: SLF001

    t0 = time.time()
    stats, _ = alg.run(verbose=False)
    elapsed = time.time() - t0
    return float(stats.best_fitness_overall), elapsed


def run_all_seeds(
    alg_name: str,
    alg_cls: Callable,
    sampling_name: str,
    n_vars: int,
    cardinality: int,
    fitness_func: Callable,
    interaction_matrix: np.ndarray,
) -> Tuple[List[float], List[float]]:
    """Repeat one configuration across ``N_RUNS`` seeds."""
    bests: List[float] = []
    times: List[float] = []
    for seed in SEEDS:
        best, elapsed = run_single_seed(
            alg_name=alg_name,
            alg_cls=alg_cls,
            sampling_name=sampling_name,
            n_vars=n_vars,
            cardinality=cardinality,
            fitness_func=fitness_func,
            interaction_matrix=interaction_matrix,
            seed=seed,
        )
        bests.append(best)
        times.append(elapsed)
    return bests, times


def main():
    name_w = 14
    samp_w = 16

    for problem_type, instance_name in PROBLEMS:
        fitness_func, n_vars, cardinality, interaction_matrix, optimal = load_problem(
            problem_type,
            instance_name,
        )
        n_edges = int(np.sum(np.triu(interaction_matrix, k=1)))

        print(f"\n{'=' * 104}")
        print(
            f"Problem: {problem_type}  Instance: {instance_name}  "
            f"(n_vars={n_vars}, cardinality={cardinality}, "
            f"prior_edges={n_edges}, optimal={optimal})"
        )
        print(
            f"pop_size={POP_SIZE}  n_gen={N_GEN}  selection_ratio={SEL_RATIO}  "
            f"n_runs={N_RUNS}  seeds={SEEDS}"
        )
        print(f"{'=' * 104}")

        tag = f"{problem_type} {instance_name}"

        for sampling_name in SAMPLING_VARIANTS:
            print(f"\n-- Sampling variant: {sampling_name}")
            for alg_name, alg_cls in ALGORITHMS:
                try:
                    bests, times = run_all_seeds(
                        alg_name=alg_name,
                        alg_cls=alg_cls,
                        sampling_name=sampling_name,
                        n_vars=n_vars,
                        cardinality=cardinality,
                        fitness_func=fitness_func,
                        interaction_matrix=interaction_matrix,
                    )
                    mean_best = float(np.mean(bests))
                    mean_time = float(np.mean(times))
                    bests_str = "[" + ", ".join(f"{b:.4f}" for b in bests) + "]"
                    print(
                        f"{alg_name:<{name_w}} {sampling_name:<{samp_w}} {tag:<20}: "
                        f"{bests_str}  mean={mean_best:.4f}  time={mean_time:.2f}s"
                    )
                except Exception as exc:
                    print(
                        f"{alg_name:<{name_w}} {sampling_name:<{samp_w}} {tag:<20}: "
                        f"ERROR -- {exc}"
                    )
                    traceback.print_exc()

    print()


if __name__ == "__main__":
    main()
