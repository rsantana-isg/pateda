"""
MOA (Markovianity Based Optimization Algorithm) for Deceptive3 Function

Implements MOA with Gibbs sampling and compares three Gibbs variants:

  - SampleGibbs (baseline)    : sequential per-individual Python loops
  - SampleGibbsBatch          : vectorised across all individuals simultaneously;
                                eliminates the outer n_samples Python loop
  - SampleGibbsWarm           : warm-starts each generation from the current
                                population and uses fewer sweeps (IT instead of
                                IT * n * log(n)); exploits temporal locality
                                across EDA generations
  - SampleGibbsBatchWarm      : combines both improvements

Timing and quality (best fitness, generation found) are compared across
all variants on Deceptive3 with n=30, pop=200.

References:
- Santana, R. (2013). "Message Passing Methods for EDAs Based on Markov Networks"
"""

import time
import numpy as np
from typing import Any, Optional

from pateda.core.eda import EDA, EDAComponents
from pateda.core.components import SamplingMethod
from pateda.core.models import Model, MarkovNetworkModel
from pateda.stop_conditions import MaxGenerations
from pateda.seeding import RandomInit
from pateda.selection import TruncationSelection
from pateda.replacement import ElitistReplacement
from pateda.learning.moa import LearnMOA
from pateda.sampling.gibbs import SampleGibbs
from pateda.learning.utils.conversions import find_acc_card, num_convert_card
from pateda.functions.discrete_binary.toy_functions.deceptive import deceptive3


# ---------------------------------------------------------------------------
# Efficient Gibbs variants
# ---------------------------------------------------------------------------

class SampleGibbsBatch(SamplingMethod):
    """
    Vectorised Gibbs sampling — updates all n_samples individuals simultaneously.

    Speedup over SampleGibbs:
    - Eliminates the outer ``for sample_idx in range(n_samples)`` Python loop.
    - For each (iteration, variable) step, all individuals are updated using
      numpy operations: mixed-radix index computation, table lookup, and
      categorical sampling are all performed as batch array operations.

    Complexity per sweep: O(IT * n_vars * n_samples) numpy ops
    vs O(n_samples * IT * n_vars * n_vars) Python ops for the baseline.
    """

    def __init__(self, n_samples: int, IT: int = 4,
                 temperature: float = 1.0, random_order: bool = True):
        self.n_samples = n_samples
        self.IT = IT
        self.temperature = temperature
        self.random_order = random_order

    def sample(self, n_vars, model, cardinality, aux_pop=None,
               aux_fitness=None, rng=None, **params):
        if rng is None:
            rng = np.random.default_rng()
        if not isinstance(model, MarkovNetworkModel):
            raise TypeError(f"SampleGibbsBatch requires MarkovNetworkModel")

        n_samples = int(params.get("n_samples", self.n_samples))
        n_iters = int(self.IT * n_vars * np.log(max(n_vars, 2)))

        # Initialise population randomly
        pop = np.zeros((n_samples, n_vars), dtype=int)
        for v in range(n_vars):
            pop[:, v] = rng.integers(0, int(cardinality[v]), size=n_samples)

        cliques = model.structure
        tables = model.parameters

        for _ in range(n_iters):
            var_order = (rng.permutation(n_vars) if self.random_order
                         else np.arange(n_vars))
            for v in var_order:
                card_v = int(cardinality[v])
                clique = cliques[v]
                table = tables[v]
                neighbors = clique[clique != v].astype(int)

                if len(neighbors) == 0:
                    # Marginal: broadcast same probs to all individuals
                    probs = np.broadcast_to(
                        table[None, :], (n_samples, card_v)
                    ).copy()
                else:
                    # Compute mixed-radix neighbour index for every individual
                    nb_idx = np.zeros(n_samples, dtype=int)
                    mult = 1
                    for nb in neighbors:
                        nb_idx += pop[:, nb] * mult
                        mult *= int(cardinality[nb])
                    # table shape: (prod_neighbour_cards, card_v)
                    probs = table[nb_idx, :]          # (n_samples, card_v)

                # Apply Boltzmann temperature
                if self.temperature != 1.0:
                    log_p = np.log(probs + 1e-10) / self.temperature
                    log_p -= log_p.max(axis=1, keepdims=True)
                    probs = np.exp(log_p)

                # Normalize row-wise
                row_sums = probs.sum(axis=1, keepdims=True)
                row_sums = np.where(row_sums <= 0, 1.0, row_sums)
                probs /= row_sums

                # Batch categorical sample via inverse-CDF
                cdf = np.cumsum(probs, axis=1)
                u = rng.random(n_samples)[:, None]
                pop[:, v] = np.argmax(cdf >= u, axis=1)

        return pop


class SampleGibbsWarm(SamplingMethod):
    """
    Warm-start Gibbs sampling.

    Instead of initialising each generation from scratch (random), the
    current population (aux_pop) is used as the starting configuration.
    Because aux_pop already comes from the previous generation, it is
    close to the stationary distribution of the new model, so far fewer
    sweeps are needed.

    Number of sweeps: IT (not IT * n * log(n)) when warm-started.
    Falls back to IT * n * log(n) sweeps with random init if aux_pop
    is unavailable (first generation).
    """

    def __init__(self, n_samples: int, IT: int = 4,
                 temperature: float = 1.0, random_order: bool = True):
        self.n_samples = n_samples
        self.IT = IT
        self.temperature = temperature
        self.random_order = random_order

    def sample(self, n_vars, model, cardinality, aux_pop=None,
               aux_fitness=None, rng=None, **params):
        if rng is None:
            rng = np.random.default_rng()
        if not isinstance(model, MarkovNetworkModel):
            raise TypeError(f"SampleGibbsWarm requires MarkovNetworkModel")

        n_samples = int(params.get("n_samples", self.n_samples))

        # Warm-start: reuse previous population when available
        if aux_pop is not None and len(aux_pop) >= n_samples:
            pop = aux_pop[:n_samples].copy()
            # Only IT sweeps needed — already near stationary distribution
            n_iters = self.IT
        else:
            pop = np.zeros((n_samples, n_vars), dtype=int)
            for v in range(n_vars):
                pop[:, v] = rng.integers(0, int(cardinality[v]), size=n_samples)
            n_iters = int(self.IT * n_vars * np.log(max(n_vars, 2)))

        cliques = model.structure
        tables = model.parameters

        for sample_idx in range(n_samples):
            config = pop[sample_idx].copy()
            for _ in range(n_iters):
                var_order = (rng.permutation(n_vars) if self.random_order
                             else np.arange(n_vars))
                for v in var_order:
                    card_v = int(cardinality[v])
                    clique = cliques[v]
                    table = tables[v]
                    neighbors = clique[clique != v].astype(int)

                    if len(neighbors) == 0:
                        probs = table
                    else:
                        nb_vals = config[neighbors].astype(int)
                        nb_cards = cardinality[neighbors].astype(int)
                        acc = find_acc_card(len(neighbors), nb_cards)
                        nb_idx = num_convert_card(nb_vals, len(neighbors), acc)
                        probs = (table if table.ndim == 1
                                 else table[nb_idx, :])

                    if self.temperature != 1.0:
                        log_p = np.log(probs + 1e-10) / self.temperature
                        probs = np.exp(log_p - log_p.max())

                    total = probs.sum()
                    if total <= 0:
                        config[v] = rng.integers(0, card_v)
                    else:
                        config[v] = rng.choice(card_v, p=probs / total)

            pop[sample_idx] = config

        return pop


class SampleGibbsBatchWarm(SamplingMethod):
    """
    Combined: vectorised across population + warm-start from aux_pop.

    Achieves the best of both variants:
    - Batch numpy ops per (iter, variable) step  → eliminates n_samples loop
    - Warm-start from previous population        → fewer sweeps needed
    """

    def __init__(self, n_samples: int, IT: int = 4,
                 temperature: float = 1.0, random_order: bool = True):
        self.n_samples = n_samples
        self.IT = IT
        self.temperature = temperature
        self.random_order = random_order

    def sample(self, n_vars, model, cardinality, aux_pop=None,
               aux_fitness=None, rng=None, **params):
        if rng is None:
            rng = np.random.default_rng()
        if not isinstance(model, MarkovNetworkModel):
            raise TypeError(f"SampleGibbsBatchWarm requires MarkovNetworkModel")

        n_samples = int(params.get("n_samples", self.n_samples))

        if aux_pop is not None and len(aux_pop) >= n_samples:
            pop = aux_pop[:n_samples].copy()
            n_iters = self.IT
        else:
            pop = np.zeros((n_samples, n_vars), dtype=int)
            for v in range(n_vars):
                pop[:, v] = rng.integers(0, int(cardinality[v]), size=n_samples)
            n_iters = int(self.IT * n_vars * np.log(max(n_vars, 2)))

        cliques = model.structure
        tables = model.parameters

        for _ in range(n_iters):
            var_order = (rng.permutation(n_vars) if self.random_order
                         else np.arange(n_vars))
            for v in var_order:
                card_v = int(cardinality[v])
                clique = cliques[v]
                table = tables[v]
                neighbors = clique[clique != v].astype(int)

                if len(neighbors) == 0:
                    probs = np.broadcast_to(
                        table[None, :], (n_samples, card_v)
                    ).copy()
                else:
                    nb_idx = np.zeros(n_samples, dtype=int)
                    mult = 1
                    for nb in neighbors:
                        nb_idx += pop[:, nb] * mult
                        mult *= int(cardinality[nb])
                    probs = table[nb_idx, :]

                if self.temperature != 1.0:
                    log_p = np.log(probs + 1e-10) / self.temperature
                    log_p -= log_p.max(axis=1, keepdims=True)
                    probs = np.exp(log_p)

                row_sums = probs.sum(axis=1, keepdims=True)
                row_sums = np.where(row_sums <= 0, 1.0, row_sums)
                probs /= row_sums

                cdf = np.cumsum(probs, axis=1)
                u = rng.random(n_samples)[:, None]
                pop[:, v] = np.argmax(cdf >= u, axis=1)

        return pop


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_moa(sampler, pop_size, n_vars, max_gen, seed, verbose=False):
    """Run one MOA experiment and return (stats, elapsed_seconds)."""
    optimal = n_vars / 3.0
    components = EDAComponents(
        seeding=RandomInit(),
        selection=TruncationSelection(ratio=0.5),
        learning=LearnMOA(k_neighbors=8, threshold_factor=1.5, prior=True),
        sampling=sampler,
        replacement=ElitistReplacement(n_elite=10),
        stop_condition=MaxGenerations(max_gen=max_gen),
    )
    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        fitness_func=deceptive3,
        cardinality=np.full(n_vars, 2),
        components=components,
        random_seed=seed,
    )
    t0 = time.time()
    stats, _ = eda.run(verbose=verbose)
    elapsed = time.time() - t0
    return stats, elapsed, optimal


# ---------------------------------------------------------------------------
# Main: single run + multi-run comparison
# ---------------------------------------------------------------------------

def demo_single():
    """Single run of the original MOA for illustration."""
    print("=" * 80)
    print("MOA Algorithm for Deceptive3 Function (single run, verbose)")
    print("=" * 80)
    n_vars, pop_size, max_gen = 30, 200, 50
    sampler = SampleGibbs(n_samples=pop_size, IT=4, temperature=1.0, random_order=True)
    stats, elapsed, optimal = run_moa(sampler, pop_size, n_vars, max_gen, seed=42, verbose=True)
    print(f"\nBest fitness: {stats.best_fitness_overall:.4f}  (optimal {optimal:.4f})")
    print(f"Generation found: {stats.generation_found}")
    print(f"Wall time: {elapsed:.2f}s")


def compare_variants(n_runs=5):
    """Compare the four Gibbs variants over multiple runs."""
    print("\n" + "=" * 80)
    print("Gibbs Sampling Variants — Time vs Quality Comparison")
    print("=" * 80)

    n_vars    = 30     # 10 × deceptive3 blocks
    pop_size  = 200    # reduced so comparison is fast
    max_gen   = 50
    optimal   = n_vars / 3.0

    variants = [
        ("Gibbs (baseline)",   lambda: SampleGibbs(n_samples=pop_size, IT=4, random_order=True)),
        ("GibbsBatch",         lambda: SampleGibbsBatch(n_samples=pop_size, IT=4, random_order=True)),
        ("GibbsWarm",          lambda: SampleGibbsWarm(n_samples=pop_size, IT=4, random_order=True)),
        ("GibbsBatchWarm",     lambda: SampleGibbsBatchWarm(n_samples=pop_size, IT=4, random_order=True)),
    ]

    print(f"\nProblem: Deceptive3, n={n_vars}, pop={pop_size}, max_gen={max_gen}, runs={n_runs}")
    print(f"Optimal fitness: {optimal:.1f}\n")

    summary = {}
    for vname, make_sampler in variants:
        best_list, gen_list, time_list, succ_list = [], [], [], []
        for run in range(n_runs):
            stats, elapsed, _ = run_moa(
                make_sampler(), pop_size, n_vars, max_gen,
                seed=100 + run, verbose=False,
            )
            best_list.append(stats.best_fitness_overall)
            gen_list.append(stats.generation_found if stats.generation_found else max_gen)
            time_list.append(elapsed)
            succ_list.append(int(abs(stats.best_fitness_overall - optimal) < 0.01))

        summary[vname] = {
            "best_mean": np.mean(best_list),
            "best_std":  np.std(best_list),
            "gen_mean":  np.mean(gen_list),
            "time_mean": np.mean(time_list),
            "time_std":  np.std(time_list),
            "success":   np.mean(succ_list) * 100,
        }

    # Print table
    print(f"{'Variant':<22} {'BestFit':>9} {'±':>5} {'Gen':>6} {'Time(s)':>9} {'±':>5} {'Succ%':>7}")
    print("-" * 70)
    baseline_time = summary["Gibbs (baseline)"]["time_mean"]
    for vname, d in summary.items():
        speedup = baseline_time / max(d["time_mean"], 1e-9)
        print(f"{vname:<22} {d['best_mean']:>9.3f} {d['best_std']:>5.3f}"
              f" {d['gen_mean']:>6.1f} {d['time_mean']:>9.3f} {d['time_std']:>5.3f}"
              f" {d['success']:>6.1f}%  (×{speedup:.1f})")

    print()
    print("Notes:")
    print("  GibbsBatch    — numpy-vectorised across population; same n_iters, less Python overhead")
    print("  GibbsWarm     — warm-starts from previous population; only IT sweeps per generation")
    print("  GibbsBatchWarm — combines vectorisation and warm-start for maximum efficiency")


if __name__ == "__main__":
    demo_single()
    compare_variants(n_runs=5)
