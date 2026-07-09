"""
Bayesian Tree EDA with K-MAP Sampling for the Ising Model

Extends bayesian_tree_ising_mpe.py to use K most probable configurations
(K-MPCs) instead of a single MAP.  Two new strategies are demonstrated:

SampleInsertKMAP (S4)
    Generate the population via PLS, then replace the K worst individuals
    with the K most probable configurations.

SampleTemplateKMAP (S5)
    Divide the population into K groups and use each K-MPC as a template:
    each individual inherits variables from its group's template with
    probability *template_prob* and resamples the rest from the model.

Both strategies extend the single-MAP methods of Santana (2013) to K
configurations, providing richer exploitation of the probabilistic model.

References
----------
Nilsson, D. (1998). An efficient algorithm for finding the M most probable
  configurations in probabilistic expert systems. Statistics and Computing, 8.
Echegoyen, C. et al. (2010). Analyzing the k Most Probable Solutions in EDAs
  Based on Bayesian Networks.
Santana, R. (2013). Message Passing Methods for Estimation of Distribution
  Algorithms Based on Markov Networks.
"""

import numpy as np
from pateda.core.eda import EDA, EDAComponents
from pateda.core.components import StopCondition
from pateda.seeding import RandomInit
from pateda.selection import TruncationSelection
from pateda.replacement import ElitistReplacement
from pateda.learning.tree import LearnTreeModel
from pateda.sampling.map_sampling import SampleInsertMAP
from pateda.sampling.kmap_sampling import SampleInsertKMAP, SampleTemplateKMAP
from pateda.sampling.fda import SampleFDA

try:
    from pateda.functions.discrete_binary.problems.ising import load_ising, eval_ising
    ISING_AVAILABLE = True
except ImportError:
    ISING_AVAILABLE = False


class StopMaxGenOrOptimum(StopCondition):
    """Stop when max generations is reached or optimal fitness is achieved."""

    def __init__(self, max_generations: int, optimal_value: float, tolerance: float = 0.1):
        self.max_generations = max_generations
        self.optimal_value = optimal_value
        self.tolerance = tolerance
        self.best_fitness_overall = -np.inf

    def should_stop(self, generation: int, population: np.ndarray,
                    fitness: np.ndarray, **params) -> bool:
        current_best = float(np.max(fitness))
        if current_best > self.best_fitness_overall:
            self.best_fitness_overall = current_best
        if generation >= self.max_generations:
            return True
        return abs(self.best_fitness_overall - self.optimal_value) <= self.tolerance

    def reset(self):
        self.best_fitness_overall = -np.inf


def create_ising_function(n_vars: int, instance: int = 1, seed: int = 0):
    """Create Ising model fitness function (loads from file or generates random)."""
    if ISING_AVAILABLE:
        try:
            lattice, inter = load_ising(n_vars, instance)
            def fitness_func(x):
                return eval_ising(x, lattice, inter)
            return fitness_func
        except FileNotFoundError:
            pass

    print(f"Creating random {n_vars}-variable Ising model (seed={seed})...")
    rng = np.random.default_rng(seed)
    grid_size = int(np.sqrt(n_vars))
    if grid_size * grid_size != n_vars:
        raise ValueError(f"n_vars must be a perfect square, got {n_vars}")

    max_neighbors = 4
    lattice = np.zeros((n_vars, max_neighbors + 1), dtype=int)
    inter = rng.uniform(-1, 1, (n_vars, max_neighbors))

    for i in range(n_vars):
        row, col = i // grid_size, i % grid_size
        neighbors = []
        if row > 0:
            neighbors.append((row - 1) * grid_size + col)
        if row < grid_size - 1:
            neighbors.append((row + 1) * grid_size + col)
        if col > 0:
            neighbors.append(row * grid_size + (col - 1))
        if col < grid_size - 1:
            neighbors.append(row * grid_size + (col + 1))
        lattice[i, 0] = len(neighbors)
        for j, nb in enumerate(neighbors):
            lattice[i, j + 1] = nb + 1

    def fitness_func(x):
        r = 0.0
        for i in range(n_vars):
            for j in range(1, int(lattice[i, 0]) + 1):
                nb = int(lattice[i, j]) - 1
                if i < nb:
                    r += inter[i, j - 1] if x[i] == x[nb] else -inter[i, j - 1]
        return r

    return fitness_func


def run_single(sampler_name: str, sampler, fitness_func, n_vars: int,
               pop_size: int, max_generations: int, optimal_value: float,
               random_seed: int = 42, verbose: bool = True) -> float:
    """Run one EDA experiment and return the best fitness found."""
    components = EDAComponents(
        seeding=RandomInit(),
        selection=TruncationSelection(ratio=0.5),
        learning=LearnTreeModel(alpha=0.1),
        sampling=sampler,
        replacement=ElitistReplacement(),
        stop_condition=StopMaxGenOrOptimum(max_generations, optimal_value),
    )
    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        fitness_func=fitness_func,
        cardinality=np.full(n_vars, 2),
        components=components,
        random_seed=random_seed,
    )
    stats, _ = eda.run(verbose=verbose)
    return stats


def demo_kmap_sampling():
    """Demonstrate Insert-KMAP and Template-KMAP on the Ising model."""
    print("=" * 70)
    print("Bayesian Tree EDA with K-MAP Sampling for the Ising Model")
    print("=" * 70)

    n_vars = 64
    pop_size = 500
    max_generations = 150
    optimal_value = 86.0  # known optimal for the standard 64-var instance
    k = 5

    print(f"\nProblem : {n_vars}-variable Ising model (8x8 lattice)")
    print(f"Pop size: {pop_size}   Max gens: {max_generations}   K: {k}\n")

    fitness_func = create_ising_function(n_vars, instance=1, seed=99)
    print()

    print("--- Running Insert-K-MAP (S4) ---")
    stats_ikmap = run_single(
        "Insert-K-MAP",
        SampleInsertKMAP(n_samples=pop_size, k=k, replace_worst=True),
        fitness_func, n_vars, pop_size, max_generations, optimal_value,
        random_seed=42, verbose=True,
    )

    print()
    print("=" * 70)
    print("RESULTS — Insert-K-MAP")
    print("=" * 70)
    total_gens = len(stats_ikmap.best_fitness) - 1
    print(f"Best fitness : {stats_ikmap.best_fitness_overall:.4f}")
    print(f"Gen found    : {stats_ikmap.generation_found}")
    print(f"Total gens   : {total_gens}")
    print()

    print("--- Running Template-K-MAP (S5) ---")
    stats_tkmap = run_single(
        "Template-K-MAP",
        SampleTemplateKMAP(n_samples=pop_size, k=k, template_prob=0.5),
        fitness_func, n_vars, pop_size, max_generations, optimal_value,
        random_seed=42, verbose=True,
    )

    print()
    print("=" * 70)
    print("RESULTS — Template-K-MAP")
    print("=" * 70)
    total_gens = len(stats_tkmap.best_fitness) - 1
    print(f"Best fitness : {stats_tkmap.best_fitness_overall:.4f}")
    print(f"Gen found    : {stats_tkmap.generation_found}")
    print(f"Total gens   : {total_gens}")

    return stats_ikmap, stats_tkmap


def run_comparison(n_runs: int = 5):
    """Compare four sampling strategies: PLS, Insert-MAP, Insert-K-MAP, Template-K-MAP."""
    print("\n" + "=" * 70)
    print("Comparison: PLS vs MAP vs Insert-K-MAP vs Template-K-MAP")
    print("=" * 70)

    n_vars = 64
    pop_size = 500
    max_generations = 150
    optimal_value = 86.0
    k = 5

    fitness_func = create_ising_function(n_vars, instance=1, seed=99)
    print()

    strategies = {
        "PLS (baseline)": lambda: SampleFDA(n_samples=pop_size),
        "Insert-MAP (S1)": lambda: SampleInsertMAP(n_samples=pop_size, map_method="bp"),
        f"Insert-{k}MAP (S4)": lambda: SampleInsertKMAP(n_samples=pop_size, k=k),
        f"Template-{k}MAP (S5)": lambda: SampleTemplateKMAP(n_samples=pop_size, k=k),
    }

    results = {name: {"fitness": [], "gen_found": []} for name in strategies}

    for name, sampler_factory in strategies.items():
        print(f"\nRunning {name} ({n_runs} runs)...")
        for run in range(n_runs):
            stats = run_single(
                name, sampler_factory(), fitness_func,
                n_vars, pop_size, max_generations, optimal_value,
                random_seed=100 + run, verbose=False,
            )
            results[name]["fitness"].append(stats.best_fitness_overall)
            results[name]["gen_found"].append(
                stats.generation_found if stats.generation_found is not None else max_generations
            )
            print(f"  Run {run + 1}/{n_runs}: best={stats.best_fitness_overall:.3f}  "
                  f"gen={results[name]['gen_found'][-1]}")

    print()
    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Strategy':<25} {'Mean fitness':>14} {'Std fitness':>12} {'Mean gen':>10}")
    print("-" * 70)
    for name, data in results.items():
        arr = np.array(data["fitness"])
        gens = np.array(data["gen_found"])
        print(f"{name:<25} {np.mean(arr):>14.3f} {np.std(arr):>12.3f} {np.mean(gens):>10.1f}")
    print()


if __name__ == "__main__":
    # Single-run demonstration of the two K-MAP strategies
    demo_kmap_sampling()

    # Multi-run comparison across four sampling strategies
    print()
    run_comparison(n_runs=3)
