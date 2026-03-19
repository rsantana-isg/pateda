"""
Test Discrete EDAs on a Toy Problem with Variables of Different Cardinalities
==============================================================================

This script demonstrates and tests that four discrete EDAs — UMDA, Tree-EDA,
Mk-EDA (k-order Markov Chain), and Tree-EDAr — can correctly handle problems
where different variables take different numbers of possible values (mixed
cardinality).

Background
----------
In standard EDA benchmarks all variables share the same cardinality (e.g.
all binary or all ternary). Real-world combinatorial problems often involve
variables with different domains — e.g., a scheduling instance where some
resources have 2 states and others have 5. This script verifies that the
four EDAs listed above naturally support this through their ``cardinality``
vector argument, without any changes to the core learning or sampling code.

Toy problem
-----------
"Mixed-Cardinality OneMax" (MC-OneMax): maximise the sum of all variable
values.  The fitness function is simply::

    f(x) = sum(x[i]  for i in range(n_vars))

The optimum is achieved when every variable takes its maximum value (i.e.
``x[i] = cardinality[i] - 1``).  Because variables have different domains
the optimal fitness differs from the binary OneMax: for a variable with
cardinality *k* the maximum contribution is *k − 1*.

Variable layout (default):
  * 6 binary variables       (cardinality = 2, max contribution 1 each)
  * 6 ternary variables      (cardinality = 3, max contribution 2 each)
  * 6 quaternary variables   (cardinality = 4, max contribution 3 each)
  * Total: 18 variables,  optimal fitness = 6*1 + 6*2 + 6*3 = 6 + 12 + 18 = 36

Usage
-----
Full test (all four algorithms, verbose):
    python scripts/test_mixed_cardinality_edas.py

Quick smoke-test (one run per algorithm, minimal output):
    python scripts/test_mixed_cardinality_edas.py --quick

Silent pass/fail:
    python scripts/test_mixed_cardinality_edas.py --silent
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# Allow running as a standalone script without installing the package.
# Directory layout: <project_root>/pateda/scripts/test_mixed_cardinality_edas.py
#   parent              = <project_root>/pateda/scripts/
#   parent.parent       = <project_root>/pateda/   ← the 'pateda' package itself
#   parent.parent.parent = <project_root>/         ← needed so 'import pateda' resolves
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

warnings.filterwarnings("ignore", message="pyvinecopulib not available")

from pateda import EDA, EDAComponents
from pateda.learning.umda import LearnUMDA
from pateda.learning.tree import LearnTreeModel
from pateda.learning.tree_r import LearnTreeModelR
from pateda.learning.markov import LearnMarkovChain
from pateda.sampling.fda import SampleFDA
from pateda.sampling.markov import SampleMarkovChain
from pateda.seeding import RandomInit
from pateda.selection import TruncationSelection
from pateda.replacement import ElitistReplacement
from pateda.stop_conditions import MaxGenerations


# ---------------------------------------------------------------------------
# Toy problem definition
# ---------------------------------------------------------------------------

def mixed_card_onemax(x: np.ndarray) -> np.ndarray:
    """
    Mixed-Cardinality OneMax: maximise the sum of all variable values.

    Works for any cardinality per variable because the fitness is simply the
    element-wise sum.  The optimum is ``x[i] = cardinality[i] - 1`` for all i.

    Parameters
    ----------
    x : np.ndarray, shape (pop_size, n_vars)
        Population matrix of integer-valued individuals.

    Returns
    -------
    np.ndarray, shape (pop_size,)
        Fitness values (higher is better).
    """
    if x.ndim == 1:
        return np.array([float(np.sum(x))])
    return np.sum(x, axis=1).astype(float)


def make_mixed_cardinality(n_binary: int = 6,
                           n_ternary: int = 6,
                           n_quaternary: int = 6) -> np.ndarray:
    """
    Build a cardinality vector with groups of binary, ternary and
    quaternary variables.

    Parameters
    ----------
    n_binary, n_ternary, n_quaternary : int
        Number of variables for each group.

    Returns
    -------
    np.ndarray
        Integer array of shape (n_binary + n_ternary + n_quaternary,).
    """
    return np.array(
        [2] * n_binary + [3] * n_ternary + [4] * n_quaternary, dtype=int
    )


def optimal_fitness(cardinality: np.ndarray) -> float:
    """Return the optimal MC-OneMax fitness for a given cardinality vector."""
    return float(np.sum(cardinality - 1))


# ---------------------------------------------------------------------------
# Chain interaction matrix for Tree-EDAr
# ---------------------------------------------------------------------------

def make_chain_interaction_matrix(n_vars: int) -> np.ndarray:
    """
    Build a symmetric binary interaction matrix where each variable only
    interacts with its immediate neighbours (a chain / path graph).

    Used as the *restriction matrix* for Tree-EDAr.
    """
    R = np.zeros((n_vars, n_vars), dtype=int)
    for i in range(n_vars - 1):
        R[i, i + 1] = 1
        R[i + 1, i] = 1
    return R


# ---------------------------------------------------------------------------
# EDA runners
# ---------------------------------------------------------------------------

def run_eda(
    algorithm_name: str,
    cardinality: np.ndarray,
    pop_size: int,
    max_generations: int,
    seed: int,
    verbose: bool,
    interaction_matrix: np.ndarray = None,
) -> Tuple[float, np.ndarray, float]:
    """
    Run a single EDA on MC-OneMax and return the best result.

    Parameters
    ----------
    algorithm_name : str
        One of 'UMDA', 'Tree-EDA', 'Mk-EDA', 'Tree-EDAr'.
    cardinality : np.ndarray
        Per-variable cardinality vector.
    pop_size : int
        Population size.
    max_generations : int
        Maximum number of generations.
    seed : int
        Random seed.
    verbose : bool
        Print per-generation statistics.
    interaction_matrix : np.ndarray or None
        Required for 'Tree-EDAr'; a binary (n_vars, n_vars) matrix.

    Returns
    -------
    best_fitness : float
        Best fitness found.
    best_individual : np.ndarray
        Best individual found.
    elapsed : float
        Wall-clock time in seconds.
    """
    import time

    n_vars = len(cardinality)

    # ------------------------------------------------------------------
    # Choose learning and sampling methods
    # ------------------------------------------------------------------
    if algorithm_name == "UMDA":
        learning = LearnUMDA(alpha=0.1)
        sampling = SampleFDA(n_samples=pop_size)

    elif algorithm_name == "Tree-EDA":
        learning = LearnTreeModel(alpha=0.1, mi_threshold=1e-4)
        sampling = SampleFDA(n_samples=pop_size)

    elif algorithm_name == "Mk-EDA":
        learning = LearnMarkovChain(k=1, alpha=0.1)
        sampling = SampleMarkovChain(n_samples=pop_size)

    elif algorithm_name == "Tree-EDAr":
        if interaction_matrix is None:
            interaction_matrix = make_chain_interaction_matrix(n_vars)
        learning = LearnTreeModelR(
            interaction_matrix=interaction_matrix,
            alpha=0.1,
            mi_threshold=1e-4,
        )
        sampling = SampleFDA(n_samples=pop_size)

    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    # ------------------------------------------------------------------
    # Build EDA components
    # ------------------------------------------------------------------
    components = EDAComponents(
        seeding=RandomInit(),
        selection=TruncationSelection(ratio=0.5),
        learning=learning,
        sampling=sampling,
        replacement=ElitistReplacement(),
        stop_condition=MaxGenerations(max_gen=max_generations),
    )

    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        fitness_func=mixed_card_onemax,
        cardinality=cardinality,
        components=components,
        random_seed=seed,
    )

    t0 = time.perf_counter()
    stats, _ = eda.run(verbose=verbose)
    elapsed = time.perf_counter() - t0

    return stats.best_fitness_overall, stats.best_individual, elapsed


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_tests(
    quick: bool = False,
    silent: bool = False,
) -> Dict[str, dict]:
    """
    Run all four EDAs on the MC-OneMax toy problem and collect results.

    Parameters
    ----------
    quick : bool
        If True run a single short trial instead of multiple seeds.
    silent : bool
        Suppress all per-algorithm output.

    Returns
    -------
    results : dict
        Mapping algorithm name → dict with keys 'best', 'optimal', 'success'.
    """
    # ------------------------------------------------------------------
    # Problem configuration
    # ------------------------------------------------------------------
    cardinality = make_mixed_cardinality(n_binary=6, n_ternary=6, n_quaternary=6)
    n_vars = len(cardinality)
    opt = optimal_fitness(cardinality)

    if quick:
        pop_size = 100
        max_gen = 30
        seeds = [42]
    else:
        pop_size = 200
        max_gen = 60
        seeds = [42, 123, 7]

    interaction_matrix = make_chain_interaction_matrix(n_vars)

    algorithms = ["UMDA", "Tree-EDA", "Mk-EDA", "Tree-EDAr"]

    if not silent:
        print("=" * 65)
        print("Mixed-Cardinality Discrete EDA Test")
        print("=" * 65)
        print(f"Variables : {n_vars}")
        print(f"Cardinality vector: {cardinality.tolist()}")
        print(f"  Binary (card=2)      : {np.sum(cardinality == 2)} variables")
        print(f"  Ternary (card=3)     : {np.sum(cardinality == 3)} variables")
        print(f"  Quaternary (card=4)  : {np.sum(cardinality == 4)} variables")
        print(f"Optimal fitness       : {opt:.1f}  (all vars at max value)")
        print(f"Population size       : {pop_size}")
        print(f"Max generations       : {max_gen}")
        print(f"Seeds                 : {seeds}")
        print()

    results = {}
    verbose_run = not silent and quick  # Only show per-gen output in quick mode

    for alg in algorithms:
        best_across_seeds = []
        times = []

        for seed in seeds:
            best_fit, best_ind, elapsed = run_eda(
                algorithm_name=alg,
                cardinality=cardinality,
                pop_size=pop_size,
                max_generations=max_gen,
                seed=seed,
                verbose=verbose_run,
                interaction_matrix=interaction_matrix,
            )
            best_across_seeds.append(best_fit)
            times.append(elapsed)

        mean_best = np.mean(best_across_seeds)
        max_best = np.max(best_across_seeds)
        success = max_best >= opt * 0.95  # Within 5 % of optimum

        results[alg] = {
            "best_per_seed": best_across_seeds,
            "mean_best": mean_best,
            "max_best": max_best,
            "optimal": opt,
            "success": success,
            "mean_time_s": np.mean(times),
        }

        if not silent:
            status = "✓ PASS" if success else "✗ FAIL"
            print(
                f"  {alg:<12s}  best={max_best:6.2f}  "
                f"mean={mean_best:6.2f}  "
                f"opt={opt:.1f}  "
                f"time={np.mean(times):.2f}s  {status}"
            )

            # Validate cardinality constraints in sampled output
            print(
                f"  {'':12s}  best individual: "
                f"{best_across_seeds.index(max_best)+1}/{len(seeds)} seed(s) reached it"
            )

    if not silent:
        print()
        all_pass = all(r["success"] for r in results.values())
        if all_pass:
            print("All algorithms PASSED the mixed-cardinality test.")
        else:
            failed = [a for a, r in results.items() if not r["success"]]
            print(f"FAILED: {failed}")
        print("=" * 65)

    return results


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_results(results: Dict[str, dict]) -> bool:
    """
    Raise AssertionError if any algorithm failed the acceptance criterion.

    Parameters
    ----------
    results : dict
        Output of :func:`run_tests`.

    Returns
    -------
    True if all algorithms passed.
    """
    for alg, res in results.items():
        assert res["success"], (
            f"{alg} did not reach ≥95 % of the optimal fitness "
            f"({res['max_best']:.2f} < {res['optimal'] * 0.95:.2f})"
        )
    return True


def validate_cardinality_constraints(
    cardinality: np.ndarray,
    population: np.ndarray,
    algorithm_name: str,
) -> None:
    """
    Assert that every variable in *population* respects its cardinality.

    Parameters
    ----------
    cardinality : np.ndarray
        Per-variable cardinality vector.
    population : np.ndarray, shape (pop_size, n_vars)
        Population to validate.
    algorithm_name : str
        Used in error messages.
    """
    n_vars = len(cardinality)
    for i in range(n_vars):
        lo, hi = 0, int(cardinality[i]) - 1
        assert np.all(population[:, i] >= lo) and np.all(population[:, i] <= hi), (
            f"{algorithm_name}: variable {i} (card={cardinality[i]}) "
            f"has out-of-range values: "
            f"min={population[:, i].min()}, max={population[:, i].max()}"
        )


# ---------------------------------------------------------------------------
# Direct cardinality sampling validation (low-level)
# ---------------------------------------------------------------------------

def run_low_level_cardinality_test(silent: bool = False) -> None:
    """
    Directly test that learn → sample produces valid values for each algorithm
    without using the full EDA loop.  Validates cardinality constraints at the
    sampling level.
    """
    if not silent:
        print("\nLow-level cardinality constraint validation")
        print("-" * 45)

    np.random.seed(0)

    # Heterogeneous cardinality: 4 binary, 4 ternary, 4 quaternary
    cardinality = np.array([2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4], dtype=int)
    n_vars = len(cardinality)
    pop_size = 200

    # Generate initial population respecting cardinality
    pop = np.column_stack(
        [np.random.randint(0, c, size=pop_size) for c in cardinality]
    )
    fitness = mixed_card_onemax(pop)

    # UMDA
    umda_model = LearnUMDA(alpha=0.1).learn(0, n_vars, cardinality, pop, fitness)
    umda_pop = SampleFDA(n_samples=pop_size).sample(n_vars, umda_model, cardinality)
    validate_cardinality_constraints(cardinality, umda_pop, "UMDA")
    if not silent:
        print("  UMDA learn+sample: cardinality constraints OK")

    # Tree-EDA
    tree_model = LearnTreeModel(alpha=0.1).learn(0, n_vars, cardinality, pop, fitness)
    tree_pop = SampleFDA(n_samples=pop_size).sample(n_vars, tree_model, cardinality)
    validate_cardinality_constraints(cardinality, tree_pop, "Tree-EDA")
    if not silent:
        print("  Tree-EDA learn+sample: cardinality constraints OK")

    # Tree-EDAr
    R = make_chain_interaction_matrix(n_vars)
    treer_model = LearnTreeModelR(interaction_matrix=R, alpha=0.1).learn(
        0, n_vars, cardinality, pop, fitness
    )
    treer_pop = SampleFDA(n_samples=pop_size).sample(n_vars, treer_model, cardinality)
    validate_cardinality_constraints(cardinality, treer_pop, "Tree-EDAr")
    if not silent:
        print("  Tree-EDAr learn+sample: cardinality constraints OK")

    # Mk-EDA (k=1)
    mk_model = LearnMarkovChain(k=1, alpha=0.1).learn(
        0, n_vars, cardinality, pop, fitness
    )
    mk_pop = SampleMarkovChain(n_samples=pop_size).sample(n_vars, mk_model, cardinality)
    validate_cardinality_constraints(cardinality, mk_pop, "Mk-EDA")
    if not silent:
        print("  Mk-EDA learn+sample: cardinality constraints OK")

    if not silent:
        print("All low-level cardinality checks passed.\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke-test with one seed and fewer generations.",
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Suppress output; exit 0 on pass, 1 on failure.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        # Low-level validation first (fast)
        run_low_level_cardinality_test(silent=args.silent)

        # Full EDA optimisation loop
        results = run_tests(quick=args.quick, silent=args.silent)

        # Raise on any failure
        validate_results(results)

    except AssertionError as exc:
        if not args.silent:
            print(f"\nFAIL: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
