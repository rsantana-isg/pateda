"""
Icosahedral braid benchmark.

Defines a benchmark in which *any* of the 60 unitary matrices of the icosahedral
group can be the target gate to approximate with a braid of Fibonacci-anyon
generators.  Each solution is a length-``n`` vector over ``{0,1,2,3}`` selecting
``{sigma_1, sigma_2, sigma_1^{-1}, sigma_2^{-1}}``; the objective maximises
``1/(1+epsilon)`` where ``epsilon = |B - T|`` is the Frobenius approximation
error (Santana et al. SOCO braid paper; Burrello et al. icosahedral group).

Used as a library::

    from braid_icosahedral_benchmark import build_benchmark
    objective, problem = build_benchmark(target_index=5, n_matrices=24)

Used as a script (positional args, seed first)::

    python braid_icosahedral_benchmark.py SEED [TARGET_INDEX] [N_MATRICES] [N_SAMPLES]

With TARGET_INDEX = -1 (default) it characterises the difficulty of all 60
targets with a random-search baseline.  Optimisation with EDAs is in the
companion script ``eda_braid_icosahedral.py``.
"""

import sys
import numpy as np

from pateda.functions.discrete_non_binary.problems.braid import (
    make_icosahedral_benchmark_problem,
    create_braid_objective_function,
    load_icosahedral_targets,
)

N_ICOSAHEDRAL = 60


def build_benchmark(target_index, n_matrices, lam=0.0, use_effective_length=False,
                    phase_invariant=False):
    """Return ``(objective, problem)`` for one icosahedral target.

    ``objective`` maximises braid fitness (``1/(1+error)`` when ``lam == 0``);
    ``problem`` is the underlying :class:`BraidProblem` (cardinality 4).
    """
    problem = make_icosahedral_benchmark_problem(target_index, n_matrices)
    objective = create_braid_objective_function(
        problem, lam=lam, use_effective_length=use_effective_length,
        phase_invariant=phase_invariant)
    return objective, problem


def random_search_baseline(problem, n_samples, rng):
    """Best (lowest) error found by sampling ``n_samples`` random braids."""
    best_err = np.inf
    best_x = None
    for _ in range(n_samples):
        x = rng.integers(0, problem.cardinality, size=problem.n_matrices)
        e = problem.error(x)
        if e < best_err:
            best_err, best_x = e, x
    return best_err, best_x


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 111
    target_index = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    n_matrices = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    n_samples = int(sys.argv[4]) if len(sys.argv) > 4 else 20000

    rng = np.random.default_rng(seed)

    print("Icosahedral braid benchmark")
    print("=" * 60)
    print(f"Seed:            {seed}")
    print(f"Braid length n:  {n_matrices}")
    print(f"Cardinality:     4  (sigma_1, sigma_2, sigma_1^-1, sigma_2^-1)")
    print(f"Random samples:  {n_samples}")
    print(f"# targets:       {N_ICOSAHEDRAL} (icosahedral group)")
    print()

    if target_index >= 0:
        objective, problem = build_benchmark(target_index, n_matrices)
        best_err, best_x = random_search_baseline(problem, n_samples, rng)
        print(f"Target {target_index}: random-search best error = {best_err:.5f}")
        print(f"  braid = {list(map(int, best_x))}")
        print(f"  effective length = {problem.effective_length(best_x)}")
        return

    # Characterise all 60 targets with the random-search baseline.
    print(f"{'target':>7}{'rand best error':>18}{'elen':>7}")
    print("-" * 32)
    errors = []
    for t in range(N_ICOSAHEDRAL):
        _, problem = build_benchmark(t, n_matrices)
        best_err, best_x = random_search_baseline(problem, n_samples, rng)
        errors.append(best_err)
        print(f"{t:>7}{best_err:>18.5f}{problem.effective_length(best_x):>7}")
    errors = np.array(errors)
    print("-" * 32)
    print(f"mean best error over 60 targets: {errors.mean():.5f}")
    print(f"min / max: {errors.min():.5f} / {errors.max():.5f}")
    print("\nNote: random search is only a difficulty baseline; use")
    print("      eda_braid_icosahedral.py to optimise with UMDA/Tree-EDA/MK-EDA.")


if __name__ == "__main__":
    main()
