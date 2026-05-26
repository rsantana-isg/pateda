"""
Compare all continuous EDAs on standard benchmark functions.

Problems (all MAXIMIZED — EDAs maximize by convention):
  1. Sphere      n_vars=10, bounds=(-5, 5)   maximize -sum(x^2)
  2. Rastrigin   n_vars=10, bounds=(-5.12, 5.12)
  3. Rosenbrock  n_vars=10, bounds=(-2, 2)

VineEDA is excluded (requires optional pyvinecopulib dependency).

Usage:
    python3.11 scripts/compare_continuous_edas.py
"""

import time
import traceback
import numpy as np

from pateda import GaussianUMDA, GaussianEDA, MixtureGaussianEDA, GMRFEDA

# ---------------------------------------------------------------------------
# Benchmark functions (all negated → maximization)
# ---------------------------------------------------------------------------

def sphere(x):
    """Maximize -sum(x^2).  Optimum at x=0, value=0."""
    return -float(np.sum(x ** 2))


def rastrigin(x):
    """Maximize -(sum(x^2 - 10*cos(2*pi*x)) + 10*n).  Optimum at x=0, value=0."""
    n = len(x)
    return -float(np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)) + 10 * n)


def rosenbrock(x):
    """Maximize -sum(100*(x[i+1]-x[i]^2)^2 + (1-x[i])^2).  Optimum at x=1, value=0."""
    total = 0.0
    for i in range(len(x) - 1):
        total += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
    return -total


# ---------------------------------------------------------------------------
# Algorithm list
# ---------------------------------------------------------------------------

ALGORITHMS = [
    ("GaussianUMDA",      GaussianUMDA,      {}),
    ("GaussianEDA",       GaussianEDA,       {}),
    ("MixtureGaussianEDA", MixtureGaussianEDA, {"n_components": 3}),
    ("GMRF-EDA",           GMRFEDA,           {"regularization": "lasso"}),
]

PROBLEMS = [
    ("Sphere",     sphere,     10, (-5.0,    5.0)),
    ("Rastrigin",  rastrigin,  10, (-5.12,   5.12)),
    ("Rosenbrock", rosenbrock, 10, (-2.0,    2.0)),
]

POP_SIZE = 200
N_GEN = 100
SEL_RATIO = 0.5
SEED = 42


def run_one(alg_cls, n_vars, bounds, fitness_func, extra_kwargs):
    alg = alg_cls(
        n_vars=n_vars,
        bounds=bounds,
        fitness_func=fitness_func,
        pop_size=POP_SIZE,
        n_gen=N_GEN,
        selection_ratio=SEL_RATIO,
        random_seed=SEED,
        **extra_kwargs,
    )
    t0 = time.time()
    stats, _ = alg.run(verbose=False)
    elapsed = time.time() - t0
    best = stats.best_fitness_overall
    mean_last = stats.mean_fitness[-1] if stats.mean_fitness else float('nan')
    return best, mean_last, elapsed


def main():
    header_width = 18
    col_w = 14

    for prob_name, fitness_func, n_vars, bounds in PROBLEMS:
        print(f"\n{'=' * 72}")
        print(f"Problem: {prob_name}  (n_vars={n_vars}, bounds={bounds})")
        print(f"{'=' * 72}")
        print(f"{'Algorithm':<{header_width}} {'Best':>{col_w}} {'MeanLast':>{col_w}} {'Time(s)':>{col_w}}")
        print("-" * (header_width + 3 * col_w + 4))

        for alg_name, alg_cls, extra_kwargs in ALGORITHMS:
            try:
                best, mean_last, elapsed = run_one(
                    alg_cls, n_vars, bounds, fitness_func, extra_kwargs
                )
                print(f"{alg_name:<{header_width}} {best:>{col_w}.4f} {mean_last:>{col_w}.4f} {elapsed:>{col_w}.2f}")
            except Exception as e:
                print(f"{alg_name:<{header_width}} {'ERROR':>{col_w}} {str(e)[:30]:>{col_w}} {'':>{col_w}}")
                traceback.print_exc()

    print()


if __name__ == "__main__":
    main()
