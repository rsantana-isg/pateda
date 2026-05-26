"""
Detailed comparison of vine copula EDAs on the 2-D off-lattice AB protein
folding model (Stillinger 1993).

Each EDA configuration is run on three Fibonacci-style AB sequences of
increasing difficulty:

  - Fibonacci-13   (n_angles = 11)
  - Fibonacci-21   (n_angles = 19)
  - Fibonacci-34   (n_angles = 32)

We compare four families of vine copula EDAs:

  1.  ``VineEDA``               R-vine + full automatic family selection.
  2.  ``CVineEDA(family)``      C-vine structure, single user-chosen family.
  3.  ``RVineEDA(family)``      R-vine structure, single user-chosen family.

A handful of representative single-family configurations are evaluated
(gaussian, gumbel, frank, clayton, joe) for both C-vine and R-vine
structures, along with the fully automatic VineEDA.  This mirrors the
encoding used in ``enhanced_edas/copula_models.py``.

For each configuration we record:
  - the best energy E* found (lower is better),
  - the mean energy of the final population,
  - the wall-clock time.

Requires the optional ``pyvinecopulib`` package::

    pip install pyvinecopulib

Usage:
    python3.11 scripts/compare_vine_copula_edas.py [n_runs]
"""

import sys
import time
import traceback
import numpy as np

from pateda import VineEDA, CVineEDA, RVineEDA
from pateda.functions.continuous import make_ab_fitness


# ---------------------------------------------------------------------------
# Experiment setup
# ---------------------------------------------------------------------------

PROBLEMS = [
    ('Fib-13', 13),
    ('Fib-21', 21),
    ('Fib-34', 34),
]

SINGLE_FAMILIES = ['gaussian', 'gumbel', 'frank', 'clayton', 'joe']

POP_SIZE = 200
N_GEN = 60
SEL_RATIO = 0.4
BOUNDS = (-np.pi, np.pi)


def make_algorithms():
    """Yield (name, factory) pairs for each vine-copula EDA variant tested."""
    yield ('R-vine auto', lambda n, fit, seed: VineEDA(
        n_vars=n, bounds=BOUNDS, fitness_func=fit,
        pop_size=POP_SIZE, n_gen=N_GEN, selection_ratio=SEL_RATIO,
        random_seed=seed,
    ))
    for fam in SINGLE_FAMILIES:
        yield (f'C-vine [{fam}]', lambda n, fit, seed, fam=fam: CVineEDA(
            n_vars=n, bounds=BOUNDS, fitness_func=fit,
            pop_size=POP_SIZE, n_gen=N_GEN, selection_ratio=SEL_RATIO,
            copula_family=fam, random_seed=seed,
        ))
    for fam in SINGLE_FAMILIES:
        yield (f'R-vine [{fam}]', lambda n, fit, seed, fam=fam: RVineEDA(
            n_vars=n, bounds=BOUNDS, fitness_func=fit,
            pop_size=POP_SIZE, n_gen=N_GEN, selection_ratio=SEL_RATIO,
            copula_family=fam, random_seed=seed,
        ))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_one(factory, n_angles, fitness_func, seed):
    alg = factory(n_angles, fitness_func, seed)
    t0 = time.time()
    stats, _ = alg.run(verbose=False)
    elapsed = time.time() - t0
    # EDAs maximise, so the actual energy is -best_fitness.
    best_energy = -float(stats.best_fitness_overall)
    mean_final_fit = float(stats.mean_fitness[-1]) if stats.mean_fitness else float('nan')
    mean_final_energy = -mean_final_fit if not np.isnan(mean_final_fit) else float('nan')
    return best_energy, mean_final_energy, elapsed


def aggregate(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float('nan'), float('nan')
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def main(n_runs: int = 3):
    name_w = 22
    col_w = 14
    print(f"Off-lattice AB protein  (pop={POP_SIZE}, n_gen={N_GEN}, "
          f"runs/config={n_runs}, sel_ratio={SEL_RATIO})")

    for prob_name, n_residues in PROBLEMS:
        fitness = make_ab_fitness(n_residues=n_residues)
        n_angles = fitness.n_angles
        print(f"\n{'=' * 78}")
        print(f"Problem: {prob_name}  (n_residues={n_residues}, n_angles={n_angles})")
        print(f"Sequence: {''.join('A' if s == 0 else 'B' for s in fitness.sequence)}")
        print(f"{'=' * 78}")
        print(f"{'Algorithm':<{name_w}} "
              f"{'best mean':>{col_w}} {'best std':>{col_w}} "
              f"{'final mean':>{col_w}} {'time(s)':>{col_w}}")
        print('-' * (name_w + 4 * col_w + 4))

        for alg_name, factory in make_algorithms():
            bests, finals, times = [], [], []
            failed = False
            for r in range(n_runs):
                seed = 100 + r
                try:
                    best, mean_f, dt = run_one(factory, n_angles, fitness, seed)
                    bests.append(best)
                    finals.append(mean_f)
                    times.append(dt)
                except ImportError as e:
                    print(f"{alg_name:<{name_w}} "
                          f"{'SKIP (pyvinecopulib missing)':>{col_w * 4}}")
                    failed = True
                    break
                except Exception as e:
                    print(f"{alg_name:<{name_w}} ERROR: {e}")
                    traceback.print_exc()
                    failed = True
                    break
            if failed:
                continue

            best_mean, best_std = aggregate(bests)
            final_mean, _ = aggregate(finals)
            time_mean = float(np.mean(times)) if times else float('nan')
            print(f"{alg_name:<{name_w}} "
                  f"{best_mean:>{col_w}.4f} {best_std:>{col_w}.4f} "
                  f"{final_mean:>{col_w}.4f} {time_mean:>{col_w}.2f}")

    print()


if __name__ == '__main__':
    n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    main(n_runs)
