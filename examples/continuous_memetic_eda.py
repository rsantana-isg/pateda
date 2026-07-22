"""
Continuous memetic EDA: combining a Gaussian EDA with SciPy local search.

A Gaussian univariate EDA is turned into a memetic algorithm by refining each
sampled population with ``scipy.optimize.minimize`` through
:class:`~pateda.local_optimization.scipy_local_search.ScipyLocalSearch`. The
example compares several SciPy methods (and the plain EDA baseline) on the
Rastrigin function.

Usage::

    python3 examples/continuous_memetic_eda.py
"""

import numpy as np

from pateda.core.eda import EDA
from pateda.core.components import EDAComponents
from pateda.seeding import RandomInit
from pateda.selection import TruncationSelection
from pateda.learning.basic_gaussian import LearnGaussianUnivariate
from pateda.sampling.basic_gaussian import SampleGaussianUnivariate
from pateda.replacement import ElitistReplacement
from pateda.stop_conditions import MaxGenerations
from pateda.local_optimization.scipy_local_search import ScipyLocalSearch
from pateda.functions.continuous import rastrigin


N_VARS = 10
POP = 40
N_GEN = 12
SEED = 1


def objective(x):
    # pateda maximizes; Rastrigin is minimized, so negate.
    return -float(rastrigin(np.asarray(x, dtype=float)))


def run(local_opt):
    bounds = np.array([[-5.12] * N_VARS, [5.12] * N_VARS])
    comp = EDAComponents(
        seeding=RandomInit(),
        selection=TruncationSelection(ratio=0.5),
        learning=LearnGaussianUnivariate(),
        sampling=SampleGaussianUnivariate(n_samples=POP),
        replacement=ElitistReplacement(n_elite=2),
        local_opt=local_opt,
        stop_condition=MaxGenerations(N_GEN),
    )
    eda = EDA(POP, N_VARS, objective, bounds, comp, random_seed=SEED)
    stats, _ = eda.run(verbose=False)
    # report the true (minimized) Rastrigin value of the best solution
    return -stats.best_fitness_overall


def main():
    configs = {
        "plain EDA (no local search)": None,
        "EDA + L-BFGS-B": ScipyLocalSearch(method="L-BFGS-B", max_iter=40),
        "EDA + Nelder-Mead": ScipyLocalSearch(method="Nelder-Mead", max_iter=40),
        "EDA + Powell": ScipyLocalSearch(method="Powell", max_iter=40),
    }
    print(f"Rastrigin ({N_VARS} vars), lower is better\n")
    print(f"{'configuration':30s} {'best Rastrigin':>16s}")
    for name, lo in configs.items():
        best = run(lo)
        print(f"{name:30s} {best:16.4f}")


if __name__ == "__main__":
    main()
