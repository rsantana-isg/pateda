"""
Evaluate UMDA, Tree-EDA and MK-EDA on the icosahedral braid benchmark.

For a chosen icosahedral target gate (or a set of them), each EDA searches for a
length-``n`` braid of Fibonacci-anyon generators that approximates the target.
The reported quantities are the best braid fitness ``1/(1+epsilon)``, the
corresponding approximation error ``epsilon = |B - T|`` and the effective braid
length after cancelling inverse pairs.

The three EDAs differ in the dependencies they model, matching the SOCO braid
paper analysis (strong dependencies between adjacent braid positions):

* UMDA      -- univariate (independent positions).
* Tree-EDA  -- bivariate dependency tree (Chow-Liu).
* MK-EDA    -- order-k Markov chain over the (ordered) braid positions.

Run (positional args, seed first)::

    python eda_braid_icosahedral.py SEED [TARGET_INDEX] [N_MATRICES] [POP_SIZE] [N_GEN]

TARGET_INDEX = -1 (default) evaluates a small representative set of targets.
"""

import sys
import numpy as np

from pateda import UMDA, TreeEDA, MKEDA
from pateda.functions.discrete_non_binary.problems.braid import (
    make_icosahedral_benchmark_problem,
    create_braid_objective_function,
)

CARDINALITY = 4  # sigma_1, sigma_2, sigma_1^-1, sigma_2^-1


def run_one(alg_name, target_index, n_matrices, pop_size, n_gen, seed):
    """Run one EDA on one icosahedral target; return (error, elen, fitness)."""
    problem = make_icosahedral_benchmark_problem(target_index, n_matrices)
    # lam = 0 -> pure error objective (fitness = 1/(1+error)).
    objective = create_braid_objective_function(problem, lam=0.0)

    common = dict(n_vars=n_matrices, cardinality=CARDINALITY, fitness_func=objective,
                  pop_size=pop_size, n_gen=n_gen, selection_ratio=0.15,
                  elitism=True, alpha=1.0, random_seed=seed)
    if alg_name == "UMDA":
        alg = UMDA(**common)
    elif alg_name == "Tree-EDA":
        alg = TreeEDA(**common)
    elif alg_name == "MK-EDA":
        alg = MKEDA(k=1, **common)
    else:
        raise ValueError(alg_name)

    stats, _ = alg.run(verbose=False)
    best_fit = float(stats.best_fitness_overall)
    best_err = 1.0 / best_fit - 1.0

    # Effective length of the best solution found across the whole run.
    best_x = stats.best_individual
    elen = problem.effective_length(best_x) if best_x is not None else n_matrices
    return best_err, elen, best_fit


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 111
    target_index = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    n_matrices = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    pop_size = int(sys.argv[4]) if len(sys.argv) > 4 else 500
    n_gen = int(sys.argv[5]) if len(sys.argv) > 5 else 60

    algorithms = ["UMDA", "Tree-EDA", "MK-EDA"]
    targets = [target_index] if target_index >= 0 else [1, 5, 25, 40]

    print("EDAs on the icosahedral braid benchmark")
    print("=" * 64)
    print(f"Seed:            {seed}")
    print(f"Braid length n:  {n_matrices}")
    print(f"Population size: {pop_size}")
    print(f"Generations:     {n_gen}")
    print(f"Objective:       maximise 1/(1+error)  (lambda = 0)")
    print()

    header = f"{'target':>7}" + "".join(f"{a + ' err':>16}" for a in algorithms)
    print(header)
    print("-" * len(header))

    agg = {a: [] for a in algorithms}
    for t in targets:
        row = f"{t:>7}"
        for a in algorithms:
            err, elen, fit = run_one(a, t, n_matrices, pop_size, n_gen, seed)
            agg[a].append(err)
            row += f"{err:>16.5f}"
        print(row)

    print("-" * len(header))
    summary = f"{'mean':>7}" + "".join(f"{np.mean(agg[a]):>16.5f}" for a in algorithms)
    print(summary)
    print()
    print("Lower error = better gate approximation. The braid problem has strong")
    print("dependencies between adjacent positions, which Tree-EDA and MK-EDA can")
    print("model but UMDA cannot.")


if __name__ == "__main__":
    main()
