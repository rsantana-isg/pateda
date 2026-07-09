"""
Demo: Bayesian-network EDAs backed by the ``bayes_nets`` library.

After migrating ``pateda`` away from pgmpy, the Bayesian-network EDAs (EBNA,
BOA) delegate structure and parameter learning to ``bayes_nets`` and sample
new solutions by ancestral sampling over the learned CPDs.

This script is a self-contained smoke / demonstration runner.  It

  1. runs EBNA and BOA on OneMax and on a 3-bit deceptive function, and
  2. shows direct use of ``bayes_nets.BayesianNetwork`` to learn a structure
     from an EDA population and draw fresh samples from it.

Usage
-----
    python bn_eda_bayes_nets_demo.py [seed]

``seed`` is the first (optional) positional argument; it defaults to 42.

Requires ``bayes_nets`` to be importable (``pip install -e edas_bayes_nets``).
"""

import sys
import numpy as np

from pateda import EDA, EDAComponents
from pateda.seeding import RandomInit
from pateda.learning import LearnEBNA, LearnBOA
from pateda.sampling import SampleBayesianNetwork
from pateda.selection import TruncationSelection
from pateda.replacement import ElitistReplacement
from pateda.stop_conditions import MaxGenerations
from pateda.functions.discrete_binary.toy_functions.onemax import onemax
from pateda.functions.discrete_binary.toy_functions.deceptive import deceptive3


# ---------------------------------------------------------------------------
# EDA runner
# ---------------------------------------------------------------------------

def run_eda(label, learner, fitness_func, n_vars, optimal,
            pop_size=300, max_gen=40, seed=42):
    """Run one Bayesian-network EDA and print a compact result line."""
    components = EDAComponents(
        seeding=RandomInit(),
        learning=learner,
        sampling=SampleBayesianNetwork(n_samples=pop_size),
        selection=TruncationSelection(ratio=0.3),
        replacement=ElitistReplacement(),
        stop_condition=MaxGenerations(max_gen=max_gen),
    )
    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        fitness_func=fitness_func,
        cardinality=np.full(n_vars, 2),
        components=components,
        random_seed=seed,
    )
    stats, _ = eda.run(verbose=False)
    best = stats.best_fitness_overall
    found = stats.generation_found if stats.generation_found is not None else max_gen
    pct = 100.0 * best / optimal if optimal else float("nan")
    print(f"  {label:<28} best={best:7.3f}/{optimal:<7.3f} ({pct:5.1f}%)  gen={found:3d}")
    return stats


# ---------------------------------------------------------------------------
# Direct bayes_nets usage
# ---------------------------------------------------------------------------

def demo_direct_bayes_nets(seed):
    """Learn a BN from a structured population and sample from it directly."""
    from bayes_nets import BayesianNetwork

    rng = np.random.default_rng(seed)
    n_vars = 6
    cardinality = np.full(n_vars, 2)

    # Build a population with a clear chain dependency 0->1->...->5.
    n = 500
    data = np.zeros((n, n_vars), dtype=int)
    data[:, 0] = rng.integers(0, 2, size=n)
    for j in range(1, n_vars):
        flip = rng.random(n) < 0.05
        data[:, j] = np.where(flip, 1 - data[:, j - 1], data[:, j - 1])

    bn = BayesianNetwork(n_vars=n_vars, cardinality=cardinality)
    bn.fit(data, method="bic", max_parents=2, alpha=1.0)

    print(f"  Learned BN: {bn}")
    print(f"  Edges (parent -> child):")
    for parent in range(n_vars):
        for child in range(n_vars):
            if bn.adjacency[parent, child]:
                print(f"     X{parent} -> X{child}")

    samples = bn.sample(n_samples=10, rng=rng)
    print(f"  10 fresh samples drawn from the learned BN:")
    for row in samples:
        print("     " + "".join(map(str, row)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(seed=42):
    print(f"Seed:             {seed}")
    print(f"Support library:  bayes_nets (replaces pgmpy)")

    n_vars = 18  # 6 blocks of 3 for deceptive3; also fine for OneMax

    print("\n=== OneMax (n_vars = {}) ===".format(n_vars))
    run_eda("EBNA (bic, alpha=0.1)",
            LearnEBNA(max_parents=2, score_metric="bic", alpha=0.1),
            onemax, n_vars, optimal=float(n_vars), seed=seed)
    run_eda("BOA  (k2,  alpha=1.0)",
            LearnBOA(max_parents=3, score_metric="k2", metric_alpha=1.0),
            onemax, n_vars, optimal=float(n_vars), seed=seed)

    print("\n=== Deceptive-3 (n_vars = {}) ===".format(n_vars))
    optimal = (n_vars // 3) * 1.0
    run_eda("EBNA (bic, alpha=0.1)",
            LearnEBNA(max_parents=2, score_metric="bic", alpha=0.1),
            deceptive3, n_vars, optimal=optimal, pop_size=500, seed=seed)
    run_eda("BOA  (k2,  alpha=1.0)",
            LearnBOA(max_parents=3, score_metric="k2", metric_alpha=1.0),
            deceptive3, n_vars, optimal=optimal, pop_size=500, seed=seed)

    print("\n=== Direct bayes_nets structure learning + sampling ===")
    demo_direct_bayes_nets(seed)


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    main(seed)
