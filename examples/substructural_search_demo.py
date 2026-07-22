"""
Substructural neighborhood search — structure-guided local search

Substructural local search
(:class:`~pateda.local_optimization.substructural_search.SubstructuralLocalSearch`)
hill-climbs over the joint values of the *substructures* (linkage groups) of the
learned model, instead of flipping single variables (Lima, Pelikan, Sastry,
Butz, Goldberg & Lobo, 2006).  Optimizing a whole building block at once escapes
the deception that traps a single-bit hill climber, providing intensification
for the EDA.

This script demonstrates:

  1. Escaping deception: on a concatenated trap (known block structure),
     single-bit steepest hill climbing vs. substructural search that optimizes
     each block jointly.

  2. Optimization on instances whose structure depends on the instance
     (Ising 2D spin glass, UBQP, 3-SAT): a Tree-EDA with no local search, with a
     single-bit hill climber, and with substructural search on the learned
     model -- at a matched local-search evaluation budget.

Usage
-----
    python3 substructural_search_demo.py [seed]
"""

import sys
import numpy as np

from pateda import EDA, EDAComponents
from pateda.seeding import RandomInit
from pateda.learning import LearnTreeModel
from pateda.sampling import SampleFDA
from pateda.selection import TruncationSelection
from pateda.replacement import ElitistReplacement
from pateda.stop_conditions import MaxGenerations
from pateda.local_optimization import (
    DeterministicHillClimber,
    SubstructuralLocalSearch,
)

from structure_problems import make_trap, make_ising, make_ubqp, make_sat


# ---------------------------------------------------------------------------
# Part 1 — escaping deception on a trap
# ---------------------------------------------------------------------------

def escape_deception(rng):
    print("=" * 78)
    print("1. Escaping deception on a concatenated trap (known block structure)")
    print("=" * 78)
    fitness, G, opt, label = make_trap(n_blocks=6, k=4)
    n = G.shape[0]
    card = np.full(n, 2)
    pop = rng.integers(0, 2, size=(30, n))
    fit = np.array([[fitness(ind)] for ind in pop], dtype=float)
    budget = 30 * 600

    print(f"  {label}, optimum={opt:.0f}\n")

    hc = DeterministicHillClimber(subset_fraction=1.0, evaluation_budget=budget,
                                  seed=int(rng.integers(1e6)))
    _, f_hc = hc.optimize(pop.copy(), fit.copy(), fitness, card)

    ss = SubstructuralLocalSearch(linkage_graph=G, neighborhood="neighborhood",
                                  subset_fraction=1.0, evaluation_budget=budget,
                                  seed=int(rng.integers(1e6)))
    _, f_ss = ss.optimize(pop.copy(), fit.copy(), fitness, card)

    print(f"  {'local search':<28} | {'mean best':>9} | {'max best':>8}")
    print("  " + "-" * 52)
    print(f"  {'single-bit steepest HC':<28} | {f_hc.mean():>9.2f} | {f_hc.max():>8.0f}")
    print(f"  {'substructural (blocks)':<28} | {f_ss.mean():>9.2f} | {f_ss.max():>8.0f}")
    print("\n  The single-bit climber is pulled into the deceptive attractor;")
    print("  optimizing each block jointly reaches the block optima.\n")


# ---------------------------------------------------------------------------
# Part 2 — optimization on instance-structured problems
# ---------------------------------------------------------------------------

def run_eda(fitness, n, local_opt_factory, seed, pop=250, gens=25):
    counter = {"n": 0}

    def wrapped(x):
        counter["n"] += 1
        return fitness(x)

    comp = EDAComponents(
        seeding=RandomInit(),
        learning=LearnTreeModel(),
        sampling=SampleFDA(n_samples=pop),
        selection=TruncationSelection(ratio=0.3),
        replacement=ElitistReplacement(),
        local_opt=None if local_opt_factory is None else local_opt_factory(seed),
        stop_condition=MaxGenerations(max_gen=gens),
    )
    eda = EDA(pop_size=pop, n_vars=n, fitness_func=wrapped,
              cardinality=np.full(n, 2), components=comp, random_seed=seed)
    stats, _ = eda.run(verbose=False)
    return stats.best_fitness_overall, counter["n"]


def optimization(base_seed, n_runs=4):
    print("=" * 78)
    print("2. Tree-EDA + local search on instance-structured problems")
    print("=" * 78)
    print("  Tree-EDA global model; local search on 30% of each population,")
    print("  budget 2000 evals/generation (matched across local searches).")
    print("  Substructural search uses the LEARNED Tree structure.\n")

    LS_BUDGET, FRAC = 2000, 0.3
    problems = [
        ("Ising", lambda s: make_ising(L=5, seed=s)),
        ("UBQP", lambda s: make_ubqp(n=25, density=0.15, seed=s)),
        ("3-SAT", lambda s: make_sat(n=25, ratio=4.0, seed=s)),
    ]
    print(f"  {'problem':<8} | {'no LS':>8} | {'single-bit HC':>13} | "
          f"{'substructural':>13}")
    print("  " + "-" * 52)
    for pname, make in problems:
        no_ls, hc, ss = [], [], []
        for r in range(n_runs):
            seed = base_seed + 100 * r
            fitness, G, opt, label = make(seed)
            n = G.shape[0]
            b0, _ = run_eda(fitness, n, None, seed)
            b1, _ = run_eda(fitness, n,
                            lambda s: DeterministicHillClimber(
                                subset_fraction=FRAC, evaluation_budget=LS_BUDGET, seed=s),
                            seed)
            b2, _ = run_eda(fitness, n,
                            lambda s: SubstructuralLocalSearch(
                                neighborhood="both", subset_fraction=FRAC,
                                evaluation_budget=LS_BUDGET, seed=s),
                            seed)
            no_ls.append(b0); hc.append(b1); ss.append(b2)
        print(f"  {pname:<8} | {np.mean(no_ls):>8.2f} | {np.mean(hc):>13.2f} | "
              f"{np.mean(ss):>13.2f}")
    print("\n  Both local searches intensify the Tree-EDA; substructural search")
    print("  additionally exploits the learned linkage groups.\n")


def main(seed=42):
    print("#" * 78)
    print("# Substructural neighborhood search: structure-guided local search")
    print(f"# seed = {seed}")
    print("#" * 78 + "\n")
    rng = np.random.default_rng(seed)
    escape_deception(rng)
    optimization(seed)


if __name__ == "__main__":
    s = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    main(s)
