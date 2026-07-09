"""
Compare UMDA, Tree-EDA and MK-EDA on the multi-objective instance families.

The three probabilistic models are compared as the *learning component* of the
same Pareto-based multi-objective EDA (NSGA-II-style crowding-distance selection),
so the only thing that changes between runs is the model of variable
dependencies:

* UMDA      -- univariate (independent variables).
* Tree-EDA  -- bivariate dependency tree (Chow-Liu).
* MK-EDA    -- order-k Markov chain over the variable ordering.

They are run on instances of the three generators implemented from
``functions/Multi_Objective_Code``:

* mNM   -- truncated-Walsh / Markov-network model.
* MNK   -- multi-objective NK landscape.
* mUBQP -- multi-objective UBQP (a hard instance built from order-5 blocks).

Quality is measured with the hypervolume of the final non-dominated front
(higher is better), using a common reference point per instance so the three
EDAs are directly comparable.

Run (positional args, seed first)::

    python eda_multiobjective_instances.py [SEED] [N_VARS] [POP_SIZE] [N_GEN]
"""

import sys
import numpy as np

from pateda.core.eda import EDA
from pateda.core.components import EDAComponents
from pateda.learning import LearnUMDA, LearnTreeModel, LearnMarkovChain
from pateda.sampling import SampleFDA, SampleMarkovChain
from pateda.selection import CrowdingDistanceSelection
from pateda.seeding import RandomInit
from pateda.stop_conditions import MaxGenerations
from pateda.multiobjective import hypervolume, reference_point_from
from pateda.multiobjective.dominance import find_pareto_set

from pateda.functions.discrete_binary.multiobjective import (
    generate_mnm, create_mnm_objective_function,
    generate_mnk, create_mnk_objective_function,
    create_mubqp_objective_function, select_hard_chunk_pairs,
    create_heavy_mubqp_from_chunks,
)

ALGORITHMS = ["UMDA", "Tree-EDA", "MK-EDA"]


def build_model(name, pop_size, alpha=1.0):
    """Return (learning_method, sampling_method) for one EDA model."""
    if name == "UMDA":
        return LearnUMDA(alpha=alpha), SampleFDA(n_samples=pop_size)
    if name == "Tree-EDA":
        return LearnTreeModel(alpha=alpha), SampleFDA(n_samples=pop_size)
    if name == "MK-EDA":
        return LearnMarkovChain(k=1, alpha=alpha), SampleMarkovChain(n_samples=pop_size)
    raise ValueError(name)


def build_instances(n_vars, seed):
    """Return a list of (name, objective_func, n_vars) multi-objective instances."""
    instances = []

    # mNM: bi-objective, orders 2 and 3, opposite signs
    mnm = generate_mnm(n_vars, max_order=3, sigma=5.0, objective_orders=[2, 3], seed=seed)
    instances.append(("mNM", create_mnm_objective_function(mnm), n_vars))

    # MNK: bi-objective, K = 2
    mnk = generate_mnk(n_vars, k=2, n_objectives=2, seed=seed)
    instances.append(("MNK", create_mnk_objective_function(mnk), n_vars))

    # mUBQP: hard instance built from order-5 blocks (heavy overlapping placement)
    hard = select_hard_chunk_pairs(max_pairs=15, n_candidates=3000, min_pareto=3, seed=seed)
    pairs = [(w1, w2) for (w1, w2, _) in hard]
    mubqp = create_heavy_mubqp_from_chunks(pairs, n_vars=n_vars, k=5,
                                           n_chunks=n_vars, seed=seed)
    instances.append(("mUBQP-hard", create_mubqp_objective_function(mubqp), n_vars))

    return instances


def run_mo_eda(objective, n_vars, model_name, pop_size, n_gen, seed):
    """Run a Pareto-based MO EDA with the given model; return its final front."""
    learner, sampler = build_model(model_name, pop_size)
    comp = EDAComponents(
        seeding=RandomInit(),
        selection=CrowdingDistanceSelection(ratio=0.5, maximize=True),
        learning=learner, sampling=sampler,
        stop_condition=MaxGenerations(n_gen),
    )
    card = 2 * np.ones(n_vars, dtype=int)
    eda = EDA(pop_size, n_vars, objective, card, comp, random_seed=seed)
    eda.run(verbose=False)
    fitness = np.atleast_2d(eda.fitness)
    idx = find_pareto_set(fitness, maximize=True, return_mask=False)
    return fitness[idx]


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    n_vars = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    pop_size = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    n_gen = int(sys.argv[4]) if len(sys.argv) > 4 else 60

    print("UMDA vs Tree-EDA vs MK-EDA on multi-objective instances")
    print("=" * 66)
    print(f"Seed:            {seed}")
    print(f"Variables:       {n_vars}")
    print(f"Population size: {pop_size}")
    print(f"Generations:     {n_gen}")
    print(f"Selection:       CrowdingDistance (Pareto-based, NSGA-II style)")
    print(f"Quality:         hypervolume of the final front (higher is better)")
    print()

    instances = build_instances(n_vars, seed)

    header = f"{'instance':<14}" + "".join(f"{a:>14}" for a in ALGORITHMS)
    print(header)
    print("-" * len(header))

    for name, objective, nv in instances:
        # collect each EDA's front, then a shared reference point per instance
        fronts = {a: run_mo_eda(objective, nv, a, pop_size, n_gen, seed) for a in ALGORITHMS}
        pooled = np.vstack([f for f in fronts.values() if f.size])
        ref = reference_point_from(pooled, maximize=True, margin=0.1)
        row = f"{name:<14}"
        for a in ALGORITHMS:
            f = fronts[a]
            hv = hypervolume(f, ref, maximize=True) if f.size else 0.0
            row += f"{hv:>14.4g}"
        print(row)

    print()
    print("Notes:")
    print("  * Hypervolume is computed against a common reference point per row,")
    print("    so values are comparable across the three EDAs within a row but")
    print("    not across rows (objectives have different scales).")
    print("  * mNM/MNK are bi-objective in [0,1]-ish ranges; mUBQP uses integer")
    print("    weights, so its hypervolume magnitude is larger.")
    print("  * Tree-EDA and MK-EDA model variable dependencies that UMDA ignores,")
    print("    which tends to help on the interaction-rich mUBQP / high-order mNM.")


if __name__ == "__main__":
    main()
