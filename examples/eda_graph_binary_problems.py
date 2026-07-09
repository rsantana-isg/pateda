"""
Discrete EDAs on binary optimization problems defined on graphs.

This example runs discrete EDAs on the four binary graph problems packaged with
pateda, each encoded with one binary variable per vertex:

* Max-Cut               (maximize weight of the cut)
* Maximum Clique        (maximize clique size, adjacency penalty)
* Maximum Independent Set (maximize set size, adjacency penalty)
* Minimum Dominating Set  (minimize set size s.t. every vertex dominated)

Instances are loaded from the packaged ``graph_instances/`` directory, located
through :func:`pateda.functions.graph_utils.graph_instances_dir`. Two learners
are compared: UMDA (univariate) and Tree-EDA (bivariate dependency tree).

Run:  python examples/eda_graph_binary_problems.py
"""

import numpy as np

from pateda.core.eda import EDA, EDAComponents
from pateda.learning import LearnUMDA, LearnTreeModel
from pateda.sampling import SampleFDA
from pateda.selection import TruncationSelection
from pateda.replacement import ElitistReplacement
from pateda.seeding import RandomInit
from pateda.stop_conditions import MaxGenerations

from pateda.functions.graph_utils import (
    graph_instances_dir,
    read_dimacs_graph,
    read_max_cut_graph,
)
from pateda.functions.discrete_binary.problems.max_cut import (
    MaxCutInstance,
    create_max_cut_objective_function,
)
from pateda.functions.discrete_binary.problems.max_clique import (
    MaxCliqueInstance,
    create_max_clique_objective_function,
)
from pateda.functions.discrete_binary.problems.max_independent_set import (
    MaxIndependentSetInstance,
    create_max_independent_set_objective_function,
)
from pateda.functions.discrete_binary.problems.dominating_set import (
    DominatingSetInstance,
    create_dominating_set_objective_function,
)


def build_problems():
    """Load one packaged instance per binary graph problem.

    Returns a list of ``(name, n_vars, objective_func)`` tuples.
    """
    problems = []

    # --- Max-Cut (weighted, own file format) ---
    mc_file = graph_instances_dir("max_cut") / "g05_60.0"
    n, adj, weights = read_max_cut_graph(str(mc_file))
    inst = MaxCutInstance(n, adj, weights)
    problems.append(("Max-Cut", n, create_max_cut_objective_function(inst)))

    # --- Maximum Clique (DIMACS) ---
    n, adj = read_dimacs_graph(str(graph_instances_dir("maximum_clique") / "gnp_30_60.clq"))
    inst = MaxCliqueInstance(n, adj)
    problems.append(("Maximum Clique", n, create_max_clique_objective_function(inst)))

    # --- Maximum Independent Set (DIMACS) ---
    n, adj = read_dimacs_graph(str(graph_instances_dir("max_independent_set") / "gnp_30_12.mis"))
    inst = MaxIndependentSetInstance(n, adj)
    problems.append(("Maximum Independent Set", n, create_max_independent_set_objective_function(inst)))

    # --- Minimum Dominating Set (DIMACS) ---
    n, adj = read_dimacs_graph(str(graph_instances_dir("dominating_set") / "gnp_30_15.ds"))
    inst = DominatingSetInstance(n, adj)
    problems.append(("Minimum Dominating Set", n, create_dominating_set_objective_function(inst)))

    return problems


def make_components(learner, pop_size):
    """Standard EDA component set for a binary problem."""
    return EDAComponents(
        seeding=RandomInit(),
        selection=TruncationSelection(ratio=0.5),
        learning=learner,
        sampling=SampleFDA(n_samples=pop_size),
        replacement=ElitistReplacement(n_elite=5),
        stop_condition=MaxGenerations(50),
    )


def run_eda(name, n_vars, objective, learner_name, learner, seed=42):
    """Run one EDA configuration on one problem and return the best fitness."""
    pop_size = 200
    cardinality = 2 * np.ones(n_vars, dtype=int)

    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        cardinality=cardinality,
        fitness_func=objective,
        components=make_components(learner, pop_size),
        random_seed=seed,
    )
    stats, _ = eda.run(verbose=False)
    return stats.best_fitness[-1], len(stats.best_fitness)


def main():
    print("Discrete EDAs on binary graph problems")
    print("=" * 60)

    problems = build_problems()
    learners = [
        ("UMDA", lambda: LearnUMDA(alpha=1.0)),
        ("Tree-EDA", lambda: LearnTreeModel()),
    ]

    header = f"{'Problem':<26}{'n':>5}{'UMDA':>14}{'Tree-EDA':>14}"
    print(header)
    print("-" * len(header))

    for name, n_vars, objective in problems:
        row = f"{name:<26}{n_vars:>5}"
        for _, make_learner in learners:
            best, gens = run_eda(name, n_vars, objective, _, make_learner())
            row += f"{best:>14.2f}"
        print(row)

    print()
    print("Notes:")
    print("  * Max-Cut / Maximum Clique / Max Independent Set are maximized.")
    print("  * Minimum Dominating Set fitness = n_nodes - set_size - penalty*undominated,")
    print("    so higher fitness means a smaller, feasible dominating set.")
    print("  * Instances live under pateda/functions/graph_instances/.")


if __name__ == "__main__":
    main()
