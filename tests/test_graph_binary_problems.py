"""Tests for discrete EDAs on binary optimization problems defined on graphs.

Covers the four binary graph problems (one binary variable per vertex):
Max-Cut, Maximum Clique, Maximum Independent Set, Minimum Dominating Set.
Each test checks (a) the objective function is correct on hand-checkable
inputs and (b) a short UMDA run executes and does not worsen the best fitness.
"""

import numpy as np
import pytest

from pateda.core.eda import EDA, EDAComponents
from pateda.learning import LearnUMDA
from pateda.sampling import SampleFDA
from pateda.selection import TruncationSelection
from pateda.replacement import ElitistReplacement
from pateda.seeding import RandomInit
from pateda.stop_conditions import MaxGenerations

from pateda.functions.graph_utils import graph_instances_dir, read_dimacs_graph, read_max_cut_graph
from pateda.functions.discrete_binary.problems.max_cut import (
    MaxCutInstance, eval_max_cut, create_max_cut_objective_function,
)
from pateda.functions.discrete_binary.problems.max_clique import (
    MaxCliqueInstance, eval_max_clique, create_max_clique_objective_function,
)
from pateda.functions.discrete_binary.problems.max_independent_set import (
    MaxIndependentSetInstance, eval_max_independent_set,
    create_max_independent_set_objective_function,
)
from pateda.functions.discrete_binary.problems.dominating_set import (
    DominatingSetInstance, eval_dominating_set, create_dominating_set_objective_function,
)


def _triangle_adj():
    """Adjacency matrix of a triangle plus one isolated vertex (4 nodes)."""
    adj = np.zeros((4, 4), dtype=bool)
    for u, v in [(0, 1), (1, 2), (0, 2)]:
        adj[u, v] = adj[v, u] = True
    return adj


# --------------------------------------------------------------------------- #
# Objective-function correctness on hand-checkable inputs
# --------------------------------------------------------------------------- #

def test_max_cut_objective_value():
    adj = _triangle_adj()
    weights = adj.astype(float)  # unit weights on triangle edges
    inst = MaxCutInstance(4, adj, weights)
    # Partition {0} vs {1,2,3}: edges (0,1) and (0,2) cross -> cut = 2
    assert eval_max_cut(np.array([1, 0, 0, 0]), inst) == pytest.approx(2.0)
    # All same side -> no edge crosses
    assert eval_max_cut(np.array([1, 1, 1, 1]), inst) == pytest.approx(0.0)


def test_max_clique_objective_value():
    inst = MaxCliqueInstance(4, _triangle_adj())
    # {0,1,2} is a clique of size 3, no violations
    assert eval_max_clique(np.array([1, 1, 1, 0]), inst) == pytest.approx(3.0)
    # {0,3} not adjacent -> one violation penalized
    val = eval_max_clique(np.array([1, 0, 0, 1]), inst)
    assert val < 2.0


def test_max_independent_set_objective_value():
    inst = MaxIndependentSetInstance(4, _triangle_adj())
    # {0,3}: not adjacent -> independent set of size 2, no penalty
    assert eval_max_independent_set(np.array([1, 0, 0, 1]), inst) == pytest.approx(2.0)
    # {0,1}: adjacent -> penalized below size
    assert eval_max_independent_set(np.array([1, 1, 0, 0]), inst) < 2.0


def test_dominating_set_objective_value():
    inst = DominatingSetInstance(4, _triangle_adj())
    n = 4
    # {0,3}: 0 dominates 1,2; 3 dominates itself -> all dominated, size 2
    assert eval_dominating_set(np.array([1, 0, 0, 1]), inst) == pytest.approx(n - 2)
    # {0}: vertex 3 isolated and not selected -> undominated, heavy penalty
    assert eval_dominating_set(np.array([1, 0, 0, 0]), inst) < 0


# --------------------------------------------------------------------------- #
# Packaged instances load and EDAs run
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("subdir,pattern", [
    ("max_independent_set", "*.mis"),
    ("dominating_set", "*.ds"),
    ("clique_covering", "*.cc"),
    ("maximum_clique", "gnp_*.clq"),
])
def test_packaged_instances_exist(subdir, pattern):
    files = sorted(graph_instances_dir(subdir).glob(pattern))
    assert files, f"no packaged instances found in {subdir}"
    n, adj = read_dimacs_graph(str(files[0]))
    assert n > 0
    assert adj.shape == (n, n)
    assert np.array_equal(adj, adj.T)  # undirected


def _short_umda(n_vars, objective, seed=1):
    pop_size = 100
    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        cardinality=2 * np.ones(n_vars, dtype=int),
        fitness_func=objective,
        components=EDAComponents(
            seeding=RandomInit(),
            selection=TruncationSelection(ratio=0.5),
            learning=LearnUMDA(alpha=1.0),
            sampling=SampleFDA(n_samples=pop_size),
            replacement=ElitistReplacement(n_elite=3),
            stop_condition=MaxGenerations(15),
        ),
        random_seed=seed,
    )
    stats, _ = eda.run(verbose=False)
    return stats


def test_umda_improves_max_cut():
    n, adj, w = read_max_cut_graph(str(graph_instances_dir("max_cut") / "g05_60.0"))
    inst = MaxCutInstance(n, adj, w)
    stats = _short_umda(n, create_max_cut_objective_function(inst))
    # Elitism guarantees the best fitness never decreases across generations.
    assert stats.best_fitness[-1] >= stats.best_fitness[0]
    assert stats.best_fitness[-1] > 0


def test_umda_runs_on_independent_set():
    n, adj = read_dimacs_graph(str(graph_instances_dir("max_independent_set") / "gnp_30_12.mis"))
    inst = MaxIndependentSetInstance(n, adj)
    stats = _short_umda(n, create_max_independent_set_objective_function(inst))
    assert stats.best_fitness[-1] >= stats.best_fitness[0]
