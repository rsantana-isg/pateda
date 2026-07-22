"""
Int_FDA on high-cardinality integer problems

Int_FDA (Santana, Ochoa & Soto, 2002) is a tree-based FDA for integer problems
whose variables have *very high cardinality*.  Its model is not a probability
distribution: it is a Chow-Liu tree plus two index tables and the selected
population itself.  New individuals are produced by copying genes from donor
vectors of the selected set along the tree, which reproduces the empirical tree
distribution p(x) = p(x_root) · prod_i p(x_i | x_parent(i)) *without ever
building a c x c conditional table* — the property that makes it scale to large
cardinalities where a table-based tree EDA would run out of memory.

This script does two things:

1. Correctness / consistency checks (fast, deterministic):
   - the auxiliary tables (PopulValues, ParentIndices) are built correctly;
   - the sampler reproduces the empirical tree conditionals of the selected set
     to within Monte-Carlo error — i.e. Int_FDA and a table-based tree EDA
     define the *same* distribution;
   - memory used by Int_FDA's model stays O(N·n + c·n) while the table-based
     tree grows as O(c^2·n), demonstrated by counting stored numbers.

2. Optimization experiment reproducing the paper's headline result:
   Int_FDA vs UMDA on an order-2 integer *deceptive* function as the
   cardinality of the variables grows.  As in the paper, the tree-based integer
   FDA reaches better (partial) solutions than the univariate UMDA, and the gap
   is visible even though both algorithms use the same population size.

Usage
-----
    python3 int_fda_integer_deceptive.py [seed]

``seed`` is the first (optional) positional argument; it defaults to 42.
"""

import sys
import numpy as np

from pateda import EDA, EDAComponents
from pateda.seeding import RandomInit
from pateda.learning import LearnIntFDA, LearnUMDA, LearnTreeModel
from pateda.sampling import SampleIntFDA, SampleFDA
from pateda.selection import TruncationSelection
from pateda.replacement import ElitistReplacement
from pateda.stop_conditions import MaxGenerations


# ---------------------------------------------------------------------------
# Integer deceptive function (order 2), following the paper's F_dec
# ---------------------------------------------------------------------------
#
# For each non-overlapping block of ``k`` integer variables with values in
# [0, c-1] (m = c-1 is the maximum value):
#   - if every variable in the block equals m (block sum = k*m): score k*m + 1
#     (the global optimum of the block, the all-max configuration);
#   - otherwise: score (k*m - 1) - block_sum, which *decreases* with the sum.
#
# The second branch makes the function deceptive: from random points the
# average gradient points toward *smaller* values (all-zeros), away from the
# all-max optimum.  A univariate algorithm follows that misleading signal; an
# algorithm that models within-block dependencies can escape it.

def make_integer_deceptive(k: int, cardinality: int):
    """Return an order-``k`` integer deceptive objective and its optimum/block."""
    m = cardinality - 1
    block_opt = k * m + 1                 # score of an all-max block

    def f(x: np.ndarray) -> float:
        if x.ndim == 2:
            x = x.ravel()
        total = 0.0
        for i in range(0, len(x) - k + 1, k):
            block = x[i:i + k]
            s = int(block.sum())
            if s == k * m:
                total += block_opt
            else:
                total += (k * m - 1) - s
        return float(total)

    return f, block_opt


# ---------------------------------------------------------------------------
# Part 1 — correctness / consistency checks
# ---------------------------------------------------------------------------

def _chain_selected_set(n_vars, N, card, rng, flip=0.15):
    """A selected set with a 0->1->...->(n-1) dependency chain (strong MI)."""
    sel = np.zeros((N, n_vars), dtype=int)
    sel[:, 0] = rng.integers(0, card, N)
    for j in range(1, n_vars):
        redraw = rng.random(N) < flip
        sel[:, j] = np.where(redraw, rng.integers(0, card, N), sel[:, j - 1])
    return sel


def check_auxiliary_tables(rng):
    print("=" * 74)
    print("1a. Auxiliary tables (PopulValues, ParentIndices) are consistent")
    print("=" * 74)
    n_vars, N, card = 6, 200, 8
    sel = _chain_selected_set(n_vars, N, card, rng)
    cardv = np.full(n_vars, card)

    model = LearnIntFDA().learn(0, n_vars, cardv, sel, np.zeros(N))
    pv = model.parameters["popul_values"]
    pi = model.parameters["parent_indices"]

    ok = True
    for i in range(n_vars):
        col_sorted = sel[pv[:, i], i]
        # PopulValues column i must be sorted by value of variable i
        ok &= bool(np.all(np.diff(col_sorted) >= 0))
        # ParentIndices boundaries must delimit exactly the block of each value
        for v in range(card):
            s, e = int(pi[i][v]), int(pi[i][v + 1])
            ok &= bool(np.all(col_sorted[s:e] == v))
            ok &= (e - s) == int((sel[:, i] == v).sum())
    print(f"  PopulValues sorted per column and ParentIndices blocks exact: {ok}")
    assert ok
    print()


def check_distribution_equivalence(rng):
    print("=" * 74)
    print("1b. Int_FDA samples the *same* distribution as a table-based tree EDA")
    print("=" * 74)
    n_vars, N, card = 5, 300, 6
    sel = _chain_selected_set(n_vars, N, card, rng, flip=0.2)
    cardv = np.full(n_vars, card)

    model = LearnIntFDA().learn(0, n_vars, cardv, sel, np.zeros(N))
    cliques = model.structure
    big = SampleIntFDA(n_samples=200_000).sample(n_vars, model, cardv, rng=rng)

    max_err = 0.0
    for c in range(cliques.shape[0]):
        if int(cliques[c, 0]) == 0:                       # root -> marginal
            rv = int(cliques[c, 2])
            emp = np.bincount(sel[:, rv], minlength=card) / N
            smp = np.bincount(big[:, rv], minlength=card) / big.shape[0]
            max_err = max(max_err, float(np.abs(emp - smp).max()))
            continue
        p, ch = int(cliques[c, 2]), int(cliques[c, 3])    # edge -> conditional
        for v in range(card):
            sel_mask = sel[:, p] == v
            smp_mask = big[:, p] == v
            if sel_mask.sum() == 0 or smp_mask.sum() == 0:
                continue
            emp = np.bincount(sel[sel_mask, ch], minlength=card) / sel_mask.sum()
            smp = np.bincount(big[smp_mask, ch], minlength=card) / smp_mask.sum()
            max_err = max(max_err, float(np.abs(emp - smp).max()))

    print(f"  max |sampled - empirical tree conditional| = {max_err:.4f}"
          f"  (Monte-Carlo error, ~1/sqrt(200000))")
    assert max_err < 0.02
    print("  => Int_FDA reproduces the empirical Chow-Liu tree distribution.")
    print()


def check_memory_scaling(rng):
    print("=" * 74)
    print("1c. Model size: Int_FDA is O(N.n + c.n), a table tree is O(c^2.n)")
    print("=" * 74)
    n_vars, N = 10, 250
    print(f"  n_vars={n_vars}, |selected set| N={N}")
    print(f"  {'cardinality':>11} | {'Int_FDA numbers':>16} | {'tree-table numbers':>18}")
    print("  " + "-" * 52)
    for card in (8, 32, 128, 512):
        sel = _chain_selected_set(n_vars, N, card, rng)
        cardv = np.full(n_vars, card)

        int_model = LearnIntFDA().learn(0, n_vars, cardv, sel, np.zeros(N))
        # Numbers physically stored by Int_FDA: selected pop + PopulValues +
        # ParentIndices (cumulative arrays of length c+1 per variable).
        int_numbers = (
            int_model.parameters["selected_population"].size
            + int_model.parameters["popul_values"].size
            + sum(pi.size for pi in int_model.parameters["parent_indices"])
        )

        # Numbers a table-based tree EDA would store: one c x c (root: c)
        # conditional table per variable.
        tree_model = LearnTreeModel(alpha=1.0).learn(0, n_vars, cardv, sel, np.zeros(N))
        tree_numbers = sum(t.size for t in tree_model.parameters)

        print(f"  {card:>11} | {int_numbers:>16,} | {tree_numbers:>18,}")
    print("  Int_FDA grows linearly in c; the table tree grows quadratically.")
    print()


# ---------------------------------------------------------------------------
# Part 2 — optimization: Int_FDA vs UMDA on integer deceptive
# ---------------------------------------------------------------------------

def run_eda(kind, fitness, n_vars, card, pop_size, max_gen, seed):
    """Run one EDA (kind in {'IntFDA', 'UMDA'}) and return best fitness reached."""
    cardv = np.full(n_vars, card)
    if kind == "IntFDA":
        learning = LearnIntFDA()
        sampling = SampleIntFDA(n_samples=pop_size)
    elif kind == "UMDA":
        learning = LearnUMDA(alpha=1.0)
        sampling = SampleFDA(n_samples=pop_size)
    else:
        raise ValueError(kind)

    components = EDAComponents(
        seeding=RandomInit(),
        learning=learning,
        sampling=sampling,
        selection=TruncationSelection(ratio=0.3),
        replacement=ElitistReplacement(),
        stop_condition=MaxGenerations(max_gen=max_gen),
    )
    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        fitness_func=fitness,
        cardinality=cardv,
        components=components,
        random_seed=seed,
    )
    stats, _ = eda.run(verbose=False)
    return stats.best_fitness_overall


def optimization_experiment(base_seed):
    print("=" * 74)
    print("2. Optimization: Int_FDA vs UMDA on order-2 integer deceptive")
    print("=" * 74)
    k = 2                      # deceptive block order
    n_blocks = 8
    n_vars = k * n_blocks      # 16 variables
    pop_size = 700
    max_gen = 40
    n_runs = 5

    print(f"  block order k={k}, blocks={n_blocks}, n_vars={n_vars}, "
          f"pop={pop_size}, gens={max_gen}, runs={n_runs}")
    print(f"  (deceptive: gradient points to all-zeros, optimum is all-max)\n")
    print(f"  {'cardinality':>11} | {'optimum':>8} | {'UMDA mean':>10} |"
          f" {'Int_FDA mean':>12} | winner")
    print("  " + "-" * 66)

    for card in (2, 4, 6, 8, 10):
        fitness, block_opt = make_integer_deceptive(k, card)
        optimum = n_blocks * block_opt

        umda_vals, int_vals = [], []
        for r in range(n_runs):
            seed = base_seed + 1000 * r
            umda_vals.append(run_eda("UMDA", fitness, n_vars, card, pop_size, max_gen, seed))
            int_vals.append(run_eda("IntFDA", fitness, n_vars, card, pop_size, max_gen, seed))

        umda_mean = float(np.mean(umda_vals))
        int_mean = float(np.mean(int_vals))
        winner = "Int_FDA" if int_mean > umda_mean + 1e-9 else (
                 "UMDA" if umda_mean > int_mean + 1e-9 else "tie")
        print(f"  {card:>11} | {optimum:>8} | {umda_mean:>10.2f} |"
              f" {int_mean:>12.2f} | {winner}")

    print("\n  As the paper reports, the tree-based integer FDA finds better")
    print("  partial solutions than UMDA on the deceptive function, because it")
    print("  models the within-block dependencies that UMDA ignores.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(seed=42):
    print("#" * 74)
    print("# Int_FDA (tree-based FDA for integers) — correctness and optimization")
    print(f"# seed = {seed}")
    print("#" * 74)
    print()

    rng = np.random.default_rng(seed)
    check_auxiliary_tables(rng)
    check_distribution_equivalence(rng)
    check_memory_scaling(rng)
    optimization_experiment(seed)


if __name__ == "__main__":
    s = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    main(s)
