"""
Tree-EDA-M: keeping only malign interactions

Tree-EDA-M (Santana, Larrañaga & Lozano, 2005) is Tree-EDA with one change: a
pairwise interaction is added to the tree only when it is *malign*.  An
interaction is

- **benign**  when the most probable joint configuration argmax p(x_i, x_j)
  coincides with the joint mode predicted by the univariate marginals alone,
  (argmax p(x_i), argmax p(x_j)) -- the dependency merely reinforces the main
  effects and can be reproduced by sampling the two variables independently;
- **malign** otherwise -- the joint mode contradicts the main effects (the
  situation the GA community calls deception), so the dependency genuinely has
  to be stored in the model.

Tree-EDA-M computes mutual information (and hence allows a tree edge) only for
malign pairs, so its model is usually a *forest*.  The benign/malign test is
defined purely through argmax of the marginal tables, so it works for
**discrete non-binary** variables of any cardinality -- which this script
demonstrates explicitly.

The script has two parts:

1. Correctness of the malign detector on hand-built marginals (binary and
   non-binary, benign and malign), including the intuition behind each case.

2. An optimization comparison of Tree-EDA vs Tree-EDA-M on a binary deceptive
   problem and on an integer (non-binary) deceptive problem: Tree-EDA-M reaches
   comparable fitness while learning a sparser model (fewer edges), reproducing
   the paper's conclusion that discarding benign interactions does not hurt.

Usage
-----
    python3 tree_eda_m_demo.py [seed]

``seed`` is the first (optional) positional argument; it defaults to 42.
"""

import sys
import numpy as np

from pateda import EDA, EDAComponents
from pateda.seeding import RandomInit
from pateda.learning import LearnTreeModel, LearnTreeModelM
from pateda.learning.utils.marginal_prob import find_marginal_prob
from pateda.sampling import SampleFDA
from pateda.selection import TruncationSelection
from pateda.replacement import ElitistReplacement
from pateda.stop_conditions import MaxGenerations
from pateda.functions.discrete_binary.toy_functions.deceptive import deceptive3


# ---------------------------------------------------------------------------
# Part 1 — malign / benign detection
# ---------------------------------------------------------------------------

def _pop_from_counts(counts, n_vars):
    """Build a 2-column population that realizes the given cell counts.

    ``counts`` maps a value tuple (a, b) to an integer number of individuals.
    """
    rows = []
    for cfg, n in counts.items():
        rows.extend([list(cfg)] * n)
    return np.array(rows, dtype=int)


def _classify(counts, cardinality):
    """Return (joint_mode, uni_modes, is_malign) for a 2-variable pair."""
    card = np.asarray(cardinality)
    pop = _pop_from_counts(counts, 2)
    univ, biv = find_marginal_prob(pop, 2, card)
    card_j = int(card[1])
    joint_mode = divmod(int(np.argmax(biv[0][1])), card_j)
    uni_modes = (int(np.argmax(univ[0])), int(np.argmax(univ[1])))
    malign = bool(LearnTreeModelM.detect_malign_mask(2, card, univ, biv)[0, 1])
    return joint_mode, uni_modes, malign


def detection_demo():
    print("=" * 76)
    print("1. Benign / malign detection (argmax of joint vs. product of marginals)")
    print("=" * 76)

    cases = [
        # label, counts, cardinality, expected_malign, intuition
        ("binary, positively correlated (both favor 1)",
         {(0, 0): 30, (0, 1): 8, (1, 0): 7, (1, 1): 55}, [2, 2], False,
         "joint mode and marginals all point to (1,1): reinforcing -> benign"),

        ("binary, deceptive pair (mass on (0,1) but marginals favor 0)",
         {(0, 0): 25, (0, 1): 30, (1, 0): 28, (1, 1): 17}, [2, 2], True,
         "marginals predict (0,0) but the joint mode is (0,1) -> malign"),

        ("cardinality 3, aligned",
         {(0, 0): 40, (0, 1): 5, (0, 2): 5, (1, 0): 5, (1, 1): 3,
          (1, 2): 2, (2, 0): 5, (2, 1): 2, (2, 2): 3}, [3, 3], False,
         "joint mode (0,0) equals the marginal modes -> benign"),

        ("cardinality 3, deceptive (joint mode hidden off the marginal modes)",
         {(0, 0): 8, (0, 1): 7, (0, 2): 6, (1, 0): 1, (1, 1): 1,
          (1, 2): 15, (2, 0): 1, (2, 1): 1, (2, 2): 1}, [3, 3], True,
         "marginals predict (0,2) but the joint mode is (1,2) -> malign"),
    ]

    all_ok = True
    for label, counts, card, expected, intuition in cases:
        joint_mode, uni_modes, malign = _classify(counts, card)
        ok = malign == expected
        all_ok &= ok
        tag = "malign" if malign else "benign"
        print(f"\n  {label}")
        print(f"    joint mode {joint_mode}, marginal modes {uni_modes}"
              f"  ->  {tag}  [{'OK' if ok else 'WRONG'}]")
        print(f"    {intuition}")
    print()
    assert all_ok, "malign detector disagreed with the expected labels"
    print("  All detection cases correct (binary and non-binary).")
    print()


# ---------------------------------------------------------------------------
# Part 2 — optimization: Tree-EDA vs Tree-EDA-M
# ---------------------------------------------------------------------------

def make_integer_deceptive(k, cardinality):
    """Order-k integer deceptive function (values in [0, cardinality-1]).

    A block scores ``k*m + 1`` when all its variables equal the maximum
    ``m = cardinality-1`` (the optimum), and ``(k*m - 1) - sum`` otherwise, so
    the average gradient misleadingly points to all-zeros: the within-block
    interactions are malign.
    """
    m = cardinality - 1
    block_opt = k * m + 1

    def f(x):
        if x.ndim == 2:
            x = x.ravel()
        total = 0.0
        for i in range(0, len(x) - k + 1, k):
            s = int(x[i:i + k].sum())
            total += block_opt if s == k * m else (k * m - 1) - s
        return float(total)

    return f, block_opt


def _count_edges(model):
    """Number of parent->child edges in a learned tree/forest model."""
    return int(np.sum(model.structure[:, 0] > 0))


def _run(learner_factory, fitness, n_vars, card, pop_size, max_gen, seed):
    """Run one EDA; return (best_fitness, edges_in_first_model)."""
    cardv = np.full(n_vars, card)
    components = EDAComponents(
        seeding=RandomInit(),
        learning=learner_factory(),
        sampling=SampleFDA(n_samples=pop_size),
        selection=TruncationSelection(ratio=0.3),
        replacement=ElitistReplacement(),
        stop_condition=MaxGenerations(max_gen=max_gen),
    )
    eda = EDA(
        pop_size=pop_size, n_vars=n_vars, fitness_func=fitness,
        cardinality=cardv, components=components, random_seed=seed,
    )
    stats, cache = eda.run(cache_config=[0, 0, 1, 0, 0], verbose=False)
    # Average number of edges over the run (model sparsity signal).
    edges = np.mean([_count_edges(m) for m in cache.models]) if cache.models else 0
    return stats.best_fitness_overall, edges


def comparison(name, fitness, optimum, n_vars, card, pop_size, base_seed,
               max_gen=40, n_runs=5):
    print("=" * 76)
    print(f"2. {name}")
    print("=" * 76)
    print(f"   n_vars={n_vars}, cardinality={card}, pop={pop_size}, "
          f"gens={max_gen}, runs={n_runs}, optimum={optimum}")
    print(f"\n   {'algorithm':<14} | {'mean best':>9} | {'%opt':>6} |"
          f" {'mean edges':>10}")
    print("   " + "-" * 50)

    algos = [
        ("Tree-EDA", lambda: LearnTreeModel()),
        ("Tree-EDA-M", lambda: LearnTreeModelM()),
    ]
    for label, factory in algos:
        bests, edges = [], []
        for r in range(n_runs):
            b, e = _run(factory, fitness, n_vars, card, pop_size,
                        max_gen, base_seed + 100 * r)
            bests.append(b)
            edges.append(e)
        mean_best = float(np.mean(bests))
        pct = 100.0 * mean_best / optimum
        print(f"   {label:<14} | {mean_best:>9.2f} | {pct:>5.1f}% |"
              f" {np.mean(edges):>10.2f}")
    print(f"\n   (a full tree on {n_vars} variables has {n_vars - 1} edges;"
          f" Tree-EDA-M keeps only malign ones)")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(seed=42):
    print("#" * 76)
    print("# Tree-EDA-M — Tree EDA using only malign interactions")
    print(f"# seed = {seed}")
    print("#" * 76)
    print()

    detection_demo()

    # Binary deceptive: within-block interactions are malign, so Tree-EDA-M
    # should keep the important edges and match Tree-EDA.
    n_vars = 18
    comparison(
        "Binary Deceptive-3", deceptive3, optimum=float(n_vars // 3),
        n_vars=n_vars, card=2, pop_size=500, base_seed=seed,
    )

    # Non-binary (integer) deceptive: demonstrates Tree-EDA-M on high-ish
    # cardinality; the deceptive blocks are malign and must be captured.
    k, n_blocks, card = 2, 6, 5
    n_vars = k * n_blocks
    fitness, block_opt = make_integer_deceptive(k, card)
    comparison(
        f"Integer Deceptive (cardinality {card})", fitness,
        optimum=float(n_blocks * block_opt), n_vars=n_vars, card=card,
        pop_size=700, base_seed=seed,
    )

    print("=" * 76)
    print("Reading the results")
    print("=" * 76)
    print("  Tree-EDA-M reaches fitness comparable to Tree-EDA while using")
    print("  fewer edges: it discards benign interactions (reproducible by")
    print("  independent sampling) and keeps only the malign/deceptive ones")
    print("  that the model actually needs -- and it does so for non-binary")
    print("  variables just as for binary ones.")
    print("=" * 76)


if __name__ == "__main__":
    s = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    main(s)
