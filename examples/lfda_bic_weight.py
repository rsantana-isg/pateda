"""
LFDA — Learning the Factorization, and Controlling How Much of It to Learn

FDA (:class:`~pateda.learning.fda.LearnFDA`) is given a factorization of the
search distribution, derived analytically from the structure of an additively
decomposable function.  LFDA (Muehlenbein & Mahnig, 1999) drops that
requirement: it *learns* a Bayesian network from the selected population at
each generation and uses the learned network as the factorization.

Structure search is greedy hill-climbing on a BIC score whose complexity
penalty carries a weighting factor w:

    BIC(B, D, w) = sum_i log P(D_i | Pa_i, theta_ML) - w * (log N / 2) * |theta|

The weight is the algorithm's main control, and it is what this script
explores.  Inside an EDA the "data set" is a single selected population --
a few hundred rows -- so the penalty that is appropriate for a large static
data set is not necessarily the one that works here:

  - w < 1  : denser networks.  Captures more dependencies, but each extra
             parent halves the data available per parameter, so the CPDs get
             noisier and sampling drifts.
  - w = 1  : standard BIC.
  - w > 1  : sparser networks.  Fewer, better-estimated parameters, at the
             risk of missing the linkage the problem actually has.

Problems
--------
1. Deceptive-3, n = 30.  Ten non-overlapping 3-bit deceptive blocks.  The
   true factorization has exactly the within-block edges, so there is a
   correct answer for the structure search to find and it is easy to check
   how much of it each weight recovers.

2. Fc2 (Mühlenbein's F5Muhl blocks), n = 30.  Six multimodal 5-bit blocks,
   from the family LFDA was developed on.  Larger blocks mean the parent sets
   needed are larger, so the penalty weight bites harder.

Alongside the sweep the script reports *linkage recovery*: the fraction of
learned edges that fall inside a true block, and the fraction of the true
within-block adjacencies that were found.  This is what distinguishes a model
that is merely dense from one that is dense in the right places.

Usage
-----
    python3 lfda_bic_weight.py [seed]

``seed`` is the first (optional) positional argument; it defaults to 42.
"""

import sys
import numpy as np

from pateda import EDA, EDAComponents
from pateda.seeding import RandomInit
from pateda.learning import LearnLFDA, LearnEBNA
from pateda.sampling import SampleBayesianNetwork
from pateda.selection import TruncationSelection
from pateda.replacement import ElitistReplacement
from pateda.stop_conditions import MaxGenerations
from pateda.functions.discrete_binary.toy_functions.deceptive import deceptive3
from pateda.functions.discrete_binary.toy_functions.additive_decomposable import fc2


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_VARS = 30
# Moderate population: the BIC penalty weight only visibly changes the learned
# model when data is scarce, which is the regime LFDA is designed for.
POP_SIZE = 150
MAX_GEN = 40
TRUNCATION = 0.3
BIC_WEIGHTS = [0.25, 0.5, 1.0, 2.0, 4.0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_eda(label, learner, fitness_func, n_vars, optimal,
            pop_size=POP_SIZE, max_gen=MAX_GEN, seed=42):
    """Run one EDA and print a compact result line."""
    components = EDAComponents(
        seeding=RandomInit(),
        learning=learner,
        sampling=SampleBayesianNetwork(n_samples=pop_size),
        selection=TruncationSelection(ratio=TRUNCATION),
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
    return best, pct, found


def block_mask(n_vars, block_size):
    """Boolean matrix marking pairs of variables inside the same true block."""
    mask = np.zeros((n_vars, n_vars), dtype=bool)
    for start in range(0, n_vars - block_size + 1, block_size):
        idx = np.arange(start, start + block_size)
        mask[np.ix_(idx, idx)] = True
    np.fill_diagonal(mask, False)
    return mask


def linkage_recovery(adj, mask):
    """Return (precision, recall) of learned edges w.r.t. the true blocks.

    precision: fraction of learned edges that lie inside a true block.
    recall:    fraction of true within-block variable pairs that got an edge.
    """
    adj = np.asarray(adj) > 0
    n_edges = int(adj.sum())
    if n_edges == 0:
        return float("nan"), 0.0
    correct = int((adj & mask).sum())
    precision = correct / n_edges
    # Undirected pairs inside blocks: mask counts each pair twice.
    n_true_pairs = int(mask.sum() // 2)
    undirected = (adj | adj.T) & mask
    recall = int(undirected.sum() // 2) / n_true_pairs
    return precision, recall


def learn_one_model(learner, fitness_func, n_vars, seed):
    """Learn a model from the first selected population of a random sample."""
    rng = np.random.default_rng(seed)
    pop = rng.integers(0, 2, size=(POP_SIZE, n_vars))
    fit = np.array([fitness_func(ind) for ind in pop])
    n_sel = int(POP_SIZE * TRUNCATION)
    order = np.argsort(fit)[::-1][:n_sel]
    return learner.learn(0, n_vars, np.full(n_vars, 2), pop[order], fit[order])


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def sweep(name, fitness_func, optimal, block_size, seed):
    """Sweep the BIC penalty weight on one problem."""
    print("\n" + "=" * 78)
    print(f"{name}  (n={N_VARS}, block size={block_size}, optimum={optimal:.3f})")
    print("=" * 78)
    print(f"  {'weight':>7}  {'edges':>5}  {'max_par':>7}  {'in-block':>8}"
          f"  {'linkage':>7}   {'best':>8}  {'%opt':>6}  {'gen':>4}")
    print("  " + "-" * 72)

    mask = block_mask(N_VARS, block_size)

    for w in BIC_WEIGHTS:
        learner = LearnLFDA(max_parents=block_size - 1, bic_weight=w, alpha=1.0)

        model = learn_one_model(learner, fitness_func, N_VARS, seed)
        adj = np.asarray(model.structure)
        n_edges = int(adj.sum())
        max_par = int(adj.sum(axis=0).max())
        precision, recall = linkage_recovery(adj, mask)

        best, pct, found = run_eda(
            f"LFDA w={w}",
            LearnLFDA(max_parents=block_size - 1, bic_weight=w, alpha=1.0),
            fitness_func, N_VARS, optimal, seed=seed,
        )

        tag = "  <- standard BIC" if w == 1.0 else ""
        print(f"  {w:7.2f}  {n_edges:5d}  {max_par:7d}  {precision:7.1%}"
              f"  {recall:6.1%}   {best:8.3f}  {pct:5.1f}%  {found:4d}{tag}")

    # EBNA reference: the same greedy BIC search without a penalty weight.
    model = learn_one_model(
        LearnEBNA(max_parents=block_size - 1, score_metric="bic", alpha=1.0),
        fitness_func, N_VARS, seed,
    )
    adj = np.asarray(model.structure)
    precision, recall = linkage_recovery(adj, mask)
    best, pct, found = run_eda(
        "EBNA", LearnEBNA(max_parents=block_size - 1, score_metric="bic", alpha=1.0),
        fitness_func, N_VARS, optimal, seed=seed,
    )
    print("  " + "-" * 72)
    print(f"  {'EBNA':>7}  {int(adj.sum()):5d}  {int(adj.sum(axis=0).max()):7d}"
          f"  {precision:7.1%}  {recall:6.1%}   {best:8.3f}  {pct:5.1f}%  {found:4d}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(seed=42):
    print("=" * 78)
    print("LFDA — effect of the BIC complexity-penalty weight")
    print("=" * 78)
    print(f"Seed:             {seed}")
    print(f"Algorithm:        LFDA (greedy BIC hill-climbing, weighted penalty)")
    print(f"Population Size:  {POP_SIZE}")
    print(f"Generations:      {MAX_GEN}")
    print(f"Selection:        Truncation (ratio={TRUNCATION})")
    print(f"Weights swept:    {BIC_WEIGHTS}")
    print()
    print("Columns: 'in-block' = share of learned edges inside a true block")
    print("         'linkage'  = share of true within-block pairs recovered")

    sweep("1. Deceptive-3", deceptive3, optimal=float(N_VARS // 3),
          block_size=3, seed=seed)
    sweep("2. Fc2 (Muehlenbein F5Muhl blocks)", fc2,
          optimal=float((N_VARS // 5) * 4.0), block_size=5, seed=seed)

    print("\n" + "=" * 78)
    print("Reading the results")
    print("=" * 78)
    print("  Edge count falls monotonically as the weight rises: the weight is")
    print("  a direct control on model complexity.")
    print("  Low weights buy linkage recall at the cost of precision -- extra")
    print("  edges are spurious, and each one costs data per parameter.")
    print("  The best-performing weight is problem dependent, which is exactly")
    print("  why LFDA exposes it rather than fixing it at 1.")
    print("=" * 78)


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    main(seed)
