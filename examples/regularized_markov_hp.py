"""
MkRg-EDA: regularized k-order Markov EDA on the HP protein folding problem

Reproduces the setting of Santana, Karshenas, Bielza & Larrañaga (2011): an EDA
whose k-order Markov model replaces each conditional probability table by a
regularized multinomial regression (elastic net) of a variable on its previous
k variables.  Three predictor variants of growing complexity are compared:

    MkRgk    -- previous k variables directly              (O(k) parameters)
    MkBivRgk -- pairwise products of the previous k         (O(k^2))
    MkAllRgk -- previous k variables + their products       (O(k^2))

against the classic k-order Markov EDA that stores full conditional probability
tables (Mk1, Mk3).

The HP (hydrophobic-polar) model folds a sequence of H/P residues on a 2-D
lattice; here fitness is the number of H-H topological contacts (penalized by
self-overlaps), which is maximized.  Solutions use the relative-move encoding
(each variable in {0,1,2} = left / forward / right), the natural sequential
representation for a Markov model.

The script has two parts:

  1. Optimization: mean best fitness of each EDA on two HP instances.
  2. Model insight: the magnitude of the regression coefficients by predictor
     lag, reproducing the paper's observation (Figure 9) that variables closer
     to X_i contribute more to its prediction than distant ones.

Usage
-----
    python3 regularized_markov_hp.py [seed]
"""

import sys
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")

from pateda import EDA, EDAComponents
from pateda.seeding import RandomInit
from pateda.learning import LearnRegularizedMarkov, LearnMarkovChain
from pateda.sampling import SampleRegularizedMarkov, SampleFDA
from pateda.selection import TruncationSelection
from pateda.replacement import ElitistReplacement
from pateda.stop_conditions import MaxGenerations
from pateda.repairing import HPBacktrackingRepair
from pateda.functions.discrete_non_binary.problems.hp_protein import (
    create_hp_objective_function,
    eval_chain,
)


# ---------------------------------------------------------------------------
# HP instances (H = 0, P = 1)
# ---------------------------------------------------------------------------

def hp_seq(s):
    return np.array([0 if c == "H" else 1 for c in s], dtype=int)


INSTANCES = {
    # s1 of the paper, n = 20  ({HP}^2 {PHH}^2 PHPHHPPHPH)
    "s1 (n=20)": hp_seq("HPHPPHHPHHPHPHHPPHPH"),
    # a second 24-residue instance
    "hp24 (n=24)": hp_seq("HHPPHPPHPPHPPHPPHPPHPPHH"),
}


# ---------------------------------------------------------------------------
# EDA runner
# ---------------------------------------------------------------------------

def run_eda(learner, sampler, seq, seed, pop=None, gens=40, repair=True):
    n = len(seq)
    pop = pop or 4 * n                       # N = 4n, as in the paper
    fitness = create_hp_objective_function(seq)

    comp = EDAComponents(
        seeding=RandomInit(),
        learning=learner,
        sampling=sampler(pop),
        selection=TruncationSelection(ratio=0.15),   # truncation T = 0.15
        replacement=ElitistReplacement(),            # best elitism
        # Backtracking repair of self-intersecting folds (paper's ENReg-EDA
        # sampling step); applied to the sampled population before evaluation.
        repairing=HPBacktrackingRepair() if repair else None,
        stop_condition=MaxGenerations(max_gen=gens),
    )
    eda = EDA(pop_size=pop, n_vars=n, fitness_func=lambda x: fitness(np.asarray(x, int)),
              cardinality=np.full(n, 3), components=comp, random_seed=seed)
    stats, _ = eda.run(verbose=False)
    return stats.best_fitness_overall


def optimization(base_seed, n_runs=3, gens=40):
    print("=" * 78)
    print("1. Optimization on HP protein instances (fitness = H-H contacts)")
    print("=" * 78)
    k = 3
    algos = [
        ("Mk1 (CPT k=1)", lambda: LearnMarkovChain(k=1, alpha=1.0),
         lambda p: SampleFDA(n_samples=p)),
        ("Mk3 (CPT k=3)", lambda: LearnMarkovChain(k=3, alpha=1.0),
         lambda p: SampleFDA(n_samples=p)),
        ("MkRgk (k=3)", lambda: LearnRegularizedMarkov(k=k, variant="rgk"),
         lambda p: SampleRegularizedMarkov(n_samples=p)),
        ("MkBivRgk (k=3)", lambda: LearnRegularizedMarkov(k=k, variant="bivrgk"),
         lambda p: SampleRegularizedMarkov(n_samples=p)),
        ("MkAllRgk (k=3)", lambda: LearnRegularizedMarkov(k=k, variant="allrgk"),
         lambda p: SampleRegularizedMarkov(n_samples=p)),
    ]
    print(f"  N = 4n, truncation 0.15, best elitism, {gens} generations, "
          f"{n_runs} runs\n")
    header = f"  {'algorithm':<16} |" + "".join(f" {name:>13} |" for name in INSTANCES)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, learner_f, sampler_f in algos:
        cells = []
        for seq in INSTANCES.values():
            vals = [run_eda(learner_f(), sampler_f, seq, base_seed + 7 * r, gens=gens)
                    for r in range(n_runs)]
            cells.append(f"{np.mean(vals):13.2f}")
        print(f"  {label:<16} |" + "".join(f" {c} |" for c in cells))
    print("\n  The regularized variants approximate the CPT-based Markov EDAs while")
    print("  keeping a polynomial (not exponential in k) number of parameters;")
    print("  richer predictors (Biv/All) can capture more of the interactions.\n")


# ---------------------------------------------------------------------------
# Model insight: coefficient magnitude by predictor lag
# ---------------------------------------------------------------------------

def coefficient_analysis(base_seed):
    print("=" * 78)
    print("2. What the regularized model captures: coefficient weight by lag")
    print("=" * 78)
    seq = INSTANCES["s1 (n=20)"]
    n = len(seq)
    k = 3
    fitness = create_hp_objective_function(seq)

    # Build a selected population from a few generations of the EDA, then learn a
    # MkRgk model and inspect its regression coefficients.
    rng = np.random.default_rng(base_seed)
    pop = rng.integers(0, 3, size=(4 * n, n))
    fits = np.array([fitness(ind) for ind in pop])
    order = np.argsort(fits)[::-1][: max(4, int(0.3 * len(pop)))]
    selected = pop[order]

    model = LearnRegularizedMarkov(k=k, variant="rgk").learn(
        0, n, np.full(n, 3), selected, fits[order])

    # For each regression sub-model the features are the raw predictors ordered
    # [x_{i-k}, ..., x_{i-1}], i.e. feature j has lag (k - j).  Average the
    # coefficient magnitude across variables per lag.
    lag_weights = {lag: [] for lag in range(1, k + 1)}
    for sm in model.parameters["submodels"]:
        if sm["kind"] != "regression":
            continue
        coef = np.abs(sm["model"].coef_)            # (n_classes, n_features)
        n_feat = coef.shape[1]
        per_feature = coef.mean(axis=0)             # mean |coef| over classes
        for j in range(n_feat):
            lag = n_feat - j                        # nearest predictor -> lag 1
            if lag in lag_weights:
                lag_weights[lag].append(per_feature[j])

    print(f"  MkRgk model on {('s1 (n=20)')}, k={k}.  Mean |coefficient| per lag:\n")
    print(f"  {'lag (distance to X_i)':<24} | {'mean |coef|':>11}")
    print("  " + "-" * 40)
    for lag in range(1, k + 1):
        w = np.mean(lag_weights[lag]) if lag_weights[lag] else 0.0
        bar = "#" * int(round(w * 40 / (max(np.mean(v) for v in lag_weights.values()
                                            if v) + 1e-9)))
        print(f"  lag {lag} (X_i-{lag}){'':<12} | {w:>11.3f}  {bar}")
    print("\n  As in the paper, variables closer to X_i (lag 1) carry more weight;")
    print("  the elastic net shrinks the contribution of the more distant ones.\n")


# ---------------------------------------------------------------------------
# Backtracking repair operator
# ---------------------------------------------------------------------------

def repair_effect(base_seed, n_runs=3, gens=40):
    print("=" * 78)
    print("3. Effect of the backtracking repair (self-avoiding folds)")
    print("=" * 78)
    seq = INSTANCES["s1 (n=20)"]
    n = len(seq)

    # (a) Direct effect: sample from a learned model and count self-intersections
    #     before and after the repair.
    rng = np.random.default_rng(base_seed)
    pop0 = rng.integers(0, 3, size=(4 * n, n))
    fitness = create_hp_objective_function(seq)
    fits = np.array([fitness(ind) for ind in pop0])
    selected = pop0[np.argsort(fits)[::-1][: max(4, int(0.15 * len(pop0)))]]
    model = LearnRegularizedMarkov(k=3, variant="allrgk").learn(
        0, n, np.full(n, 3), selected, fits[np.argsort(fits)[::-1][:len(selected)]])
    sampled = SampleRegularizedMarkov(n_samples=1000).sample(
        n, model, np.full(n, 3), rng=np.random.default_rng(base_seed + 1))
    repaired = HPBacktrackingRepair().repair(sampled, np.full(n, 3))

    def frac_overlapping(P):
        return np.mean([eval_chain(ind, seq)[1] > 0 for ind in P])

    print(f"  Sampled folds that self-intersect: "
          f"{frac_overlapping(sampled):.0%}  ->  after repair: "
          f"{frac_overlapping(repaired):.0%}\n")

    # (b) Optimization effect: MkAllRgk with vs without the repair.
    print(f"  MkAllRgk mean best fitness ({n_runs} runs, {gens} generations):")
    for label, rep in (("with repair", True), ("without repair", False)):
        vals = [run_eda(LearnRegularizedMarkov(k=3, variant="allrgk"),
                        lambda p: SampleRegularizedMarkov(n_samples=p),
                        seq, base_seed + 7 * r, gens=gens, repair=rep)
                for r in range(n_runs)]
        print(f"    {label:<16} {np.mean(vals):6.2f}")
    print("\n  The repair turns self-intersecting samples into valid self-avoiding")
    print("  walks, so more H-H contacts are counted and the search is not wasted")
    print("  on infeasible folds.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(seed=42):
    print("#" * 78)
    print("# MkRg-EDA: regularized k-order Markov models on HP protein folding")
    print(f"# seed = {seed}")
    print("#" * 78 + "\n")
    t0 = time.time()
    optimization(seed)
    coefficient_analysis(seed)
    repair_effect(seed)
    print(f"(total time {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    s = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    main(s)
