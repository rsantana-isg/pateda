"""
run_bn_eda.py -- one cluster job of the BN-EDA comparison study.

Runs a single Bayesian-network EDA (one BN structure-learning algorithm) on one
optimization problem for one seed, and saves the full per-generation behaviour
needed to compare BN learning algorithms and to test whether *search-distribution
quality metrics* (Spearman correlation, log-likelihood, KL divergence, skeleton
F1) are good surrogates of the *actual* EDA performance.

All the compared EDAs share every component EXCEPT the learning algorithm:

    seeding       RandomInit
    selection     TruncationSelection(ratio = 0.5)            (T = 0.5)
    weighting     selection_weighting = "boltzmann"           (weighted counts)
    learning      LearnEBNA(score_metric = ALGORITHM, ...)    (the only variable)
    sampling      SampleBayesianNetwork
    replacement   ElitistReplacement(n_elite = 1)
    stop          MaxGenerations(100)                         (100 generations)
    pop_size      POP_MULT * n_vars                           (default 10 * n)

Usage (positional args, seed first -- project convention)
---------------------------------------------------------
    python3 scripts/run_bn_eda.py SEED PROBLEM ALGORITHM [OUT_DIR] \
            [POP_MULT] [N_GEN] [MAX_PARENTS] [BETA]

    SEED         integer RNG seed
    PROBLEM      "<Family>_<n>" as in eda_cluster_results.csv (e.g. Ising_100)
    ALGORITHM    bayes_nets BN learning method (e.g. bic, sartre, univ_bn)
    OUT_DIR      output directory (default: results/bn_eda_cluster)
    POP_MULT     population multiplier; pop_size = POP_MULT * n   (default 10)
    N_GEN        number of generations                            (default 100)
    MAX_PARENTS  max parents per variable in the BN               (default 5)
    BETA         Boltzmann inverse-temperature for weighting      (default 1.0)

Output
------
A self-describing JSON file
    bneda_<problem>_<algorithm>_s<seed>.json
holding the run configuration, a per-generation trajectory, and a run-level
summary (see ``build_output``).  A companion one-line ``.dat`` with the run
summary is also written for quick aggregation.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Make pateda (src/), the problem registry (examples/) and bayes_nets importable
# ---------------------------------------------------------------------------
def _setup_paths():
    """Make the vendored problem registry and pateda importable.

    * The problem registry (``bn_eda_problems.py``) lives next to this script.
    * ``pateda`` / ``bayes_nets`` are expected to be **installed** (this is the
      normal case on the cluster).  As a dev convenience, if a source checkout of
      pateda (``.../src/pateda``) is found in an ancestor directory it is put
      first on ``sys.path`` so the local sources are used; otherwise nothing is
      added and the installed packages are used.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)          # so `import bn_eda_problems` works

    d = here
    for _ in range(8):
        src = os.path.join(d, "src")
        if os.path.isfile(os.path.join(src, "pateda", "__init__.py")):
            if src not in sys.path:
                sys.path.insert(0, src)
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.dirname(here)          # project root (parent of scripts/)


_REPO = _setup_paths()

import numpy as np  # noqa: E402  (after path setup)
from bn_eda_problems import parse_problem                      # noqa: E402
from pateda.core.eda import EDA                                # noqa: E402
from pateda.core.components import EDAComponents, LearningMethod, StatisticsMethod  # noqa: E402
from pateda.seeding import RandomInit                          # noqa: E402
from pateda.selection import TruncationSelection               # noqa: E402
from pateda.replacement import ElitistReplacement             # noqa: E402
from pateda.stop_conditions import MaxGenerations              # noqa: E402
from pateda.learning.ebna import LearnEBNA                     # noqa: E402
from pateda.sampling.bayesian_network import SampleBayesianNetwork  # noqa: E402


DEFAULT_OUT = os.path.join(_REPO, "results", "bn_eda_cluster")


# ---------------------------------------------------------------------------
# Search-distribution quality metrics (computed live each generation)
# ---------------------------------------------------------------------------
def bn_log_probs(model, X: np.ndarray, card: np.ndarray) -> np.ndarray:
    """Joint log P(x) under the learned BN for every row of X (from the CPDs)."""
    X = np.asarray(X, dtype=int)
    lp = np.zeros(len(X))
    params = model.parameters
    for v in range(X.shape[1]):
        pa = params[v]["parents"]
        cpd = params[v]["cpd"]
        if not pa:
            p = cpd[X[:, v]]
        else:
            idx = np.zeros(len(X), dtype=int)
            mult = 1
            for pp in pa:
                idx += X[:, pp] * mult
                mult *= int(card[pp])
            p = cpd[idx, X[:, v]]
        lp += np.log(np.clip(p, 1e-12, None))
    return lp


def boltzmann_target(f: np.ndarray, beta: float) -> np.ndarray:
    """Boltzmann probability p ∝ exp(beta * z), z = standardised fitness."""
    f = np.asarray(f, dtype=float)
    s = f.std()
    if s <= 0:
        return np.full(len(f), 1.0 / len(f))
    z = (f - f.mean()) / s
    w = np.exp(beta * z - np.max(beta * z))
    return w / w.sum()


def distribution_metrics(model, X, f, card, beta):
    """(spearman, mean_loglik, kl) of the BN over the set (X, f)."""
    lp = bn_log_probs(model, X, card)
    ll = float(lp.mean())
    if np.all(lp == lp[0]) or np.all(f == f[0]):
        sp = float("nan")
    else:
        sp = float(spearmanr(lp, f).correlation)
    p_data = boltzmann_target(f, beta)
    q = np.exp(lp - lp.max())
    q = q / q.sum()
    q = np.clip(q, 1e-300, None)
    kl = float(np.sum(p_data * np.log(p_data / q)))
    return sp, ll, kl


def skeleton_f1(adjacency: np.ndarray, true_mat: np.ndarray) -> float:
    """Undirected-edge F1 of the learned structure vs the true interaction graph."""
    def sk(a):
        s = ((np.asarray(a) != 0) | (np.asarray(a).T != 0)).astype(int)
        np.fill_diagonal(s, 0)
        return s
    L, T = sk(adjacency), sk(true_mat)
    tp = np.sum((L == 1) & (T == 1)) / 2
    fp = np.sum((L == 1) & (T == 0)) / 2
    fn = np.sum((T == 1) & (L == 0)) / 2
    if tp + fp + fn == 0:
        return 1.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0


def population_diversity(pop: np.ndarray) -> float:
    """Mean per-variable normalised entropy (0 = converged, 1 = uniform)."""
    pop = np.asarray(pop, dtype=int)
    n, m = pop.shape
    ents = []
    for j in range(m):
        _, counts = np.unique(pop[:, j], return_counts=True)
        probs = counts / n
        h = -np.sum(probs * np.log(np.clip(probs, 1e-12, None)))
        k = len(counts)
        ents.append(h / np.log(k) if k > 1 else 0.0)
    return float(np.mean(ents))


# ---------------------------------------------------------------------------
# Components that record the per-generation behaviour
# ---------------------------------------------------------------------------
class _Recorder:
    """Shared scratch space between the stats stash and the metered learner."""
    def __init__(self):
        self.pop = None
        self.pop_fitness = None
        self.rows = []


class PopulationStash(StatisticsMethod):
    """Stash the current full population/fitness so the learner can score it.

    ``collect`` is called by the EDA *before* selection+learning in the same
    generation, so the metered learner (below) sees the population the model
    will have to reproduce.
    """
    def __init__(self, recorder: "_Recorder"):
        self.rec = recorder

    def collect(self, generation, population, fitness, model=None, **params):
        self.rec.pop = np.asarray(population, dtype=int)
        self.rec.pop_fitness = np.asarray(fitness, dtype=float).reshape(-1)
        return {}


class MeteredBNLearn(LearningMethod):
    """Wrap ``LearnEBNA``: time the learning and record model-quality metrics.

    * ``*_sel`` metrics are computed on the *selected* set the model was learned
      from (the training distribution).
    * ``*_pop`` metrics are computed on the *full* population of the same
      generation (a mild held-out test: the population includes the unselected
      individuals the model never weighted).
    """
    def __init__(self, algorithm, cardinality, true_mat, beta,
                 max_parents, recorder: "_Recorder"):
        self.inner = LearnEBNA(score_metric=algorithm, max_parents=max_parents,
                               alpha=1.0)
        self.card = np.asarray(cardinality, dtype=int)
        self.true_mat = true_mat
        self.beta = beta
        self.rec = recorder

    def learn(self, generation, n_vars, cardinality, population, fitness, **params):
        t0 = time.time()
        model = self.inner.learn(generation, n_vars, cardinality,
                                 population, fitness, **params)
        learn_time = time.time() - t0

        sel = np.asarray(population, dtype=int)
        sel_f = np.asarray(fitness, dtype=float).reshape(-1)
        sp_sel, ll_sel, kl_sel = distribution_metrics(
            model, sel, sel_f, self.card, self.beta)

        if self.rec.pop is not None:
            sp_pop, ll_pop, kl_pop = distribution_metrics(
                model, self.rec.pop, self.rec.pop_fitness, self.card, self.beta)
            pop_div = population_diversity(self.rec.pop)
        else:
            sp_pop = ll_pop = kl_pop = float("nan")
            pop_div = float("nan")

        adj = np.asarray(model.structure)
        sk = ((adj != 0) | (adj.T != 0)).astype(int)
        np.fill_diagonal(sk, 0)

        self.rec.rows.append(dict(
            gen=int(generation),
            learn_time=learn_time,
            edges=int(sk.sum() // 2),
            f1=skeleton_f1(adj, self.true_mat),
            n_selected=int(sel.shape[0]),
            pop_diversity=pop_div,
            sp_sel=sp_sel, ll_sel=ll_sel, kl_sel=kl_sel,
            sp_pop=sp_pop, ll_pop=ll_pop, kl_pop=kl_pop,
        ))
        return model


# ---------------------------------------------------------------------------
# Run one BN-EDA
# ---------------------------------------------------------------------------
# BN learners that only support binary variables (verified against bayes_nets).
BINARY_ONLY_ALGORITHMS = {"binotears"}


def run(problem_name, algorithm, seed, pop_mult, n_gen, max_parents, beta):
    family, n_str = problem_name.rsplit("_", 1)
    n = int(n_str)
    problem = parse_problem(family, n)
    pop_size = pop_mult * problem.n_vars

    # Fail fast (before any learning) on a known-incompatible combination, e.g.
    # a binary-only learner on a non-binary problem (Braid has cardinality 4).
    if algorithm in BINARY_ONLY_ALGORITHMS and int(np.max(problem.cardinality)) > 2:
        raise SystemExit(
            f"Incompatible: '{algorithm}' only supports binary variables, but "
            f"'{problem_name}' has cardinality {int(np.max(problem.cardinality))}. "
            f"Skip this (algorithm, problem) pair (the launcher already excludes it)."
        )

    # seed everything reachable (EDA rng + numpy global used by some learners)
    np.random.seed(seed)
    rec = _Recorder()

    components = EDAComponents(
        seeding=RandomInit(),
        selection=TruncationSelection(ratio=0.5),
        learning=MeteredBNLearn(algorithm, problem.cardinality,
                                problem.interaction_matrix, beta,
                                max_parents, rec),
        sampling=SampleBayesianNetwork(n_samples=pop_size),
        replacement=ElitistReplacement(n_elite=1),
        stop_condition=MaxGenerations(n_gen),
        statistics=PopulationStash(rec),
    )
    eda = EDA(
        pop_size=pop_size,
        n_vars=problem.n_vars,
        fitness_func=problem.fitness_func,
        cardinality=problem.cardinality,
        components=components,
        random_seed=seed,
        selection_weighting="boltzmann",
        weighting_beta=beta,
    )

    t0 = time.time()
    stats, _ = eda.run(verbose=False)
    wall_time = time.time() - t0

    # merge the fitness trajectory into the per-generation records
    traj = []
    for i, row in enumerate(rec.rows):
        row = dict(row)
        if i < len(stats.best_fitness):
            row["best_fitness"] = float(stats.best_fitness[i])
            row["mean_fitness"] = float(stats.mean_fitness[i])
            row["std_fitness"] = float(stats.std_fitness[i])
        traj.append(row)

    return problem, pop_size, stats, wall_time, traj


def build_output(problem_name, problem, algorithm, seed, pop_size, n_gen,
                 max_parents, beta, stats, wall_time, traj):
    def _series(key):
        return [r.get(key) for r in traj]

    best_curve = [r.get("best_fitness") for r in traj if r.get("best_fitness") is not None]
    # trapezoidal area under the best-so-far curve, normalised by #generations
    # (mean best-so-far) -- an "anytime performance" score independent of scale.
    if best_curve:
        a = np.asarray(best_curve, dtype=float)
        area = float(np.sum((a[:-1] + a[1:]) / 2.0)) if len(a) > 1 else float(a[0])
        auc = area / (len(a) - 1) if len(a) > 1 else float(a[0])
    else:
        auc = float("nan")

    def _nanmean(vals):
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        return float(np.mean(vals)) if vals else float("nan")

    summary = dict(
        best_fitness=float(stats.best_fitness_overall),
        generation_found=int(stats.generation_found) if stats.generation_found is not None else -1,
        final_best=float(best_curve[-1]) if best_curve else float("nan"),
        auc_best=auc,
        total_wall_time=wall_time,
        total_learn_time=float(np.sum(_series("learn_time"))),
        mean_learn_time=_nanmean(_series("learn_time")),
        mean_edges=_nanmean(_series("edges")),
        final_edges=_series("edges")[-1] if traj else None,
        mean_f1=_nanmean(_series("f1")),
        # surrogate metrics: means over the run and their final values
        mean_sp_sel=_nanmean(_series("sp_sel")), final_sp_sel=_series("sp_sel")[-1] if traj else None,
        mean_sp_pop=_nanmean(_series("sp_pop")), final_sp_pop=_series("sp_pop")[-1] if traj else None,
        mean_ll_pop=_nanmean(_series("ll_pop")),
        mean_kl_pop=_nanmean(_series("kl_pop")),
        final_diversity=_series("pop_diversity")[-1] if traj else None,
    )

    return dict(
        problem=problem_name,
        family=problem.name,
        n_vars=int(problem.n_vars),
        max_cardinality=int(np.max(problem.cardinality)),
        algorithm=algorithm,
        seed=int(seed),
        pop_size=int(pop_size),
        n_gen=int(n_gen),
        selection="truncation_0.5",
        weighting="boltzmann",
        weighting_beta=beta,
        max_parents=max_parents,
        summary=summary,
        trajectory=traj,
    )


def main():
    if len(sys.argv) < 4:
        raise SystemExit(
            "Usage: run_bn_eda.py SEED PROBLEM ALGORITHM "
            "[OUT_DIR] [POP_MULT] [N_GEN] [MAX_PARENTS] [BETA]"
        )
    seed = int(sys.argv[1])
    problem_name = sys.argv[2]
    algorithm = sys.argv[3]
    out_dir = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_OUT
    pop_mult = int(sys.argv[5]) if len(sys.argv) > 5 else 10
    n_gen = int(sys.argv[6]) if len(sys.argv) > 6 else 100
    max_parents = int(sys.argv[7]) if len(sys.argv) > 7 else 5
    beta = float(sys.argv[8]) if len(sys.argv) > 8 else 1.0

    os.makedirs(out_dir, exist_ok=True)
    stem = f"bneda_{problem_name}_{algorithm}_s{seed}"
    json_path = os.path.join(out_dir, stem + ".json")
    dat_path = os.path.join(out_dir, stem + ".dat")

    print(f"Seed:             {seed}")
    print(f"Problem:          {problem_name}")
    print(f"Algorithm:        {algorithm}")
    print(f"Population mult:   {pop_mult}  (pop_size = {pop_mult} * n)")
    print(f"Generations:      {n_gen}")
    print(f"Max parents:      {max_parents}")
    print(f"Weighting beta:   {beta}")

    t0 = time.time()
    problem, pop_size, stats, wall_time, traj = run(
        problem_name, algorithm, seed, pop_mult, n_gen, max_parents, beta)
    out = build_output(problem_name, problem, algorithm, seed, pop_size, n_gen,
                       max_parents, beta, stats, wall_time, traj)

    with open(json_path, "w") as fh:
        json.dump(out, fh, indent=1)

    s = out["summary"]
    with open(dat_path, "w") as fh:
        fh.write(f"# problem={problem_name} algorithm={algorithm} seed={seed} "
                 f"pop_size={pop_size} n_gen={n_gen}\n")
        fh.write("# best_fitness generation_found total_wall_time total_learn_time "
                 "mean_sp_pop mean_ll_pop mean_kl_pop mean_f1 final_edges\n")
        fh.write(f"{s['best_fitness']} {s['generation_found']} {s['total_wall_time']} "
                 f"{s['total_learn_time']} {s['mean_sp_pop']} {s['mean_ll_pop']} "
                 f"{s['mean_kl_pop']} {s['mean_f1']} {s['final_edges']}\n")

    print(f"\nBest fitness:     {s['best_fitness']}  (gen {s['generation_found']})")
    print(f"Total wall time:  {wall_time:.1f} s  (learn {s['total_learn_time']:.1f} s)")
    print(f"mean sp_pop:      {s['mean_sp_pop']:.4f}   mean F1: {s['mean_f1']:.4f}")
    print(f"Wrote:            {json_path}")
    print(f"Elapsed total:    {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
