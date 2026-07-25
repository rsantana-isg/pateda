"""
Compare the five HBOA-Light variants (A1-A5) against the EBNA family, LFDA, BOA
and SARTRE-EDA on three structured discrete problems.

Algorithms (11)
---------------
  BN score-and-search / constraint-based:
    EBNA_BIC   penalized ML, warm-started add/delete/reverse search (departs LFDA)
    EBNA_K2    penalized-K2 (EBNA_K2+pen) with the Etxeberria parent bound
    EBNA_PC    constraint-based (PC-Stable)
    LFDA       add-only greedy BIC from scratch each generation (tunable penalty)
    BOA        K2/BDe greedy Bayesian network
    SARTRE     order-based sparse-additive BN (SARTRE-EDA)
  HBOA-Light (compact decision-tree/graph local structure + RTS niching):
    A1_dt      decision-tree CPDs (no leaf merging)
    A2_mi      decision graphs + top-k mutual-information candidate pruning
    A3_fast    decision graphs + cached-statistics (IIG) split scoring
    A4_mdl     bounded decision graphs grown with the cheaper MDL split score
    A5_ndg     non-search decision-graph construction (Tree-in-Tree / NDG)

Problems (all maximization, binary)
-----------------------------------
  Deceptive3   30 variables (10 non-overlapping deceptive 3-bit blocks; opt = 10)
  Trap5        30 variables (6 non-overlapping trap-5 blocks;         opt = 30)
  UBQP         100 variables (packaged Beasley OR-Library ``bqp100``; opt = 3955)

Deceptive3 and Trap5 are additively decomposable with a known block structure --
the regime where modelling higher-order dependencies (BN / decision-graph EDAs)
pays off over univariate search; UBQP has pairwise structure.

Cost note
---------
The search-based decision-graph variants (A2_mi, A3_fast, A4_mdl) and the exact
BN learners (EBNA_*, BOA) are the expensive ones at n = 100 (UBQP); A1_dt and
especially A5_ndg stay cheap there (see ``docs/A1_benchmark_note.md`` /
``docs/Fast_DG_Learning.md``).  Reduce ``N_RUNS`` / ``n_gen`` or restrict the
algorithm set (4th argument) for a quick pass; A2_mi prunes candidates to the
top-k neighbours, so it scales far better than A3_fast/A4_mdl at n = 100.

Outputs (in ``results/hboa_light/``)
------------------------------------
  final_fitness.csv          per-run best fitness (raw).
  summary.csv                mean/std/success-rate/%opt/time per (problem, alg).
  table_final.tex            LaTeX table (mean +- std, best per problem bold,
                             \\tabcolsep{10pt}).
  kruskal_wallis.csv         Kruskal-Wallis across algorithms per problem
                             (+ Dunn post hoc CSVs if scikit_posthocs is installed).

Usage (all arguments optional, positional):
    python scripts/compare_hboa_light.py [n_runs] [algs] [problems]

    n_runs    number of independent runs per (problem, algorithm)  (default 10)
    algs      comma-separated algorithm names or "all"             (default all)
    problems  comma-separated problem names or "all"               (default all)

Examples:
    python scripts/compare_hboa_light.py
    python scripts/compare_hboa_light.py 5 A1_dt,A5_ndg,EBNA_BIC Deceptive3,Trap5
"""

import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from pateda import (
    EBNA_BIC, EBNA_K2, EBNA_PC, LFDA, BOA, SARTRE_EDA,
    HBOA_Light_A1, HBOA_Light_A2, HBOA_Light_A3, HBOA_Light_A4, HBOA_Light_A5,
)
from pateda.functions.discrete_binary.toy_functions.deceptive import deceptive3
from pateda.functions.discrete_binary.toy_functions.trap import trap_n
from pateda.functions.discrete_binary.problems.ubqp import load_ubqp_benchmark_instance


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, "results", "hboa_light"))

N_RUNS = 10
SEL_RATIO = 0.5
BASE_SEED = 1


# ---------------------------------------------------------------------------
# Problems: name -> (n_vars, fitness_func, optimum, pop_size, n_gen)
# ---------------------------------------------------------------------------
def _make_problems():
    ubqp_inst, ubqp_opt = load_ubqp_benchmark_instance("bqp100")
    ubqp_opt = float(ubqp_opt) if str(ubqp_opt).replace(".", "").isdigit() else None

    def f_deceptive3(x):
        return float(deceptive3(np.asarray(x, dtype=int)))

    def f_trap5(x):
        return float(trap_n(np.asarray(x, dtype=int), 5))

    def f_ubqp(x):
        return float(np.ravel(ubqp_inst.evaluate(np.asarray(x, dtype=int)))[0])

    return {
        # name:        (n_vars, fitness,      optimum, pop_size, n_gen)
        "Deceptive3":  (30,     f_deceptive3, 10.0,    200,      50),
        "Trap5":       (30,     f_trap5,      30.0,    200,     50),
        "UBQP":        (100,    f_ubqp,       ubqp_opt, 200,    50),
    }


# ---------------------------------------------------------------------------
# Algorithms: name -> builder(n_vars, fitness, pop_size, n_gen, sel_ratio, seed)
# ---------------------------------------------------------------------------
def _build(cls, **extra):
    def build(n_vars, fitness, pop_size, n_gen, sel_ratio, seed):
        return cls(n_vars=n_vars, cardinality=2, fitness_func=fitness,
                   pop_size=pop_size, n_gen=n_gen, selection_ratio=sel_ratio,
                   random_seed=seed, **extra)
    return build


ALGORITHMS = {
    "EBNA_BIC": _build(EBNA_BIC),
    "EBNA_K2":  _build(EBNA_K2),
    "EBNA_PC":  _build(EBNA_PC),
    "LFDA":     _build(LFDA),
    "BOA":      _build(BOA),
    "SARTRE":   _build(SARTRE_EDA),
    "A1_dt":    _build(HBOA_Light_A1),
    "A2_mi":    _build(HBOA_Light_A2),
    "A3_fast":  _build(HBOA_Light_A3),
    "A4_mdl":   _build(HBOA_Light_A4),
    "A5_ndg":   _build(HBOA_Light_A5),
}
ALG_ORDER = list(ALGORITHMS.keys())


# ---------------------------------------------------------------------------
def run_single(build, n_vars, fitness, pop_size, n_gen, seed):
    """One run; return (best_fitness, elapsed_seconds)."""
    alg = build(n_vars, fitness, pop_size, n_gen, SEL_RATIO, seed)
    t0 = time.time()
    stats, _ = alg.run(verbose=False)
    return float(stats.best_fitness_overall), time.time() - t0


def parse_args(argv, problems):
    n_runs = N_RUNS
    algs = list(ALG_ORDER)
    probs = list(problems.keys())
    if len(argv) > 1:
        n_runs = int(argv[1])
    if len(argv) > 2 and argv[2].lower() != "all":
        algs = argv[2].split(",")
        unknown = [a for a in algs if a not in ALGORITHMS]
        if unknown:
            raise ValueError(f"Unknown algorithms: {unknown}. Known: {ALG_ORDER}")
    if len(argv) > 3 and argv[3].lower() != "all":
        probs = argv[3].split(",")
        unknown = [p for p in probs if p not in problems]
        if unknown:
            raise ValueError(f"Unknown problems: {unknown}. Known: {list(problems)}")
    return n_runs, algs, probs


def write_latex_table(summary, algs, probs, path):
    """mean +- std final fitness, algorithms x problems, best per problem bold."""
    lines = [
        "% Final best fitness (mean +- std over runs).  Best mean per problem in bold.",
        "\\setlength{\\tabcolsep}{10pt}",
        "\\begin{tabular}{l" + "r" * len(probs) + "}",
        "\\hline",
        "Algorithm & " + " & ".join(probs) + " \\\\",
        "\\hline",
    ]
    best = {p: max(summary[(a, p)]["mean"] for a in algs) for p in probs}
    for a in algs:
        cells = []
        for p in probs:
            m, s = summary[(a, p)]["mean"], summary[(a, p)]["std"]
            body = f"{m:.3f} \\pm {s:.3f}"
            if np.isclose(m, best[p]):
                body = f"\\mathbf{{{body}}}"
            cells.append(f"${body}$")
        lines.append(f"{a} & " + " & ".join(cells) + " \\\\")
    lines += ["\\hline", "\\end{tabular}"]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def write_kruskal(raw, algs, probs, out_dir):
    try:
        import scikit_posthocs as sp
    except ImportError:
        sp = None
        print("  note: scikit_posthocs not installed; skipping Dunn tests")
    rows = []
    for p in probs:
        groups = [raw[(a, p)] for a in algs if len(raw[(a, p)]) > 0]
        if len(groups) < 2:
            continue
        try:
            stat, pval = scipy_stats.kruskal(*groups)
        except ValueError:
            stat, pval = np.nan, 1.0
        if np.isnan(pval):
            pval = 1.0
        rows.append({"problem": p, "H_statistic": stat, "p_value": pval})
        if sp is not None and pval < 0.05:
            df = pd.DataFrame(
                [(a, v) for a in algs for v in raw[(a, p)]],
                columns=["algorithm", "best"],
            )
            dunn = sp.posthoc_dunn(df, val_col="best", group_col="algorithm",
                                   p_adjust="holm")
            dunn.to_csv(os.path.join(out_dir, f"dunn_{p}.csv"))
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "kruskal_wallis.csv"), index=False)


def main():
    problems = _make_problems()
    n_runs, algs, probs = parse_args(sys.argv, problems)
    seeds = list(range(BASE_SEED, BASE_SEED + n_runs))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("HBOA-Light comparison vs EBNA family / LFDA / BOA / SARTRE-EDA")
    print(f"Output dir:       {OUTPUT_DIR}")
    print(f"Algorithms:       {algs}")
    print(f"Problems:         {probs}")
    print(f"Runs (seeds):     {seeds}")
    print(f"Selection ratio:  {SEL_RATIO}")
    for p in probs:
        n_vars, _, opt, pop, gen = problems[p]
        print(f"  {p:<11} n={n_vars:<4} pop={pop:<5} gen={gen:<4} "
              f"optimum={opt}  budget={pop * (gen + 1)} evals")

    raw = {}          # (alg, problem) -> list of best fitness
    summary = {}      # (alg, problem) -> dict of stats
    raw_records = []  # per-run rows for final_fitness.csv

    for p in probs:
        n_vars, fitness, opt, pop, gen = problems[p]
        print(f"\n{'=' * 78}\nProblem: {p}  (n={n_vars}, optimum={opt})\n{'=' * 78}")
        print(f"{'algorithm':<10} {'mean':>10} {'std':>9} {'best':>10} "
              f"{'succ':>6} {'time/run':>9}")
        for a in algs:
            build = ALGORITHMS[a]
            bests, times = [], []
            for seed in seeds:
                try:
                    b, t = run_single(build, n_vars, fitness, pop, gen, seed)
                except Exception as exc:
                    print(f"{a:<10} run seed={seed} ERROR -- {exc}")
                    traceback.print_exc()
                    continue
                bests.append(b)
                times.append(t)
                raw_records.append({"problem": p, "algorithm": a, "seed": seed,
                                    "n_vars": n_vars, "pop_size": pop, "n_gen": gen,
                                    "best": b, "optimum": opt, "time_s": t})
            if not bests:
                continue
            bests = np.asarray(bests, dtype=float)
            raw[(a, p)] = list(bests)
            succ = (float(np.mean(np.isclose(bests, opt))) if opt is not None
                    else float("nan"))
            pct = (float(np.mean(bests) / opt) if opt else float("nan"))
            summary[(a, p)] = {
                "mean": float(np.mean(bests)), "std": float(np.std(bests)),
                "best": float(np.max(bests)), "success_rate": succ,
                "pct_opt": pct, "time_per_run": float(np.mean(times)),
            }
            print(f"{a:<10} {np.mean(bests):>10.3f} {np.std(bests):>9.3f} "
                  f"{np.max(bests):>10.3f} {succ:>6.2f} {np.mean(times):>8.2f}s")

    # ---- outputs ----
    pd.DataFrame(raw_records).to_csv(
        os.path.join(OUTPUT_DIR, "final_fitness.csv"), index=False)
    srows = [{"algorithm": a, "problem": p, **summary[(a, p)]}
             for a in algs for p in probs if (a, p) in summary]
    pd.DataFrame(srows).to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)
    present_probs = [p for p in probs if any((a, p) in summary for a in algs)]
    present_algs = [a for a in algs if any((a, p) in summary for p in present_probs)]
    if present_algs and present_probs:
        write_latex_table(summary, present_algs, present_probs,
                          os.path.join(OUTPUT_DIR, "table_final.tex"))
        write_kruskal(raw, present_algs, present_probs, OUTPUT_DIR)

    print(f"\nDone.  Results in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
