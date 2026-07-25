"""
Controlled comparison of BN-based EDA *learning* methods under a **matched**
selection and replacement scheme -- to isolate the effect of the learning model
from the effect of the diversity mechanism (restricted-tournament replacement).

Motivation
----------
In ``scripts/compare_hboa_light.py`` the traditional BN EDAs (EBNA_*, LFDA, BOA,
SARTRE) use **elitist** replacement, while the HBOA-Light variants (A1-A5) use
**restricted-tournament (RTS) niching** -- the diversity mechanism hBOA adds to
BOA.  On deceptive problems the light variants reach ~100% success while the
traditional ones stall at ~9.2-9.7/10 (0% success).  That comparison therefore
confounds *the learning model* with *the replacement method*.

This script removes the confound: **every method uses the same seeding, the same
truncation selection, and the same replacement**, and (by default) the same
``max_parents`` budget.  Only the BN structure/parameter learner -- and the
sampler matched to its representation (tabular vs compact local-structure CPDs)
-- changes.  By default **all methods use RestrictedTournamentReplacement (RTS)**
-- the diversity mechanism the light variants already have -- so the test is
direct: does giving the *traditional* learners RTS make them succeed too?::

    python scripts/compare_bn_learning_matched.py                 # RTS for all
    python scripts/compare_bn_learning_matched.py 10 all all both # add elitist contrast

If the traditional learners' success rate jumps to the light variants' level
under RTS, the difference was the diversity mechanism (RTS), not the model; if a
gap persists, it is the model.

Matched components (identical for every method)
-----------------------------------------------
  seeding      RandomInit
  selection    TruncationSelection(ratio=0.5)
  replacement  RestrictedTournamentReplacement(window_size)  ("rts", default) -- or
               ElitistReplacement(n_elite=1)                 ("elitist") for contrast
  max_parents  common value for all methods (default 6; set to isolate the
               learner under an equal parent budget)

Only the learner (+ matched sampler) varies:
  tabular BN (SampleBayesianNetwork):   EBNA_BIC, EBNA_K2, EBNA_PC, LFDA, BOA, SARTRE
  local-structure BN (SampleLocalStructureBN): A1_dt, A2_mi, A3_fast, A4_mdl, A5_ndg

Problems, pop_size, n_gen and optima are imported unchanged from
``compare_hboa_light.py`` so the numbers are directly comparable.

Usage (all optional, positional):
    python scripts/compare_bn_learning_matched.py [n_runs] [algs] [problems] [replacement]

    n_runs       runs per (problem, algorithm, replacement)   (default 10)
    algs         comma-separated names or "all"               (default all)
    problems     comma-separated names or "all"               (default all)
    replacement  "rts", "elitist", or "both"                  (default rts: RTS for all)

Outputs (in ``results/bn_matched/``): final_fitness.csv, summary.csv,
table_success.tex (success rate, methods x replacement per problem) and
kruskal_wallis.csv (+ Dunn CSVs if scikit_posthocs is installed).
"""

import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from pateda import EDA, EDAComponents
from pateda.seeding.random_init import RandomInit
from pateda.selection.truncation import TruncationSelection
from pateda.replacement.elitist import ElitistReplacement
from pateda.replacement.niching import RestrictedTournamentReplacement
from pateda.stop_conditions.max_generations import MaxGenerations

from pateda.learning.ebna import LearnEBNA
from pateda.learning.lfda import LearnLFDA
from pateda.learning.boa import LearnBOA
from pateda.learning.bn_extra import LearnSARTRE
from pateda.learning.hboa import LearnHBOALight
from pateda.sampling.bayesian_network import SampleBayesianNetwork, SampleLocalStructureBN

# Reuse the exact problems / pop_size / n_gen / optima of the light comparison.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from compare_hboa_light import _make_problems  # noqa: E402

OUTPUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, "results", "bn_matched"))

N_RUNS = 10
SEL_RATIO = 0.5
BASE_SEED = 1
MAX_PARENTS = 6      # common parent budget for every method (control variable)
ALPHA = 1.0
# RestrictedTournamentReplacement (RTS) for ALL methods is the default: it is the
# diversity mechanism the light variants already use, so applying it uniformly
# isolates the BN *learning* method.  "elitist"/"both" remain available for contrast.
REPLACEMENTS = ["rts"]


def _window_size(pop_size: int) -> int:
    """hBOA-style restricted-tournament window (~ pop/20, at least 5)."""
    return max(5, pop_size // 20)


# ---------------------------------------------------------------------------
# Learner factories: name -> (make_learner(max_parents, alpha), uses_local_structure)
# Factories (not instances) so every run gets a FRESH learner -- important
# because EBNA_BIC's warm start caches the previous generation's DAG.
# ---------------------------------------------------------------------------
ALGORITHMS = {
    # ---- tabular BN (score-and-search / constraint-based) ----
    "EBNA_BIC": (lambda mp, a: LearnEBNA(max_parents=mp, score_metric="bic",
                                         alpha=a, warm_start=True), False),
    "EBNA_K2":  (lambda mp, a: LearnEBNA(max_parents=mp, score_metric="k2_pen",
                                         alpha=a, penalty="bic"), False),
    "EBNA_PC":  (lambda mp, a: LearnEBNA(max_parents=mp, score_metric="stable_pc",
                                         alpha=a), False),
    "LFDA":     (lambda mp, a: LearnLFDA(max_parents=mp, bic_weight=1.0, alpha=a),
                 False),
    "BOA":      (lambda mp, a: LearnBOA(max_parents=mp), False),
    "SARTRE":   (lambda mp, a: LearnSARTRE(max_parents=mp, alpha=a), False),
    # ---- compact local-structure BN (HBOA-Light) ----
    "A1_dt":    (lambda mp, a: LearnHBOALight(method="dt", local_structure="dt",
                                              max_parents=mp, alpha=a), True),
    "A2_mi":    (lambda mp, a: LearnHBOALight(method="dg", local_structure="dg",
                                              max_parents=mp, alpha=a,
                                              candidate_parents="mi:10"), True),
    "A3_fast":  (lambda mp, a: LearnHBOALight(method="dg", local_structure="dg",
                                              max_parents=mp, alpha=a,
                                              fast_local_scoring=True), True),
    "A4_mdl":   (lambda mp, a: LearnHBOALight(method="dg", local_structure="dg",
                                              max_parents=mp, alpha=a,
                                              split_score="mdl", max_leaves=32), True),
    "A5_ndg":   (lambda mp, a: LearnHBOALight(method="dg_ndg", local_structure="dg",
                                              max_parents=mp, alpha=a), True),
}
ALG_ORDER = list(ALGORITHMS.keys())


def make_replacement(kind: str, pop_size: int):
    if kind == "elitist":
        return ElitistReplacement(n_elite=1)
    if kind == "rts":
        return RestrictedTournamentReplacement(window_size=_window_size(pop_size))
    raise ValueError(f"replacement must be 'elitist' or 'rts', got {kind!r}")


def run_single(alg_name, replacement, n_vars, fitness, pop_size, n_gen, seed):
    """One run with matched components; return (best_fitness, elapsed_s)."""
    make_learner, uses_local = ALGORITHMS[alg_name]
    learner = make_learner(MAX_PARENTS, ALPHA)
    sampler = (SampleLocalStructureBN(n_samples=pop_size) if uses_local
               else SampleBayesianNetwork(n_samples=pop_size))
    components = EDAComponents(
        seeding=RandomInit(),
        selection=TruncationSelection(ratio=SEL_RATIO),
        learning=learner,
        sampling=sampler,
        stop_condition=MaxGenerations(n_gen),
        replacement=make_replacement(replacement, pop_size),
    )
    card = np.full(n_vars, 2)
    eda = EDA(pop_size, n_vars, fitness, card, components, random_seed=seed)
    t0 = time.time()
    stats, _ = eda.run(verbose=False)
    return float(stats.best_fitness_overall), time.time() - t0


def parse_args(argv, problems):
    n_runs, algs = N_RUNS, list(ALG_ORDER)
    probs = list(problems.keys())
    reps = list(REPLACEMENTS)
    if len(argv) > 1:
        n_runs = int(argv[1])
    if len(argv) > 2 and argv[2].lower() != "all":
        algs = argv[2].split(",")
        bad = [a for a in algs if a not in ALGORITHMS]
        if bad:
            raise ValueError(f"Unknown algorithms: {bad}. Known: {ALG_ORDER}")
    if len(argv) > 3 and argv[3].lower() != "all":
        probs = argv[3].split(",")
        bad = [p for p in probs if p not in problems]
        if bad:
            raise ValueError(f"Unknown problems: {bad}. Known: {list(problems)}")
    if len(argv) > 4:
        val = argv[4].lower()
        if val == "both":
            reps = ["elitist", "rts"]
        elif val in ("elitist", "rts"):
            reps = [val]
        else:
            raise ValueError("replacement must be 'rts', 'elitist' or 'both'")
    return n_runs, algs, probs, reps


def write_success_table(summary, algs, probs, reps, path):
    """Success rate, algorithms x (problem, replacement).  Best per column bold."""
    cols = [(p, r) for p in probs for r in reps]
    header = "Algorithm & " + " & ".join(f"{p}/{r}" for p, r in cols) + " \\\\"
    best = {(p, r): max(summary[(a, p, r)]["success_rate"]
                        for a in algs if (a, p, r) in summary)
            for p, r in cols}
    lines = [
        "% Success rate (fraction of runs reaching the optimum) under matched",
        "% selection+replacement; only the BN learning method changes.",
        "\\setlength{\\tabcolsep}{10pt}",
        "\\begin{tabular}{l" + "r" * len(cols) + "}",
        "\\hline", header, "\\hline",
    ]
    for a in algs:
        cells = []
        for p, r in cols:
            if (a, p, r) not in summary:
                cells.append("--")
                continue
            v = summary[(a, p, r)]["success_rate"]
            body = f"{v:.2f}"
            cells.append(f"$\\mathbf{{{body}}}$" if np.isclose(v, best[(p, r)])
                         else f"${body}$")
        lines.append(f"{a} & " + " & ".join(cells) + " \\\\")
    lines += ["\\hline", "\\end{tabular}"]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def write_kruskal(raw, algs, probs, reps, out_dir):
    """KW across learners per (problem, replacement)."""
    try:
        import scikit_posthocs as sp
    except ImportError:
        sp = None
        print("  note: scikit_posthocs not installed; skipping Dunn tests")
    rows = []
    for p in probs:
        for r in reps:
            groups = [raw[(a, p, r)] for a in algs if raw.get((a, p, r))]
            if len(groups) < 2:
                continue
            try:
                stat, pval = scipy_stats.kruskal(*groups)
            except ValueError:
                stat, pval = np.nan, 1.0
            if np.isnan(pval):
                pval = 1.0
            rows.append({"problem": p, "replacement": r,
                         "H_statistic": stat, "p_value": pval})
            if sp is not None and pval < 0.05:
                df = pd.DataFrame([(a, v) for a in algs for v in raw.get((a, p, r), [])],
                                  columns=["algorithm", "best"])
                sp.posthoc_dunn(df, val_col="best", group_col="algorithm",
                                p_adjust="holm").to_csv(
                    os.path.join(out_dir, f"dunn_{p}_{r}.csv"))
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "kruskal_wallis.csv"), index=False)


def main():
    problems = _make_problems()
    n_runs, algs, probs, reps = parse_args(sys.argv, problems)
    seeds = list(range(BASE_SEED, BASE_SEED + n_runs))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Controlled BN-learning comparison (matched selection + replacement)")
    print(f"Output dir:       {OUTPUT_DIR}")
    print(f"Algorithms:       {algs}")
    print(f"Problems:         {probs}")
    print(f"Replacements:     {reps}   (only difference across passes)")
    print(f"Selection:        TruncationSelection(ratio={SEL_RATIO})  [same for all]")
    print(f"max_parents:      {MAX_PARENTS}  [same for all]")
    print(f"Runs (seeds):     {seeds}")

    raw, summary, records = {}, {}, []

    for p in probs:
        n_vars, fitness, opt, pop, gen = problems[p]
        print(f"\n{'=' * 84}\nProblem: {p}  (n={n_vars}, optimum={opt}, "
              f"pop={pop}, gen={gen})\n{'=' * 84}")
        print(f"{'algorithm':<10} {'replace':<8} {'mean':>9} {'std':>8} "
              f"{'best':>9} {'succ':>6} {'time/run':>9}")
        for a in algs:
            for r in reps:
                bests, times = [], []
                for seed in seeds:
                    try:
                        b, t = run_single(a, r, n_vars, fitness, pop, gen, seed)
                    except Exception as exc:
                        print(f"{a:<10} {r:<8} seed={seed} ERROR -- {exc}")
                        traceback.print_exc()
                        continue
                    bests.append(b)
                    times.append(t)
                    records.append({"problem": p, "algorithm": a, "replacement": r,
                                    "seed": seed, "best": b, "optimum": opt,
                                    "time_s": t})
                if not bests:
                    continue
                bests = np.asarray(bests, float)
                raw[(a, p, r)] = list(bests)
                succ = (float(np.mean(np.isclose(bests, opt)))
                        if opt is not None else float("nan"))
                summary[(a, p, r)] = {"mean": float(np.mean(bests)),
                                      "std": float(np.std(bests)),
                                      "best": float(np.max(bests)),
                                      "success_rate": succ,
                                      "time_per_run": float(np.mean(times))}
                print(f"{a:<10} {r:<8} {np.mean(bests):>9.3f} {np.std(bests):>8.3f} "
                      f"{np.max(bests):>9.3f} {succ:>6.2f} {np.mean(times):>8.2f}s")

    pd.DataFrame(records).to_csv(os.path.join(OUTPUT_DIR, "final_fitness.csv"),
                                 index=False)
    pd.DataFrame([{"algorithm": a, "problem": p, "replacement": r, **summary[(a, p, r)]}
                  for (a, p, r) in summary]).to_csv(
        os.path.join(OUTPUT_DIR, "summary.csv"), index=False)
    p_ok = [p for p in probs if any((a, p, r) in summary for a in algs for r in reps)]
    a_ok = [a for a in algs if any((a, p, r) in summary for p in p_ok for r in reps)]
    if a_ok and p_ok:
        write_success_table(summary, a_ok, p_ok, reps,
                            os.path.join(OUTPUT_DIR, "table_success.tex"))
        write_kruskal(raw, a_ok, p_ok, reps, OUTPUT_DIR)

    print(f"\nDone.  Results in {OUTPUT_DIR}")
    if len(reps) == 2:
        print("Compare the 'elitist' vs 'rts' rows per method: if the traditional")
        print("learners' success rate jumps under 'rts', the gap was the diversity")
        print("mechanism (restricted tournament), not the BN model.")


if __name__ == "__main__":
    main()
