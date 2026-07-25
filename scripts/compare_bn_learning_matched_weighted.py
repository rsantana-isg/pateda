"""
Controlled comparison of BN-based EDA learning methods under a **matched**
selection + replacement scheme *and* **Boltzmann weighting of the selected
individuals during model learning** (customized selection).

This is the weighted counterpart of ``scripts/compare_bn_learning_matched.py``:
everything is identical -- same three problems (Deceptive3, Trap5, UBQP), same
seeding, same ``TruncationSelection(0.5)``, the same replacement for all methods
(RestrictedTournamentReplacement by default), the same ``max_parents=6`` budget,
and the same 11 learners with their matched samplers -- **except** that each
model is now learned with the selected individuals re-weighted by a Boltzmann
distribution of their (standardised) fitness (``selection_weighting="boltzmann"``,
``weighting_beta=1`` i.e. T=1).  Only the BN learning method varies.

The weighting is applied by the EDA core (it turns the selected fitness into a
per-individual probability ``p`` that every learner uses to weight its counts /
CPD tables), so no per-learner change is needed.

Usage (all optional, positional):
    python scripts/compare_bn_learning_matched_weighted.py [n_runs] [algs] [problems] [replacement]

    (arguments identical to compare_bn_learning_matched.py; replacement defaults
     to "rts" = RestrictedTournamentReplacement for all.)

Outputs (in ``results/bn_matched_weighted/``): final_fitness.csv, summary.csv,
table_success.tex and kruskal_wallis.csv, as in the unweighted version.
"""

import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

from pateda import EDA, EDAComponents
from pateda.seeding.random_init import RandomInit
from pateda.selection.truncation import TruncationSelection
from pateda.stop_conditions.max_generations import MaxGenerations
from pateda.sampling.bayesian_network import SampleBayesianNetwork, SampleLocalStructureBN

# Reuse the whole controlled-comparison machinery (registry, matched components,
# tables, problems) from the unweighted script; only the learning weighting adds.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import compare_bn_learning_matched as M  # noqa: E402

OUTPUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir,
                                          "results", "bn_matched_weighted"))

# Customized selection: Boltzmann weighting of the selected set, inverse
# temperature beta = 1 (T = 1) -- matches the "BZ"/"RTS" schemes of the PBO study.
WEIGHTING = "boltzmann"
WEIGHTING_BETA = 1.0


def run_single(alg_name, replacement, n_vars, fitness, pop_size, n_gen, seed):
    """One run with matched components + Boltzmann-weighted learning."""
    make_learner, uses_local = M.ALGORITHMS[alg_name]
    learner = make_learner(M.MAX_PARENTS, M.ALPHA)
    sampler = (SampleLocalStructureBN(n_samples=pop_size) if uses_local
               else SampleBayesianNetwork(n_samples=pop_size))
    components = EDAComponents(
        seeding=RandomInit(),
        selection=TruncationSelection(ratio=M.SEL_RATIO),
        learning=learner,
        sampling=sampler,
        stop_condition=MaxGenerations(n_gen),
        replacement=M.make_replacement(replacement, pop_size),
    )
    card = np.full(n_vars, 2)
    eda = EDA(pop_size, n_vars, fitness, card, components, random_seed=seed,
              selection_weighting=WEIGHTING, weighting_beta=WEIGHTING_BETA)
    t0 = time.time()
    stats, _ = eda.run(verbose=False)
    return float(stats.best_fitness_overall), time.time() - t0


def main():
    problems = M._make_problems()
    n_runs, algs, probs, reps = M.parse_args(sys.argv, problems)
    seeds = list(range(M.BASE_SEED, M.BASE_SEED + n_runs))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Controlled BN-learning comparison — MATCHED components + "
          "Boltzmann-weighted learning")
    print(f"Output dir:       {OUTPUT_DIR}")
    print(f"Weighting:        {WEIGHTING} (beta={WEIGHTING_BETA}, T={1/WEIGHTING_BETA:g})")
    print(f"Algorithms:       {algs}")
    print(f"Problems:         {probs}")
    print(f"Replacements:     {reps}")
    print(f"Selection:        TruncationSelection(ratio={M.SEL_RATIO})  [same for all]")
    print(f"max_parents:      {M.MAX_PARENTS}  [same for all]")
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
                                    "time_s": t, "weighting": WEIGHTING})
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
        M.write_success_table(summary, a_ok, p_ok, reps,
                              os.path.join(OUTPUT_DIR, "table_success.tex"))
        M.write_kruskal(raw, a_ok, p_ok, reps, OUTPUT_DIR)

    print(f"\nDone.  Results in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
