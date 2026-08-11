"""
A/B test of scale-aware target-degree ESS modulation for the new MN-FDA.

Compares four graph-calibration modes on n in {64, 100, 625} (a function
subset), under BZ (Boltzmann weighting), for MN-FDA and MN-FDA-P:

    noESS  : effective size = raw N                     (best at small n)
    ESS    : effective size = Kish ESS of the weights   (best at large n)
    TD6    : target-degree modulation, target mean degree = 6
    TD10   : target-degree modulation, target mean degree = 10

The question: does a target-degree mode match/beat noESS at small n AND ESS at
large n (i.e. recover the large-n win without the small-n regression)?

Because fitness scales differ per function, modes are compared by **average rank**
(per (function, dim); mean over seeds; 1 = best).  Runs 14 in parallel.

Usage: python3 scripts/test_mnfda_td.py [n_seeds] [dims] [fids]
"""

import os
import sys
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FTimeout

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
OUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, "results", "mnfda_td"))
POP, NGEN, MAXCLIQUE = 200, 50, 5
N_WORKERS, RUN_TIMEOUT = 14, 300

MODES = {  # name -> (use_ess, target_degree)
    "noESS": (False, None),
    "ESS":   (True,  None),
    "TD6":   (False, 6.0),
    "TD10":  (False, 10.0),
}


def _run_one(args):
    alg, mode, dim, fid, seed = args
    import ioh
    from pateda import EDA, EDAComponents
    from pateda.seeding.random_init import RandomInit
    from pateda.selection.truncation import TruncationSelection
    from pateda.replacement.elitist import ElitistReplacement
    from pateda.stop_conditions.max_generations import MaxGenerations
    from pateda.learning.mnfda import LearnMNFDA
    from pateda.sampling.fda import SampleFDA
    from pateda.sampling.fda_mpc import SampleFDAWithMPC

    use_ess, td = MODES[mode]
    prob = ioh.get_problem(fid, instance=1, dimension=dim,
                           problem_class=ioh.ProblemClass.PBO)

    def fit(x):
        return float(prob(np.asarray(x, dtype=int)))

    learner = LearnMNFDA(max_clique_size=MAXCLIQUE, use_ess=use_ess,
                         target_degree=td, random_state=seed)
    sampler = SampleFDAWithMPC(n_samples=POP) if alg == "MNFDAP" else SampleFDA(n_samples=POP)
    comp = EDAComponents(
        seeding=RandomInit(), selection=TruncationSelection(ratio=0.5),
        learning=learner, sampling=sampler,
        stop_condition=MaxGenerations(NGEN),
        replacement=ElitistReplacement(n_elite=1))
    eda = EDA(POP, dim, fit, np.full(dim, 2), comp, random_seed=seed,
              selection_weighting="boltzmann", weighting_beta=1.0)
    t0 = time.time()
    stats, _ = eda.run(verbose=False)
    return {"alg": alg, "mode": mode, "dim": dim, "fid": fid, "seed": seed,
            "best": float(stats.best_fitness_overall), "time_s": time.time() - t0}


def main():
    import pandas as pd
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    dims = ([int(d) for d in sys.argv[2].split(",")] if len(sys.argv) > 2
            else [64, 100, 625])
    fids = ([int(f) for f in sys.argv[3].split(",")] if len(sys.argv) > 3
            else [1, 4, 16, 18, 23, 25])
    seeds = list(range(1, n_seeds + 1))
    os.makedirs(OUT_DIR, exist_ok=True)
    tasks = [(alg, mode, dim, fid, seed)
             for alg in ("MNFDA", "MNFDAP") for mode in MODES
             for dim in dims for fid in fids for seed in seeds]
    print(f"target-degree A/B: {len(tasks)} runs "
          f"(2 algs x {len(MODES)} modes x {len(dims)} dims x {len(fids)} funcs "
          f"x {n_seeds} seeds), pop={POP} gen={NGEN}, BZ, {N_WORKERS} workers")

    records = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS, max_tasks_per_child=1) as ex:
        futs = {ex.submit(_run_one, t): t for t in tasks}
        for i, fut in enumerate(list(futs), 1):
            t = futs[fut]
            try:
                records.append(fut.result(timeout=RUN_TIMEOUT))
            except (FTimeout, Exception) as exc:
                records.append({"alg": t[0], "mode": t[1], "dim": t[2], "fid": t[3],
                                "seed": t[4], "best": np.nan, "time_s": np.nan})
            if i % 50 == 0:
                print(f"  {i}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(OUT_DIR, "mnfda_td.csv"), index=False)

    print(f"\n=== average rank per mode (1=best), per algorithm x dimension "
          f"===  total {time.time()-t0:.0f}s")
    modes = list(MODES)
    for alg in ("MNFDA", "MNFDAP"):
        for dim in dims:
            sub = df[(df.alg == alg) & (df.dim == dim)]
            # mean over seeds per (fid, mode)
            piv = sub.groupby(["fid", "mode"])["best"].mean().unstack()[modes]
            # rank modes within each function (higher best = rank 1)
            ranks = piv.rank(axis=1, ascending=False, method="min")
            avg = ranks.mean()
            order = " | ".join(f"{m}={avg[m]:.2f}" for m in modes)
            best_mode = avg.idxmin()
            print(f"  {alg:6s} n={dim:<4}  avg-rank: {order}   -> best: {best_mode}")
    print(f"\nCSV: {os.path.join(OUT_DIR,'mnfda_td.csv')}")


if __name__ == "__main__":
    main()
