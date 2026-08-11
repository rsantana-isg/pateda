"""
Parallel test of the new MN-FDA and MN-FDA-P, with and without the effective
sample size (ESS) in the chi-square adjacency, on the previously-hanging regime
(n=625, Boltzmann selection) and a smaller control (n=100).

For every (algorithm, ess, dimension, function, seed) it runs one EDA under the
BZ scheme (Boltzmann weighting + elitist replacement), recording the final best
fitness, wall time, whether it finished (a per-run timeout guards against any
residual hang), and the process peak RSS (memory).  Runs 14 in parallel.

Prints, per (algorithm, dimension): mean final fitness with vs without ESS, the
number of functions where ESS wins / ties / loses, and feasibility (all finished,
max memory).  Writes results/mnfda_ess/mnfda_ess.csv.

Usage:
    python3 scripts/test_mnfda_ess.py [n_seeds] [dims] [fids]
      n_seeds  default 3
      dims     default 625,100
      fids     default all (1..25)
"""

import os
import sys
import time
import resource
import numpy as np
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FTimeout

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

OUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, "results", "mnfda_ess"))
POP, NGEN, SEL_RATIO, MAXCLIQUE = 200, 50, 0.5, 5
RUN_TIMEOUT = 240        # seconds; a run exceeding this is treated as a hang
N_WORKERS = 14


def _run_one(args):
    """One (alg, ess, dim, fid, seed) run.  Top-level for ProcessPool."""
    alg, ess, dim, fid, seed = args
    import ioh
    from pateda import MNFDA, MNFDAP
    from compare_weighted_edas_pbo import apply_selection_method
    cls = MNFDAP if alg == "MNFDAP" else MNFDA
    prob = ioh.get_problem(fid, instance=1, dimension=dim,
                           problem_class=ioh.ProblemClass.PBO)

    def fit(x):
        return float(prob(np.asarray(x, dtype=int)))

    a = cls(n_vars=dim, cardinality=2, fitness_func=fit, pop_size=POP,
            n_gen=NGEN, max_clique_size=MAXCLIQUE, use_ess=ess, random_seed=seed)
    apply_selection_method(a._eda, "BZ", POP)      # Boltzmann weighting
    t0 = time.time()
    stats, _ = a.run(verbose=False)
    dt = time.time() - t0
    mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return {"alg": alg, "ess": ess, "dim": dim, "fid": fid, "seed": seed,
            "best": float(stats.best_fitness_overall), "time_s": dt,
            "mem_mb": mem_mb, "finished": True}


def main():
    import pandas as pd
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    dims = ([int(d) for d in sys.argv[2].split(",")] if len(sys.argv) > 2
            else [625, 100])
    fids = (list(range(1, 26))
            if len(sys.argv) <= 3 or sys.argv[3].lower() == "all"
            else [int(f) for f in sys.argv[3].split(",")])
    seeds = list(range(1, n_seeds + 1))
    os.makedirs(OUT_DIR, exist_ok=True)

    tasks = [(alg, ess, dim, fid, seed)
             for alg in ("MNFDA", "MNFDAP")
             for ess in (False, True)
             for dim in dims
             for fid in fids
             for seed in seeds]
    print(f"MN-FDA ESS test: {len(tasks)} runs "
          f"(2 algs x 2 ess x {len(dims)} dims x {len(fids)} funcs x {n_seeds} seeds), "
          f"pop={POP} gen={NGEN} clique={MAXCLIQUE}, BZ (Boltzmann), {N_WORKERS} workers")

    records, hangs = [], []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS, max_tasks_per_child=1) as ex:
        futs = {ex.submit(_run_one, t): t for t in tasks}
        done = 0
        for fut in list(futs):
            t = futs[fut]
            try:
                records.append(fut.result(timeout=RUN_TIMEOUT))
            except FTimeout:
                hangs.append(t)
                records.append({"alg": t[0], "ess": t[1], "dim": t[2], "fid": t[3],
                                "seed": t[4], "best": np.nan, "time_s": np.nan,
                                "mem_mb": np.nan, "finished": False})
            except Exception as exc:
                records.append({"alg": t[0], "ess": t[1], "dim": t[2], "fid": t[3],
                                "seed": t[4], "best": np.nan, "time_s": np.nan,
                                "mem_mb": np.nan, "finished": False,
                                "error": type(exc).__name__})
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(tasks)} done ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(OUT_DIR, "mnfda_ess.csv"), index=False)

    print(f"\n=== FEASIBILITY (fixed the hang?) ===  total time {time.time()-t0:.0f}s")
    fin = df["finished"].mean() * 100
    print(f"  finished: {int(df['finished'].sum())}/{len(df)} ({fin:.1f}%)   "
          f"unfinished/hangs: {len(hangs)}")
    print(f"  peak process memory over all runs: {df['mem_mb'].max():.0f} MB")
    print(f"  mean time/run: {df['time_s'].mean():.1f}s   max: {df['time_s'].max():.1f}s")

    print("\n=== ESS vs no-ESS: mean final fitness, and per-function wins ===")
    ok = df[df["finished"]]
    for alg in ("MNFDA", "MNFDAP"):
        for dim in dims:
            sub = ok[(ok["alg"] == alg) & (ok["dim"] == dim)]
            # per-function mean over seeds, ess vs no-ess
            piv = sub.groupby(["fid", "ess"])["best"].mean().unstack()
            if piv.empty or True not in piv.columns or False not in piv.columns:
                continue
            wins = int((piv[True] > piv[False] + 1e-9).sum())
            ties = int(np.isclose(piv[True], piv[False]).sum())
            loss = int((piv[True] < piv[False] - 1e-9).sum())
            mean_no, mean_es = piv[False].mean(), piv[True].mean()
            print(f"  {alg:6s} n={dim:<4}  mean-final  noESS={mean_no:8.2f}  "
                  f"ESS={mean_es:8.2f}   ESS wins/ties/losses over funcs = "
                  f"{wins}/{ties}/{loss}")
    print(f"\nCSV: {os.path.join(OUT_DIR, 'mnfda_ess.csv')}")


if __name__ == "__main__":
    main()
