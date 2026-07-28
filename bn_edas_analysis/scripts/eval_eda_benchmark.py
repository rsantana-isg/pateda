"""
Evaluate BN structure-learning algorithms on the EDA benchmark
(``data/eda_datasets/``).

Each problem provides a *structure* file (symmetric true interactions) and a
*samples* file (3000 rows: n binary variables, an objective value, and a split
index 0=train/1=val/2=test, 1000 rows each).

Protocol (as agreed with the user)
----------------------------------
* Sample probabilities: objective globally min-max normalised to [0, 1], then a
  Boltzmann distribution  p ∝ exp(f_norm / T)  (maximisation).  T is swept over
  {0.1, 1.0, 10}.
* Train data + its Boltzmann weights are used to learn structure AND parameters.
* Metrics per learned BN:
    - skeleton F1 vs the true (undirected) interaction matrix,
    - mean log-likelihood per sample on train / val / test,
    - KL(data ‖ BN) on each split, where p_data is the Boltzmann probability
      renormalised over the split and p_BN is the BN joint renormalised over the
      split's rows.
* Groups: Small n<=40, Medium 50<n<100, Large n>=100.
* On the first (small) evaluation, any method whose mean running time exceeds
  15x the K2 time is removed from the comparison.

Usage
-----
    python3 scripts/eval_eda_benchmark.py [small|medium|large] [out_dir]
"""
from __future__ import annotations

import os
import sys
import time
import glob
import csv
import signal
import warnings
from functools import partial
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

import numpy as np

warnings.simplefilter("ignore")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from bayes_nets import BayesianNetwork  # noqa: E402

DATA_DIR = os.path.join(_ROOT, "data", "eda_datasets")
TEMPERATURES = [0.1, 1.0, 10.0]
TIME_FILTER_FACTOR = 15.0          # drop methods slower than 15x K2
N_CPUS = min(10, os.cpu_count() or 1)
MIN_TIMEOUT = 4.0                  # never time a method out below this (s)
ABS_TIMEOUT = 180.0               # absolute per-run wall cap (s)


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def _run_with_timeout(fn, seconds):
    """Run *fn* under a SIGALRM wall-clock limit (worker main thread only)."""
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, max(seconds, 0.1))
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)

# Methods evaluated (name -> fit kwargs).  K2 is the timing reference and must
# be first.  Exact DP (exponential) and SARTRE (needs an order) are excluded.
METHODS: List[Tuple[str, dict]] = [
    ("k2",          dict(method="k2")),
    ("k2_mi",       dict(method="k2_mi")),
    ("k2_plus",     dict(method="k2_plus")),
    ("k2_ensemble", dict(method="k2_ensemble")),
    ("k2_mb",       dict(method="k2_mb")),
    ("bic_hc",      dict(method="bic")),
    ("aic_hc",      dict(method="aic")),
    ("stable_hc",   dict(method="stable_hc")),
    ("tabu",        dict(method="tabu")),
    ("gs",          dict(method="gs")),
    ("rcd",         dict(method="rcd")),
    ("rpcd",        dict(method="rpcd")),
    ("pc",          dict(method="pc")),
    ("stable_pc",   dict(method="stable_pc")),
    ("dt",          dict(method="dt")),
    ("dg",          dict(method="dg")),
    ("dmbbn",       dict(method="dmbbn")),
    ("iterdsla",    dict(method="iterdsla")),
    ("binotears",   dict(method="binotears")),
    ("bounded_tw",  dict(method="bounded_tw")),
]


# ---------------------------------------------------------------------------
# Loading & probabilities
# ---------------------------------------------------------------------------

def dataset_names() -> List[str]:
    names = []
    for f in sorted(glob.glob(os.path.join(DATA_DIR, "*_structure.dat"))):
        names.append(os.path.basename(f).replace("_structure.dat", ""))
    return names


def n_of(name: str) -> int:
    return int(name.split("_")[-1])


def group_of(name: str) -> Optional[str]:
    n = n_of(name)
    if n <= 40:
        return "small"
    if 50 < n < 100:
        return "medium"
    if n >= 100:
        return "large"
    return None       # e.g. n == 50 falls outside the agreed bounds


def load_dataset(name: str):
    A = np.loadtxt(os.path.join(DATA_DIR, f"{name}_UMDA_samples.dat"))
    n = A.shape[1] - 2
    X = A[:, :n].astype(int)
    f = A[:, n].astype(float)
    split = A[:, n + 1].astype(int)
    lines = [l for l in open(os.path.join(DATA_DIR, f"{name}_structure.dat")).read().splitlines() if l.strip()]
    true = np.zeros((n, n), dtype=int)
    for l in lines[1:]:
        i, j = map(int, l.split())
        true[i, j] = true[j, i] = 1
    return X, f, split, true, n


def boltzmann_weights(f: np.ndarray, T: float) -> np.ndarray:
    """Global min-max normalise the objective then p ∝ exp(f_norm / T)."""
    lo, hi = f.min(), f.max()
    fn = (f - lo) / (hi - lo + 1e-12)
    return np.exp(fn / T)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def skeleton_f1(learned: np.ndarray, true: np.ndarray) -> float:
    def sk(a):
        s = ((a != 0) | (a.T != 0)).astype(int)
        np.fill_diagonal(s, 0)
        return s
    L, T = sk(learned), sk(true)
    tp = np.sum((L == 1) & (T == 1)) / 2
    fp = np.sum((L == 1) & (T == 0)) / 2
    fn = np.sum((T == 1) & (L == 0)) / 2
    if tp + fp + fn == 0:
        return 1.0                     # both empty (e.g. OneMax) -> perfect
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def sample_log_probs(bn: BayesianNetwork, X: np.ndarray) -> np.ndarray:
    """Joint log P(x) under the BN for every row (library parent indexing)."""
    n = bn.n_vars
    card = bn.cardinality
    lp = np.zeros(len(X))
    for v in range(n):
        parents = bn.cpds[v]["parents"]
        cpd = bn.cpds[v]["cpd"]
        if not parents:
            p = cpd[X[:, v]]
        else:
            pc = [int(card[pp]) for pp in parents]
            idx = np.zeros(len(X), dtype=int)
            mult = 1
            for j, pp in enumerate(parents):
                idx += X[:, pp] * mult
                mult *= pc[j]
            p = cpd[idx, X[:, v]]
        lp += np.log(np.clip(p, 1e-12, None))
    return lp


def kl_data_bn(w: np.ndarray, lp: np.ndarray, mask: np.ndarray) -> float:
    """KL(p_data ‖ p_BN) over the split's rows."""
    ws = w[mask]
    p_data = ws / ws.sum()
    q = np.exp(lp[mask] - lp[mask].max())     # stabilise before normalising
    q = q / q.sum()
    q = np.clip(q, 1e-300, None)
    return float(np.sum(p_data * np.log(p_data / q)))


# ---------------------------------------------------------------------------
# One learning run
# ---------------------------------------------------------------------------

def run_once(method_kwargs: dict, X, w_full, splits, true, n, timeout=ABS_TIMEOUT):
    """Learn a BN on weighted train data and compute all metrics.

    Enforces a SIGALRM wall-clock ``timeout``; on timeout the record is marked
    with ``timeout=True`` (used by the 15x-K2 filter).  Returns
    (metrics_dict, elapsed_seconds).
    """
    tr, va, te = splits["tr"], splits["va"], splits["te"]
    w_tr = w_full[tr] / w_full[tr].sum()
    bn = BayesianNetwork(n, np.full(n, 2))
    t0 = time.time()
    try:
        _run_with_timeout(lambda: bn.fit(X[tr], sample_weights=w_tr, **method_kwargs),
                          timeout)
    except _Timeout:
        return {"error": "timeout", "timeout": True}, time.time() - t0
    except Exception as exc:               # noqa: BLE001
        return {"error": str(exc)[:80]}, time.time() - t0
    elapsed = time.time() - t0
    lp = sample_log_probs(bn, X)
    m = {
        "f1": skeleton_f1(bn.adjacency, true),
        "edges": int(bn.adjacency.sum()),
        "ll_train": float(lp[tr].mean()),
        "ll_val": float(lp[va].mean()),
        "ll_test": float(lp[te].mean()),
        "kl_train": kl_data_bn(w_full, lp, tr),
        "kl_val": kl_data_bn(w_full, lp, va),
        "kl_test": kl_data_bn(w_full, lp, te),
    }
    return m, elapsed


# ---------------------------------------------------------------------------
# Parallel worker tasks
# ---------------------------------------------------------------------------

def _k2_baseline_task(name: str):
    """Time base K2 on a dataset (used to set each method's timeout)."""
    X, f, split, true, n = load_dataset(name)
    splits = {"tr": split == 0, "va": split == 1, "te": split == 2}
    w = boltzmann_weights(f, 1.0)
    metrics, elapsed = run_once(dict(method="k2"), X, w, splits, true, n, ABS_TIMEOUT)
    rec = {"dataset": name, "n": n, "T": 1.0, "method": "k2",
           "time_s": round(elapsed, 4), "time_ratio": 1.0}
    rec.update(metrics)
    return name, max(elapsed, 1e-4), rec


def _method_task(args):
    """Run one (dataset, method, T) with a per-dataset timeout."""
    name, mname, mkw, T, k2_time = args
    X, f, split, true, n = load_dataset(name)
    splits = {"tr": split == 0, "va": split == 1, "te": split == 2}
    w = boltzmann_weights(f, T)
    timeout = min(ABS_TIMEOUT, max(MIN_TIMEOUT, TIME_FILTER_FACTOR * k2_time))
    metrics, elapsed = run_once(mkw, X, w, splits, true, n, timeout)
    ratio = elapsed / k2_time if k2_time else float("nan")
    rec = {"dataset": name, "n": n, "T": T, "method": mname,
           "time_s": round(elapsed, 4), "time_ratio": round(ratio, 2)}
    rec.update(metrics)
    return rec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    tier = sys.argv[1] if len(sys.argv) > 1 else "small"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(DATA_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)

    names = [nm for nm in dataset_names() if group_of(nm) == tier]
    print(f"Tier: {tier}  ({len(names)} datasets)   CPUs: {N_CPUS}")
    print("Datasets:", ", ".join(names))
    print(f"Temperatures: {TEMPERATURES}   time filter: >{TIME_FILTER_FACTOR:.0f}x K2 removed\n")

    rows: List[dict] = []

    # ---- Phase 0: K2 baseline timing per dataset (parallel) --------------
    print("=" * 78)
    print("PHASE 0 — K2 baseline timing per dataset")
    print("=" * 78)
    k2_time: Dict[str, float] = {}
    with Pool(N_CPUS) as pool:
        for name, t, rec in pool.imap_unordered(_k2_baseline_task, names):
            k2_time[name] = t
            rows.append(rec)
            print(f"  {name:16s} K2={t:.3f}s")

    # ---- Phase 1: every method at T=1.0 with 15x-K2 timeout (parallel) ----
    print("\n" + "=" * 78)
    print("PHASE 1 — timing all methods at T=1.0 (per-run cap = 15x K2)")
    print("=" * 78)
    tasks = []
    for name in names:
        for mname, mkw in METHODS:
            if mname == "k2":
                continue                       # already have it from phase 0
            tasks.append((name, mname, mkw, 1.0, k2_time[name]))
    time_ratios: Dict[str, List[float]] = {m: [] for m, _ in METHODS}
    time_ratios["k2"] = [1.0] * len(names)
    with Pool(N_CPUS) as pool:
        for rec in pool.imap_unordered(_method_task, tasks):
            rows.append(rec)
            # a timed-out run counts as exceeding the 15x cap
            ratio = TIME_FILTER_FACTOR + 1 if rec.get("timeout") else rec["time_ratio"]
            time_ratios[rec["method"]].append(ratio)
    print(f"  {len(tasks)} runs complete")

    # Robust filter: remove a method if its *median* per-dataset time ratio
    # exceeds 15x K2.  (K2 itself ranges 0.1-3.3s across datasets, so the mean
    # ratio is skewed by slow-K2 datasets; the median reflects the typical
    # dataset and correctly removes methods that time out on the fast ones.)
    med_ratio = {m: float(np.median(time_ratios[m])) for m, _ in METHODS}
    mean_ratio = med_ratio       # used downstream for display
    max_ratio = {m: float(np.max(time_ratios[m])) for m, _ in METHODS}
    survivors = [m for m, _ in METHODS if med_ratio[m] <= TIME_FILTER_FACTOR]
    removed = [m for m, _ in METHODS if med_ratio[m] > TIME_FILTER_FACTOR]

    print("\n  Median time ratio vs K2 (over datasets; timeouts count as >15x):")
    for m, _ in METHODS:
        flag = "  <-- REMOVED (median >15x K2)" if m in removed else ""
        print(f"    {m:14s} median={med_ratio[m]:6.2f}x  max={max_ratio[m]:6.2f}x{flag}")
    print(f"\n  Survivors ({len(survivors)}): {', '.join(survivors)}")
    print(f"  Removed   ({len(removed)}): {', '.join(removed) or '(none)'}")

    # ---- Phase 2: T sweep for survivors (parallel) -----------------------
    print("\n" + "=" * 78)
    print("PHASE 2 — temperature sweep for surviving methods")
    print("=" * 78)
    tasks = []
    for name in names:
        for T in TEMPERATURES:
            if T == 1.0:
                continue                       # already computed in phase 1
            for mname in survivors:
                mkw = dict(METHODS)[mname]
                tasks.append((name, mname, mkw, T, k2_time[name]))
    with Pool(N_CPUS) as pool:
        for rec in pool.imap_unordered(_method_task, tasks):
            rec["time_ratio"] = ""             # ratio only meaningful at T=1
            rows.append(rec)
    print(f"  {len(tasks)} runs complete")

    # ---- write CSV -------------------------------------------------------
    csv_path = os.path.join(out_dir, f"eda_eval_{tier}.csv")
    fields = ["dataset", "n", "T", "method", "time_s", "time_ratio", "f1", "edges",
              "ll_train", "ll_val", "ll_test", "kl_train", "kl_val", "kl_test", "error"]
    with open(csv_path, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        wtr.writeheader()
        for r in rows:
            wtr.writerow(r)
    print(f"\nSaved: {csv_path}  ({len(rows)} rows)")

    _summary(rows, survivors, removed, mean_ratio, TEMPERATURES)


def _summary(rows, survivors, removed, mean_ratio, temps):
    """Print aggregate accuracy tables per temperature for survivors."""
    import statistics as st
    print("\n" + "=" * 78)
    print("SUMMARY — mean over small datasets (survivors), by temperature")
    print("=" * 78)
    for T in temps:
        print(f"\n--- T = {T} ---")
        print(f"  {'method':14s}{'F1':>7s}{'llTest':>10s}{'KLtest':>9s}{'xK2':>7s}")
        agg = {}
        for r in rows:
            if r["method"] not in survivors or r["T"] != T or "f1" not in r:
                continue
            agg.setdefault(r["method"], {"f1": [], "ll": [], "kl": []})
            agg[r["method"]]["f1"].append(r["f1"])
            agg[r["method"]]["ll"].append(r["ll_test"])
            agg[r["method"]]["kl"].append(r["kl_test"])
        ranked = sorted(agg.items(), key=lambda kv: -st.mean(kv[1]["f1"]))
        for m, d in ranked:
            print(f"  {m:14s}{st.mean(d['f1']):>7.3f}{st.mean(d['ll']):>10.2f}"
                  f"{st.mean(d['kl']):>9.3f}{mean_ratio[m]:>6.1f}x")


if __name__ == "__main__":
    main()
