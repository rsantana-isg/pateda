"""
gen_eval_eda_benchmark.py — single cluster job of the extended EDA benchmark.

Cluster-friendly variant of ``scripts/eval_eda_benchmark.py``: it evaluates
*one* (problem, train_set, algorithm, seed, temperature) combination and writes
a self-describing result file.  The companion generator
``slurm/launch_eval_eda_benchmark.py`` prints one ``sbatch`` line per
combination for the whole grid.

Usage (positional args, no flags)
---------------------------------
    python3 scripts/gen_eval_eda_benchmark.py PROBLEM TRAIN_SET ALGORITHM SEED TEMPERATURE [OUT_DIR]

    PROBLEM      dataset name in data/eda_datasets/ (e.g. Braid_36)
    TRAIN_SET    which split (0, 1 or 2) is used as the training pool
    ALGORITHM    BN learning method (see ALGORITHMS in the launcher)
    SEED         integer RNG seed (controls the 80% selection and the learner)
    TEMPERATURE  Boltzmann temperature (e.g. 0.1, 1.0, 10)
    OUT_DIR      output directory (default: results/eda_eval_cluster)

Protocol
--------
1.  Randomly select 80% of the solutions in split ``TRAIN_SET``.
2.  Compute their per-solution probabilities: the objective of the *selected*
    solutions is min-max normalised (per set) and turned into a Boltzmann
    distribution  p ∝ exp(f_norm / T).  These are the ``sample_weights`` used
    to learn the BN (structure **and** parameters).
3.  Predict probabilities (the BN joint) for every solution of each of the
    three splits (0, 1, 2).  The training-pool split is evaluated on 100% of
    its rows (the 80% used for learning plus the 20% held out).
4.  For every split compute three prediction metrics; the skeleton F1 is a
    property of the learned *structure* and is therefore computed once.  The
    learning wall-clock time is also recorded.

Output
------
A file  ``eval_<problem>_tr<train_set>_<algorithm>_T<temp>_s<seed>.dat``  whose
name encodes the inputs and whose body is a vector of **11 values**:

    f1  time  ll_s0 kl_s0 sp_s0  ll_s1 kl_s1 sp_s1  ll_s2 kl_s2 sp_s2

where, per split s ∈ {0,1,2}:
    ll_s  = mean log-likelihood of the BN over split s,
    kl_s  = KL(p_data ‖ p_BN) over split s,
    sp_s  = Spearman correlation between predicted and original probabilities.

The two leading ``#`` comment lines make the file both human- and
``np.loadtxt``-readable (which skips ``#`` lines and returns the 11 numbers).
"""
from __future__ import annotations

import os
import sys
import time
import traceback

import numpy as np
from scipy.stats import spearmanr

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _add_bayes_nets_to_path():
    """Make the local ``bayes_nets`` source package importable.

    This script does NOT require ``bayes_nets`` to be pip-installed (no PyPI
    publish needed): it imports the package straight from the source folder
    shipped with the repository.  We search a few likely locations for the
    folder that contains ``bayes_nets/__init__.py`` and put it first on
    ``sys.path`` so the local source shadows any installed distribution.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        _ROOT,                       # repo root (…/scripts/..) — normal layout
        here,                        # a bayes_nets/ folder next to this script
        os.path.dirname(_ROOT),      # one level above the repo root
        os.getcwd(),                 # current working directory
    ]
    seen = set()
    ordered = [c for c in candidates if not (c in seen or seen.add(c))]
    for cand in ordered:
        if os.path.isfile(os.path.join(cand, "bayes_nets", "__init__.py")):
            if cand not in sys.path:
                sys.path.insert(0, cand)
            return cand, ordered
    return None, ordered


_PKG_ROOT, _SEARCHED = _add_bayes_nets_to_path()
try:
    from bayes_nets import BayesianNetwork  # noqa: E402
except ImportError as exc:
    _looked = "\n  ".join(
        os.path.join(c, "bayes_nets", "__init__.py") for c in _SEARCHED
    )
    raise SystemExit(
        "ERROR: could not import the 'bayes_nets' package.\n\n"
        "This script uses the bayes_nets/ SOURCE FOLDER directly — no pip\n"
        "install and no PyPI/GitHub publish is required — but that folder was\n"
        "not found on this machine.  Copy the repository's bayes_nets/ package\n"
        "folder to the repo root so that one of these files exists:\n\n"
        "  " + _looked + "\n\n"
        "For example, from your local machine (no GitHub needed):\n"
        "  rsync -av bayes_nets/ <cluster>:" + os.path.join(_ROOT, "bayes_nets") + "/\n"
        "  # or:  tar czf bn.tgz bayes_nets && scp bn.tgz <cluster>:"
        + _ROOT + " && (cd " + _ROOT + " && tar xzf bn.tgz)\n\n"
        f"Original import error: {exc}"
    )

DATA_DIR = os.path.join(_ROOT, "data", "eda_datasets")
DEFAULT_OUT = os.path.join(_ROOT, "results", "eda_eval_cluster")
TRAIN_FRACTION = 0.8
SPLITS = (0, 1, 2)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(name: str):
    """Return (X, f, split, true_adjacency, n, cardinality) for dataset *name*.

    Cardinality is inferred from the **full** dataset (``max state + 1`` per
    column) so that states appearing only in the validation/test splits are
    always representable in the learned CPDs.
    """
    A = np.loadtxt(os.path.join(DATA_DIR, f"{name}_UMDA_samples.dat"))
    n = A.shape[1] - 2
    X = A[:, :n].astype(int)
    f = A[:, n].astype(float)
    split = A[:, n + 1].astype(int)
    cardinality = np.maximum(X.max(axis=0) + 1, 2).astype(int)
    struct_path = os.path.join(DATA_DIR, f"{name}_structure.dat")
    lines = [l for l in open(struct_path).read().splitlines() if l.strip()]
    true = np.zeros((n, n), dtype=int)
    for l in lines[1:]:                      # first line is a header/count
        i, j = map(int, l.split())
        true[i, j] = true[j, i] = 1
    return X, f, split, true, n, cardinality


def boltzmann_weights(f_sub: np.ndarray, T: float) -> np.ndarray:
    """Per-set min-max normalise *f_sub* then p ∝ exp(f_norm / T) (unnormalised)."""
    lo, hi = float(f_sub.min()), float(f_sub.max())
    fn = (f_sub - lo) / (hi - lo + 1e-12)
    return np.exp(fn / T)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def skeleton_f1(learned: np.ndarray, true: np.ndarray) -> float:
    """Undirected-edge F1 of the learned structure vs the true structure."""
    def sk(a):
        s = ((a != 0) | (a.T != 0)).astype(int)
        np.fill_diagonal(s, 0)
        return s
    L, T = sk(learned), sk(true)
    tp = np.sum((L == 1) & (T == 1)) / 2
    fp = np.sum((L == 1) & (T == 0)) / 2
    fn = np.sum((T == 1) & (L == 0)) / 2
    if tp + fp + fn == 0:
        return 1.0                           # both empty (e.g. OneMax): perfect
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def sample_log_probs(bn: BayesianNetwork, X: np.ndarray) -> np.ndarray:
    """Joint log P(x) under the BN for every row of *X*."""
    card = bn.cardinality
    lp = np.zeros(len(X))
    for v in range(bn.n_vars):
        parents = bn.cpds[v]["parents"]
        cpd = bn.cpds[v]["cpd"]
        if not parents:
            p = cpd[X[:, v]]
        else:
            idx = np.zeros(len(X), dtype=int)
            mult = 1
            for pp in parents:
                idx += X[:, pp] * mult
                mult *= int(card[pp])
            p = cpd[idx, X[:, v]]
        lp += np.log(np.clip(p, 1e-12, None))
    return lp


def kl_data_bn(orig_prob: np.ndarray, lp: np.ndarray) -> float:
    """KL(p_data ‖ p_BN) over a split's rows.

    ``p_data`` is the (renormalised) original Boltzmann probability and
    ``p_BN`` is the BN joint renormalised over the same rows.
    """
    p_data = orig_prob / orig_prob.sum()
    q = np.exp(lp - lp.max())                # stabilise before normalising
    q = q / q.sum()
    q = np.clip(q, 1e-300, None)
    return float(np.sum(p_data * np.log(p_data / q)))


def spearman_pred_orig(lp: np.ndarray, orig_prob: np.ndarray) -> float:
    """Spearman correlation between predicted and original probabilities.

    Rank-based, hence invariant to the (monotone) exp / normalisation, so the
    predicted log-probability ``lp`` can be used directly.
    """
    if np.all(lp == lp[0]) or np.all(orig_prob == orig_prob[0]):
        return float("nan")
    rho = spearmanr(lp, orig_prob).correlation
    return float(rho)


# ---------------------------------------------------------------------------
# Core evaluation of one combination
# ---------------------------------------------------------------------------

def evaluate(problem, train_set, algorithm, seed, temperature):
    """Learn one BN and return (vector11, elapsed, extra_info)."""
    X, f, split, true, n, cardinality = load_dataset(problem)
    rng = np.random.default_rng(seed)

    # 1. select 80% of the training-pool split
    pool = np.where(split == train_set)[0]
    if pool.size == 0:
        raise ValueError(f"train_set {train_set} has no rows in {problem}")
    n_sel = max(1, int(round(TRAIN_FRACTION * pool.size)))
    sel = rng.choice(pool, size=n_sel, replace=False)

    # 2. per-set Boltzmann probabilities of the selected solutions (learning weights)
    w_learn = boltzmann_weights(f[sel], temperature)
    w_learn = w_learn / w_learn.sum()

    # 3. learn BN structure + parameters (timed); seed forwarded to the learner
    bn = BayesianNetwork(n, cardinality)
    t0 = time.time()
    bn.learn_structure(
        X[sel], method=algorithm, sample_weights=w_learn, seed=seed
    )
    bn.learn_parameters(X[sel], sample_weights=w_learn)
    elapsed = time.time() - t0

    # 4. structural F1 (once) + per-split prediction metrics
    f1 = skeleton_f1(bn.adjacency, true)

    per_set = []
    for s in SPLITS:
        idx = np.where(split == s)[0]
        lp = sample_log_probs(bn, X[idx])
        orig = boltzmann_weights(f[idx], temperature)
        ll = float(lp.mean())
        kl = kl_data_bn(orig, lp)
        sp = spearman_pred_orig(lp, orig)
        per_set.append((ll, kl, sp))

    vector = [f1, elapsed]
    for ll, kl, sp in per_set:
        vector.extend([ll, kl, sp])
    extra = dict(n=n, edges=int(bn.adjacency.sum()), n_selected=n_sel)
    return vector, elapsed, extra


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

VALUE_NAMES = [
    "f1", "time",
    "ll_s0", "kl_s0", "sp_s0",
    "ll_s1", "kl_s1", "sp_s1",
    "ll_s2", "kl_s2", "sp_s2",
]


def result_filename(problem, train_set, algorithm, seed, temp_str) -> str:
    return f"eval_{problem}_tr{train_set}_{algorithm}_T{temp_str}_s{seed}.dat"


def write_result(path, vector, problem, train_set, algorithm, seed, temp_str, extra):
    header_params = (
        f"# problem={problem} train_set={train_set} algorithm={algorithm} "
        f"temperature={temp_str} seed={seed} "
        f"n={extra.get('n', '?')} edges={extra.get('edges', '?')} "
        f"n_selected={extra.get('n_selected', '?')}"
    )
    header_cols = "# " + " ".join(VALUE_NAMES)
    body = " ".join(f"{v:.8g}" for v in vector)
    content = header_params + "\n" + header_cols + "\n" + body + "\n"
    # Write to a temporary file in the same directory, then atomically replace
    # the target.  os.replace overwrites any existing file and is robust even
    # when a job wrapper has redirected stdout onto the same path: the final
    # result file always contains exactly this 11-value vector.
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w") as fh:
        fh.write(content)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 6:
        print(__doc__)
        sys.exit(1)

    problem = sys.argv[1]
    train_set = int(sys.argv[2])
    algorithm = sys.argv[3]
    seed = int(sys.argv[4])
    temp_str = sys.argv[5]                    # keep the raw string for the filename
    temperature = float(temp_str)
    out_dir = sys.argv[6] if len(sys.argv) > 6 else DEFAULT_OUT
    # Resolve to an absolute path so the result location never depends on the
    # current working directory (which the cluster wrapper may set elsewhere).
    out_dir = os.path.abspath(out_dir)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir, result_filename(problem, train_set, algorithm, seed, temp_str)
    )

    print(f"Problem:          {problem}")
    print(f"Train set:        {train_set}")
    print(f"Algorithm:        {algorithm}")
    print(f"Seed:             {seed}")
    print(f"Temperature:      {temperature}")
    print(f"Output:           {out_path}")

    # Always run and overwrite the result file (no skip-if-exists check).

    t_start = time.time()
    try:
        vector, elapsed, extra = evaluate(
            problem, train_set, algorithm, seed, temperature
        )
        print(f"Learning time:    {elapsed:.4f}s   edges={extra['edges']}   "
              f"n_selected={extra['n_selected']}")
    except Exception as exc:                  # noqa: BLE001
        traceback.print_exc()
        # Record a failed run so the grid is complete and re-launches skip it.
        elapsed = time.time() - t_start
        vector = [float("nan"), elapsed] + [float("nan")] * 9
        extra = dict(error=str(exc)[:120])
        print(f"ERROR: {exc}")

    write_result(out_path, vector, problem, train_set, algorithm,
                 seed, temp_str, extra)
    print("Vector: " + " ".join(f"{v:.6g}" for v in vector))
    print(f"Saved:  {out_path}")


if __name__ == "__main__":
    main()
