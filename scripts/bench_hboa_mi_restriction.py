"""Phase 0 (baseline) + Phase 1 (MI candidate restriction) for HBOA-Light A1-A5.

Roadmap goals: (1) heavily promote / bound the moral-triangulated treewidth, and
(2) reduce the learning cost of the decision-tree/graph variants -- with the
*same* mechanism for all five variants.  The lever is the ``candidate_parents``
switch already in ``LearnHBOALight`` (``bayes_nets`` ``candidate_parents="mi:k"``,
used today only by A2): compute pairwise mutual information once and restrict
each variable's parent search to its top-K MI neighbours.  Fewer candidates
=> shorter split-search (goal 2) *and* smaller families / sparser moral graph
=> smaller treewidth (goal 1).

For each variant and each mode (baseline = no restriction, vs ``mi:12`` /
``mi:8``) we learn one structure on a realistic captured selected population
(LABS f18) and record: learning time, scenario-1 treewidth (moralize +
triangulate max clique), scenario-2 family clique (max in-degree + 1), edge
count, and a dense-tabular BIC of the structure (a comparable fit proxy).  A
per-learn wall-clock cap flags the variants/sizes the *baseline* cannot reach --
the scaling story Phase 1 is meant to fix.

Usage: python3 scripts/bench_hboa_mi_restriction.py [dims] [modes] [timeout_s]
       python3 scripts/bench_hboa_mi_restriction.py 64,100,256 base,mi:12,mi:8 500
"""
import os
import sys
import time
import signal
from concurrent.futures import ProcessPoolExecutor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
OUT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, "results", "bn_learning"))
POP = 200

# The five HBOA-Light variants, by their LearnHBOALight construction kwargs
# (verbatim from the HBOA_Light_A* wrappers, minus candidate_parents which is
# the Phase-1 variable).  A2's report default is mi:10; here every variant is
# run both without restriction (baseline) and with it, to isolate the MI effect.
VARIANTS = {
    "A1_dt":   dict(method="dt",     local_structure="dt"),
    "A2_dg":   dict(method="dg",     local_structure="dg"),
    "A3_fast": dict(method="dg",     local_structure="dg", fast_local_scoring=True),
    "A4_mdl":  dict(method="dg",     local_structure="dg", max_leaves=32,
                    split_score="mdl"),
    "A5_ndg":  dict(method="dg_ndg", local_structure="dg"),
}


def _capture_population(dim, seed=7):
    """A realistic (partially converged) selected population from a cheap run."""
    import numpy as np, ioh
    import compare_bn_variants_pbo as C
    prob = ioh.get_problem(18, instance=1, dimension=dim,
                           problem_class=ioh.ProblemClass.PBO)
    eda = C.build_configured_eda("EBNA_K2", "BZ", prob, POP, 3, 0.5, seed)
    cap = {}
    ol = eda.components.learning.learn

    def lw(g, nv, card, pop, fit, **kw):
        m = ol(g, nv, card, pop, fit, **kw)
        cap["pop"] = pop.copy(); cap["card"] = np.asarray(card); return m
    eda.components.learning.learn = lw
    eda.run(verbose=False)
    return cap["pop"], cap["card"]


def _run(args):
    label, kw, mode, data, card, timeout_s = args
    import numpy as np
    from pateda.learning.hboa import LearnHBOALight
    from bench_bn_learning_constraints import _treewidth, _bic

    cp = None if mode == "base" else mode

    def _handler(signum, frame):
        raise TimeoutError()
    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(timeout_s))
    try:
        L = LearnHBOALight(max_parents=6, alpha=1.0, candidate_parents=cp, **kw)
        t0 = time.perf_counter()
        m = L.learn(0, data.shape[1], card, data, None)
        t_learn = time.perf_counter() - t0
        signal.alarm(0)
    except TimeoutError:
        return {"variant": label, "mode": mode, "t_learn_s": float("inf"),
                "treewidth": -1, "family_clique": -1, "edges": -1,
                "bic": float("nan"), "status": f"timeout>{int(timeout_s)}s"}
    adj = m.structure
    tw, _ = _treewidth(adj, card)
    return {"variant": label, "mode": mode, "t_learn_s": t_learn,
            "treewidth": tw, "family_clique": int(adj.sum(0).max()) + 1,
            "edges": int(adj.sum()), "bic": _bic(adj, data, card), "status": "ok"}


def main():
    import numpy as np, pandas as pd
    dims = [int(d) for d in sys.argv[1].split(",")] if len(sys.argv) > 1 else [64, 100, 256]
    modes = sys.argv[2].split(",") if len(sys.argv) > 2 else ["base", "mi:12", "mi:8"]
    timeout_s = float(sys.argv[3]) if len(sys.argv) > 3 else 500
    os.makedirs(OUT, exist_ok=True)

    all_rows = []
    for dim in dims:
        print(f"\n=== n={dim}: capturing LABS f18 selected population ...",
              flush=True)
        data, card = _capture_population(dim)
        print(f"    captured {data.shape}; learning A1-A5 x {modes} "
              f"(timeout {timeout_s:.0f}s) ...", flush=True)
        tasks = [(lbl, kw, md, data, card, timeout_s)
                 for lbl, kw in VARIANTS.items() for md in modes]
        rows = []
        with ProcessPoolExecutor(max_workers=min(14, len(tasks)),
                                 max_tasks_per_child=1) as ex:
            for r in ex.map(_run, tasks):
                r["dim"] = dim
                rows.append(r); all_rows.append(r)
                tw = "  --" if r["treewidth"] < 0 else f"{r['treewidth']:4d}"
                tl = "   timeout" if r["t_learn_s"] == float("inf") else f"{r['t_learn_s']:8.2f}s"
                print(f"    {r['variant']:8s} {r['mode']:6s}  learn={tl}  "
                      f"tw(sc1)={tw}  fam(sc2)={r['family_clique']:2d}  "
                      f"edges={r['edges']:4d}  [{r['status']}]", flush=True)
        # per-dim speed/treewidth summary vs baseline
        df = pd.DataFrame(rows)
        print(f"    --- n={dim} MI effect vs baseline ---")
        for lbl in VARIANTS:
            b = df[(df["variant"] == lbl) & (df["mode"] == "base")]
            if b.empty:
                continue
            bt, btw = b.iloc[0]["t_learn_s"], b.iloc[0]["treewidth"]
            for md in [m for m in modes if m != "base"]:
                r = df[(df["variant"] == lbl) & (df["mode"] == md)]
                if r.empty:
                    continue
                rt, rtw = r.iloc[0]["t_learn_s"], r.iloc[0]["treewidth"]
                spd = (bt / rt) if (rt and np.isfinite(bt) and rt > 0) else float("nan")
                print(f"      {lbl:8s} {md:6s}: speed x{spd:5.1f}  "
                      f"tw {btw}->{rtw}")

    pd.DataFrame(all_rows).to_csv(os.path.join(OUT, "hboa_mi_restriction.csv"),
                                  index=False)
    print(f"\nCSV: {os.path.join(OUT, 'hboa_mi_restriction.csv')}")


if __name__ == "__main__":
    main()
