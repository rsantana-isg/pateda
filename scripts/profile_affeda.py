"""
Computational-time profile of AffEDA (affinity-propagation factorization).

Times each component of the AffEDA learning pipeline plus PLS sampling on
Deceptive3 under the weighted-PBO configuration (proportional/FP weighting,
elitist replacement), for a sweep of problem sizes, so one can see which
component drives the cost as n grows.

Components timed (per generation):
    marginals (find_marginal_prob), MI matrix (Python i,j,k,l loop),
    affinity-propagation factorization (recursive AP clustering),
    clique probability tables, PLS sampling.

Usage:
    python3.11 scripts/profile_affeda.py [seed] [sizes] [pop_size] [n_gen] [max_clique_size]
    defaults: seed=1  sizes=60,120,240,624  pop_size=200  n_gen=4  max_clique_size=5
"""

import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, os.pardir, "src"))

from pateda.functions.discrete_binary.toy_functions.deceptive import deceptive3
from pateda.selection.weighting import compute_selection_probabilities
from pateda.sampling.fda import SampleFDA
from pateda.learning.affinity import LearnAffinityFactorization, _fast_clique_tables
from pateda.learning.utils.mnfda_fast import compute_mi_matrix_fast
from pateda.learning.utils.weights import count_weights_from_p


def profile_learn(L, sel, card, n, p):
    """Time the (optimized, approaches A/B) AffEDA learn steps."""
    w = count_weights_from_p(p, sel.shape[0])
    r = {}
    t = time.perf_counter()                       # approach A: vectorized MI
    mi = compute_mi_matrix_fast(sel, card, w)
    r["MI_matrix"] = time.perf_counter() - t
    pref = np.median(mi)
    t = time.perf_counter()
    cliques_list = L._recursive_factorization(mi, np.arange(n), pref)
    r["affinity_prop"] = time.perf_counter() - t
    cliques = L._create_clique_structure(cliques_list, n)
    t = time.perf_counter()                       # approach B: vectorized tables
    tables = _fast_clique_tables(cliques, sel, card, w, L.alpha)
    r["clique_tables"] = time.perf_counter() - t
    from pateda.core.models import FactorizedModel
    model = FactorizedModel(structure=cliques, parameters=tables, metadata={})
    stats = {"n_cliques": len(cliques_list),
             "max_size": max(len(c) for c in cliques_list)}
    return model, r, stats


COMPONENTS = ["MI_matrix", "affinity_prop", "clique_tables", "sample"]


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    sizes = ([int(s) for s in sys.argv[2].split(",")]
             if len(sys.argv) > 2 else [60, 120, 240, 624])
    ps = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    n_gen = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    mcs = int(sys.argv[5]) if len(sys.argv) > 5 else 5

    print(f"Seed: {seed}   AffEDA (max_clique_size={mcs}, alpha=1.0)   pop={ps}   "
          f"weighting=proportional   gens/avg={n_gen}\n")

    for n in sizes:
        card = 2 * np.ones(n, int)
        S = SampleFDA(n_samples=ps)
        L = LearnAffinityFactorization(max_clique_size=mcs, alpha=1.0)
        rng = np.random.default_rng(seed)
        pop = rng.integers(0, 2, size=(ps, n))
        fit = np.array([deceptive3(x) for x in pop]).reshape(-1, 1)
        acc = {c: 0.0 for c in COMPONENTS}
        last = {}
        t_size = time.perf_counter()
        for g in range(n_gen):
            order = np.argsort(-fit[:, 0])
            sel = pop[order[:ps // 2]]; selfit = fit[order[:ps // 2]]
            p = compute_selection_probabilities(selfit, mode="proportional", beta=1.0)
            m, r, stats = profile_learn(L, sel, card, n, p)
            t = time.perf_counter(); npop = S.sample(n, m, card, rng=rng)
            r["sample"] = time.perf_counter() - t
            nfit = np.array([deceptive3(x) for x in npop]).reshape(-1, 1)
            allp = np.vstack([pop, npop]); allf = np.vstack([fit, nfit])
            keep = np.argsort(-allf[:, 0])[:ps]; pop = allp[keep]; fit = allf[keep]
            for c in COMPONENTS:
                acc[c] += r[c]
            last = stats
        avg = {c: acc[c] / n_gen for c in COMPONENTS}
        total = sum(avg.values())
        print(f"=== n = {n}  (cliques={last['n_cliques']}, max_size={last['max_size']}, "
              f"total/gen={total:.2f}s) ===")
        for c in COMPONENTS:
            print(f"    {c:14s}: {avg[c]*1000:9.1f} ms  ({100*avg[c]/total:4.1f}%)")
        print(flush=True)


if __name__ == "__main__":
    main()
