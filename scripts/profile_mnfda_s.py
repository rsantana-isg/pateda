"""
Computational-time profile of MN-FDA-S (simplified MN-FDA).

Runs a faithful EDA loop (proportional/FP weighting + elitist replacement, as in
the weighted-PBO study) on Deceptive3 at n = 624 and times each component of the
MN-FDA-S learning pipeline plus PLS sampling, printing the number of cliques and
the maximum clique size.  It repeats the profile for several target clique sizes
so one can see whether (and how) the cost scales with the clique size k.

Components timed (per generation):
    MI matrix, chi2 graph, build per-variable cliques, subsumption removal,
    clique ordering, structure+prune, probability tables, PLS sampling.

Usage (all positional, optional):
    python3.11 scripts/profile_mnfda_s.py [seed] [n_vars] [pop_size] [ks] [n_gen]

    seed       RNG seed                          (default 1)
    n_vars     problem size                      (default 624; multiple of 3)
    pop_size   population size                   (default 200)
    ks         comma-separated clique sizes      (default 3,4,5)
    n_gen      generations to average over       (default 8)
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
from pateda.core.models import FactorizedModel
from pateda.learning.utils.weights import count_weights_from_p
from pateda.learning.utils.markov_network import convert_cliques_to_factorized_structure
from pateda.learning.utils.mnfda_fast import (
    compute_mi_matrix_fast, chi2_adjacency, compute_clique_tables_fast,
    order_cliques_for_sampling_fast, prune_empty_cliques,
    build_per_variable_cliques, remove_subsumed_cliques,
)


def learn_timed(sel, card, k, threshold=0.05, prior=True, p=None):
    """One MN-FDA-S learn, returning (model, per-component-times, stats)."""
    w = count_weights_from_p(p, sel.shape[0])
    r = {}
    t = time.perf_counter(); mi = compute_mi_matrix_fast(sel, card, w); r["MI"] = time.perf_counter() - t
    t = time.perf_counter(); adj = chi2_adjacency(mi, sel.shape[0], threshold); r["chi2"] = time.perf_counter() - t
    t = time.perf_counter(); cl = build_per_variable_cliques(mi, adj, k); r["build_cliques"] = time.perf_counter() - t
    t = time.perf_counter(); cl = remove_subsumed_cliques(cl); r["subsumption"] = time.perf_counter() - t
    n_raw = len(cl)
    t = time.perf_counter(); o = order_cliques_for_sampling_fast(cl); r["order"] = time.perf_counter() - t
    t = time.perf_counter()
    st = convert_cliques_to_factorized_structure(cl, o); st = prune_empty_cliques(st)
    r["structure"] = time.perf_counter() - t
    t = time.perf_counter(); tb = compute_clique_tables_fast(sel, st, card, w, prior); r["tables"] = time.perf_counter() - t
    stats = {"n_cliques": int(st.shape[0]), "n_cliques_raw": n_raw,
             "max_size": max(len(c) for c in cl)}
    return FactorizedModel(structure=st, parameters=tb, metadata={}), r, stats


COMPONENTS = ["MI", "chi2", "build_cliques", "subsumption", "order",
              "structure", "tables", "sample"]


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 624
    ps = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    ks = ([int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [3, 4, 5])
    n_gen = int(sys.argv[5]) if len(sys.argv) > 5 else 8

    print(f"Seed:            {seed}")
    print(f"Algorithm:       MN-FDA-S (per-variable cliques, chi2, PLS)")
    print(f"Problem:         Deceptive3, n={n}")
    print(f"Population Size: {ps}   Weighting: proportional (FP)   Gens/avg: {n_gen}")
    print(f"Clique sizes k:  {ks}\n")

    card = 2 * np.ones(n, int)
    S = SampleFDA(n_samples=ps)

    for k in ks:
        rng = np.random.default_rng(seed)
        pop = rng.integers(0, 2, size=(ps, n))
        fit = np.array([deceptive3(x) for x in pop]).reshape(-1, 1)
        acc = {c: 0.0 for c in COMPONENTS}
        last = {}
        for g in range(n_gen):
            order = np.argsort(-fit[:, 0])
            sel = pop[order[:ps // 2]]; selfit = fit[order[:ps // 2]]
            p = compute_selection_probabilities(selfit, mode="proportional", beta=1.0)
            m, r, stats = learn_timed(sel, card, k, p=p)
            t = time.perf_counter(); npop = S.sample(n, m, card, rng=rng); r["sample"] = time.perf_counter() - t
            nfit = np.array([deceptive3(x) for x in npop]).reshape(-1, 1)
            allp = np.vstack([pop, npop]); allf = np.vstack([fit, nfit])
            keep = np.argsort(-allf[:, 0])[:ps]; pop = allp[keep]; fit = allf[keep]
            for c in COMPONENTS:
                acc[c] += r[c]
            last = stats
        avg = {c: acc[c] / n_gen for c in COMPONENTS}
        learn_total = sum(avg[c] for c in COMPONENTS if c != "sample")
        total = learn_total + avg["sample"]
        print(f"--- k = {k} ---")
        print(f"  cliques (productive/raw): {last['n_cliques']} / {last['n_cliques_raw']}"
              f"   max clique size: {last['max_size']}   best fitness: {fit[:,0].max():.0f}/{n//3}")
        print(f"  per-generation time (avg over {n_gen} gens):")
        for c in COMPONENTS:
            print(f"      {c:14s}: {avg[c]*1000:8.1f} ms  ({100*avg[c]/total:4.1f}%)")
        print(f"      {'LEARN total':14s}: {learn_total*1000:8.1f} ms")
        print(f"      {'TOTAL/gen':14s}: {total*1000:8.1f} ms\n")


if __name__ == "__main__":
    main()
