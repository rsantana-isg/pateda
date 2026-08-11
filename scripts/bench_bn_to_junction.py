"""
Benchmark: is transforming a learned Bayesian network into an MN-FDA-style
junction (clique) factorization + block sampling faster than the current
per-solution BN sampler?

For each tabular BN-based EDA (EBNA_BIC/K2/PC, LFDA_mp6, BOA_mp6, SARTRE_mp6)
we run the EDA for a few generations to obtain a *realistic* (partially
converged) learned BN, capture the selected population and the learned model,
and then time three ways of producing one new population of POP solutions:

  A  persol   : current sampler, SampleBayesianNetwork (one solution at a time)
  A' vec      : a vectorized ancestral sampler over the SAME BN (topological
                order, per-variable inverse-CDF over the whole population) --
                the "simple win" that needs no transform
  B  transform: bayes_nets.bn_to_factorization(adjacency, card, cpds,
                data=selected_pop) -> moralize -> triangulate -> junction tree
                -> FDA clique tables (CliqueFactorization)
  B' block    : SampleFDA over the transformed factorization (block sampling,
                exactly MN-FDA's method)

We report, per (alg, dim): the BN density (mean/max parents), the transformed
model's treewidth (max clique size) and #cliques, and the four timings.  The
question the user asked: would B+B' beat A, and how long does the transform
take?  We also flag the *fidelity* caveat: with data= given the clique tables
are EMPIRICAL (re-estimated from the selected population), so B does not sample
the BN's own distribution -- to do that faithfully one must calibrate the
junction tree to the BN CPDs, which is exponential in treewidth.

Usage: python3 scripts/bench_bn_to_junction.py [dims] [ngen] [reps]
"""

import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
OUT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, "results", "bn_junction"))
POP = 200
TABULAR = ["EBNA_BIC", "EBNA_K2", "EBNA_PC", "LFDA_mp6", "BOA_mp6", "SARTRE_mp6"]


def _vectorized_ancestral_sample(adjacency, cpds, cardinality, n_samples, rng):
    """Sample n_samples from the BN, all solutions at once, per variable in
    topological order (inverse-CDF over the whole population)."""
    import numpy as np
    n = adjacency.shape[0]
    # topological order
    indeg = adjacency.sum(axis=0).astype(int).copy()
    order, ready = [], [v for v in range(n) if indeg[v] == 0]
    while ready:
        v = ready.pop()
        order.append(v)
        for w in np.where(adjacency[v] != 0)[0]:
            indeg[w] -= 1
            if indeg[w] == 0:
                ready.append(int(w))
    pop = np.zeros((n_samples, n), dtype=int)
    for var in order:
        info = cpds[var]
        parents = list(info["parents"])
        cpd = np.asarray(info["cpd"])
        k = int(cardinality[var])
        if not parents:
            cdf = np.cumsum(cpd)
            u = rng.random(n_samples)
            pop[:, var] = np.searchsorted(cdf, u).clip(max=k - 1)
        else:
            pcard = [int(cardinality[p]) for p in parents]
            mult = np.ones(len(parents), dtype=int)
            for j in range(len(parents) - 2, -1, -1):
                mult[j] = mult[j + 1] * pcard[j + 1]
            config = (pop[:, parents] * mult).sum(axis=1)
            probs = cpd[config]                       # (n_samples, k)
            cdf = np.cumsum(probs, axis=1)
            u = rng.random(n_samples)[:, None]
            pop[:, var] = (u < cdf).argmax(axis=1)
    return pop


def _bench_one(args):
    alg, dim = args
    import numpy as np, ioh
    import compare_bn_variants_pbo as C
    from pateda.sampling.bayesian_network import SampleBayesianNetwork
    from pateda.sampling.fda import SampleFDA
    from pateda.core.models import FactorizedModel
    from bayes_nets.factorization import bn_to_factorization

    prob = ioh.get_problem(18, instance=1, dimension=dim,
                           problem_class=ioh.ProblemClass.PBO)
    eda = C.build_configured_eda(alg, "BZ", prob, POP, 4, 0.5, 7)

    # capture last learned model + the (selected) population it was fit on
    cap = {}
    orig_learn = eda.components.learning.learn

    def learn_wrap(generation, n_vars, cardinality, population, fitness, **kw):
        model = orig_learn(generation, n_vars, cardinality, population, fitness, **kw)
        cap["model"] = model
        cap["pop"] = population.copy()
        cap["card"] = np.asarray(cardinality)
        return model
    eda.components.learning.learn = learn_wrap
    eda.run(verbose=False)

    model, sel_pop, card = cap["model"], cap["pop"], cap["card"]
    adjacency, cpds = model.structure, model.parameters
    n = adjacency.shape[0]
    nparents = adjacency.sum(axis=0)
    rng = np.random.default_rng(0)

    def timeit(fn, reps=5):
        best = float("inf")
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        return best

    # A: current per-solution sampler
    persol = SampleBayesianNetwork(n_samples=POP)
    t_persol = timeit(lambda: persol.sample(n, model, card, n_samples=POP,
                                            rng=np.random.default_rng(0)))
    # A': vectorized ancestral over the same BN
    t_vec = timeit(lambda: _vectorized_ancestral_sample(
        adjacency, cpds, card, POP, np.random.default_rng(0)))

    # B: transform BN -> junction/clique factorization (empirical tables from data).
    # NOTE: with the common max_parents=6 the moralize->triangulate step can produce
    # a very large maximum clique (treewidth); the resulting 2^tw clique table makes
    # both the transform and the block sampler blow up.  We therefore time the
    # transform once, read off the treewidth, and only attempt the (potentially
    # exponential) reps/block-sampling when the treewidth is tractable.  A treewidth
    # above TW_CAP is itself the answer: the MN-FDA route is infeasible there.
    TW_CAP = 20  # 2^20 ~ 1e6 clique-table entries / overlap-config loop iterations

    t0 = time.perf_counter()
    fac = bn_to_factorization(adjacency, card, cpds, data=sel_pop,
                              max_clique_width=None)
    t_transform = time.perf_counter() - t0          # single measured transform
    treewidth = max((len(c) for c in fac.cliques), default=1)
    n_cliques = len(fac.cliques)
    if t_transform < 1.0:                            # cheap: refine with a few reps
        t_transform = timeit(
            lambda: bn_to_factorization(adjacency, card, cpds, data=sel_pop,
                                        max_clique_width=None), reps=4)

    # B': block sampling (SampleFDA) over the transformed model -- only if tractable.
    if treewidth <= TW_CAP:
        fmodel = FactorizedModel(structure=fac.structure, parameters=fac.tables,
                                 metadata={})
        block = SampleFDA(n_samples=POP)
        try:
            t_block = timeit(lambda: block.sample(n, fmodel, card, n_samples=POP,
                                                  rng=np.random.default_rng(0)),
                             reps=3)
        except Exception:
            t_block = float("nan")
        block_status = "ok"
    else:
        t_block = float("inf")                       # 2^tw block sampling infeasible
        block_status = f"skipped_tw>{TW_CAP}"

    return {
        "alg": alg, "dim": dim,
        "mean_parents": float(nparents.mean()),
        "max_parents": int(nparents.max()),
        "tw_junction": int(treewidth), "n_cliques": int(n_cliques),
        "t_persol_ms": 1000 * t_persol,
        "t_vec_ms": 1000 * t_vec,
        "t_transform_ms": 1000 * t_transform,
        "t_block_ms": 1000 * t_block,
        "block_status": block_status,
    }


def main():
    import pandas as pd
    dims = [int(d) for d in sys.argv[1].split(",")] if len(sys.argv) > 1 else [64, 100]
    os.makedirs(OUT, exist_ok=True)
    tasks = [(a, d) for a in TABULAR for d in dims]
    print(f"bench BN->junction: {len(tasks)} (alg,dim), f18 LABS, pop={POP}")
    rows = []
    with ProcessPoolExecutor(max_workers=14, max_tasks_per_child=1) as ex:
        for r in ex.map(_bench_one, tasks):
            rows.append(r)
            print(f"  {r['alg']:11s} n={r['dim']:<4} "
                  f"parents(mean/max)={r['mean_parents']:.2f}/{r['max_parents']} "
                  f"tw={r['tw_junction']} cliques={r['n_cliques']} | "
                  f"persol={r['t_persol_ms']:.1f} vec={r['t_vec_ms']:.2f} "
                  f"transform={r['t_transform_ms']:.1f} block={r['t_block_ms']:.2f} ms "
                  f"[{r['block_status']}]",
                  flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "bn_junction.csv"), index=False)

    print("\n=== verdict per row: transform+block vs current persol vs vec ===")
    for _, r in df.iterrows():
        tb = r["t_transform_ms"] + r["t_block_ms"]
        print(f"  {r['alg']:11s} n={r['dim']:<4}  "
              f"persol={r['t_persol_ms']:7.1f}  vec={r['t_vec_ms']:6.2f}  "
              f"transform+block={tb:7.1f}  "
              f"(transform alone={r['t_transform_ms']:.1f})  "
              f"-> vec is {r['t_persol_ms']/max(r['t_vec_ms'],1e-9):.0f}x persol; "
              f"transform+block is "
              f"{'FASTER' if tb < r['t_persol_ms'] else 'SLOWER'} than persol")
    print(f"\nCSV: {os.path.join(OUT,'bn_junction.csv')}")


if __name__ == "__main__":
    main()
