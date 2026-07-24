"""
Compare the plug-and-play BN-based EDAs of pateda on two decomposable binary
problems: Trap5 (order-5 trap, n=50, optimum 50) and UBQP-50.

Algorithms compared (every BN-based EDA uses max_parents=5)
-----------------------------------------------------------
* UMDA                      -- univariate baseline (no dependencies)
* EBNA, BOA, LFDA           -- the existing score-and-search BN-based EDAs
* HBOA                      -- decision-graph local structure + niching (faithful)
* SARTRE_EDA                -- sparse additive / group-lasso structure learning
* BINOTEARS_EDA             -- differentiable structure learning (binary only)
* PCBN_EDA                  -- constraint-based PC-Stable, bounded conditioning order
* HSARTRE_EDA, HBINOTEARS_EDA -- hBOA-style (decision graph + niching) upgrades

For each (problem, algorithm) it runs several seeds, printing the best-so-far
fitness at every generation while the run proceeds, and finally reports the
mean/std of the best fitness, the success rate (fraction of runs reaching the
known optimum, Trap5 only), and the mean wall-clock time.  Algorithms are then
ranked per problem by mean best fitness.

Run with the interpreter that has ``bayes_nets`` installed (python3.11 here):

    python3.11 scripts/test_new_bn_edas.py [n_gen] [pop_size] [n_seeds]
"""
from __future__ import annotations

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _HERE)  # bn_eda_problems

import numpy as np

from pateda import (
    UMDA, EBNA, BOA, LFDA, HBOA,
    SARTRE_EDA, BINOTEARS_EDA, PCBN_EDA, HSARTRE_EDA, HBINOTEARS_EDA,
)
from pateda.core.components import StatisticsMethod
from pateda.functions.discrete_binary.toy_functions.trap import trap_n
from pateda.functions.discrete_binary.toy_functions.deceptive import deceptive3
from pateda.functions.discrete_binary.toy_functions.checkerboard import checkerboard

import bn_eda_problems as P

# All Bayesian-network learners use max_parents=5 in this comparison.
MAX_PARENTS = 5


class _ProgressPrinter(StatisticsMethod):
    """Prints the best-so-far fitness at every generation while the EDA runs."""

    def __init__(self, tag: str):
        self.tag = tag
        self.best = -np.inf

    def collect(self, generation, population, fitness, model=None, **params):
        gen_best = float(np.max(np.asarray(fitness, dtype=float).reshape(-1)))
        if gen_best > self.best:
            self.best = gen_best
        print(f"    [{self.tag}] gen {generation:3d}  best_so_far={self.best:.4f}",
              flush=True)
        return {}


# --- problems ---------------------------------------------------------------
def make_problems():

    n_check = 100
    check = dict(
        name="Check (n=100)", n_vars=n_check, cardinality=2,
        fitness_func=lambda x: float(checkerboard(np.asarray(x, dtype=int), 10)),
        optimum=1000.0,
    )
    n_trap = 50
    trap5 = dict(
        name="Trap5 (n=50)", n_vars=n_trap, cardinality=2,
        fitness_func=lambda x: float(trap_n(np.asarray(x, dtype=int), 5)),
        optimum=50.0,
    )
    n_dec = 60
    dec3 = dict(
        name="Dec3 (n=60)", n_vars=n_dec, cardinality=2,
        fitness_func=lambda x: float(deceptive3(np.asarray(x, dtype=int))),
        optimum=20.0,
    )
    ubqp_prob = P.parse_problem("UBQP", 50)
    ubqp50 = dict(
        name="UBQP-50", n_vars=ubqp_prob.n_vars, cardinality=2,
        fitness_func=ubqp_prob.fitness_func, optimum=None,
    )
    ubqp_prob = P.parse_problem("UBQP", 100)
    ubqp100 = dict(
        name="UBQP-100", n_vars=ubqp_prob.n_vars, cardinality=2,
        fitness_func=ubqp_prob.fitness_func, optimum=None,
    )
    return [check,dec3,ubqp100,ubqp50,trap5]


# --- algorithms -------------------------------------------------------------
# extra kwargs per algorithm.  Every BN-based EDA uses max_parents=5; UMDA
# (univariate) has no such parameter.
_MP = {"max_parents": MAX_PARENTS}
ALGORITHMS = [
    ("UMDA", UMDA, {}),
    ("EBNA", EBNA, _MP),
    ("BOA", BOA, _MP),
    #("LFDA", LFDA, _MP),
    #("HBOA", HBOA, _MP),
    ("SARTRE_EDA", SARTRE_EDA, _MP),
    ("BINOTEARS_EDA", BINOTEARS_EDA, _MP),
    ("PCBN_EDA", PCBN_EDA, _MP),
    ("HSARTRE_EDA", HSARTRE_EDA, _MP),
    ("HBINOTEARS_EDA", HBINOTEARS_EDA, _MP),
]


def run_one(Cls, extra, prob, pop_size, n_gen, seed, tag):
    eda = Cls(n_vars=prob["n_vars"], cardinality=prob["cardinality"],
              fitness_func=prob["fitness_func"], pop_size=pop_size,
              n_gen=n_gen, random_seed=seed, **extra)
    # Print the best-so-far every generation while the run proceeds.
    eda._eda.components.statistics = _ProgressPrinter(tag)
    t0 = time.time()
    stats, _ = eda.run(verbose=False)
    return float(stats.best_fitness_overall), time.time() - t0


def main():
    n_gen = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    pop_size = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    seeds = list(range(1, n_seeds + 1))

    print(f"pop_size={pop_size}  n_gen={n_gen}  seeds={seeds}\n")
    problems = make_problems()

    for prob in problems:
        print(f"===== {prob['name']}"
              + (f"  (optimum {prob['optimum']:g})" if prob["optimum"] else "")
              + " =====")
        rows = []
        for name, Cls, extra in ALGORITHMS:
            bests, times, fails = [], [], 0
            for s in seeds:
                try:
                    tag = f"{prob['name'].split()[0]}|{name}|s{s}"
                    b, dt = run_one(Cls, extra, prob, pop_size, n_gen, s, tag)
                    bests.append(b); times.append(dt)
                except Exception as e:            # keep going on a single failure
                    fails += 1
                    print(f"  [warn] {name} seed {s}: {type(e).__name__}: {e}")
            if not bests:
                rows.append((name, np.nan, np.nan, np.nan, np.nan, fails))
                continue
            bests = np.array(bests)
            succ = (float(np.mean(bests >= prob["optimum"] - 1e-9))
                    if prob["optimum"] is not None else np.nan)
            rows.append((name, bests.mean(), bests.std(), succ,
                         float(np.mean(times)), fails))

        # rank by mean best (desc), print table
        order = sorted(range(len(rows)),
                       key=lambda i: (-rows[i][1] if not np.isnan(rows[i][1]) else 1e18))
        hdr = f"  {'rank':>4} {'algorithm':16} {'mean_best':>10} {'std':>8} {'succ':>6} {'time(s)':>8}"
        print(hdr)
        for r, i in enumerate(order, 1):
            name, mu, sd, succ, tm, fails = rows[i]
            succ_s = f"{succ:5.2f}" if not (isinstance(succ, float) and np.isnan(succ)) else "   --"
            flag = f"  (!{fails} failed)" if fails else ""
            print(f"  {r:>4} {name:16} {mu:10.3f} {sd:8.3f} {succ_s:>6} {tm:8.2f}{flag}")
        print()


if __name__ == "__main__":
    main()
