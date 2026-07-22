"""
Compare multi-objective EDA paradigms on a shared bi-objective problem.

Three approaches are compared:

* NSGA-II-style  : dominance + crowding-distance selection (CrowdingDistanceSelection)
* SMS-EMOA-style : hypervolume indicator-based selection (IndicatorBasedSelection)
* MOEA/D         : decomposition into scalar sub-problems (MOEAD driver)

All three build their offspring from a UMDA model. Each run's final
approximation set is scored with the hypervolume and IGD indicators from
``pateda.multiobjective.indicators`` (against the true Pareto front of the
instance), and the fronts are plotted.

Usage::

    python3 examples/multiobjective_variants_comparison.py [output.pdf]
"""

import sys
import numpy as np

from pateda.core.eda import EDA
from pateda.core.components import EDAComponents, CacheConfig
from pateda.seeding import RandomInit
from pateda.selection.crowding import CrowdingDistanceSelection
from pateda.selection.indicator_based import IndicatorBasedSelection
from pateda.selection.truncation import TruncationSelection
from pateda.learning.umda import LearnUMDA
from pateda.sampling.fda import SampleFDA
from pateda.replacement.elitist import ElitistReplacement
from pateda.stop_conditions import MaxGenerations
from pateda.multiobjective.moead import MOEAD
from pateda.multiobjective.indicators import hypervolume, igd
from pateda.selection.utils.pareto import find_pareto_set


N_VARS = 20
POP = 200
N_GEN = 25
N_WEIGHTS = 80
SEED = 1


def make_instance(n_vars, seed):
    """A bi-objective 0/1 problem with two conflicting linear objectives."""
    rng = np.random.default_rng(seed)
    w1 = rng.uniform(0.2, 1.0, size=n_vars)
    w2 = rng.uniform(0.2, 1.0, size=n_vars)

    def f(x):
        return np.array([float(x @ w1), float(x @ w2)])

    return f, w1, w2


def true_front(w1, w2):
    """The true Pareto front, obtained by scalarizing over many weightings.

    For a linear objective ``lam*w1 + (1-lam)*w2`` over ``{0,1}^n`` the optimum is
    ``x_i = 1`` iff the per-variable score is positive, so sweeping ``lam`` traces
    the exact front.
    """
    n = len(w1)
    pts = []
    for lam in np.linspace(0, 1, 200):
        score = lam * w1 + (1 - lam) * w2
        x = (score > 0).astype(int)
        pts.append([x @ w1, x @ w2])
    pts.append([np.ones(n) @ w1, np.ones(n) @ w2])
    pts.append([0.0, 0.0])
    pts = np.array(pts)
    idx = find_pareto_set(pts)
    return pts[idx]


def run_selection_eda(selection, f, cardinality):
    comp = EDAComponents(
        seeding=RandomInit(),
        selection=selection,
        learning=LearnUMDA(alpha=1.0),
        sampling=SampleFDA(n_samples=POP),
        replacement=ElitistReplacement(n_elite=1),
        stop_condition=MaxGenerations(N_GEN),
    )
    eda = EDA(POP, N_VARS, f, cardinality, comp, random_seed=SEED)
    _, cache = eda.run(
        cache_config=CacheConfig(cache_populations=True, cache_fitness=True),
        verbose=False,
    )
    pop = np.vstack(cache.populations)
    fit = np.vstack(cache.fitness_values)
    idx = find_pareto_set(fit)
    return fit[idx]


def run_moead(f, cardinality):
    comp = EDAComponents(
        seeding=RandomInit(),
        selection=TruncationSelection(ratio=0.5),   # unused by MOEA/D but required
        learning=LearnUMDA(alpha=1.0),
        sampling=SampleFDA(n_samples=1),
        stop_condition=MaxGenerations(N_GEN),
    )
    driver = MOEAD(
        n_vars=N_VARS, cardinality=cardinality, fitness_func=f, components=comp,
        n_obj=2, n_weights=N_WEIGHTS, neighbourhood_size=15, n_gen=N_GEN,
        maximize=True, random_seed=SEED,
    )
    res = driver.run(verbose=False)
    return res.pareto_objectives


def main(out_path=None):
    f, w1, w2 = make_instance(N_VARS, SEED)
    cardinality = np.full(N_VARS, 2)
    pf = true_front(w1, w2)
    ref = np.array([0.0, 0.0])   # objectives are non-negative and maximized

    fronts = {
        "NSGA-II (crowding)": run_selection_eda(
            CrowdingDistanceSelection(n_select=POP // 2), f, cardinality),
        "SMS-EMOA (indicator)": run_selection_eda(
            IndicatorBasedSelection(n_select=POP // 2), f, cardinality),
        "MOEA/D": run_moead(f, cardinality),
    }

    print(f"{'approach':22s} {'HV':>12s} {'IGD':>10s}  #front")
    for name, front in fronts.items():
        hv = hypervolume(front, reference=ref, maximize=True)
        ind = igd(front, pf, maximize=True)
        print(f"{name:22s} {hv:12.2f} {ind:10.4f}  {len(front):5d}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(pf[:, 0], pf[:, 1], c="lightgray", s=60, label="true front", zorder=1)
        markers = {"NSGA-II (crowding)": "o", "SMS-EMOA (indicator)": "s", "MOEA/D": "^"}
        for name, front in fronts.items():
            order = np.argsort(front[:, 0])
            ax.plot(front[order, 0], front[order, 1], markers[name] + "-",
                    ms=5, lw=1, label=name, zorder=2)
        ax.set_xlabel("objective 1 (maximize)")
        ax.set_ylabel("objective 2 (maximize)")
        ax.legend(fontsize=8)
        path = out_path or "mo_variants_comparison.pdf"
        fig.savefig(path, bbox_inches="tight")
        print("wrote", path)
    except Exception as exc:  # pragma: no cover
        print("plot skipped:", exc)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
