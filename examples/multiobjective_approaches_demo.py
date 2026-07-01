"""
Demonstration of the three multi-objective optimisation paradigms in pateda.

This script runs, on the same discrete benchmark, the three classical
multi-objective strategies -- all on top of pateda probabilistic models -- and
compares the resulting Pareto-front approximations with the hypervolume
indicator:

* **Pareto-based**        -> standard EDA + ``CrowdingDistanceSelection`` (NSGA-II
                             style non-dominated sorting with crowding distance).
* **Indicator-based**     -> standard EDA + ``IndicatorBasedSelection`` (IBEA with
                             the additive epsilon indicator, or SMS-EMOA-style
                             hypervolume contribution).
* **Decomposition-based** -> the ``MOEAD`` driver, which reuses the *same*
                             learning/sampling components, so any pateda model
                             works inside MOEA/D.

The same probabilistic model (e.g. UMDA, Tree-EDA) is plugged into every
paradigm, illustrating the model-agnostic design.

Usage::

    python multiobjective_approaches_demo.py --problem deceptive --model umda \\
        --n_vars 20 --pop_size 200 --n_gen 30 --seed 42 [--plot]
"""

import argparse
import numpy as np

from pateda.core.eda import EDA
from pateda.core.components import EDAComponents
from pateda.seeding import RandomInit
from pateda.learning import LearnUMDA, LearnTreeModel, LearnEBNA
from pateda.sampling import SampleFDA, SampleBayesianNetwork
from pateda.selection import CrowdingDistanceSelection, IndicatorBasedSelection
from pateda.stop_conditions.max_generations import MaxGenerations

from pateda.functions.discrete.multiobjective import (
    make_mo_onemax_zeromax,
    make_mo_deceptive,
    make_mubqp,
    mo_pareto_front_onemax_zeromax,
)
from pateda.multiobjective import (
    MOEAD,
    hypervolume,
    find_pareto_set,
    igd,
)


# --------------------------------------------------------------------------
# Problem / model factories
# --------------------------------------------------------------------------

def build_problem(name, n_vars, seed):
    """Return (fitness_func, reference_front_or_None) for a benchmark."""
    if name == "onemax_zeromax":
        return make_mo_onemax_zeromax(n_vars), mo_pareto_front_onemax_zeromax(n_vars)
    if name == "deceptive":
        return make_mo_deceptive(n_vars, block_size=5), None
    if name == "mubqp":
        f, _ = make_mubqp(n_vars, n_objectives=2, density=0.4, rho=-0.2, seed=seed)
        return f, None
    raise ValueError(f"Unknown problem '{name}'")


def build_model(model_name, pop_size, alpha=1.0):
    """Return (learning_method, sampling_method) for the requested model."""
    if model_name == "umda":
        return LearnUMDA(alpha=alpha), SampleFDA(n_samples=pop_size)
    if model_name == "tree":
        return LearnTreeModel(alpha=alpha), SampleFDA(n_samples=pop_size)
    if model_name == "ebna":
        return LearnEBNA(), SampleBayesianNetwork(n_samples=pop_size)
    raise ValueError(f"Unknown model '{model_name}'")


# --------------------------------------------------------------------------
# Front extraction / quality
# --------------------------------------------------------------------------

def front_of(fitness, maximize=True):
    """Non-dominated objective vectors of a fitness array."""
    fitness = np.atleast_2d(fitness)
    idx = find_pareto_set(fitness, maximize=maximize, return_mask=False)
    return fitness[idx]


def run_pareto(problem_f, n_vars, card, pop_size, n_gen, model_name, seed):
    learner, sampler = build_model(model_name, pop_size)
    comp = EDAComponents(
        seeding=RandomInit(),
        selection=CrowdingDistanceSelection(ratio=0.5, maximize=True),
        learning=learner, sampling=sampler,
        stop_condition=MaxGenerations(n_gen),
    )
    eda = EDA(pop_size, n_vars, problem_f, card, comp, random_seed=seed)
    stats, cache = eda.run(verbose=False)
    return front_of(eda.fitness)


def run_indicator(problem_f, n_vars, card, pop_size, n_gen, model_name, seed,
                  indicator="epsilon"):
    learner, sampler = build_model(model_name, pop_size)
    comp = EDAComponents(
        seeding=RandomInit(),
        selection=IndicatorBasedSelection(ratio=0.5, maximize=True,
                                          indicator=indicator),
        learning=learner, sampling=sampler,
        stop_condition=MaxGenerations(n_gen),
    )
    eda = EDA(pop_size, n_vars, problem_f, card, comp, random_seed=seed)
    stats, cache = eda.run(verbose=False)
    return front_of(eda.fitness)


def run_decomposition(problem_f, n_vars, card, pop_size, n_gen, model_name, seed,
                      scalarization="tchebycheff", model_scope="neighbourhood"):
    learner, sampler = build_model(model_name, pop_size)
    comp = EDAComponents(
        seeding=RandomInit(),
        selection=CrowdingDistanceSelection(),  # unused by MOEA/D
        learning=learner, sampling=sampler,
        stop_condition=MaxGenerations(n_gen),
    )
    moead = MOEAD(
        n_vars, card, problem_f, comp, n_obj=2, n_weights=pop_size,
        neighbourhood_size=max(5, pop_size // 10), scalarization=scalarization,
        maximize=True, n_gen=n_gen, model_scope=model_scope, random_seed=seed,
    )
    res = moead.run(verbose=False)
    return res.pareto_objectives if res.pareto_objectives.size else np.empty((0, 2))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", default="deceptive",
                        choices=["onemax_zeromax", "deceptive", "mubqp"])
    parser.add_argument("--model", default="umda",
                        choices=["umda", "tree", "ebna"])
    parser.add_argument("--n_vars", type=int, default=20)
    parser.add_argument("--pop_size", type=int, default=200)
    parser.add_argument("--n_gen", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", action="store_true",
                        help="Save a Pareto-front comparison figure (.pdf).")
    args = parser.parse_args()

    print(f"Seed:             {args.seed}")
    print(f"Problem:          {args.problem}")
    print(f"Model:            {args.model}")
    print(f"Number of vars:   {args.n_vars}")
    print(f"Population Size:  {args.pop_size}")
    print(f"Generations:      {args.n_gen}")
    print()

    problem_f, ref_front = build_problem(args.problem, args.n_vars, args.seed)
    card = np.full(args.n_vars, 2)

    fronts = {}
    fronts["Pareto (NSGA-II)"] = run_pareto(
        problem_f, args.n_vars, card, args.pop_size, args.n_gen, args.model, args.seed)
    fronts["Indicator (IBEA)"] = run_indicator(
        problem_f, args.n_vars, card, args.pop_size, args.n_gen, args.model, args.seed)
    fronts["Decomposition (MOEA/D)"] = run_decomposition(
        problem_f, args.n_vars, card, args.pop_size, args.n_gen, args.model, args.seed)

    # Common reference point: nadir of all fronts, pushed slightly worse.
    all_pts = np.vstack([f for f in fronts.values() if f.size]) if any(
        f.size for f in fronts.values()) else np.zeros((1, 2))
    ref = all_pts.min(axis=0) - 0.1 * (all_pts.max(axis=0) - all_pts.min(axis=0) + 1.0)

    print(f"{'Approach':<26}{'|front|':>9}{'Hypervolume':>16}" +
          ("{:>14}".format("IGD") if ref_front is not None else ""))
    print("-" * (51 + (14 if ref_front is not None else 0)))
    for name, front in fronts.items():
        hv = hypervolume(front, ref, maximize=True) if front.size else 0.0
        line = f"{name:<26}{len(front):>9}{hv:>16.3f}"
        if ref_front is not None:
            ig = igd(front, ref_front, maximize=True) if front.size else float("inf")
            line += f"{ig:>14.4f}"
        print(line)

    if args.plot:
        _plot(fronts, ref_front, args)


def _plot(fronts, ref_front, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[plot skipped: matplotlib not available]")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    markers = {"Pareto (NSGA-II)": "o", "Indicator (IBEA)": "s",
               "Decomposition (MOEA/D)": "^"}
    if ref_front is not None:
        order = np.argsort(ref_front[:, 0])
        ax.plot(ref_front[order, 0], ref_front[order, 1], "k--",
                lw=1, alpha=0.5, label="True front")
    for name, front in fronts.items():
        if front.size:
            ax.scatter(front[:, 0], front[:, 1], s=40,
                       marker=markers.get(name, "o"), label=name)
    ax.set_xlabel("$f_1$", fontsize=14)
    ax.set_ylabel("$f_2$", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=11)
    fig.tight_layout()
    out = f"mo_fronts_{args.problem}_{args.model}.pdf"
    fig.savefig(out)
    print(f"\nFigure saved to {out}")


if __name__ == "__main__":
    main()
