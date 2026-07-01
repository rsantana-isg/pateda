"""
Single-experiment runner for multi-objective EDAs (cluster-friendly).

Positional arguments only, seed first, so it can be driven from SLURM launchers.
Runs one (approach, model) configuration on one discrete multi-objective
problem and prints the configuration plus the final hypervolume.

Usage::

    python run_mo_eda.py SEED APPROACH PROBLEM MODEL N_VARS POP_SIZE N_GEN [SCALARIZATION]

where
    SEED          integer random seed
    APPROACH      pareto | indicator | decomposition
    PROBLEM       onemax_zeromax | deceptive | mubqp
    MODEL         umda | tree | ebna
    N_VARS        number of binary variables
    POP_SIZE      population size (== number of weights for decomposition)
    N_GEN         number of generations
    SCALARIZATION optional, decomposition only: tchebycheff | weighted_sum | pbi

Example::

    python run_mo_eda.py 42 decomposition deceptive umda 20 200 40 tchebycheff
"""

import sys
import numpy as np

from pateda.core.eda import EDA
from pateda.core.components import EDAComponents
from pateda.seeding import RandomInit
from pateda.learning import LearnUMDA, LearnTreeModel, LearnEBNA
from pateda.sampling import SampleFDA, SampleBayesianNetwork
from pateda.selection import CrowdingDistanceSelection, IndicatorBasedSelection
from pateda.stop_conditions.max_generations import MaxGenerations

from pateda.functions.discrete.multiobjective import (
    make_mo_onemax_zeromax, make_mo_deceptive, make_mubqp,
)
from pateda.multiobjective import MOEAD, hypervolume, find_pareto_set


def build_problem(name, n_vars, seed):
    if name == "onemax_zeromax":
        return make_mo_onemax_zeromax(n_vars)
    if name == "deceptive":
        return make_mo_deceptive(n_vars, block_size=5)
    if name == "mubqp":
        f, _ = make_mubqp(n_vars, n_objectives=2, density=0.4, rho=-0.2, seed=seed)
        return f
    raise ValueError(f"Unknown problem '{name}'")


def build_model(model_name, pop_size):
    if model_name == "umda":
        return LearnUMDA(alpha=1.0), SampleFDA(n_samples=pop_size)
    if model_name == "tree":
        return LearnTreeModel(alpha=1.0), SampleFDA(n_samples=pop_size)
    if model_name == "ebna":
        return LearnEBNA(), SampleBayesianNetwork(n_samples=pop_size)
    raise ValueError(f"Unknown model '{model_name}'")


def main(argv):
    if len(argv) < 7:
        print(__doc__)
        sys.exit(1)

    seed = int(argv[0])
    approach = argv[1]
    problem = argv[2]
    model = argv[3]
    n_vars = int(argv[4])
    pop_size = int(argv[5])
    n_gen = int(argv[6])
    scalarization = argv[7] if len(argv) > 7 else "tchebycheff"

    print(f"Seed:             {seed}")
    print(f"Approach:         {approach}")
    print(f"Problem:          {problem}")
    print(f"Model:            {model}")
    print(f"Number of vars:   {n_vars}")
    print(f"Population Size:  {pop_size}")
    print(f"Generations:      {n_gen}")
    if approach == "decomposition":
        print(f"Scalarization:    {scalarization}")

    problem_f = build_problem(problem, n_vars, seed)
    card = np.full(n_vars, 2)
    learner, sampler = build_model(model, pop_size)

    if approach == "pareto":
        comp = EDAComponents(
            seeding=RandomInit(),
            selection=CrowdingDistanceSelection(ratio=0.5, maximize=True),
            learning=learner, sampling=sampler,
            stop_condition=MaxGenerations(n_gen))
        eda = EDA(pop_size, n_vars, problem_f, card, comp, random_seed=seed)
        eda.run(verbose=False)
        fitness = eda.fitness
    elif approach == "indicator":
        comp = EDAComponents(
            seeding=RandomInit(),
            selection=IndicatorBasedSelection(ratio=0.5, maximize=True,
                                              indicator="epsilon"),
            learning=learner, sampling=sampler,
            stop_condition=MaxGenerations(n_gen))
        eda = EDA(pop_size, n_vars, problem_f, card, comp, random_seed=seed)
        eda.run(verbose=False)
        fitness = eda.fitness
    elif approach == "decomposition":
        comp = EDAComponents(
            seeding=RandomInit(), selection=CrowdingDistanceSelection(),
            learning=learner, sampling=sampler,
            stop_condition=MaxGenerations(n_gen))
        moead = MOEAD(
            n_vars, card, problem_f, comp, n_obj=2, n_weights=pop_size,
            neighbourhood_size=max(5, pop_size // 10),
            scalarization=scalarization, maximize=True, n_gen=n_gen,
            model_scope="neighbourhood", random_seed=seed)
        res = moead.run(verbose=False)
        fitness = res.pareto_objectives
    else:
        raise ValueError(f"Unknown approach '{approach}'")

    fitness = np.atleast_2d(fitness)
    idx = find_pareto_set(fitness, maximize=True, return_mask=False)
    front = fitness[idx]
    ref = front.min(axis=0) - 0.1 * (front.max(axis=0) - front.min(axis=0) + 1.0)
    hv = hypervolume(front, ref, maximize=True)

    print(f"Front size:       {len(front)}")
    print(f"Hypervolume:      {hv:.4f}")


if __name__ == "__main__":
    main(sys.argv[1:])
