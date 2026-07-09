"""
Bi-objective quasiparticle braid approximation.

The single-objective braid problem trades approximation error against braid
length through a scalar ``lambda`` (see
:mod:`pateda.functions.discrete_non_binary.problems.braid`).  Here the two goals
are kept explicit as a bi-objective problem:

    minimise  epsilon(x) = |B(x) - T|        (approximation error)
    minimise  l(x)                            (number of matrices in the braid)

where ``l`` is either the raw braid length or the *effective* length ``elen``
after cancelling adjacent inverse pairs.  The two objectives conflict: shorter
braids generally approximate the target worse.

To match pateda's ``maximize=True`` multi-objective convention, the objective
callable returns a 2-vector to be **maximised**::

    (  1 / (1 + epsilon) ,  1 / l  )

Both components lie in ``(0, 1]`` and reach 1 for, respectively, a perfect
approximation and a length-1 braid.  Helper functions expose the raw
``(epsilon, l)`` pair for reporting the true Pareto front.
"""

from typing import Callable, Optional
import numpy as np

from pateda.functions.discrete_non_binary.problems.braid import (
    BraidProblem,
    make_fibonacci_braid_problem,
    make_icosahedral_benchmark_problem,
    braid_error,
    effective_length,
)


def braid_raw_objectives(x: np.ndarray, problem: BraidProblem,
                         use_effective_length: bool = True,
                         phase_invariant: bool = False) -> np.ndarray:
    """Return the raw ``(epsilon, length)`` objective pair (both minimised)."""
    eps = braid_error(x, problem.generators, problem.target, phase_invariant=phase_invariant)
    if use_effective_length:
        length = effective_length(x, problem.inverse_index)
    else:
        length = len(np.asarray(x).ravel())
    return np.array([eps, float(max(1, length))])


def braid_biobjective(x: np.ndarray, problem: BraidProblem,
                      use_effective_length: bool = True,
                      phase_invariant: bool = False) -> np.ndarray:
    """Maximised 2-vector ``(1/(1+epsilon), 1/length)`` for one solution."""
    eps, length = braid_raw_objectives(x, problem, use_effective_length, phase_invariant)
    return np.array([1.0 / (1.0 + eps), 1.0 / length])


def create_braid_biobjective_function(problem: BraidProblem,
                                      use_effective_length: bool = True,
                                      phase_invariant: bool = False
                                      ) -> Callable[[np.ndarray], np.ndarray]:
    """Create a bi-objective braid fitness function for multi-objective EDAs.

    Objective 1 rewards low approximation error, objective 2 rewards short
    braids; both are maximised.  Accepts a single individual (1-D -> 2-vector)
    or a population (2-D -> ``(pop, 2)`` array).
    """
    def objective(population: np.ndarray) -> np.ndarray:
        population = np.asarray(population)
        if population.ndim == 1:
            return braid_biobjective(population, problem, use_effective_length, phase_invariant)
        out = np.empty((population.shape[0], 2))
        for i in range(population.shape[0]):
            out[i, :] = braid_biobjective(population[i], problem,
                                          use_effective_length, phase_invariant)
        return out

    return objective


def make_fibonacci_braid_biobjective(target: np.ndarray, n_matrices: int,
                                     use_effective_length: bool = True,
                                     phase_invariant: bool = False):
    """Convenience builder: Fibonacci-anyon bi-objective braid problem.

    Returns ``(objective_func, problem)``.
    """
    problem = make_fibonacci_braid_problem(target, n_matrices)
    return create_braid_biobjective_function(
        problem, use_effective_length=use_effective_length,
        phase_invariant=phase_invariant), problem


def make_icosahedral_braid_biobjective(target_index: int, n_matrices: int,
                                       use_effective_length: bool = True,
                                       phase_invariant: bool = False,
                                       instances_dir: Optional[str] = None):
    """Convenience builder: bi-objective braid for an icosahedral target.

    Returns ``(objective_func, problem)``.
    """
    problem = make_icosahedral_benchmark_problem(target_index, n_matrices,
                                                 instances_dir=instances_dir)
    return create_braid_biobjective_function(
        problem, use_effective_length=use_effective_length,
        phase_invariant=phase_invariant), problem
