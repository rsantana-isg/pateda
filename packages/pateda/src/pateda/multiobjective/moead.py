"""
MOEA/D: a decomposition-based driver that works with *any* pateda model.

This is the decomposition-based counterpart of :class:`pateda.core.eda.EDA`.
Instead of a single global selection/learning/sampling cycle, it decomposes the
problem into ``N`` scalar sub-problems (one weight vector each) and optimises
them cooperatively, exploiting neighbourhood structure (Zhang & Li, 2007).

The novelty here is that reproduction is delegated to pateda's existing
``LearningMethod`` / ``SamplingMethod`` components, exactly as in the standard
EDA engine.  Any probabilistic model already available in pateda (UMDA, Tree,
EBNA, MN-FDA, Gaussian, mixtures, ...) therefore plugs straight in -- this is
the EDA realisation of MOEA/D-GM (probabilistic graphical models inside
MOEA/D).  Two reproduction scopes are offered:

* ``"neighbourhood"`` (default) -- a fresh model is learnt from each
  sub-problem's mating pool (its neighbourhood) and one offspring is sampled.
  This specialises the model to each region of the Pareto front.
* ``"global"`` -- one model is learnt per generation from the whole set of
  current solutions and used to sample one offspring per sub-problem.  Cheaper
  (a single learning step per generation) and closer to a decomposition-guided
  EDA.

The driver is representation-agnostic: pass a 1-D ``cardinality`` array for
discrete problems or a ``(2, n_vars)`` bounds array for continuous ones.  (As
requested, permutation problems are handled in a separate derived package and
are out of scope here.)
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np

from pateda.core.components import EDAComponents
from pateda.multiobjective.scalarization import scalarize
from pateda.multiobjective.weights import generate_weights, weight_neighbourhoods
from pateda.multiobjective.archive import ParetoArchive
from pateda.multiobjective.indicators import hypervolume

__all__ = ["MOEAD", "MOEADResult"]


@dataclass
class MOEADResult:
    """Outcome of a MOEA/D run."""

    pareto_solutions: np.ndarray
    pareto_objectives: np.ndarray
    population: np.ndarray
    obj_values: np.ndarray
    ideal_point: np.ndarray
    archive_size_history: List[int] = field(default_factory=list)
    hypervolume_history: List[float] = field(default_factory=list)


class MOEAD:
    """Decomposition-based multi-objective EDA driver.

    Args:
        n_vars: Number of variables.
        cardinality: 1-D cardinalities (discrete) or ``(2, n_vars)`` bounds
            (continuous).
        fitness_func: Maps an individual to an objective vector of length ``m``.
        components: An :class:`EDAComponents` providing ``learning`` and
            ``sampling`` (and optionally ``seeding``, ``repairing``,
            ``mutation``).  Selection / replacement / stop_condition are unused.
        n_obj: Number of objectives.
        n_weights: Number of weight vectors / sub-problems.
        neighbourhood_size: Neighbourhood size ``T``.
        scalarization: ``"tchebycheff"``, ``"weighted_sum"`` or ``"pbi"``.
        maximize: Direction of optimisation.
        n_gen: Number of generations.
        nr: Maximum number of neighbours a single offspring may replace.
        delta: Probability of drawing the mating pool from the neighbourhood
            (otherwise the whole population is used).
        model_scope: ``"neighbourhood"`` or ``"global"`` (see module docstring).
        theta: PBI penalty parameter (ignored for other scalarizations).
        archive_capacity: Optional cap on the external archive size.
        hv_reference: Optional reference point to track hypervolume per
            generation.
        random_seed: Seed for reproducibility.
    """

    def __init__(
        self,
        n_vars: int,
        cardinality: Union[np.ndarray, List],
        fitness_func: Callable,
        components: EDAComponents,
        n_obj: int = 2,
        n_weights: int = 100,
        neighbourhood_size: int = 20,
        scalarization: str = "tchebycheff",
        maximize: bool = True,
        n_gen: int = 100,
        nr: int = 2,
        delta: float = 0.9,
        model_scope: str = "neighbourhood",
        theta: float = 5.0,
        archive_capacity: Optional[int] = None,
        hv_reference: Optional[np.ndarray] = None,
        random_seed: Optional[int] = None,
    ):
        self.n_vars = n_vars
        self.cardinality = np.asarray(cardinality)
        self.fitness_func = fitness_func
        self.components = components
        self.n_obj = n_obj
        self.n_weights = n_weights
        self.neighbourhood_size = neighbourhood_size
        self.scalarization = scalarization
        self.maximize = maximize
        self.n_gen = n_gen
        self.nr = nr
        self.delta = delta
        if model_scope not in ("neighbourhood", "global"):
            raise ValueError("model_scope must be 'neighbourhood' or 'global'")
        self.model_scope = model_scope
        self.theta = theta
        self.archive_capacity = archive_capacity
        self.hv_reference = hv_reference
        self.rng = np.random.default_rng(random_seed)

        self._continuous = self.cardinality.ndim == 2

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _seed_population(self, n: int) -> np.ndarray:
        """Create an initial population of ``n`` solutions."""
        if self.components.seeding is not None:
            return self.components.seeding.seed(
                self.n_vars, n, self.cardinality, rng=self.rng,
                **self.components.seeding_params,
            )
        if self._continuous:
            lo, hi = self.cardinality[0], self.cardinality[1]
            return self.rng.uniform(lo, hi, size=(n, self.n_vars))
        pop = np.empty((n, self.n_vars), dtype=int)
        for v in range(self.n_vars):
            pop[:, v] = self.rng.integers(0, int(self.cardinality[v]), size=n)
        return pop

    def _evaluate(self, individual: np.ndarray) -> np.ndarray:
        return np.atleast_1d(np.asarray(self.fitness_func(individual), dtype=float))

    def _update_ideal(self, obj_values: np.ndarray) -> None:
        if self.ideal_point is None:
            self.ideal_point = obj_values.copy()
        elif self.maximize:
            self.ideal_point = np.maximum(self.ideal_point, obj_values)
        else:
            self.ideal_point = np.minimum(self.ideal_point, obj_values)

    def _scalar(self, obj_values: np.ndarray, weight: np.ndarray) -> float:
        return scalarize(
            obj_values, weight, ideal=self.ideal_point,
            method=self.scalarization, maximize=self.maximize, theta=self.theta,
        )

    def _learn(self, generation: int, pool_pop: np.ndarray, pool_fit: np.ndarray):
        return self.components.learning.learn(
            generation, self.n_vars, self.cardinality,
            pool_pop, pool_fit, **self.components.learning_params,
        )

    def _sample_one(self, model, pool_pop: np.ndarray, pool_fit: np.ndarray) -> np.ndarray:
        params = dict(self.components.sampling_params)
        params["n_samples"] = 1
        children = self.components.sampling.sample(
            self.n_vars, model, self.cardinality,
            aux_pop=pool_pop, aux_fitness=pool_fit, rng=self.rng, **params,
        )
        child = np.atleast_2d(children)[0]
        if self.components.repairing is not None:
            child = self.components.repairing.repair(
                child.reshape(1, -1), self.cardinality,
                **self.components.repairing_params,
            )[0]
        if self.components.mutation is not None:
            child = self.components.mutation.mutate(
                self.n_vars, self.cardinality, child.reshape(1, -1),
                **self.components.mutation_params,
            )[0]
        return child

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, verbose: bool = False) -> MOEADResult:
        """Execute MOEA/D and return a :class:`MOEADResult`."""
        # 1. Weights, neighbourhoods, initial population
        self.weights = generate_weights(self.n_obj, self.n_weights)
        n = len(self.weights)
        self.neighbours = weight_neighbourhoods(self.weights, self.neighbourhood_size)

        self.population = self._seed_population(n)
        self.obj_values = np.array([self._evaluate(ind) for ind in self.population])

        self.ideal_point = None
        for ov in self.obj_values:
            self._update_ideal(ov)

        archive = ParetoArchive(maximize=self.maximize, capacity=self.archive_capacity)
        archive.add_population(self.population, self.obj_values)

        result = MOEADResult(
            pareto_solutions=np.array([]), pareto_objectives=np.array([]),
            population=self.population, obj_values=self.obj_values,
            ideal_point=self.ideal_point,
        )

        # 2. Generations
        for gen in range(self.n_gen):
            global_model = None
            if self.model_scope == "global":
                global_model = self._learn(gen, self.population, self.obj_values)

            order = self.rng.permutation(n)
            for i in order:
                use_neighbourhood = self.rng.random() < self.delta
                pool_idx = self.neighbours[i] if use_neighbourhood else np.arange(n)

                if self.model_scope == "global":
                    child = self._sample_one(
                        global_model, self.population[pool_idx],
                        self.obj_values[pool_idx],
                    )
                else:
                    model = self._learn(
                        gen, self.population[pool_idx], self.obj_values[pool_idx]
                    )
                    child = self._sample_one(
                        model, self.population[pool_idx], self.obj_values[pool_idx]
                    )

                child_obj = self._evaluate(child)
                self._update_ideal(child_obj)

                # Update at most nr neighbours whose sub-problem the child improves
                replace_idx = self.rng.permutation(pool_idx)
                updated = 0
                for j in replace_idx:
                    if updated >= self.nr:
                        break
                    if self._scalar(child_obj, self.weights[j]) <= self._scalar(
                        self.obj_values[j], self.weights[j]
                    ):
                        self.population[j] = np.array(child).copy()
                        self.obj_values[j] = child_obj.copy()
                        updated += 1

                archive.add(child, child_obj)

            result.archive_size_history.append(archive.size)
            if self.hv_reference is not None:
                _, objs = archive.get_front()
                hv = hypervolume(objs, self.hv_reference, self.maximize) if objs.size else 0.0
                result.hypervolume_history.append(hv)

            if verbose and (gen + 1) % max(1, self.n_gen // 10) == 0:
                msg = f"MOEA/D gen {gen + 1}/{self.n_gen}  archive={archive.size}"
                if result.hypervolume_history:
                    msg += f"  HV={result.hypervolume_history[-1]:.4f}"
                print(msg)

        sols, objs = archive.get_front()
        result.pareto_solutions = sols
        result.pareto_objectives = objs
        result.population = self.population
        result.obj_values = self.obj_values
        result.ideal_point = self.ideal_point
        return result
