"""
MOEA/D: Multi-Objective Evolutionary Algorithm Based on Decomposition.

A Python implementation for permutation-based multi-objective problems,
using Mallows models or simple crossover operators for solution generation.

The algorithm decomposes the multi-objective problem into a set of single-
objective subproblems using weight vectors, and optimises them simultaneously
by exploiting neighbourhood structure.

References:
    [1] Q. Zhang, H. Li. "MOEA/D: A Multiobjective Evolutionary Algorithm
        Based on Decomposition". IEEE TEVC, 2007.
    [2] J. Ceberio et al. "Multi-objective decomposition-based Mallows Models".
"""

import numpy as np
from typing import List, Callable, Optional, Tuple, Dict, Any

from src.multiobjective.scalarization import weighted_sum, tchebycheff
from src.multiobjective.pareto import ParetoArchive, dominates
from src.distributions.mallows import MallowsKendall
from src.utils.permutations import random_permutation


class MOEAD:
    """MOEA/D framework for permutation-based multi-objective optimisation.

    This implementation supports:
    - Uniform weight vector generation for 2 objectives
    - Neighbourhood-based mating selection
    - Mallows model or crossover-based reproduction
    - Weighted sum or Tchebycheff scalarization
    """

    def __init__(
        self,
        objectives: List[Callable],
        n: int,
        n_subproblems: int = 50,
        neighbourhood_size: int = 10,
        scalarization: str = "tchebycheff",
        minimize: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Args:
            objectives: List of objective functions, each taking a
                        0-indexed permutation and returning a float.
            n: Permutation size.
            n_subproblems: Number of subproblems (weight vectors).
            neighbourhood_size: Size of each subproblem's neighbourhood.
            scalarization: "weighted_sum" or "tchebycheff".
            minimize: If True, objectives are minimised.
            seed: Random seed.
        """
        self.objectives = objectives
        self.n_obj = len(objectives)
        self.n = n
        self.n_subproblems = n_subproblems
        self.neighbourhood_size = min(neighbourhood_size, n_subproblems)
        self.scalarization = scalarization
        self.minimize = minimize
        self.rng = np.random.default_rng(seed)

        # Initialised by run()
        self.weights = None
        self.neighbours = None
        self.population = None
        self.obj_values = None
        self.reference_point = None
        self.archive = ParetoArchive(minimize=minimize)

    def _generate_weights(self) -> np.ndarray:
        """Generate uniform weight vectors for 2 objectives."""
        if self.n_obj == 2:
            w = np.linspace(0, 1, self.n_subproblems)
            return np.column_stack([w, 1.0 - w])
        else:
            # For m > 2, use random weights on the simplex
            raw = self.rng.dirichlet(np.ones(self.n_obj), self.n_subproblems)
            return raw

    def _compute_neighbourhoods(self):
        """Compute neighbourhood indices for each subproblem."""
        N = self.n_subproblems
        T = self.neighbourhood_size
        dists = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                dists[i, j] = np.linalg.norm(self.weights[i] - self.weights[j])
        self.neighbours = np.argsort(dists, axis=1)[:, :T]

    def _evaluate(self, perm: np.ndarray) -> np.ndarray:
        """Evaluate all objectives for a permutation."""
        return np.array([f(perm) for f in self.objectives])

    def _scalar_value(self, obj_vals: np.ndarray, weight: np.ndarray) -> float:
        """Compute scalarised fitness."""
        if self.scalarization == "tchebycheff":
            return tchebycheff(obj_vals, weight, self.reference_point, self.minimize)
        else:
            return weighted_sum(obj_vals, weight, self.reference_point, self.minimize)

    def _update_reference_point(self, obj_vals: np.ndarray):
        """Update the ideal reference point."""
        if self.reference_point is None:
            self.reference_point = obj_vals.copy()
        else:
            if self.minimize:
                self.reference_point = np.minimum(self.reference_point, obj_vals)
            else:
                self.reference_point = np.maximum(self.reference_point, obj_vals)

    def _initialize_population(self):
        """Create random initial population."""
        self.population = np.array([
            self.rng.permutation(self.n) for _ in range(self.n_subproblems)
        ])
        self.obj_values = np.array([
            self._evaluate(p) for p in self.population
        ])
        for ov in self.obj_values:
            self._update_reference_point(ov)

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Order crossover (OX) for permutations.

        Selects a random segment from parent1 and fills remaining
        positions using the order from parent2.
        """
        n = self.n
        child = -np.ones(n, dtype=int)
        start = self.rng.integers(0, n)
        end = self.rng.integers(start + 1, n + 1)
        child[start:end] = parent1[start:end]

        fill_values = [v for v in parent2 if v not in child[start:end]]
        pos = 0
        for i in range(n):
            if child[i] == -1:
                child[i] = fill_values[pos]
                pos += 1
        return child

    def _mutate(self, perm: np.ndarray, prob: float = 0.1) -> np.ndarray:
        """Swap mutation: swap two random positions with given probability."""
        if self.rng.random() < prob:
            result = perm.copy()
            i, j = self.rng.choice(self.n, size=2, replace=False)
            result[i], result[j] = result[j], result[i]
            return result
        return perm.copy()

    def run(
        self,
        n_generations: int,
        use_mallows: bool = False,
        mutation_prob: float = 0.1,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Execute the MOEA/D algorithm.

        Args:
            n_generations: Number of generations.
            use_mallows: If True, use Mallows model for reproduction
                         instead of crossover.
            mutation_prob: Probability of swap mutation.
            verbose: Print progress every 10 generations.

        Returns:
            Dictionary with keys:
            - "pareto_solutions": non-dominated permutations
            - "pareto_objectives": their objective vectors
            - "final_population": last generation's population
            - "final_objectives": last generation's objective values
        """
        # Initialization
        self.weights = self._generate_weights()
        self._compute_neighbourhoods()
        self._initialize_population()

        # Add initial population to archive
        for i in range(self.n_subproblems):
            self.archive.add(self.population[i], self.obj_values[i])

        # Main loop
        for gen in range(n_generations):
            indices = list(range(self.n_subproblems))
            self.rng.shuffle(indices)

            for i in indices:
                # Select parents from neighbourhood
                nb = self.neighbours[i]

                if use_mallows and len(nb) >= 3:
                    # Learn a Mallows model from neighbourhood solutions
                    nb_pop = self.population[nb]
                    mallows = MallowsKendall()
                    mallows.fit(nb_pop)
                    child = mallows.sample(1, self.rng)[0]
                    child = self._mutate(child, mutation_prob)
                else:
                    # Crossover two parents
                    p1_idx = self.rng.choice(nb)
                    p2_idx = self.rng.choice(nb)
                    child = self._crossover(
                        self.population[p1_idx],
                        self.population[p2_idx],
                    )
                    child = self._mutate(child, mutation_prob)

                # Evaluate child
                child_obj = self._evaluate(child)
                self._update_reference_point(child_obj)

                # Update neighbouring solutions
                for j in nb:
                    w = self.weights[j]
                    child_scalar = self._scalar_value(child_obj, w)
                    current_scalar = self._scalar_value(self.obj_values[j], w)
                    if child_scalar < current_scalar:
                        self.population[j] = child.copy()
                        self.obj_values[j] = child_obj.copy()

                # Update archive
                self.archive.add(child, child_obj)

            if verbose and (gen + 1) % 10 == 0:
                print(f"Generation {gen + 1}/{n_generations}, "
                      f"archive size: {self.archive.size}")

        sols, objs = self.archive.get_front()
        return {
            "pareto_solutions": sols,
            "pareto_objectives": objs,
            "final_population": self.population,
            "final_objectives": self.obj_values,
        }
