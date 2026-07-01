"""
MEDA/D-MK: Multi-objective Decomposition-based EDA using Kernels of
Mallows Models.

Implements Algorithm 1 from Zangari et al., "Multiobjective decomposition-
based Mallows Models estimation of distribution algorithm", Information
Sciences 397-398 (2017) 137-154.

Each subproblem k keeps a Mallows Kernel centred on its current solution.
A new sample is drawn from this kernel and used to update the neighbourhood.
"""

import numpy as np
from typing import List, Callable, Optional, Dict, Any, Tuple

from src.multiobjective.scalarization import weighted_sum, tchebycheff
from src.multiobjective.pareto import ParetoArchive
from src.distributions.mallows import MallowsCayley
from src.utils.permutations import random_permutation


class MEDA_D_MK:
    """MEDA/D-MK framework for bi-objective permutation optimisation.

    Parameters
    ----------
    objectives : list of callables
        Each callable takes a 0-indexed permutation and returns a float.
    n : int
        Permutation size.
    n_subproblems : int
        Number of weight vectors / subproblems (N in the paper).
    neighbourhood_size : int
        T – how many closest weight vectors form each neighbourhood.
    theta : float
        Spread parameter for the Mallows Kernel (fixed across generations).
    nr : int
        Maximum number of neighbours a new solution can replace.
    scalarization : str
        "weighted_sum" or "tchebycheff".
    shake_threshold : int
        Number of generations without improvement before shaking a subproblem.
    shake_strength : int
        Number of random insert moves applied during shaking.
    seed : int or None
        Random seed.
    """

    def __init__(
        self,
        objectives: List[Callable],
        n: int,
        n_subproblems: int = 50,
        neighbourhood_size: int = 10,
        theta: float = 1.0,
        nr: int = 2,
        scalarization: str = "tchebycheff",
        shake_threshold: int = 20,
        shake_strength: int = 3,
        minimize: Tuple[bool, ...] = None,
        seed: Optional[int] = None,
    ):
        self.objectives = objectives
        self.n_obj = len(objectives)
        self.n = n
        self.N = n_subproblems
        self.T = min(neighbourhood_size, n_subproblems)
        self.theta = theta
        self.nr = nr
        self.scalarization = scalarization
        self.shake_threshold = shake_threshold
        self.shake_strength = shake_strength
        # Default to minimization for all objectives if not specified
        if minimize is None:
            minimize = tuple([True] * self.n_obj)
        self.minimize = minimize
        self.rng = np.random.default_rng(seed)

        # State initialised in run()
        self.weights = None
        self.neighbours = None
        self.population = None
        self.obj_values = None
        self.reference_point = None
        self.worst_point = None
        self.archive = ParetoArchive(minimize=all(self.minimize))

    # ------------------------------------------------------------------
    # Weight vectors and neighbourhoods
    # ------------------------------------------------------------------

    def _generate_weights(self) -> np.ndarray:
        if self.n_obj == 2:
            w = np.linspace(0, 1, self.N)
            return np.column_stack([w, 1.0 - w])
        raw = self.rng.dirichlet(np.ones(self.n_obj), self.N)
        return raw

    def _compute_neighbourhoods(self):
        dists = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                dists[i, j] = np.linalg.norm(self.weights[i] - self.weights[j])
        self.neighbours = np.argsort(dists, axis=1)[:, : self.T]

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def _evaluate(self, perm: np.ndarray) -> np.ndarray:
        return np.array([f(perm) for f in self.objectives])

    def _scalar_value(self, obj_vals: np.ndarray, weight: np.ndarray) -> float:
        if self.scalarization == "tchebycheff":
            return self._tchebycheff_normalised(obj_vals, weight)
        return self._weighted_sum_normalised(obj_vals, weight)

    def _weighted_sum_normalised(self, obj_vals, weight):
        z = self.reference_point
        w = self.worst_point
        alpha = 0.6
        denom = w - z
        denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)
        normalised = (obj_vals - alpha * z) / denom
        return float(np.dot(weight, normalised))

    def _tchebycheff_normalised(self, obj_vals, weight):
        z = self.reference_point
        w = self.worst_point
        alpha = 0.6
        denom = w - z
        denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)
        normalised = (obj_vals - alpha * z) / denom
        wt = np.where(weight > 1e-10, weight, 1e-10)
        return float(np.max(wt * normalised))

    def _update_reference(self, obj_vals: np.ndarray):
        """Update reference point component-wise based on minimize flags."""
        if self.reference_point is None:
            self.reference_point = obj_vals.copy()
        else:
            for i in range(self.n_obj):
                if self.minimize[i]:
                    # For minimization: track best (minimum) value seen
                    self.reference_point[i] = min(self.reference_point[i], obj_vals[i])
                else:
                    # For maximization: track best (maximum) value seen
                    self.reference_point[i] = max(self.reference_point[i], obj_vals[i])

    def _update_worst(self, obj_vals: np.ndarray):
        """Update worst point component-wise based on minimize flags."""
        if self.worst_point is None:
            self.worst_point = obj_vals.copy()
        else:
            for i in range(self.n_obj):
                if self.minimize[i]:
                    # For minimization: track worst (maximum) value seen
                    self.worst_point[i] = max(self.worst_point[i], obj_vals[i])
                else:
                    # For maximization: track worst (minimum) value seen
                    self.worst_point[i] = min(self.worst_point[i], obj_vals[i])

    # ------------------------------------------------------------------
    # Mallows Kernel sampling (Section 3.3 of the paper)
    # ------------------------------------------------------------------

    def _sample_from_kernel(self, centre: np.ndarray) -> np.ndarray:
        """Sample a single permutation from a Mallows Kernel centred on *centre*."""
        mallows = MallowsCayley()
        mallows.n = self.n
        mallows.consensus = centre.copy()
        mallows.theta = self.theta
        # Build x-vector probabilities for Cayley distance
        j = np.arange(1, self.n)
        mallows.x_probs = 1.0 / (j * np.exp(-self.theta) + 1)
        return mallows.sample(1, self.rng)[0]

    # ------------------------------------------------------------------
    # Insert-based perturbation (Section 4.3 of the paper)
    # ------------------------------------------------------------------

    def _insert_move(self, perm: np.ndarray) -> np.ndarray:
        """Apply one random insert-based move."""
        result = perm.copy()
        n = len(result)
        src = self.rng.integers(0, n)
        dst = self.rng.integers(0, n)
        if src == dst:
            return result
        val = result[src]
        result = np.delete(result, src)
        result = np.insert(result, dst, val)
        return result

    def _shake(self, perm: np.ndarray) -> np.ndarray:
        """Apply several random insert moves (controlled perturbation)."""
        result = perm.copy()
        for _ in range(self.shake_strength):
            result = self._insert_move(result)
        return result

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(
        self,
        n_generations: int,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Execute the MEDA/D-MK algorithm.

        Returns a dictionary with keys:
            pareto_solutions, pareto_objectives, final_population, final_objectives
        """
        # 1. Initialisation
        self.weights = self._generate_weights()
        self._compute_neighbourhoods()

        self.population = np.array([
            self.rng.permutation(self.n) for _ in range(self.N)
        ])
        self.obj_values = np.array([self._evaluate(p) for p in self.population])

        for ov in self.obj_values:
            self._update_reference(ov)
            self._update_worst(ov)

        # EP initialisation
        self.archive = ParetoArchive(minimize=all(self.minimize))
        for i in range(self.N):
            self.archive.add(self.population[i], self.obj_values[i])

        # Track stagnation for shaking
        no_improve = np.zeros(self.N, dtype=int)

        # 2. Main loop
        for gen in range(n_generations):
            order = list(range(self.N))
            self.rng.shuffle(order)

            for k in order:
                # Learning: Mallows Kernel centred on current solution
                centre = self.population[k]
                child = self._sample_from_kernel(centre)

                # Optional single insert move (low probability)
                if self.rng.random() < 0.1:
                    child = self._insert_move(child)

                # Duplicate avoidance within neighbourhood
                nb = self.neighbours[k]
                max_trials = self.T
                trial = 0
                while trial < max_trials:
                    if not any(np.array_equal(child, self.population[r]) for r in nb):
                        break
                    child = self._sample_from_kernel(centre)
                    trial += 1

                child_obj = self._evaluate(child)
                self._update_reference(child_obj)
                self._update_worst(child_obj)

                # Update neighbours (capped by nr)
                updated = 0
                improved_k = False
                for r in nb:
                    if updated >= self.nr:
                        break
                    w = self.weights[r]
                    if self._scalar_value(child_obj, w) < self._scalar_value(
                        self.obj_values[r], w
                    ):
                        self.population[r] = child.copy()
                        self.obj_values[r] = child_obj.copy()
                        updated += 1
                        if r == k:
                            improved_k = True

                # Shaking
                if improved_k:
                    no_improve[k] = 0
                else:
                    no_improve[k] += 1
                    if no_improve[k] >= self.shake_threshold:
                        self.population[k] = self._shake(self.population[k])
                        self.obj_values[k] = self._evaluate(self.population[k])
                        self._update_reference(self.obj_values[k])
                        self._update_worst(self.obj_values[k])
                        no_improve[k] = 0

                self.archive.add(child, child_obj)

            if verbose and (gen + 1) % 10 == 0:
                print(
                    f"MEDA/D-MK  gen {gen + 1}/{n_generations}  "
                    f"archive={self.archive.size}"
                )

        sols, objs = self.archive.get_front()
        return {
            "pareto_solutions": sols,
            "pareto_objectives": objs,
            "final_population": self.population,
            "final_objectives": self.obj_values,
        }
