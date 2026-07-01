"""
Fourier-enhanced MOEA/D for multi-objective permutation problems.

This module integrates Fourier permutation decomposition (FPD) surrogates
with the MOEA/D framework for bi-objective optimisation.

Key ideas:
----------
1. **Per-objective Fourier models**: Two Fourier decomposition models
   (one per objective) are learned from the current population.

2. **Scalarised Fourier evaluation**: For each subproblem the two
   per-objective Fourier models are combined into a scalarised score
   using the subproblem's weight vector.  This cheap evaluation is used
   to rank candidate solutions.

3. **Pool-based candidate generation**: A pool of candidate solutions
   is generated via crossover (default) or Mallows sampling from the
   neighbourhood.  The best candidate according to the scalarised
   Fourier model is selected for true evaluation.

References:
    [1] F. Chicano et al. "Fourier Transform-based Surrogates for
        Permutation Problems". GECCO 2023.
    [2] J. Ceberio et al. "Multi-objective decomposition-based Mallows
        Models".
"""

import numpy as np
from typing import List, Callable, Optional, Dict, Any, Type

from perm_pateda.multiobjective.moead import MOEAD
from perm_pateda.multiobjective.scalarization import weighted_sum, tchebycheff
from perm_pateda.fourier.surrogate import FourierSurrogate
from perm_pateda.distributions.mallows import MallowsKendall
from perm_pateda.fourier.probability_distribution import (
    PermutationDistribution,
    TSPPermutationDistribution,
)


class FourierMOEAD(MOEAD):
    """MOEA/D with Fourier surrogate assistance.

    Extends the base MOEA/D by periodically building per-objective Fourier
    decomposition models and using their scalarised combination to
    pre-screen candidate solutions.
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
        surrogate_order: int = 1,
        surrogate_retrain_interval: int = 10,
        prescreening_factor: int = 5,
        distribution_class: Type[PermutationDistribution] = TSPPermutationDistribution,
    ):
        super().__init__(
            objectives, n, n_subproblems, neighbourhood_size,
            scalarization, minimize, seed,
        )
        self.surrogate_order = surrogate_order
        self.surrogate_retrain_interval = surrogate_retrain_interval
        self.prescreening_factor = prescreening_factor
        self.dist_cls = distribution_class
        # Per-objective Fourier models (rebuilt periodically)
        self.fourier_models: List[Optional[PermutationDistribution]] = [None, None]

    # ------------------------------------------------------------------
    # Build per-objective Fourier models from the current population
    # ------------------------------------------------------------------

    def _build_fourier_models(self) -> None:
        """Learn one Fourier decomposition model per objective.

        Each model is a problem-specific PermutationDistribution learned
        from the current population (fitness-weighted resampling).
        """
        self.fourier_models = []
        for obj_idx in range(self.n_obj):
            vals = np.array([
                self.objectives[obj_idx](p) for p in self.population
            ])

            # Fitness-weighted resampling
            if self.minimize:
                ranks = np.argsort(np.argsort(vals))
            else:
                ranks = np.argsort(np.argsort(-vals))
            weights = (len(ranks) - ranks).astype(float)
            weights /= weights.sum()

            n_resampled = max(len(self.population), 50)
            indices = self.rng.choice(
                len(self.population), size=n_resampled, p=weights
            )
            weighted_samples = self.population[indices]

            model = self.dist_cls(self.n, self.surrogate_order)
            model.learning_FD(weighted_samples)
            self.fourier_models.append(model)

    # ------------------------------------------------------------------
    # Pool-based surrogate selection for a subproblem
    # ------------------------------------------------------------------

    def _build_surrogate_for_subproblem(
        self,
        idx: int,
        use_mallows: bool = False,
        mutation_prob: float = 0.1,
    ) -> np.ndarray:
        """Build a candidate pool and select the best via scalarised Fourier model.

        1. Generate a pool of candidate solutions from the neighbourhood
           using crossover (default) or Mallows sampling.
        2. Score each candidate using the scalarised linear combination
           of the two per-objective Fourier decomposition models.
        3. Return the candidate with the best scalarised score.

        Parameters
        ----------
        idx : int
            Subproblem index.
        use_mallows : bool
            If True, use Mallows sampling instead of crossover.
        mutation_prob : float
            Swap mutation probability applied to each candidate.

        Returns
        -------
        np.ndarray
            The best candidate permutation according to the Fourier model.
        """
        nb = self.neighbours[idx]
        w = self.weights[idx]

        # Step 1: Generate a pool of candidate permutations
        candidates = []
        for _ in range(self.prescreening_factor):
            if use_mallows and len(nb) >= 3:
                nb_pop = self.population[nb]
                mallows = MallowsKendall()
                mallows.fit(nb_pop)
                c = mallows.sample(1, self.rng)[0]
            else:
                p1 = self.population[self.rng.choice(nb)]
                p2 = self.population[self.rng.choice(nb)]
                c = self._crossover(p1, p2)
            candidates.append(self._mutate(c, mutation_prob))

        candidates = np.array(candidates)

        # Step 2: Score candidates using the scalarised Fourier model.
        # The scalarised model is the weighted combination of the two
        # per-objective Fourier decomposition models evaluated on each
        # candidate — NOT a model trained on the neighbourhood.
        if any(m is None for m in self.fourier_models):
            return candidates[0]

        scores = np.zeros(len(candidates))
        for i, cand in enumerate(candidates):
            obj_approx = np.zeros(self.n_obj)
            for obj_idx, model in enumerate(self.fourier_models):
                obj_approx[obj_idx] = model._evaluate_all(
                    cand.reshape(1, -1)
                )[0]
            scores[i] = self._scalar_value(obj_approx, w)

        # Step 3: Select the candidate with the best (lowest) score
        best_idx = np.argmin(scores)
        return candidates[best_idx]

    def run(
        self,
        n_generations: int,
        use_mallows: bool = False,
        mutation_prob: float = 0.1,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Execute the Fourier-enhanced MOEA/D algorithm.

        Args:
            n_generations: Number of generations.
            use_mallows: Use Mallows model for candidate generation.
            mutation_prob: Swap mutation probability.
            verbose: Print progress.

        Returns:
            Dictionary with Pareto front and population data.
        """
        self.weights = self._generate_weights()
        self._compute_neighbourhoods()
        self._initialize_population()

        for i in range(self.n_subproblems):
            self.archive.add(self.population[i], self.obj_values[i])

        for gen in range(n_generations):
            # Periodically rebuild per-objective Fourier models
            if gen % self.surrogate_retrain_interval == 0:
                self._build_fourier_models()

            indices = list(range(self.n_subproblems))
            self.rng.shuffle(indices)

            for i in indices:
                # Use the pool-based Fourier pre-screening
                child = self._build_surrogate_for_subproblem(
                    i, use_mallows=use_mallows, mutation_prob=mutation_prob
                )

                # True evaluation
                child_obj = self._evaluate(child)
                self._update_reference_point(child_obj)

                # Update neighbours
                nb = self.neighbours[i]
                for j in nb:
                    w = self.weights[j]
                    if self._scalar_value(child_obj, w) < \
                       self._scalar_value(self.obj_values[j], w):
                        self.population[j] = child.copy()
                        self.obj_values[j] = child_obj.copy()

                self.archive.add(child, child_obj)

            if verbose and (gen + 1) % 10 == 0:
                print(f"[FourierMOEAD] Gen {gen + 1}/{n_generations}, "
                      f"archive: {self.archive.size}")

        sols, objs = self.archive.get_front()
        return {
            "pareto_solutions": sols,
            "pareto_objectives": objs,
            "final_population": self.population,
            "final_objectives": self.obj_values,
        }
