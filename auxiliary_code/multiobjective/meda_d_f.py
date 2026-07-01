"""
MEDA/D-F: Multi-objective Decomposition-based EDA using Fourier
Decomposition Models.

For each generation the algorithm:
1. Learns one Fourier distribution model per objective from the current
   neighbourhood population (using the problem-specific *PermutationDistribution
   classes).
2. For each subproblem k with weight vector lambda_k = (w1, w2), builds
   a combined probability:
       P_k(sigma) = w1 * P1(sigma) + w2 * P2(sigma)
   (linear combination of the two per-objective distributions).
3. Generates candidates via crossover+mutation and selects the one with
   the highest score under P_k.

The combined distribution is evaluated directly on crossover-generated
candidates using the per-objective Fourier models.
"""

import numpy as np
from math import factorial
from typing import List, Callable, Optional, Dict, Any, Type, Tuple

from src.multiobjective.scalarization import weighted_sum, tchebycheff
from src.multiobjective.pareto import ParetoArchive
from src.fourier.probability_distribution import (
    PermutationDistribution,
    QAPPermutationDistribution,
    LOPPermutationDistribution,
    TSPPermutationDistribution,
    STSPPermutationDistribution,
    sample_from_distribution,
    _compute_freq_matrix,
    _compute_pair_tensor_marginal_fill,
    _compute_pair_tensor_uniform_fill,
)
from src.utils.permutations import random_permutation


class MEDA_D_F:
    """MEDA/D-F framework for bi-objective permutation optimisation.

    Follows MOEA/D algorithm structure except for solution generation:
    - Uses Fourier Decomposition models instead of crossover/mutation
    - Samples from weighted combination of per-objective probability models
    - All other aspects (reference point tracking, scalarization, neighbor
      replacement) follow MOEA/D implementation

    Parameters
    ----------
    objectives : list of callables
        Each callable takes a 0-indexed permutation and returns a float.
    n : int
        Permutation size.
    distribution_class : subclass of PermutationDistribution
        The problem-specific Fourier distribution class to use.
        One of TSPPermutationDistribution, QAPPermutationDistribution,
        LOPPermutationDistribution, STSPPermutationDistribution.
    n_subproblems : int
        Number of weight vectors / subproblems.
    neighbourhood_size : int
        Size of each subproblem's neighbourhood.
    truncation_order : int
        Fourier truncation order (0, 1, or 2).
    nr : int
        (Deprecated as of v1.0, no longer used) Kept for API compatibility.
        The algorithm now replaces ALL improving neighbors like MOEA/D.
    scalarization : str
        "weighted_sum" or "tchebycheff".
    minimize : bool
        If True, objectives are minimised. If False, maximised.
    seed : int or None
        Random seed.
    use_global_models : bool
        If True, use incremental learning across generations.
    use_weighted_learning : bool
        If True, use fitness-weighted learning.
    """

    def __init__(
        self,
        objectives: List[Callable],
        n: int,
        distribution_class: Type[PermutationDistribution] = TSPPermutationDistribution,
        n_subproblems: int = 50,
        neighbourhood_size: int = 10,
        truncation_order: int = 2,
        nr: int = 2,
        scalarization: str = "tchebycheff",
        minimize: bool = True,
        seed: Optional[int] = None,
        use_global_models: bool = False,
        use_weighted_learning: bool = False,
    ):
        self.objectives = objectives
        self.n_obj = len(objectives)
        self.n = n
        self.dist_cls = distribution_class
        self.N = n_subproblems
        self.T = min(neighbourhood_size, n_subproblems)
        self.truncation_order = truncation_order
        self.nr = nr
        self.scalarization = scalarization
        # Handle both boolean and tuple for backward compatibility
        # MOEA/D uses a single boolean for all objectives
        if isinstance(minimize, tuple):
            # For tuple, verify all are the same (MOEA/D-like behavior)
            if not all(m == minimize[0] for m in minimize):
                raise ValueError(
                    "MEDA_D_F (like MOEA/D) requires all objectives to have "
                    "the same minimize direction. Got: {}".format(minimize)
                )
            self.minimize = minimize[0]
        else:
            self.minimize = minimize
        self.rng = np.random.default_rng(seed)

        # Incremental FD parameters
        self.use_global_models = use_global_models
        self.use_weighted_learning = use_weighted_learning

        # State initialised in run()
        self.weights = None
        self.neighbours = None
        self.population = None
        self.obj_values = None
        self.reference_point = None
        self.archive = ParetoArchive(minimize=self.minimize)
        # Global FD statistics per objective (initialised in run() when needed)
        self._global_stats: Optional[List[Optional[Dict]]] = None

        # Phase 2: Incremental surrogate learning state
        # These maintain persistent surrogates across generations
        self._persistent_surrogates: Optional[List] = None  # One per objective
        self._accumulated_training_data: Optional[List[Dict]] = None  # Training history per objective
        self._surrogate_window_size: int = 500  # Maximum training samples to keep

    # ------------------------------------------------------------------
    # Weight vectors and neighbourhoods  (same as MEDA/D-MK)
    # ------------------------------------------------------------------

    def _generate_weights(self) -> np.ndarray:
        if self.n_obj == 2:
            w = np.linspace(0, 1, self.N)
            return np.column_stack([w, 1.0 - w])
        return self.rng.dirichlet(np.ones(self.n_obj), self.N)

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
        """Compute scalarised fitness using standard functions."""
        if self.scalarization == "tchebycheff":
            return tchebycheff(obj_vals, weight, self.reference_point, self.minimize)
        else:
            return weighted_sum(obj_vals, weight, self.reference_point, self.minimize)

    def _update_reference(self, obj_vals: np.ndarray):
        """Update the ideal reference point."""
        if self.reference_point is None:
            self.reference_point = obj_vals.copy()
        else:
            if self.minimize:
                self.reference_point = np.minimum(self.reference_point, obj_vals)
            else:
                self.reference_point = np.maximum(self.reference_point, obj_vals)

    # ------------------------------------------------------------------
    # Crossover and mutation operators
    # ------------------------------------------------------------------

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Order crossover (OX) for permutations."""
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

    # ------------------------------------------------------------------
    # Fourier distribution learning and combined sampling
    # ------------------------------------------------------------------

    def _learn_fourier_models(
        self, pop_subset: np.ndarray, use_weighted_learning: bool = False
    ) -> List[PermutationDistribution]:
        """Learn one Fourier distribution per objective from the population subset.

        For each objective we rank the subset, weight the permutations by
        fitness (better solutions appear more often), and call from_samples.
        The model is also augmented with random permutations so that the
        distribution can assign probability mass to novel solutions.
        """
        models = []
        for obj_idx in range(self.n_obj):
            vals = np.array([self.objectives[obj_idx](p) for p in pop_subset])
            if use_weighted_learning:                
                if self.minimize:                   
                    weights = (np.max(vals) - vals).astype(float)
                else:
                    weights = vals.astype(float)               
            else:
                # Convert objective values to sampling weights
                # For minimization: smaller values are better (rank 0 = best)
                # For maximization: larger values are better (rank 0 = best)
                if self.minimize:
                    ranks = np.argsort(np.argsort(vals))  # 0 = best for minimization
                else:
                    ranks = np.argsort(np.argsort(-vals))  # 0 = best for maximization
                weights = (len(ranks) - ranks).astype(float)

            # Normalize weights, handling the case when sum is zero
            weight_sum = weights.sum()
            if weight_sum > 0:
                weights /= weight_sum
            else:
                # If all weights are zero (e.g., all solutions have same fitness),
                # use uniform distribution
                weights = np.ones(len(weights)) / len(weights)

            # Build a weighted sample set: repeat permutations proportionally.
            # Also inject random permutations so the learned Fourier
            # coefficients can steer probability to unseen solutions.
            n_resampled = max(len(pop_subset), 50)          
            indices = self.rng.choice(
                len(pop_subset), size=n_resampled, p=weights
            )
            weighted_samples = pop_subset[indices]
            model = self.dist_cls.from_samples(weighted_samples, self.truncation_order)
            models.append(model)

        return models

    # ------------------------------------------------------------------
    # Incremental Fourier learning (use_global_models=True)
    # ------------------------------------------------------------------

    def _compute_temporary_statistics(
        self, samples: np.ndarray
    ) -> Dict[str, Any]:
        """Compute freq_matrix, pair_tensor, and mean from samples.

        Uses the appropriate conventions for the distribution class:
        - TSP/STSP: zero freq_matrix, uniform-fill pair tensor, mean = 1/n
        - LOP: computed freq_matrix, marginal-fill pair tensor, mean = 1/n
        - QAP/base: computed freq_matrix, marginal-fill pair tensor, mean = 1/n!
        """
        n = self.n

        # Mean: class-specific
        if self.dist_cls in (
            TSPPermutationDistribution,
            STSPPermutationDistribution,
            LOPPermutationDistribution,
        ):
            mean = 1.0 / n if n <= 20 else 1e-6
        else:  # QAP / base
            mean = 1.0 / factorial(n) if n <= 20 else 1e-6

        # Freq matrix: TSP/STSP don't use first-order statistics
        if self.dist_cls in (TSPPermutationDistribution, STSPPermutationDistribution):
            freq_matrix = np.zeros((n, n))
        else:
            freq_matrix = _compute_freq_matrix(samples, n)

        # Pair tensor: TSP/STSP use uniform fill, QAP/LOP use marginal fill
        pair_tensor = None
        if self.truncation_order >= 2:
            if self.dist_cls in (
                TSPPermutationDistribution,
                STSPPermutationDistribution,
            ):
                pair_tensor = _compute_pair_tensor_uniform_fill(samples, n)
            else:
                pair_tensor = _compute_pair_tensor_marginal_fill(
                    samples, n, freq_matrix
                )

        return {"mean": mean, "freq_matrix": freq_matrix, "pair_tensor": pair_tensor}

    def _learn_fourier_models_incremental(
        self, gen: int, use_weighted_learning: bool = False
    ) -> List[PermutationDistribution]:
        """Learn FD models with incremental blending of statistics.

        Instead of relearning from scratch each generation, maintains
        global statistics (freq_matrix, pair_tensor, mean) per objective
        and blends them with temporary statistics from the current
        population:

            global = alpha * global + (1 - alpha) * temporary

        where alpha = gen / (gen + 1) for gen > 0 (alpha = 0 for gen 0).

        When use_weighted_learning is True, the temporary statistics are
        computed from fitness-weighted resampled permutations.
        """
        #alpha = gen / (gen + 1) if gen > 0 else 0.0
        alpha = 0.9  if gen > 0 else 0.0

        models = []
        for obj_idx in range(self.n_obj):
            # Determine samples for temporary FD
            vals = np.array([
                self.objectives[obj_idx](p) for p in self.population
            ])
            if use_weighted_learning:               
                if self.minimize:                   
                    weights = (np.max(vals) - vals).astype(float)
                else:
                    weights = vals.astype(float)               
               
            else:
                # Convert objective values to sampling weights
                # For minimization: smaller values are better (rank 0 = best)
                # For maximization: larger values are better (rank 0 = best)
                if self.minimize:
                    ranks = np.argsort(np.argsort(vals))  # 0 = best for minimization
                else:
                    ranks = np.argsort(np.argsort(-vals))  # 0 = best for maximization
                weights = (len(ranks) - ranks).astype(float)
            weights /= weights.sum()
            n_resampled = max(len(self.population), 50)
            indices = self.rng.choice(
                    len(self.population), size=n_resampled, p=weights
            )
            samples = self.population[indices]    

            # Compute temporary statistics from current population
            temp_stats = self._compute_temporary_statistics(samples)

            # Blend with global statistics
            gs = self._global_stats[obj_idx]
            if gs is None:
                # First generation: initialise global from temporary
                self._global_stats[obj_idx] = {
                    "mean": temp_stats["mean"],
                    "freq_matrix": temp_stats["freq_matrix"].copy(),
                    "pair_tensor": (
                        temp_stats["pair_tensor"].copy()
                        if temp_stats["pair_tensor"] is not None
                        else None
                    ),
                }
            else:
                # Incremental blend
                gs["mean"] = alpha * gs["mean"] + (1 - alpha) * temp_stats["mean"]
                gs["freq_matrix"] = (
                    alpha * gs["freq_matrix"]
                    + (1 - alpha) * temp_stats["freq_matrix"]
                )
                if gs["pair_tensor"] is not None and temp_stats["pair_tensor"] is not None:
                    gs["pair_tensor"] = (
                        alpha * gs["pair_tensor"]
                        + (1 - alpha) * temp_stats["pair_tensor"]
                    )

            # Create model with blended statistics
            blended = self._global_stats[obj_idx]
            model = self.dist_cls(self.n, self.truncation_order)
            model._mean = blended["mean"]
            model._freq_matrix = blended["freq_matrix"]
            model._pair_tensor = blended["pair_tensor"]

            # Set up training data for surrogate models
            # This is required when evaluation_method="surrogate"
            model._training_perms = samples.copy()
            model._training_values = model._evaluate_all(samples)

            # Compute probabilities on population + random permutations
            #n_random = max(len(self.population) // 4, 10)
            #random_perms = np.array([
            #    self.rng.permutation(self.n) for _ in range(n_random)
            #])
            #candidates = np.vstack([self.population, random_perms])
            #model.sampling_FD(candidates)

            models.append(model)

        return models

    # ------------------------------------------------------------------
    # Phase 2: Incremental surrogate learning with warm-start
    # ------------------------------------------------------------------

    def _accumulate_training_data(
        self,
        obj_idx: int,
        new_perms: np.ndarray,
        new_values: np.ndarray
    ) -> None:
        """Accumulate training data for surrogate model with sliding window.

        Parameters
        ----------
        obj_idx : int
            Objective index (0 or 1).
        new_perms : np.ndarray, shape (m, n)
            New training permutations to add.
        new_values : np.ndarray, shape (m,)
            Fourier values for the new permutations.
        """
        if self._accumulated_training_data is None:
            # Initialize storage for all objectives
            self._accumulated_training_data = [
                {"perms": [], "values": []} for _ in range(self.n_obj)
            ]

        data = self._accumulated_training_data[obj_idx]
        data["perms"].append(new_perms)
        data["values"].append(new_values)

        # Apply sliding window: keep only last N samples
        total_samples = sum(len(p) for p in data["perms"])
        while total_samples > self._surrogate_window_size and len(data["perms"]) > 1:
            # Remove oldest batch
            removed_perms = data["perms"].pop(0)
            data["values"].pop(0)
            total_samples -= len(removed_perms)

    def _get_accumulated_training_data(
        self,
        obj_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get accumulated training data for an objective.

        Parameters
        ----------
        obj_idx : int
            Objective index.

        Returns
        -------
        all_perms : np.ndarray, shape (m, n)
            All accumulated training permutations (within window).
        all_values : np.ndarray, shape (m,)
            All accumulated training values.
        """
        if self._accumulated_training_data is None or not self._accumulated_training_data[obj_idx]["perms"]:
            return np.empty((0, self.n), dtype=int), np.empty(0, dtype=float)

        data = self._accumulated_training_data[obj_idx]
        all_perms = np.vstack(data["perms"])
        all_values = np.concatenate(data["values"])
        return all_perms, all_values

    def _train_surrogate_incremental(
        self,
        obj_idx: int,
        model: PermutationDistribution,
        alpha: float,
        use_sparse_features: bool = False,
        min_feature_count: int = 2
    ):
        """Train or update surrogate model incrementally with warm-start.

        Parameters
        ----------
        obj_idx : int
            Objective index.
        model : PermutationDistribution
            Current Fourier model with training data.
        alpha : float
            Lasso regularization parameter.
        use_sparse_features : bool, optional
            If True, use only observed features (default: False).
        min_feature_count : int, optional
            Minimum feature appearances to include (default: 2).

        Returns
        -------
        FourierSurrogate
            Trained surrogate model.
        """
        from src.fourier.surrogate import FourierSurrogate
        from sklearn.linear_model import Lasso

        # Accumulate new training data from current model
        if model._training_perms is not None:
            self._accumulate_training_data(
                obj_idx,
                model._training_perms,
                model._training_values
            )

        # Get all accumulated training data
        all_perms, all_values = self._get_accumulated_training_data(obj_idx)

        if len(all_perms) == 0:
            raise ValueError(f"No training data available for objective {obj_idx}")

        # Initialize persistent surrogates if needed
        if self._persistent_surrogates is None:
            self._persistent_surrogates = [None] * self.n_obj

        # Determine surrogate order
        surrogate_order = min(model.truncation_order, 2) if model.truncation_order > 0 else 1

        # Check if we can use warm-start
        use_warm_start = (
            self._persistent_surrogates[obj_idx] is not None
            and self._persistent_surrogates[obj_idx].model is not None
        )

        if use_warm_start:
            # Reuse existing surrogate structure and warm-start from previous coefficients
            surrogate = self._persistent_surrogates[obj_idx]
            # Update parameters if needed
            surrogate.alpha = alpha
            surrogate.use_sparse_features = use_sparse_features
            surrogate.min_feature_count = min_feature_count
        else:
            # Create new surrogate with sparse feature support
            surrogate = FourierSurrogate(
                n=self.n,
                max_order=surrogate_order,
                alpha=alpha,
                use_sparse_features=use_sparse_features,
                min_feature_count=min_feature_count
            )

        # Compute features for all accumulated data
        X = surrogate._features_batch(all_perms)
        y = all_values

        # Apply sparse feature selection if enabled
        if use_sparse_features:
            surrogate._identify_active_features(X)
            X = surrogate._extract_sparse_features(X)

        # Check if we can use warm-start
        # Warm-start requires the feature dimension to remain constant
        # With sparse features, the number of active features can change between generations
        can_use_warm_start = (
            use_warm_start
            and surrogate.model is not None
            and surrogate._feature_dim == X.shape[1]  # Feature dimension must match
        )

        # Train with warm-start if available and feature dimensions match
        if can_use_warm_start:
            # Update existing model with warm-start
            surrogate.model.set_params(alpha=alpha, warm_start=True)
            surrogate.model.fit(X, y)
        else:
            # Train new model (either first time or feature dimension changed)
            surrogate.model = Lasso(
                alpha=alpha,
                fit_intercept=True,
                max_iter=1000,
                warm_start=True  # Enable for future incremental updates
            )
            surrogate.model.fit(X, y)

        surrogate._feature_dim = X.shape[1]

        # Store for next generation
        self._persistent_surrogates[obj_idx] = surrogate

        return surrogate

    def _precompute_combined_data(
        self,
        models: List[PermutationDistribution],
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Precompute combined permutation set and per-model probabilities.

        This is called once per generation to avoid recomputing the same
        data for each subproblem.

        Returns
        -------
        combined_perms : np.ndarray
            Union of all permutations from all models (de-duplicated).
        model_probs : List[np.ndarray]
            For each model, the probability of each permutation in combined_perms.
        """
        # Collect the union of permutation sets from all models
        perm_sets = [m.all_perms for m in models]

        # Build a combined permutation set (union by rows, de-duplicated)
        seen = set()
        combined_list = []
        for ps in perm_sets:
            for p in ps:
                key = tuple(p)
                if key not in seen:
                    seen.add(key)
                    combined_list.append(p)
        combined_perms = np.array(combined_list)

        # Evaluate each model's probability on the combined set
        model_probs = []
        for model in models:
            probs_for_model = np.array([
                model.evaluate_probability(p) for p in combined_perms
            ])
            model_probs.append(probs_for_model)

        return combined_perms, model_probs

    def _sample_combined(
        self,
        combined_perms: np.ndarray,
        model_probs: List[np.ndarray],
        weight_vector: np.ndarray,
    ) -> np.ndarray:
        """Sample one permutation from the linear combination of per-objective models.

        P_k(sigma) = w1 * P1(sigma) + w2 * P2(sigma)

        Uses precomputed combined permutation set and model probabilities to
        compute the weighted combination and sample.

        Parameters
        ----------
        combined_perms : np.ndarray
            Union of all permutations from all models (precomputed).
        model_probs : List[np.ndarray]
            For each model, the probability of each permutation in combined_perms
            (precomputed).
        weight_vector : np.ndarray
            Weight vector for the current subproblem.

        Returns
        -------
        np.ndarray
            Sampled permutation.
        """
        # Compute the weighted combination of model probabilities
        combined_probs = np.zeros(len(combined_perms))
        for obj_idx, probs_for_model in enumerate(model_probs):
            combined_probs += weight_vector[obj_idx] * probs_for_model

        # Normalise and sample via the library function
        combined_probs = np.maximum(combined_probs, 0)
        total = combined_probs.sum()
        if total < 1e-15:
            combined_probs = np.ones(len(combined_perms)) / len(combined_perms)
        else:
            combined_probs /= total

        return sample_from_distribution(combined_perms, combined_probs, 1, self.rng)[0]

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(
        self,
        n_generations: int,
        mutation_prob: float = 0.1,
        verbose: bool = False,
        use_weighted_learning: Optional[bool] = None,
        use_global_models: Optional[bool] = None,
        evaluation_method: str = "fourier",
        surrogate_alpha: float = 1e-6,
        use_incremental_surrogates: bool = True,
        use_sparse_features: bool = True,
        min_feature_count: int = 2,
    ) -> Dict[str, Any]:
        """Execute the MEDA/D-F algorithm.

        Args:
            n_generations: Number of generations to run.
            mutation_prob: Probability of swap mutation on each candidate.
            verbose: Whether to print progress information.
            use_weighted_learning: If True, use direct fitness-weighted learning
                                   (_learn_weighted_fourier_models). If False,
                                   use rank-based resampling with augmentation
                                   (_learn_fourier_models). If None, uses value from __init__.
            use_global_models: If True, learn Fourier models from all evaluated
                              solutions so far. If False, learn only from current
                              population (default behavior). If None, uses value from __init__.
            evaluation_method: Method for evaluating candidates. Options:
                              - "fourier": Use direct Fourier decomposition (_evaluate_all)
                              - "surrogate": Use surrogate model (_evaluate_all_as_surrogates)
                              Default: "fourier"
            surrogate_alpha: Lasso regularization parameter for surrogate models
                            (only used when evaluation_method="surrogate").
                            Default: 1e-6
            use_incremental_surrogates: If True, use Phase 2 incremental surrogate learning
                                       with warm-start and sliding window (default: True).
                                       If False, use Phase 1 per-generation caching.
                                       Only applies when evaluation_method="surrogate" AND
                                       use_global_models=True (Phase 2 requires global models).
            use_sparse_features: If True, use only observed features in surrogate models
                                (default: True). Reduces dimensionality by 60-70% and
                                improves training speed by 8-10×.
                                Only applies when evaluation_method="surrogate".
            min_feature_count: Minimum times a feature must appear to be included
                              (default: 2). Only used when use_sparse_features=True.

        Returns a dictionary with keys:
            pareto_solutions, pareto_objectives, final_population, final_objectives
        """
        # Use constructor values as defaults if not specified
        if use_weighted_learning is None:
            use_weighted_learning = self.use_weighted_learning
        if use_global_models is None:
            use_global_models = self.use_global_models

        # 1. Initialisation
        self.weights = self._generate_weights()
        self._compute_neighbourhoods()

        self.population = np.array([
            self.rng.permutation(self.n) for _ in range(self.N)
        ])
        self.obj_values = np.array([self._evaluate(p) for p in self.population])

        for ov in self.obj_values:
            self._update_reference(ov)

        # Reset archive for this run
        self.archive = ParetoArchive(minimize=self.minimize)
        for i in range(self.N):
            self.archive.add(self.population[i], self.obj_values[i])

        # Initialise global statistics storage when using incremental mode
        if use_global_models:
            self._global_stats = [None] * self.n_obj
            self.all_evaluated_solutions = [p.copy() for p in self.population]

        # Initialize Phase 2 incremental surrogate state
        if evaluation_method == "surrogate" and use_incremental_surrogates:
            self._persistent_surrogates = None
            self._accumulated_training_data = None

        # 2. Main loop
        for gen in range(n_generations):
            # Learn two Fourier models (one per objective).
            if use_global_models:
                fourier_models = self._learn_fourier_models_incremental(gen, use_weighted_learning)
            else:
                fourier_models = self._learn_fourier_models(self.population, use_weighted_learning)

            # Surrogate training optimization
            generation_surrogates = None
            if evaluation_method == "surrogate":
                # Phase 2 incremental learning only works with global models
                # because it accumulates training data across generations
                if use_incremental_surrogates and use_global_models:
                    # Phase 2: Incremental learning with warm-start and sliding window
                    # Accumulates training data across generations (2-5× additional speedup)
                    # With sparse features: 8-10× faster training via dimensionality reduction
                    generation_surrogates = [
                        self._train_surrogate_incremental(
                            obj_idx, model, surrogate_alpha,
                            use_sparse_features, min_feature_count
                        )
                        for obj_idx, model in enumerate(fourier_models)
                    ]
                else:
                    # Phase 1: Train surrogates once per generation
                    # Avoids retraining for each subproblem (100× speedup)
                    # With sparse features: 8-10× faster training via dimensionality reduction
                    generation_surrogates = [
                        model.train_surrogate(
                            alpha=surrogate_alpha,
                            use_sparse_features=use_sparse_features,
                            min_feature_count=min_feature_count
                        )
                        for model in fourier_models
                    ]

            order = list(range(self.N))
            self.rng.shuffle(order)

            for k in order:
                nb = self.neighbours[k]
                w_k = self.weights[k]

                # Generate m = 2*len(nb) candidates via crossover + mutation
                m = 2 * len(nb)
                candidates = np.empty((m, self.n), dtype=int)
                for ci in range(m):
                    p1_idx = self.rng.choice(nb)
                    p2_idx = self.rng.choice(nb)
                    c = self._crossover(
                        self.population[p1_idx],
                        self.population[p2_idx],
                    )
                    candidates[ci] = self._mutate(c, mutation_prob)



                # Score candidates with the combined Fourier model
                combined_scores = np.zeros(m)
                for obj_idx, model in enumerate(fourier_models):
                    if evaluation_method == "surrogate":
                        # Use precomputed surrogate (Phase 1 optimization)
                        combined_scores += w_k[obj_idx] * model._evaluate_all_as_surrogates(
                            candidates,
                            alpha=surrogate_alpha,
                            precomputed_surrogate=generation_surrogates[obj_idx]
                        )
                        if self.minimize:
                            ind = np.argmin(combined_scores)
                        else:
                            ind = np.argmax(combined_scores)
                        
                    elif evaluation_method == "fourier":
                        combined_scores += w_k[obj_idx] * model._evaluate_all(candidates)
                        combined_scores = combined_scores - np.min(combined_scores) + 10**(-6)
                        combined_scores = combined_scores/np.sum(combined_scores)          
                        ind = self.rng.choice(np.arange(combined_scores.shape[0]), size=None, p=combined_scores)
                        
                    else:
                        raise ValueError(
                            f"Unknown evaluation_method: {evaluation_method}. "
                            "Valid options are: 'fourier', 'surrogate'"
                        )

                
                #combined_scores = 2**combined_scores
                # Select the candidate with the highest score
                #child = candidates[np.argmax(combined_scores)]

                                
                child = candidates[ind]
                #print("Gen", ind, combined_scores[ind], child)
                

                child_obj = self._evaluate(child)
                self._update_reference(child_obj)

                # Track this evaluated solution for global models
                if use_global_models:
                    self.all_evaluated_solutions.append(child.copy())

                # Update neighbouring solutions (replace ALL that improve)
                for r in nb:
                    w = self.weights[r]
                    if self._scalar_value(child_obj, w) < self._scalar_value(
                        self.obj_values[r], w
                    ):
                        self.population[r] = child.copy()
                        self.obj_values[r] = child_obj.copy()

                self.archive.add(child, child_obj)

            if verbose and (gen + 1) % 10 == 0:
                print(
                    f"MEDA/D-F   gen {gen + 1}/{n_generations}  "
                    f"archive={self.archive.size}"
                )

        sols, objs = self.archive.get_front()
        return {
            "pareto_solutions": sols,
            "pareto_objectives": objs,
            "final_population": self.population,
            "final_objectives": self.obj_values,
        }
