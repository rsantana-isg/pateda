"""
Affinity-based Factorization for EDAs

Implements factorization methods using affinity propagation clustering
to identify groups of related variables based on mutual information.

Based on MATEDA-2.0 FactAffinity.m and FactAffinityElim.m

References:
    - Frey, B. J., & Dueck, D. (2007). Clustering by passing messages between
      data points. Science, 315(5814), 972-976.
    - Santana, R. (2011). Estimation of Distribution Algorithms: A New
      Evolutionary Computation Approach for Graph Matching Problems.
"""

from typing import Any, List, Tuple, Optional
import numpy as np
from sklearn.cluster import AffinityPropagation

from pateda.core.components import LearningMethod
from pateda.core.models import FactorizedModel
from pateda.learning.utils.marginal_prob import find_marginal_prob


class LearnAffinityFactorization(LearningMethod):
    """
    Learn factorization using affinity propagation clustering

    This method uses affinity propagation on a mutual information matrix
    to discover groups of related variables. Variables with high mutual
    information are clustered together into cliques/factors.

    The algorithm recursively applies affinity propagation when clusters
    exceed the size constraint.
    """

    def __init__(
        self,
        max_clique_size: int = 5,
        preference: Optional[float] = None,
        damping: float = 0.5,
        max_iter: int = 200,
        convergence_iter: int = 15,
        alpha: float = 0.0,
        recursive: bool = True,
        max_recursion_depth: int = 10,
    ):
        """
        Initialize affinity-based factorization learning

        Args:
            max_clique_size: Maximum size of factors/cliques (default: 5)
            preference: Preference parameter for affinity propagation.
                       Controls number of clusters (higher = more clusters).
                       If None, uses median of similarities.
            damping: Damping factor in [0.5, 1) to avoid numerical oscillations
            max_iter: Maximum number of iterations for affinity propagation
            convergence_iter: Number of iterations with no change for convergence
            alpha: Smoothing parameter for probability estimation
            recursive: Whether to recursively factor large clusters
            max_recursion_depth: Maximum recursion depth to prevent infinite loops
        """
        self.max_clique_size = max_clique_size
        self.preference = preference
        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.alpha = alpha
        self.recursive = recursive
        self.max_recursion_depth = max_recursion_depth

    def _compute_mutual_information_matrix(
        self,
        population: np.ndarray,
        n_vars: int,
        cardinality: np.ndarray,
        univ_prob: List[np.ndarray],
        biv_prob: List[List[np.ndarray]],
    ) -> np.ndarray:
        """
        Compute mutual information matrix from marginal probabilities

        Args:
            population: Population data
            n_vars: Number of variables
            cardinality: Variable cardinalities
            univ_prob: List of univariate probability distributions
            biv_prob: List of lists containing bivariate probability distributions

        Returns:
            Symmetric matrix of mutual information values
        """
        mi_matrix = np.zeros((n_vars, n_vars))

        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                # Compute mutual information from marginal probabilities
                mi = 0.0
                card_i = int(cardinality[i])
                card_j = int(cardinality[j])

                for k in range(card_i):
                    for l in range(card_j):
                        # Get bivariate probability P(X_i=k, X_j=l)
                        idx = card_j * k + l
                        p_ij = biv_prob[i][j][idx]
                        p_i = univ_prob[i][k]
                        p_j = univ_prob[j][l]

                        if p_ij > 0 and p_i > 0 and p_j > 0:
                            mi += p_ij * np.log2(p_ij / (p_i * p_j))

                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi

        return mi_matrix

    def _affinity_clustering(
        self,
        similarity_matrix: np.ndarray,
        preference: Optional[float] = None,
        add_noise: bool = False,
    ) -> Tuple[np.ndarray, bool]:
        """
        Perform affinity propagation clustering

        Args:
            similarity_matrix: Symmetric similarity matrix
            preference: Preference parameter (if None, uses median)
            add_noise: Whether to add small random noise to break ties

        Returns:
            Tuple of (cluster labels, converged flag)
        """
        # Add small noise to help convergence if requested
        if add_noise:
            noise = np.random.randn(*similarity_matrix.shape) * 0.01 * np.mean(
                similarity_matrix
            )
            similarity_matrix = similarity_matrix + noise

        # Set preference
        if preference is None:
            pref = np.median(similarity_matrix)
        else:
            pref = preference

        # Run affinity propagation
        try:
            ap = AffinityPropagation(
                damping=self.damping,
                max_iter=self.max_iter,
                convergence_iter=self.convergence_iter,
                preference=pref,
                affinity="precomputed",
                random_state=None,
            )
            ap.fit(similarity_matrix)
            labels = ap.labels_
            converged = ap.n_iter_ < self.max_iter
        except Exception:
            # If clustering fails, put all in one cluster
            labels = np.zeros(similarity_matrix.shape[0], dtype=int)
            converged = False

        return labels, converged

    def _recursive_factorization(
        self,
        mi_matrix: np.ndarray,
        var_indices: np.ndarray,
        preference: Optional[float],
        depth: int = 0,
    ) -> List[np.ndarray]:
        """
        Recursively factorize variables using affinity propagation

        Args:
            mi_matrix: Mutual information matrix for variables
            var_indices: Current variable indices being factorized
            preference: Preference parameter for affinity propagation
            depth: Current recursion depth

        Returns:
            List of cliques (variable groups)
        """
        n_vars = len(var_indices)
        cliques = []

        # Add noise to MI matrix if deep recursion to help convergence
        add_noise = depth >= self.max_recursion_depth

        # Cluster variables using affinity propagation
        labels, converged = self._affinity_clustering(
            mi_matrix, preference, add_noise=add_noise
        )

        # Group variables by cluster
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

        for label in unique_labels:
            cluster_mask = labels == label
            cluster_vars = var_indices[cluster_mask]
            cluster_size = len(cluster_vars)

            if cluster_size > self.max_clique_size and depth < self.max_recursion_depth:
                # Recursively factorize large clusters
                cluster_mi = mi_matrix[np.ix_(cluster_mask, cluster_mask)]

                # Adjust preference for recursion
                if preference is None:
                    sub_pref = np.median(cluster_mi)
                else:
                    mi_range = np.max(cluster_mi) - np.min(cluster_mi)
                    sub_pref = np.random.rand() * mi_range + np.min(cluster_mi)

                sub_cliques = self._recursive_factorization(
                    cluster_mi, cluster_vars, sub_pref, depth + 1
                )
                cliques.extend(sub_cliques)
            else:
                # Create clique for this cluster
                cliques.append(cluster_vars)

        return cliques

    def _create_clique_structure(
        self, cliques_list: List[np.ndarray], n_vars: int
    ) -> np.ndarray:
        """
        Convert list of cliques to standard clique array format

        Args:
            cliques_list: List of arrays, each containing variable indices in a clique
            n_vars: Total number of variables

        Returns:
            Array where each row is [n_overlap, n_new, overlap_vars..., new_vars...]
        """
        max_clique_size = max(len(c) for c in cliques_list)
        # Format: [n_overlap, n_new, variables...]
        cliques_array = np.zeros((len(cliques_list), max_clique_size + 2), dtype=int)

        for i, clique in enumerate(cliques_list):
            n_vars_in_clique = len(clique)
            # First clique has no overlap
            cliques_array[i, 0] = 0  # n_overlap
            cliques_array[i, 1] = n_vars_in_clique  # n_new
            cliques_array[i, 2 : 2 + n_vars_in_clique] = sorted(clique)

        return cliques_array

    def _learn_clique_parameters(
        self,
        cliques: np.ndarray,
        population: np.ndarray,
        cardinality: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Learn probability tables for each clique

        Args:
            cliques: Clique structure array
            population: Population data
            cardinality: Variable cardinalities

        Returns:
            List of probability tables
        """
        tables = []
        n_samples = population.shape[0]

        for i in range(cliques.shape[0]):
            n_overlap = int(cliques[i, 0])
            n_new = int(cliques[i, 1])
            clique_vars = cliques[i, 2 : 2 + n_overlap + n_new].astype(int)
            clique_vars = clique_vars[clique_vars >= 0]  # Remove padding

            # Get cardinalities for variables in this clique
            clique_cards = cardinality[clique_vars].astype(int)
            total_card = np.prod(clique_cards)

            # Count occurrences of each configuration
            counts = np.zeros(total_card)

            for sample in population:
                # Convert variable values to a single index
                idx = 0
                multiplier = 1
                for j, var in enumerate(clique_vars):
                    val = int(sample[var])
                    idx += val * multiplier
                    multiplier *= clique_cards[j]

                counts[int(idx)] += 1

            # Apply Laplace smoothing
            if self.alpha > 0:
                counts += self.alpha

            # Normalize to probabilities
            probs = counts / np.sum(counts)
            tables.append(probs)

        return tables

    def learn(
        self,
        generation: int,
        n_vars: int,
        cardinality: np.ndarray,
        population: np.ndarray,
        fitness: np.ndarray,
        **params: Any,
    ) -> FactorizedModel:
        """
        Learn factorization using affinity propagation

        Args:
            generation: Current generation number
            n_vars: Number of variables
            cardinality: Variable cardinalities
            population: Selected population to learn from
            fitness: Fitness values (not used)
            **params: Additional parameters

        Returns:
            Learned FactorizedModel with affinity-based factorization
        """
        # Learn univariate and bivariate marginal probabilities
        univ_prob, biv_prob = find_marginal_prob(population, n_vars, cardinality)

        # Compute mutual information matrix
        mi_matrix = self._compute_mutual_information_matrix(
            population, n_vars, cardinality, univ_prob, biv_prob
        )

        # Determine preference
        pref = self.preference
        if pref is None:
            pref = np.median(mi_matrix)

        # Perform recursive factorization
        var_indices = np.arange(n_vars)
        if self.recursive:
            cliques_list = self._recursive_factorization(mi_matrix, var_indices, pref)
        else:
            # Non-recursive: single level of clustering
            labels, _ = self._affinity_clustering(mi_matrix, pref)
            cliques_list = []
            for label in np.unique(labels):
                cluster_vars = var_indices[labels == label]
                if len(cluster_vars) <= self.max_clique_size:
                    cliques_list.append(cluster_vars)
                else:
                    # Split large clusters into smaller ones
                    for i in range(0, len(cluster_vars), self.max_clique_size):
                        cliques_list.append(
                            cluster_vars[i : i + self.max_clique_size]
                        )

        # Convert to standard clique format
        cliques = self._create_clique_structure(cliques_list, n_vars)

        # Learn parameters for each clique
        tables = self._learn_clique_parameters(cliques, population, cardinality)

        # Create and return model
        model = FactorizedModel(
            structure=cliques,
            parameters=tables,
            metadata={
                "generation": generation,
                "model_type": "AffinityFactorization",
                "max_clique_size": self.max_clique_size,
                "n_cliques": len(cliques_list),
                "mi_matrix": mi_matrix,
                "preference": pref,
            },
        )

        return model


class LearnAffinityFactorizationElim(LearningMethod):
    """
    Learn factorization using affinity propagation with elimination strategy

    Similar to LearnAffinityFactorization but uses an elimination strategy
    where large clusters are eliminated and recursively processed together,
    rather than individually. This can lead to different factorizations.
    """

    def __init__(
        self,
        max_clique_size: int = 5,
        preference: Optional[float] = None,
        damping: float = 0.9,
        max_iter: int = 200,
        convergence_iter: int = 15,
        alpha: float = 0.0,
        max_recursion_depth: int = 10,
        max_convergence_retries: int = 10,
    ):
        """
        Initialize affinity-based factorization with elimination

        Args:
            max_clique_size: Maximum size of factors/cliques
            preference: Preference parameter for affinity propagation
            damping: Damping factor (higher for better convergence)
            max_iter: Maximum iterations for affinity propagation
            convergence_iter: Iterations with no change for convergence
            alpha: Smoothing parameter for probability estimation
            max_recursion_depth: Maximum recursion depth
            max_convergence_retries: Max retries if clustering doesn't converge
        """
        self.max_clique_size = max_clique_size
        self.preference = preference
        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.alpha = alpha
        self.max_recursion_depth = max_recursion_depth
        self.max_convergence_retries = max_convergence_retries

    def _compute_mutual_information_matrix(
        self,
        population: np.ndarray,
        n_vars: int,
        cardinality: np.ndarray,
        univ_prob: List[np.ndarray],
        biv_prob: List[List[np.ndarray]],
    ) -> np.ndarray:
        """Compute mutual information matrix (same as LearnAffinityFactorization)"""
        mi_matrix = np.zeros((n_vars, n_vars))

        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                mi = 0.0
                card_i = int(cardinality[i])
                card_j = int(cardinality[j])

                for k in range(card_i):
                    for l in range(card_j):
                        idx = card_j * k + l
                        p_ij = biv_prob[i][j][idx]
                        p_i = univ_prob[i][k]
                        p_j = univ_prob[j][l]

                        if p_ij > 0 and p_i > 0 and p_j > 0:
                            mi += p_ij * np.log2(p_ij / (p_i * p_j))

                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi

        return mi_matrix

    def _affinity_clustering_with_retry(
        self, similarity_matrix: np.ndarray, preference: Optional[float]
    ) -> Tuple[np.ndarray, bool]:
        """
        Perform affinity propagation with retries on non-convergence

        Args:
            similarity_matrix: Similarity matrix
            preference: Preference parameter

        Returns:
            Tuple of (cluster labels, converged flag)
        """
        converged = False
        labels = None
        n_retries = 0

        while not converged and n_retries < self.max_convergence_retries:
            # Add noise on retries to help convergence
            if n_retries > 0:
                noise = np.random.randn(*similarity_matrix.shape) * 0.01 * np.mean(
                    np.abs(similarity_matrix)
                )
                sim_matrix = similarity_matrix + noise
            else:
                sim_matrix = similarity_matrix

            pref = preference if preference is not None else np.median(sim_matrix)

            try:
                ap = AffinityPropagation(
                    damping=self.damping,
                    max_iter=self.max_iter,
                    convergence_iter=self.convergence_iter,
                    preference=pref,
                    affinity="precomputed",
                    random_state=None,
                )
                ap.fit(sim_matrix)
                labels = ap.labels_
                converged = ap.n_iter_ < self.max_iter
            except Exception:
                labels = np.zeros(similarity_matrix.shape[0], dtype=int)
                converged = False

            n_retries += 1

        if labels is None:
            labels = np.zeros(similarity_matrix.shape[0], dtype=int)

        return labels, converged

    def _elimination_factorization(
        self,
        mi_matrix: np.ndarray,
        var_indices: np.ndarray,
        preference: Optional[float],
        depth: int = 0,
    ) -> List[np.ndarray]:
        """
        Factorize using elimination strategy

        Variables are clustered, and small clusters are kept as cliques.
        Large clusters are collected and recursively factorized together.

        Args:
            mi_matrix: Mutual information matrix
            var_indices: Variable indices
            preference: Preference parameter
            depth: Current recursion depth

        Returns:
            List of cliques
        """
        n_vars = len(var_indices)
        cliques = []

        # Add noise at deep recursion levels
        if depth >= self.max_recursion_depth:
            noise = np.random.randn(*mi_matrix.shape) * np.mean(mi_matrix) * 0.1
            mi_matrix = mi_matrix + noise

        # Cluster variables
        labels, converged = self._affinity_clustering_with_retry(mi_matrix, preference)

        # Separate clusters by size
        small_clusters = []
        large_cluster_vars = []

        for label in np.unique(labels):
            cluster_mask = labels == label
            cluster_vars = var_indices[cluster_mask]

            if len(cluster_vars) <= self.max_clique_size:
                small_clusters.append(cluster_vars)
            else:
                large_cluster_vars.extend(cluster_vars)

        # Add small clusters as cliques
        cliques.extend(small_clusters)

        # Recursively process large clusters together
        if len(large_cluster_vars) > 0:
            if depth < self.max_recursion_depth:
                large_vars = np.array(large_cluster_vars)

                # Create sub-matrix for large cluster variables
                var_to_idx = {v: i for i, v in enumerate(var_indices)}
                large_indices = np.array([var_to_idx[v] for v in large_vars])
                sub_mi = mi_matrix[np.ix_(large_indices, large_indices)]

                # Use median as preference for recursion
                sub_pref = np.median(sub_mi)

                # Recursive call
                sub_cliques = self._elimination_factorization(
                    sub_mi, large_vars, sub_pref, depth + 1
                )
                cliques.extend(sub_cliques)
            else:
                # Reached max recursion depth, add all remaining large vars as single clique
                # to ensure all variables are covered
                cliques.append(np.array(large_cluster_vars))

        return cliques

    def _create_clique_structure(
        self, cliques_list: List[np.ndarray], n_vars: int
    ) -> np.ndarray:
        """Convert list of cliques to standard array format"""
        max_clique_size = max(len(c) for c in cliques_list)
        cliques_array = np.zeros((len(cliques_list), max_clique_size + 2), dtype=int)

        for i, clique in enumerate(cliques_list):
            n_vars_in_clique = len(clique)
            cliques_array[i, 0] = 0  # n_overlap
            cliques_array[i, 1] = n_vars_in_clique  # n_new
            cliques_array[i, 2 : 2 + n_vars_in_clique] = sorted(clique)

        return cliques_array

    def _learn_clique_parameters(
        self,
        cliques: np.ndarray,
        population: np.ndarray,
        cardinality: np.ndarray,
    ) -> List[np.ndarray]:
        """Learn probability tables for each clique"""
        tables = []
        n_samples = population.shape[0]

        for i in range(cliques.shape[0]):
            n_overlap = int(cliques[i, 0])
            n_new = int(cliques[i, 1])
            clique_vars = cliques[i, 2 : 2 + n_overlap + n_new].astype(int)
            clique_vars = clique_vars[clique_vars >= 0]

            clique_cards = cardinality[clique_vars].astype(int)
            total_card = np.prod(clique_cards)

            counts = np.zeros(total_card)

            for sample in population:
                idx = 0
                multiplier = 1
                for j, var in enumerate(clique_vars):
                    val = int(sample[var])
                    idx += val * multiplier
                    multiplier *= clique_cards[j]

                counts[int(idx)] += 1

            if self.alpha > 0:
                counts += self.alpha

            probs = counts / np.sum(counts)
            tables.append(probs)

        return tables

    def learn(
        self,
        generation: int,
        n_vars: int,
        cardinality: np.ndarray,
        population: np.ndarray,
        fitness: np.ndarray,
        **params: Any,
    ) -> FactorizedModel:
        """
        Learn factorization using affinity propagation with elimination

        Args:
            generation: Current generation number
            n_vars: Number of variables
            cardinality: Variable cardinalities
            population: Selected population to learn from
            fitness: Fitness values (not used)
            **params: Additional parameters

        Returns:
            Learned FactorizedModel
        """
        # Learn marginal probabilities
        univ_prob, biv_prob = find_marginal_prob(population, n_vars, cardinality)

        # Compute mutual information matrix
        mi_matrix = self._compute_mutual_information_matrix(
            population, n_vars, cardinality, univ_prob, biv_prob
        )

        # Determine preference
        pref = self.preference if self.preference is not None else np.median(mi_matrix)

        # Perform elimination-based factorization
        var_indices = np.arange(n_vars)
        cliques_list = self._elimination_factorization(mi_matrix, var_indices, pref)

        # Convert to standard format
        cliques = self._create_clique_structure(cliques_list, n_vars)

        # Learn parameters
        tables = self._learn_clique_parameters(cliques, population, cardinality)

        # Create model
        model = FactorizedModel(
            structure=cliques,
            parameters=tables,
            metadata={
                "generation": generation,
                "model_type": "AffinityFactorizationElim",
                "max_clique_size": self.max_clique_size,
                "n_cliques": len(cliques_list),
                "mi_matrix": mi_matrix,
                "preference": pref,
            },
        )

        return model
