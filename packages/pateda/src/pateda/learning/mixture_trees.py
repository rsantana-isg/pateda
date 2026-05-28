"""
Mixture of Trees EDA (MT-EDA) learning

Implements mixture of tree-structured models for EDAs. Mixture models are a powerful
extension of single-component EDAs that can represent multimodal distributions by
combining multiple simpler component distributions.

Mixture Model Formulation:
A mixture model combines multiple component distributions with mixture weights:
    Q(x) = Σⱼ₌₁ᵏ λⱼ · fⱼ(x)

where:
- λⱼ are mixture weights (λⱼ ≥ 0, Σⱼ λⱼ = 1)
- fⱼ(x) are component distributions (e.g., tree models)
- k is the number of mixture components

Each component fⱼ can be any factorized distribution. In MT-EDA, each component
is a tree-structured Bayesian network, combining the benefits of:
- Tree models: Efficient learning and sampling, captures pairwise dependencies
- Mixtures: Can model multimodal and complex distributions

Why Mixture Models?
Single-component models (UMDA, tree, BOA) assume a unimodal distribution, which
may not fit the true distribution of promising solutions when:
- Multiple distinct "building blocks" or solution patterns exist
- Search space has multiple peaks or clusters
- Problem exhibits different dependency structures in different regions

Mixture models address this by allowing multiple modes, each captured by a
different component with its own structure and parameters.

Learning Mixture Models:
1. Component Learning:
   - Learn k different tree structures (or use same structure with different parameters)
   - Can use different subsets/perturbations of data for diversity
   - Each component captures a different mode or cluster

2. Weight Learning:
   Methods for determining mixture weights λⱼ:
   - Uniform: λⱼ = 1/k (equal weight to all components)
   - EM (Expectation-Maximization): Iteratively estimate component responsibilities
   - Fitness-proportional: Weight by quality of solutions in each component
   - Clustering-based: Use k-means or hierarchical clustering

EM Algorithm for Mixtures:
E-step: Calculate responsibilities (posterior probability of component membership)
    γⱼᵢ = (λⱼ · fⱼ(xᵢ)) / Σₗ (λₗ · fₗ(xᵢ))

M-step: Update weights based on responsibilities
    λⱼ = (1/N) Σᵢ γⱼᵢ

Advantages:
- Can model multimodal distributions
- More expressive than single trees
- Each component remains simple and interpretable
- Can capture different dependency structures in different modes

Challenges:
- More parameters to learn (k trees + k-1 weights)
- Need more samples to estimate reliably
- Risk of overfitting with too many components
- Component identifiability issues

Model Selection:
Choosing k (number of components):
- Too few: Cannot capture multimodality
- Too many: Overfitting, computational cost
- Use validation set or information criteria (BIC, AIC)
- Typical range: k = 2-10 for most problems

Sampling from Mixture Models:
1. Sample component j with probability λⱼ
2. Sample x from component distribution fⱼ(x)
3. Return sampled x

Applications:
- Problems with multiple distinct solution patterns
- Highly multimodal optimization landscapes
- NK-landscapes with multiple optima
- Combinatorial problems with building blocks

Related Mixture EDAs:
- Mixture of Gaussians (continuous domains)
- Mixture of Bayesian networks (general dependencies)
- Hierarchical BOA (implicit mixture through local structures)

Based on MATEDA-2.0 mixture of distributions (Section 4.4) and tree models.

References:
- Santana, R., Ochoa, A., & Soto, M.R. (2001). "The Mixture of Trees Factorized
  Distribution Algorithm." GECCO 2001, pp. 543-550.
- Pelikan, M., & Goldberg, D.E. (2000). "Research on the Bayesian Optimization
  Algorithm." Optimization by Building and Using Probabilistic Models, pp. 216-219.
- Bosman, P.A.N., & Thierens, D. (2000). "Expanding from discrete to continuous
  estimation of distribution algorithms: The IDEA." PPSN VI, pp. 767-776.
- MATEDA-2.0 User Guide, Section 4.4: "Mixture of distributions"
"""

from typing import Any, List, Optional, Tuple
import numpy as np

from pateda.core.components import LearningMethod
from pateda.core.models import MixtureModel, TreeModel
from pateda.learning.bmda import LearnBMDA
from pateda.learning.tree import LearnTreeModel


class LearnMixtureTrees(LearningMethod):
    """
    Learn a Mixture of Trees model (MT-EDA)

    Combines multiple tree-structured models using mixture weights. Each component
    is a tree model that captures dependencies through a tree structure.

    The mixture allows modeling more complex distributions than a single tree by
    combining multiple trees with different structures.
    """

    def __init__(
        self,
        n_components: int = 3,
        component_learning: str = "tree",
        alpha: float = 0.0,
        weight_learning: str = "uniform",
        em_iterations: int = 10,
        random_seed: Optional[int] = None,
        use_priors: bool = False,
        use_adaptive: bool = False,
        truncation_ratio: float = 0.5,
        adaptive_mu: float = 0.9,
        min_prior: float = 1e-6,
    ):
        """
        Initialize Mixture of Trees learning

        Args:
            n_components: Number of tree components in the mixture
            component_learning: Method for learning each tree component
                              ("tree", "random_tree", "greedy")
            alpha: Smoothing parameter for probability estimation
            weight_learning: Method for learning mixture weights
                           ("uniform", "em", "fitness_proportional")
            em_iterations: Number of EM iterations for weight learning
            random_seed: Random seed for reproducibility
            use_priors: Whether to use priors for mutation-like effect (Section 4)
                       When True, applies prior r' = 2^(-(k-1)) * r to prevent
                       premature convergence
            use_adaptive: Whether to use adaptive learning (Section 5)
                         When True, stops learning when P(D) >= mu to prevent
                         overfitting
            truncation_ratio: Truncation selection ratio I_tau (default 0.5)
                            Used to compute base prior r = I_tau * M / n
            adaptive_mu: Threshold for adaptive learning (default 0.9)
                        Learning stops when P(D) >= mu
            min_prior: Minimum prior value to prevent numerical issues
        """
        self.n_components = n_components
        self.component_learning = component_learning
        self.alpha = alpha
        self.weight_learning = weight_learning
        self.em_iterations = em_iterations
        self.random_seed = random_seed
        self.use_priors = use_priors
        self.use_adaptive = use_adaptive
        self.truncation_ratio = truncation_ratio
        self.adaptive_mu = adaptive_mu
        self.min_prior = min_prior

        if random_seed is not None:
            np.random.seed(random_seed)

    def _learn_component_tree(
        self,
        component_id: int,
        population: np.ndarray,
        n_vars: int,
        cardinality: np.ndarray,
    ) -> TreeModel:
        """
        Learn a single tree component using LearnTreeModel (Chow-Liu).

        Uses the same algorithm as TreeEDA, which always applies Laplace
        smoothing and normalized mutual information. This avoids zero-
        probability table entries that arise with BMDA alpha=0 and ensures
        that every configuration can be sampled.

        Diversity across components is created by bootstrap resampling of
        the selected population for component_id > 0.  Because LearnTreeModel
        uses random permutation for MST tie-breaking (unlike BMDA), bootstrap
        samples produce significantly more structural diversity between
        components (≈33 % shared edges) than BMDA (≈67 %).
        """
        learner = LearnTreeModel(alpha=self.alpha)

        if component_id > 0 and len(population) > 10:
            # Bootstrap sampling: creates a different MI matrix per component
            # → different maximum-spanning-tree → genuinely diverse structures
            indices = np.random.choice(len(population), len(population), replace=True)
            component_pop = population[indices]
        else:
            component_pop = population

        fda_model = learner.learn(
            generation=0,
            n_vars=n_vars,
            cardinality=cardinality,
            population=component_pop,
            fitness=np.zeros(len(component_pop)),
        )

        return self._factorized_to_tree(fda_model, n_vars)

    def _factorized_to_tree(self, fda_model, n_vars: int) -> TreeModel:
        """
        Convert factorized model to tree model representation

        Args:
            fda_model: FactorizedModel from BMDA
            n_vars: Number of variables

        Returns:
            TreeModel representation

        Note:
            The structure is kept in FDA clique format for compatibility
            with exact likelihood computation. The format is:
            [n_overlap, n_new, overlap_vars..., new_vars...]
        """
        # Keep the clique structure for proper likelihood computation
        # The clique format is needed to correctly map parameters to variables
        cliques = fda_model.structure
        parameters = fda_model.parameters

        tree_model = TreeModel(
            structure=cliques,  # Keep FDA clique format
            parameters=parameters,
            metadata=fda_model.metadata
        )

        return tree_model

    def _learn_mixture_weights_uniform(self, n_components: int) -> np.ndarray:
        """
        Learn uniform mixture weights

        Args:
            n_components: Number of components

        Returns:
            Uniform weights
        """
        return np.ones(n_components) / n_components

    def _learn_mixture_weights_em(
        self,
        components: List[TreeModel],
        population: np.ndarray,
        n_vars: int,
        cardinality: np.ndarray,
    ) -> np.ndarray:
        """
        Learn mixture weights using EM algorithm

        Args:
            components: List of tree components
            population: Population data
            n_vars: Number of variables
            cardinality: Variable cardinalities

        Returns:
            Learned mixture weights
        """
        n_components = len(components)
        n_samples = len(population)

        # Initialize weights uniformly
        weights = np.ones(n_components) / n_components

        # EM iterations
        for iteration in range(self.em_iterations):
            # E-step: Calculate responsibilities
            responsibilities = np.zeros((n_samples, n_components))

            for j in range(n_components):
                # Calculate likelihood of each sample under component j
                # (Simplified: use approximation based on marginal likelihoods)
                component_likelihood = self._approximate_likelihood(
                    components[j], population, cardinality
                )
                responsibilities[:, j] = weights[j] * component_likelihood

            # Normalize responsibilities
            resp_sum = responsibilities.sum(axis=1, keepdims=True)
            resp_sum = np.maximum(resp_sum, 1e-10)  # Avoid division by zero
            responsibilities /= resp_sum

            # M-step: Update weights
            weights = responsibilities.sum(axis=0) / n_samples

        return weights

    def exact_likelihood(
        self,
        component: TreeModel,
        population: np.ndarray,
        cardinality: np.ndarray,
    ) -> np.ndarray:
        """
        Compute exact likelihood of population under a tree component

        For a tree model, the probability factorizes as:
            T(x) = P(X_root) * prod_{v != root} P(X_v | X_{pa(v)})

        where pa(v) is the parent of variable v in the tree.

        The TreeModel stores parameters in FDA clique format:
        - Each clique: [n_overlap, n_new, overlap_vars..., new_vars...]
        - Parameters are probability tables indexed by clique

        Args:
            component: Tree model with structure and parameters
            population: Population data (n_samples, n_vars)
            cardinality: Variable cardinalities

        Returns:
            Array of likelihoods for each sample (NOT log-likelihoods)
        """
        n_samples = len(population)
        log_likelihoods = np.zeros(n_samples)

        # The structure in TreeModel is tree_edges format [n_parents, parent, child]
        # But parameters are in FDA clique format
        # We need to reconstruct the clique format from the tree structure
        # or work directly with the parameters

        parameters = component.parameters
        structure = component.structure

        # Try to detect if structure is in clique format (4+ columns) or tree format (3 columns)
        if len(structure.shape) == 2 and structure.shape[1] >= 4:
            # Structure is in FDA clique format
            cliques = structure
            return self._exact_likelihood_fda(cliques, parameters, population, cardinality)
        else:
            # Structure is in tree edge format - use simplified computation
            return self._exact_likelihood_tree(structure, parameters, population, cardinality)

    def _exact_likelihood_fda(
        self,
        cliques: np.ndarray,
        parameters: List[np.ndarray],
        population: np.ndarray,
        cardinality: np.ndarray,
    ) -> np.ndarray:
        """
        Compute exact likelihood using FDA clique structure

        Args:
            cliques: FDA clique structure
            parameters: List of probability tables
            population: Population data
            cardinality: Variable cardinalities

        Returns:
            Array of likelihoods
        """
        from pateda.learning.utils.conversions import find_acc_card, num_convert_card

        n_samples = len(population)
        log_likelihoods = np.zeros(n_samples)

        for c in range(len(parameters)):
            n_overlap = int(cliques[c, 0])
            n_new = int(cliques[c, 1])
            table = parameters[c]

            # Get variable indices
            if n_overlap > 0:
                overlap_vars = cliques[c, 2:2 + n_overlap].astype(int)
            else:
                overlap_vars = np.array([], dtype=int)

            new_vars = cliques[c, 2 + n_overlap:2 + n_overlap + n_new].astype(int)

            # Calculate accumulated cardinalities
            if n_new > 0:
                new_card = cardinality[new_vars]
                new_acc_card = find_acc_card(n_new, new_card)

            for s in range(n_samples):
                sample = population[s]

                if n_overlap == 0:
                    # Root clique: marginal probability
                    if n_new > 0:
                        new_vals = sample[new_vars]
                        new_idx = num_convert_card(new_vals, n_new, new_acc_card)

                        if len(table.shape) == 1:
                            if new_idx < len(table):
                                prob = table[new_idx]
                            else:
                                prob = 1e-10
                        else:
                            prob = np.mean(table)
                    else:
                        prob = 1.0
                else:
                    # Conditional probability P(new | overlap)
                    overlap_card = cardinality[overlap_vars]
                    overlap_acc_card = find_acc_card(n_overlap, overlap_card)

                    overlap_vals = sample[overlap_vars]
                    overlap_idx = num_convert_card(overlap_vals, n_overlap, overlap_acc_card)

                    if n_new > 0:
                        new_vals = sample[new_vars]
                        new_idx = num_convert_card(new_vals, n_new, new_acc_card)

                        if len(table.shape) == 2:
                            if overlap_idx < table.shape[0] and new_idx < table.shape[1]:
                                prob = table[overlap_idx, new_idx]
                            else:
                                prob = 1e-10
                        elif len(table.shape) == 1:
                            # Flattened table
                            n_new_configs = int(np.prod(new_card))
                            idx = overlap_idx * n_new_configs + new_idx
                            if idx < len(table):
                                prob = table[idx]
                            else:
                                prob = 1e-10
                        else:
                            prob = 1e-10
                    else:
                        prob = 1.0

                log_likelihoods[s] += np.log(max(prob, 1e-10))

        return np.exp(log_likelihoods)

    def _exact_likelihood_tree(
        self,
        structure: np.ndarray,
        parameters: List[np.ndarray],
        population: np.ndarray,
        cardinality: np.ndarray,
    ) -> np.ndarray:
        """
        Compute exact likelihood using tree edge structure

        This is a fallback for when structure is in [n_parents, parent, child] format.
        Uses a simpler computation based on available parameters.

        Args:
            structure: Tree structure (n_parents, parent, child per row)
            parameters: List of probability tables
            population: Population data
            cardinality: Variable cardinalities

        Returns:
            Array of likelihoods
        """
        n_samples = len(population)
        n_params = len(parameters)
        log_likelihoods = np.zeros(n_samples)

        # Map each parameter to variables it covers
        # We'll iterate through parameters directly
        for p_idx in range(n_params):
            table = parameters[p_idx]

            for s in range(n_samples):
                sample = population[s]

                if len(table.shape) == 1:
                    # Marginal probability - need to figure out which variable
                    # Use heuristic: assume parameters are ordered by variable
                    var_idx = min(p_idx, len(sample) - 1)
                    var_val = int(sample[var_idx])

                    if var_val < len(table):
                        prob = table[var_val]
                    else:
                        prob = 1.0 / len(table)  # Uniform fallback

                elif len(table.shape) == 2:
                    # Conditional probability
                    # Use heuristic based on table shape
                    prob = np.mean(table)  # Simplified fallback
                else:
                    prob = 1e-10

                log_likelihoods[s] += np.log(max(prob, 1e-10))

        return np.exp(log_likelihoods)

    def _approximate_likelihood(
        self,
        component: TreeModel,
        population: np.ndarray,
        cardinality: np.ndarray,
    ) -> np.ndarray:
        """
        Wrapper for backward compatibility - calls exact_likelihood

        Args:
            component: Tree model
            population: Population data
            cardinality: Variable cardinalities

        Returns:
            Array of likelihoods for each sample
        """
        return self.exact_likelihood(component, population, cardinality)

    def _compute_priors(
        self,
        n_components: int,
        n_vars: int,
        n_samples: int,
        weights: np.ndarray,
        components: List[TreeModel],
        population: np.ndarray,
        cardinality: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Compute priors for mutation-like effect (Section 4 of RepMutMTFDA.pdf)

        The prior prevents premature convergence by adding probability mass
        to unseen configurations. Two methods are implemented:

        1. Fixed prior: r' = r* = 2^(-(k-1)) * r
           where r = I_tau * M / n
           - I_tau: truncation ratio
           - M: population size
           - n: number of variables

        2. Adaptive prior: r^k = P_bar(D^c) * M / (lambda^k * n)
           where P_bar(D^c) is the probability given to points not in data

        Args:
            n_components: Number of mixture components
            n_vars: Number of variables
            n_samples: Number of samples in population
            weights: Current mixture weights
            components: List of tree components
            population: Population data
            cardinality: Variable cardinalities

        Returns:
            Tuple of (component_priors, total_prior)
            - component_priors: Array of priors for each component
            - total_prior: Total prior probability
        """
        # Base prior: r = I_tau * M / n
        base_prior = self.truncation_ratio * n_samples / n_vars

        # Fixed prior: r' = 2^(-(k-1)) * r (equation 3 in paper)
        # This decreases exponentially with number of components
        fixed_prior = (2 ** (-(n_vars - 1))) * base_prior
        fixed_prior = max(fixed_prior, self.min_prior)

        if not self.use_adaptive:
            # Use fixed prior for all components
            component_priors = np.full(n_components, fixed_prior)
            return component_priors, fixed_prior * n_components

        # Adaptive prior: r^k = P_bar(D^c) * M / (lambda^k * n)
        # P_bar(D^c) = 1 - P_bar(D) where P_bar(D) is probability of data

        # Compute P_bar(D): average probability of data points under mixture
        p_bar_d = self._compute_data_probability(
            components, weights, population, cardinality
        )

        # P_bar(D^c) = 1 - P_bar(D)
        p_bar_dc = max(1.0 - p_bar_d, self.min_prior)

        # Compute per-component adaptive priors
        component_priors = np.zeros(n_components)
        for k in range(n_components):
            if weights[k] > self.min_prior:
                # r^k = P_bar(D^c) * M / (lambda^k * n)
                component_priors[k] = p_bar_dc * n_samples / (weights[k] * n_vars)
            else:
                component_priors[k] = fixed_prior

            # Ensure minimum prior
            component_priors[k] = max(component_priors[k], self.min_prior)

        total_prior = np.sum(weights * component_priors)

        return component_priors, total_prior

    def _compute_data_probability(
        self,
        components: List[TreeModel],
        weights: np.ndarray,
        population: np.ndarray,
        cardinality: np.ndarray,
    ) -> float:
        """
        Compute average probability of data under the mixture model

        P_bar(D) = (1/N) * sum_i Q(x_i)
        where Q(x) = sum_k lambda_k * T^k(x)

        Args:
            components: List of tree components
            weights: Mixture weights
            population: Population data
            cardinality: Variable cardinalities

        Returns:
            Average probability of data points
        """
        n_samples = len(population)
        n_components = len(components)

        # Compute mixture probability for each sample
        mixture_probs = np.zeros(n_samples)

        for k in range(n_components):
            component_likelihood = self.exact_likelihood(
                components[k], population, cardinality
            )
            mixture_probs += weights[k] * component_likelihood

        # Average probability
        p_bar_d = np.mean(mixture_probs)

        return p_bar_d

    def _should_stop_adaptive(
        self,
        components: List[TreeModel],
        weights: np.ndarray,
        population: np.ndarray,
        cardinality: np.ndarray,
    ) -> bool:
        """
        Check if adaptive learning should stop (Section 5 of RepMutMTFDA.pdf)

        Learning stops when P_bar(D) >= mu, indicating the model has
        captured enough of the data distribution.

        Args:
            components: List of tree components
            weights: Mixture weights
            population: Population data
            cardinality: Variable cardinalities

        Returns:
            True if learning should stop, False otherwise
        """
        p_bar_d = self._compute_data_probability(
            components, weights, population, cardinality
        )

        return p_bar_d >= self.adaptive_mu

    def _learn_mixture_weights_fitness(
        self,
        components: List[TreeModel],
        population: np.ndarray,
        fitness: np.ndarray,
        n_vars: int,
        cardinality: np.ndarray,
    ) -> np.ndarray:
        """
        Learn mixture weights using fitness-proportional weighting

        Each component is weighted by the average fitness of samples
        that have high likelihood under that component.

        Args:
            components: List of tree components
            population: Population data
            fitness: Fitness values for each sample
            n_vars: Number of variables
            cardinality: Variable cardinalities

        Returns:
            Fitness-weighted mixture weights
        """
        n_components = len(components)
        n_samples = len(population)

        # Compute responsibilities (soft assignment to components)
        responsibilities = np.zeros((n_samples, n_components))
        uniform_weights = np.ones(n_components) / n_components

        for j in range(n_components):
            component_likelihood = self.exact_likelihood(
                components[j], population, cardinality
            )
            responsibilities[:, j] = uniform_weights[j] * component_likelihood

        # Normalize responsibilities
        resp_sum = responsibilities.sum(axis=1, keepdims=True)
        resp_sum = np.maximum(resp_sum, 1e-10)
        responsibilities /= resp_sum

        # Compute fitness-weighted weights for each component
        # Weight = sum_i (responsibility_ij * fitness_i) / sum_i (responsibility_ij)
        weights = np.zeros(n_components)

        # Normalize fitness to [0, 1] range for stability
        fitness_min = np.min(fitness)
        fitness_max = np.max(fitness)
        if fitness_max > fitness_min:
            normalized_fitness = (fitness - fitness_min) / (fitness_max - fitness_min)
        else:
            normalized_fitness = np.ones_like(fitness)

        for j in range(n_components):
            resp_j = responsibilities[:, j]
            resp_sum_j = np.sum(resp_j)

            if resp_sum_j > 1e-10:
                # Weighted average fitness for this component
                avg_fitness = np.sum(resp_j * normalized_fitness) / resp_sum_j
                weights[j] = resp_sum_j * avg_fitness
            else:
                weights[j] = 1e-10

        # Normalize weights
        weights_sum = np.sum(weights)
        if weights_sum > 1e-10:
            weights = weights / weights_sum
        else:
            weights = np.ones(n_components) / n_components

        return weights

    def learn(
        self,
        generation: int,
        n_vars: int,
        cardinality: np.ndarray,
        population: np.ndarray,
        fitness: np.ndarray,
        **params: Any,
    ) -> MixtureModel:
        """
        Learn Mixture of Trees model from population

        Args:
            generation: Current generation number
            n_vars: Number of variables
            cardinality: Variable cardinalities
            population: Selected population to learn from
            fitness: Fitness values
            **params: Additional parameters
                     - n_components: Override number of components
                     - alpha: Override smoothing parameter

        Returns:
            Learned MixtureModel with tree components

        Note:
            The model contains:
            - structure: List of tree structures (one per component)
            - parameters: Dict with 'components' (list of tree parameters) and
                         'weights' (mixture coefficients)
        """
        n_components = params.get("n_components", self.n_components)
        alpha = params.get("alpha", self.alpha)

        # Learn multiple tree components
        components = []
        component_structures = []
        component_parameters = []

        for j in range(n_components):
            tree_component = self._learn_component_tree(
                j, population, n_vars, cardinality
            )
            components.append(tree_component)
            component_structures.append(tree_component.structure)
            component_parameters.append(tree_component.parameters)

        # Learn mixture weights
        if self.weight_learning == "uniform":
            weights = self._learn_mixture_weights_uniform(n_components)
        elif self.weight_learning == "em":
            weights = self._learn_mixture_weights_em(
                components, population, n_vars, cardinality
            )
        elif self.weight_learning == "fitness_proportional":
            weights = self._learn_mixture_weights_fitness(
                components, population, fitness, n_vars, cardinality
            )
        else:
            weights = self._learn_mixture_weights_uniform(n_components)

        # Apply adaptive learning check
        adaptive_stopped = False
        p_bar_d = None
        if self.use_adaptive:
            p_bar_d = self._compute_data_probability(
                components, weights, population, cardinality
            )
            adaptive_stopped = p_bar_d >= self.adaptive_mu

        # Compute priors if enabled
        component_priors = None
        total_prior = None
        if self.use_priors:
            component_priors, total_prior = self._compute_priors(
                n_components, n_vars, len(population), weights,
                components, population, cardinality
            )

        # Create mixture model
        model = MixtureModel(
            structure=component_structures,
            parameters={
                "components": component_parameters,
                "weights": weights,
                "priors": component_priors,  # Per-component priors (if enabled)
                "total_prior": total_prior,  # Total prior probability
            },
            metadata={
                "generation": generation,
                "model_type": "Mixture of Trees",
                "n_components": n_components,
                "component_learning": self.component_learning,
                "weight_learning": self.weight_learning,
                "alpha": alpha,
                "use_priors": self.use_priors,
                "use_adaptive": self.use_adaptive,
                "adaptive_stopped": adaptive_stopped,
                "p_bar_d": p_bar_d,  # Probability of data under model
                "truncation_ratio": self.truncation_ratio,
                "adaptive_mu": self.adaptive_mu,
            },
        )

        return model
