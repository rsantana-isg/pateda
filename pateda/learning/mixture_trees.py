"""
Mixture of Trees EDA (MT-EDA) learning

Implements mixture of tree-structured models for EDAs. A mixture model combines
multiple component distributions with mixture coefficients:

Q(x) = Σ_j λ_j · f_j(x)

where λ_j are mixture weights and f_j(x) are tree-structured component distributions.

Based on MATEDA-2.0 mixture of distributions (section 4.4) and tree models.

References:
    - R. Santana, A. Ochoa, M.R. Soto: The Mixture of Trees Factorized Distribution Algorithm.
      GECCO 2001.
"""

from typing import Any, List, Optional
import numpy as np

from pateda.core.components import LearningMethod
from pateda.core.models import MixtureModel, TreeModel
from pateda.learning.bmda import LearnBMDA


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
        """
        self.n_components = n_components
        self.component_learning = component_learning
        self.alpha = alpha
        self.weight_learning = weight_learning
        self.em_iterations = em_iterations
        self.random_seed = random_seed

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
        Learn a single tree component

        Args:
            component_id: Component identifier
            population: Population data
            n_vars: Number of variables
            cardinality: Variable cardinalities

        Returns:
            Learned TreeModel
        """
        # Use BMDA tree learning with different initializations for diversity
        learner = LearnBMDA(
            structure=None,
            structure_learning=self.component_learning,
            alpha=self.alpha
        )

        # Add perturbation to create diverse trees
        # Option 1: Use different random subsets of population
        if component_id > 0 and len(population) > 10:
            # Bootstrap sampling for diversity
            indices = np.random.choice(len(population), len(population), replace=True)
            component_pop = population[indices]
        else:
            component_pop = population

        # Learn factorized model (BMDA returns FactorizedModel, we'll convert to TreeModel)
        fda_model = learner.learn(
            generation=0,
            n_vars=n_vars,
            cardinality=cardinality,
            population=component_pop,
            fitness=np.zeros(len(component_pop))  # Not used
        )

        # Convert FactorizedModel to TreeModel structure
        tree_model = self._factorized_to_tree(fda_model, n_vars)

        return tree_model

    def _factorized_to_tree(self, fda_model, n_vars: int) -> TreeModel:
        """
        Convert factorized model to tree model representation

        Args:
            fda_model: FactorizedModel from BMDA
            n_vars: Number of variables

        Returns:
            TreeModel representation
        """
        # For tree models, the structure is a list of parent-child relationships
        # Extract from cliques (bivariate marginals in a tree)
        cliques = fda_model.structure
        parameters = fda_model.parameters

        # Build tree structure: each variable has at most one parent
        # Structure format: [n_parents, parent_idx, child_idx] for each edge
        tree_edges = []

        for i in range(len(parameters)):
            n_overlap = int(cliques[i, 0])
            n_new = int(cliques[i, 1])

            if n_overlap == 1 and n_new == 1:
                # This is a parent-child relationship
                parent = int(cliques[i, 2])
                child = int(cliques[i, 3])
                tree_edges.append([1, parent, child])
            elif n_overlap == 0:
                # Root node(s)
                for j in range(n_new):
                    var = int(cliques[i, 2 + j])
                    tree_edges.append([0, -1, var])  # -1 indicates root

        # Convert to numpy array
        if tree_edges:
            structure = np.array(tree_edges, dtype=int)
        else:
            # Empty tree (all independent)
            structure = np.array([[0, -1, i] for i in range(n_vars)], dtype=int)

        tree_model = TreeModel(
            structure=structure,
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

    def _approximate_likelihood(
        self,
        component: TreeModel,
        population: np.ndarray,
        cardinality: np.ndarray,
    ) -> np.ndarray:
        """
        Approximate likelihood of population under a tree component

        Args:
            component: Tree model
            population: Population data
            cardinality: Variable cardinalities

        Returns:
            Approximate log-likelihoods for each sample
        """
        n_samples = len(population)
        log_likelihoods = np.zeros(n_samples)

        # Simplified likelihood based on marginal probabilities
        # For a proper implementation, would need to compute full tree likelihood
        # Here we use sum of log probabilities from the tables
        parameters = component.parameters

        for i, table in enumerate(parameters):
            if len(table.shape) == 1:
                # Marginal distribution
                for s in range(n_samples):
                    # Assuming the table corresponds to a variable
                    # This is a simplified approximation
                    prob = np.mean(table)  # Placeholder
                    log_likelihoods[s] += np.log(max(prob, 1e-10))
            else:
                # Conditional distribution (simplified)
                for s in range(n_samples):
                    prob = np.mean(table)  # Placeholder
                    log_likelihoods[s] += np.log(max(prob, 1e-10))

        return np.exp(log_likelihoods)

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
            # Use fitness to weight components (not implemented yet)
            weights = self._learn_mixture_weights_uniform(n_components)
        else:
            weights = self._learn_mixture_weights_uniform(n_components)

        # Create mixture model
        model = MixtureModel(
            structure=component_structures,
            parameters={
                "components": component_parameters,
                "weights": weights,
            },
            metadata={
                "generation": generation,
                "model_type": "Mixture of Trees",
                "n_components": n_components,
                "component_learning": self.component_learning,
                "weight_learning": self.weight_learning,
                "alpha": alpha,
            },
        )

        return model
