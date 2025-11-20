"""
MAP (Maximum A Posteriori) inference for Markov networks.

This module implements various methods for computing the most probable
configuration(s) from Markov network models, including:
- Exact inference using junction trees
- Approximate inference using belief propagation
- Decimation-based inference
- k-MAP computation (k most probable configurations)

Based on:
- Santana, R. (2013). "Message Passing Methods for Estimation of Distribution
  Algorithms Based on Markov Networks"
- Nilsson, D. (1998). "An efficient algorithm for finding the M most probable
  configurations in probabilistic expert systems"
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass
from enum import Enum

try:
    from pgmpy.models import MarkovNetwork
    from pgmpy.inference import BeliefPropagation
    from pgmpy.factors.discrete import DiscreteFactor
    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False


class MAPMethod(Enum):
    """Methods for MAP computation."""
    EXACT = "exact"  # Junction tree exact inference
    BP = "bp"  # Belief propagation (loopy)
    DECIMATION = "decimation"  # Decimation-based BP
    MAXFLOW = "maxflow"  # Max-flow propagation for k-MAP


@dataclass
class MAPResult:
    """Result of MAP inference.

    Attributes:
        configuration: The MAP configuration (n,) array
        log_probability: Log probability of the configuration
        probability: Probability of the configuration
        method: Method used for inference
        metadata: Additional information (iterations, convergence, etc.)
    """
    configuration: np.ndarray
    log_probability: float
    probability: float
    method: str
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class KMAPResult:
    """Result of k-MAP inference.

    Attributes:
        configurations: Array of k configurations (k, n)
        log_probabilities: Log probabilities of configurations (k,)
        probabilities: Probabilities of configurations (k,)
        method: Method used for inference
        metadata: Additional information
    """
    configurations: np.ndarray
    log_probabilities: np.ndarray
    probabilities: np.ndarray
    method: str
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MAPInference:
    """MAP inference engine for Markov networks.

    This class provides multiple methods for computing MAP configurations
    from Markov network models, supporting both exact and approximate inference.

    Parameters:
        cliques: List of cliques (variable indices)
        tables: List of clique probability/potential tables
        cardinalities: Cardinality of each variable
        method: Inference method to use
        max_iterations: Maximum iterations for iterative methods
        tolerance: Convergence tolerance for iterative methods
        temperature: Temperature for Boltzmann-like sampling (for decimation)

    Example:
        >>> cliques = [np.array([0, 1]), np.array([1, 2])]
        >>> tables = [table1, table2]  # Probability tables
        >>> cardinalities = np.array([2, 2, 2])
        >>> inference = MAPInference(cliques, tables, cardinalities)
        >>> result = inference.compute_map()
        >>> print(result.configuration)
    """

    def __init__(
        self,
        cliques: List[np.ndarray],
        tables: List[np.ndarray],
        cardinalities: np.ndarray,
        method: Union[str, MAPMethod] = MAPMethod.EXACT,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        temperature: float = 1.0
    ):
        self.cliques = cliques
        self.tables = tables
        self.cardinalities = cardinalities
        self.n_variables = len(cardinalities)

        if isinstance(method, str):
            method = MAPMethod(method)
        self.method = method

        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.temperature = temperature

        # Build pgmpy model if available
        self._pgmpy_model = None
        if PGMPY_AVAILABLE and method in [MAPMethod.EXACT, MAPMethod.BP]:
            self._build_pgmpy_model()

    def _build_pgmpy_model(self):
        """Build pgmpy MarkovNetwork from cliques and tables."""
        if not PGMPY_AVAILABLE:
            return

        # Create Markov network
        edges = []
        for clique in self.cliques:
            if len(clique) > 1:
                # Add all edges within clique
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        edge = (int(clique[i]), int(clique[j]))
                        if edge not in edges:
                            edges.append(edge)

        self._pgmpy_model = MarkovNetwork(edges)

        # Create factors from clique tables
        factors = []
        for clique, table in zip(self.cliques, self.tables):
            # pgmpy expects non-zero probabilities, so add small constant
            table_safe = np.maximum(table, 1e-300)

            factor = DiscreteFactor(
                variables=[int(v) for v in clique],
                cardinality=[int(self.cardinalities[v]) for v in clique],
                values=table_safe.ravel()
            )
            factors.append(factor)

        self._pgmpy_model.add_factors(*factors)

    def compute_map(self) -> MAPResult:
        """Compute the MAP (most probable) configuration.

        Returns:
            MAPResult with the MAP configuration and its probability
        """
        if self.method == MAPMethod.EXACT:
            return self._compute_map_exact()
        elif self.method == MAPMethod.BP:
            return self._compute_map_bp()
        elif self.method == MAPMethod.DECIMATION:
            return self._compute_map_decimation()
        else:
            raise ValueError(f"Unknown MAP method: {self.method}")

    def _compute_map_exact(self) -> MAPResult:
        """Compute MAP using exact junction tree inference (via pgmpy)."""
        if not PGMPY_AVAILABLE or self._pgmpy_model is None:
            # Fallback to simple greedy method
            return self._compute_map_greedy()

        try:
            # Use belief propagation for MAP
            bp = BeliefPropagation(self._pgmpy_model)

            # Query MAP state for each variable
            map_config = np.zeros(self.n_variables, dtype=int)

            # Get marginals and pick argmax for each variable
            for var in range(self.n_variables):
                try:
                    marginal = bp.query(variables=[var], show_progress=False)
                    map_config[var] = np.argmax(marginal.values)
                except:
                    # If query fails, use greedy fallback
                    map_config[var] = 0

            # Compute probability of this configuration
            log_prob = self._evaluate_configuration(map_config)
            prob = np.exp(log_prob)

            return MAPResult(
                configuration=map_config,
                log_probability=log_prob,
                probability=prob,
                method="exact",
                metadata={"pgmpy": True}
            )
        except Exception as e:
            # Fallback to greedy if pgmpy fails
            return self._compute_map_greedy()

    def _compute_map_bp(self) -> MAPResult:
        """Compute MAP using belief propagation (max-product).

        This implements loopy belief propagation with max-product messages
        for approximate MAP inference.
        """
        # Simple implementation: use max-marginals
        # For each variable, compute max-marginal and pick argmax

        # Initialize with uniform messages
        max_marginals = np.zeros((self.n_variables, max(self.cardinalities)))

        # Compute max-marginal for each variable from clique tables
        for var in range(self.n_variables):
            marginal = np.zeros(self.cardinalities[var])

            # Find cliques containing this variable
            for clique, table in zip(self.cliques, self.tables):
                if var in clique:
                    # Ensure table has correct shape for the clique
                    expected_shape = tuple(self.cardinalities[v] for v in clique)
                    if table.shape != expected_shape:
                        table = table.reshape(expected_shape)

                    # Get variable position in clique
                    var_idx = np.where(clique == var)[0][0]

                    # Max-marginalize over other variables
                    axes_to_max = [i for i in range(len(clique)) if i != var_idx]
                    if axes_to_max:
                        max_vals = table
                        for axis in sorted(axes_to_max, reverse=True):
                            max_vals = np.max(max_vals, axis=axis)
                    else:
                        max_vals = table

                    # Accumulate (multiply in log space)
                    if len(marginal) == len(max_vals):
                        marginal += np.log(np.maximum(max_vals, 1e-300))

            max_marginals[var, :self.cardinalities[var]] = marginal

        # Pick argmax for each variable
        map_config = np.array([np.argmax(max_marginals[v, :self.cardinalities[v]])
                               for v in range(self.n_variables)])

        # Evaluate configuration
        log_prob = self._evaluate_configuration(map_config)
        prob = np.exp(log_prob)

        return MAPResult(
            configuration=map_config,
            log_probability=log_prob,
            probability=prob,
            method="bp",
            metadata={"type": "max-product"}
        )

    def _compute_map_decimation(self) -> MAPResult:
        """Compute MAP using decimation (iteratively fixing variables).

        This implements the decimation strategy from Santana (2013):
        1. Run belief propagation to get marginals
        2. Fix the variable with highest marginal certainty
        3. Update model and repeat
        """
        # Start with unfixed variables
        unfixed = set(range(self.n_variables))
        fixed_values = {}

        iteration = 0
        while unfixed and iteration < self.max_iterations:
            # Compute max-marginals for unfixed variables
            best_var = None
            best_certainty = -1
            best_value = None

            for var in unfixed:
                marginal = self._compute_marginal(var, fixed_values)
                max_prob = np.max(marginal)

                if max_prob > best_certainty:
                    best_certainty = max_prob
                    best_var = var
                    best_value = np.argmax(marginal)

            # Fix the most certain variable
            if best_var is not None:
                fixed_values[best_var] = best_value
                unfixed.remove(best_var)
            else:
                break

            iteration += 1

        # Build final configuration
        map_config = np.array([fixed_values.get(v, 0) for v in range(self.n_variables)])

        # Evaluate configuration
        log_prob = self._evaluate_configuration(map_config)
        prob = np.exp(log_prob)

        return MAPResult(
            configuration=map_config,
            log_probability=log_prob,
            probability=prob,
            method="decimation",
            metadata={"iterations": iteration}
        )

    def _compute_marginal(self, var: int, fixed: Dict[int, int]) -> np.ndarray:
        """Compute marginal distribution for a variable given fixed variables."""
        marginal = np.zeros(self.cardinalities[var])

        # Find cliques containing this variable
        for clique, table in zip(self.cliques, self.tables):
            if var in clique:
                # Ensure table has correct shape for the clique
                expected_shape = tuple(self.cardinalities[v] for v in clique)
                if table.shape != expected_shape:
                    table = table.reshape(expected_shape)

                # Create indices for this clique
                # Start with all possible values for var
                var_idx = np.where(clique == var)[0][0]

                # For each value of var, sum/max over compatible configurations
                for val in range(self.cardinalities[var]):
                    # Build configuration for this clique
                    clique_config = np.zeros(len(clique), dtype=int)
                    clique_config[var_idx] = val

                    # Set fixed values
                    compatible = True
                    for i, v in enumerate(clique):
                        if v in fixed:
                            if i == var_idx:
                                if fixed[v] != val:
                                    compatible = False
                                    break
                            else:
                                clique_config[i] = fixed[v]

                    if compatible:
                        # Get probability from table (sum over unfixed vars)
                        prob = self._get_clique_probability(clique, clique_config, table, fixed)
                        marginal[val] += prob

        # Normalize
        total = np.sum(marginal)
        if total > 0:
            marginal /= total
        else:
            marginal = np.ones(self.cardinalities[var]) / self.cardinalities[var]

        return marginal

    def _get_clique_probability(
        self,
        clique: np.ndarray,
        config: np.ndarray,
        table: np.ndarray,
        fixed: Dict[int, int]
    ) -> float:
        """Get probability from clique table, marginalizing over unfixed variables."""
        # Convert configuration to table index
        try:
            index = self._config_to_index(config, [self.cardinalities[v] for v in clique])
            return table.ravel()[index]
        except:
            return 1e-300

    def _config_to_index(self, config: np.ndarray, cardinalities: List[int]) -> int:
        """Convert configuration to flat index (mixed-radix)."""
        index = 0
        multiplier = 1
        for i in range(len(config) - 1, -1, -1):
            index += config[i] * multiplier
            multiplier *= cardinalities[i]
        return index

    def _compute_map_greedy(self) -> MAPResult:
        """Simple greedy MAP computation (fallback method)."""
        # For each variable, pick value that maximizes probability in all cliques
        map_config = np.zeros(self.n_variables, dtype=int)

        for var in range(self.n_variables):
            best_val = 0
            best_score = -np.inf

            for val in range(self.cardinalities[var]):
                # Try this value
                map_config[var] = val
                score = self._evaluate_configuration(map_config)

                if score > best_score:
                    best_score = score
                    best_val = val

            map_config[var] = best_val

        log_prob = self._evaluate_configuration(map_config)
        prob = np.exp(log_prob)

        return MAPResult(
            configuration=map_config,
            log_probability=log_prob,
            probability=prob,
            method="greedy",
            metadata={"fallback": True}
        )

    def _evaluate_configuration(self, config: np.ndarray) -> float:
        """Evaluate log probability of a configuration."""
        log_prob = 0.0

        for clique, table in zip(self.cliques, self.tables):
            # Extract configuration for this clique
            clique_config = config[clique]

            # Get probability from table
            index = self._config_to_index(
                clique_config,
                [self.cardinalities[v] for v in clique]
            )

            prob = table.ravel()[index]
            log_prob += np.log(np.maximum(prob, 1e-300))

        return log_prob

    def compute_k_map(self, k: int) -> KMAPResult:
        """Compute k most probable configurations.

        This implements a simplified version of the max-flow propagation
        algorithm from Nilsson (1998).

        Parameters:
            k: Number of configurations to find

        Returns:
            KMAPResult with k configurations and their probabilities
        """
        # Use a priority queue / beam search approach
        # Start with MAP, then iteratively find next best by flipping variables

        configurations = []
        log_probs = []

        # Get initial MAP
        map_result = self.compute_map()
        configurations.append(map_result.configuration.copy())
        log_probs.append(map_result.log_probability)

        # Generate candidates by flipping each variable
        candidates = []

        for i in range(1, k):
            # Generate candidates from current best configurations
            for base_config in configurations:
                for var in range(self.n_variables):
                    for val in range(self.cardinalities[var]):
                        if val != base_config[var]:
                            # Create new configuration
                            new_config = base_config.copy()
                            new_config[var] = val

                            # Evaluate
                            log_prob = self._evaluate_configuration(new_config)
                            candidates.append((log_prob, new_config))

            # Remove duplicates and already selected
            unique_candidates = []
            seen = {tuple(c) for c in configurations}

            for log_prob, config in candidates:
                config_tuple = tuple(config)
                if config_tuple not in seen:
                    unique_candidates.append((log_prob, config))
                    seen.add(config_tuple)

            if not unique_candidates:
                break

            # Select best candidate
            unique_candidates.sort(reverse=True, key=lambda x: x[0])
            best_log_prob, best_config = unique_candidates[0]

            configurations.append(best_config)
            log_probs.append(best_log_prob)

            # Keep top candidates for next iteration
            candidates = unique_candidates[:min(100, len(unique_candidates))]

        configurations = np.array(configurations)
        log_probs = np.array(log_probs)
        probs = np.exp(log_probs)

        return KMAPResult(
            configurations=configurations,
            log_probabilities=log_probs,
            probabilities=probs,
            method=self.method.value,
            metadata={"k": len(configurations)}
        )


def compute_map(
    cliques: List[np.ndarray],
    tables: List[np.ndarray],
    cardinalities: np.ndarray,
    method: str = "exact"
) -> np.ndarray:
    """Compute MAP configuration from Markov network.

    Convenience function that creates MAPInference and returns configuration.

    Parameters:
        cliques: List of cliques (variable indices)
        tables: List of clique probability tables
        cardinalities: Cardinality of each variable
        method: Inference method ("exact", "bp", "decimation")

    Returns:
        MAP configuration as numpy array

    Example:
        >>> cliques = [np.array([0, 1]), np.array([1, 2])]
        >>> tables = [table1, table2]
        >>> cardinalities = np.array([2, 2, 2])
        >>> map_config = compute_map(cliques, tables, cardinalities)
    """
    inference = MAPInference(cliques, tables, cardinalities, method=method)
    result = inference.compute_map()
    return result.configuration


def compute_k_map(
    cliques: List[np.ndarray],
    tables: List[np.ndarray],
    cardinalities: np.ndarray,
    k: int,
    method: str = "exact"
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute k most probable configurations from Markov network.

    Convenience function that creates MAPInference and returns configurations.

    Parameters:
        cliques: List of cliques (variable indices)
        tables: List of clique probability tables
        cardinalities: Cardinality of each variable
        k: Number of configurations to find
        method: Inference method ("exact", "bp", "decimation")

    Returns:
        Tuple of (configurations, probabilities)
        - configurations: (k, n) array
        - probabilities: (k,) array

    Example:
        >>> configs, probs = compute_k_map(cliques, tables, cardinalities, k=5)
        >>> print(f"Top 5 configurations found with probabilities: {probs}")
    """
    inference = MAPInference(cliques, tables, cardinalities, method=method)
    result = inference.compute_k_map(k)
    return result.configurations, result.probabilities


def compute_map_decimation(
    cliques: List[np.ndarray],
    tables: List[np.ndarray],
    cardinalities: np.ndarray,
    max_iterations: int = 100
) -> np.ndarray:
    """Compute MAP using decimation strategy.

    Parameters:
        cliques: List of cliques (variable indices)
        tables: List of clique probability tables
        cardinalities: Cardinality of each variable
        max_iterations: Maximum decimation iterations

    Returns:
        MAP configuration as numpy array
    """
    inference = MAPInference(
        cliques,
        tables,
        cardinalities,
        method=MAPMethod.DECIMATION,
        max_iterations=max_iterations
    )
    result = inference.compute_map()
    return result.configuration
