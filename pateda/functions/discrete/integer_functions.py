"""
Integer Representation Benchmark Functions for EDAs

This module contains benchmark functions designed specifically for testing
Estimation of Distribution Algorithms (EDAs) with integer representation
(non-binary discrete variables with cardinality > 2).

These functions extend binary benchmarks to multi-valued domains, testing
the ability of EDAs to model dependencies among integer-valued variables.

Functions included:
- Generalized OneMax: Sum of normalized integer values
- Generalized Deceptive: Extension of k-deceptive for integers
- Integer NK Landscape: NK landscapes with integer-valued variables
- Categorical Peak: Finding a specific categorical pattern
- Multi-Level Trap: Trap function with multiple discrete levels
"""

import numpy as np
from typing import Callable, Optional, Union, List


def integer_onemax(x: np.ndarray, cardinality: int = 4) -> float:
    """
    Generalized OneMax for integer representation

    Sums all values normalized by maximum possible value (cardinality - 1).
    The optimal solution is all variables at their maximum value.

    Args:
        x: Integer vector with values in [0, cardinality-1]
        cardinality: Number of possible values for each variable

    Returns:
        Sum of all elements (optimal = n * (cardinality - 1))

    Example:
        >>> x = np.array([3, 3, 3, 3])  # cardinality=4
        >>> integer_onemax(x, cardinality=4)
        12.0
    """
    if x.ndim == 2:
        x = x.flatten()
    return float(np.sum(x))


def integer_leading_blocks(x: np.ndarray, cardinality: int = 4, block_size: int = 3) -> float:
    """
    Leading Blocks function for integer representation

    Counts the number of consecutive blocks from the beginning where
    all variables have the maximum value.

    Args:
        x: Integer vector with values in [0, cardinality-1]
        cardinality: Number of possible values for each variable
        block_size: Size of each block to check

    Returns:
        Number of complete leading blocks with all max values

    Example:
        >>> x = np.array([3, 3, 3, 3, 3, 3, 0, 0, 0])  # cardinality=4
        >>> integer_leading_blocks(x, cardinality=4, block_size=3)
        2.0
    """
    if x.ndim == 2:
        x = x.flatten()

    n = len(x)
    max_val = cardinality - 1
    blocks = 0

    for i in range(0, n - block_size + 1, block_size):
        block = x[i:i + block_size]
        if np.all(block == max_val):
            blocks += 1
        else:
            break  # Stop at first incomplete block

    return float(blocks)


def integer_partition_max(
    vector: np.ndarray,
    k: int,
    cardinality: int
) -> float:
    """
    Evaluate a single partition of the generalized max function

    Args:
        vector: Integer vector for this partition (length k)
        k: Size of the partition
        cardinality: Number of possible values

    Returns:
        Partition contribution (k if all max, else sum)
    """
    max_val = cardinality - 1
    s = np.sum(vector)
    optimal_sum = k * max_val

    if s == optimal_sum:
        return float(optimal_sum + k)  # Bonus for optimal
    else:
        return float(s)


def integer_max_blocks(
    vector: np.ndarray,
    k: int = 3,
    cardinality: int = 4
) -> float:
    """
    Integer Max Blocks function

    Rewards solutions where all variables in each block have maximum value.
    Similar to trap function but for integer representation.

    Args:
        vector: Integer solution vector (1D array)
        k: Number of variables in each partition (default: 3)
        cardinality: Number of possible values (default: 4)

    Returns:
        Sum of partition values

    Example:
        >>> x = np.array([3, 3, 3, 2, 2, 2])  # cardinality=4
        >>> integer_max_blocks(x, k=3, cardinality=4)
        18.0  # First block: 9+3=12, second block: 6
    """
    if vector.ndim == 2:
        vector = vector.flatten()

    n = len(vector)
    total = 0.0

    for i in range(0, n, k):
        partition = vector[i:i + k]
        total += integer_partition_max(partition, k, cardinality)

    return total


def integer_categorical_match(
    x: np.ndarray,
    target_pattern: Optional[np.ndarray] = None,
    cardinality: int = 4
) -> float:
    """
    Categorical Pattern Match function

    Counts matches between solution and a target pattern.
    Can be used to test ability to converge to specific categorical values.

    Args:
        x: Integer solution vector
        target_pattern: Target pattern to match (default: alternating 0,1,2,3,...)
        cardinality: Number of possible values

    Returns:
        Number of positions matching the target pattern

    Example:
        >>> x = np.array([0, 1, 2, 3, 0, 1])
        >>> integer_categorical_match(x, cardinality=4)
        6.0  # All match alternating pattern
    """
    if x.ndim == 2:
        x = x.flatten()

    n = len(x)

    if target_pattern is None:
        # Default: alternating pattern 0, 1, 2, ..., cardinality-1, 0, 1, ...
        target_pattern = np.array([i % cardinality for i in range(n)])

    matches = np.sum(x == target_pattern)
    return float(matches)


def integer_plateau_search(
    x: np.ndarray,
    cardinality: int = 4,
    block_size: int = 3
) -> float:
    """
    Integer Plateau Search function

    Tests ability to search on plateaus. Within each block, only the
    sum matters (creating large plateaus), but reaching the maximum
    sum gives a bonus.

    Args:
        x: Integer solution vector
        cardinality: Number of possible values
        block_size: Size of each block

    Returns:
        Fitness based on block sums with bonuses for maximal blocks

    Example:
        >>> x = np.array([1, 2, 3, 0, 3, 3])  # cardinality=4
        >>> integer_plateau_search(x, cardinality=4, block_size=3)
        # First block sum=6, second block sum=6+bonus
    """
    if x.ndim == 2:
        x = x.flatten()

    n = len(x)
    max_val = cardinality - 1
    max_block_sum = block_size * max_val
    total = 0.0

    for i in range(0, n, block_size):
        block = x[i:i + block_size]
        block_sum = np.sum(block)
        total += block_sum

        # Bonus for maximal block
        if block_sum == max_block_sum:
            total += max_val

    return total


def integer_dependency_chain(
    x: np.ndarray,
    cardinality: int = 4
) -> float:
    """
    Integer Dependency Chain function

    Tests ability to model sequential dependencies.
    Rewards when x[i+1] = (x[i] + 1) mod cardinality.

    Args:
        x: Integer solution vector
        cardinality: Number of possible values

    Returns:
        Count of correct sequential dependencies + sum of values

    Example:
        >>> x = np.array([0, 1, 2, 3, 0, 1, 2, 3])  # cardinality=4
        >>> integer_dependency_chain(x, cardinality=4)
        # 7 correct dependencies + sum(x) = 7 + 12 = 19.0
    """
    if x.ndim == 2:
        x = x.flatten()

    n = len(x)

    # Count correct dependencies
    dependencies = 0
    for i in range(n - 1):
        if x[i + 1] == (x[i] + 1) % cardinality:
            dependencies += 1

    # Add base fitness from sum
    base_fitness = np.sum(x)

    return float(dependencies * cardinality + base_fitness)


def integer_multi_level_trap(
    vector: np.ndarray,
    k: int = 3,
    cardinality: int = 4
) -> float:
    """
    Multi-Level Trap function for integer representation

    Extension of the binary trap function to multiple discrete levels.
    Creates deceptive structure at each level.

    Partition fitness:
    - If all variables at max (cardinality-1): return k * (cardinality-1) + bonus
    - Otherwise: reward for being close to all zeros (deceptive)

    Args:
        vector: Integer solution vector (1D array)
        k: Number of variables in each partition (default: 3)
        cardinality: Number of possible values (default: 4)

    Returns:
        Sum of trap values over all partitions

    Example:
        >>> x = np.array([3, 3, 3, 0, 0, 0])  # cardinality=4, k=3
        >>> integer_multi_level_trap(x, k=3, cardinality=4)
        # First partition optimal, second suboptimal
    """
    if vector.ndim == 2:
        vector = vector.flatten()

    n = len(vector)
    max_val = cardinality - 1
    optimal_sum = k * max_val
    total = 0.0

    for i in range(0, n, k):
        partition = vector[i:i + k]
        partition_sum = np.sum(partition)

        if partition_sum == optimal_sum:
            # Global optimum for this partition
            total += optimal_sum + k  # Bonus for reaching max
        else:
            # Deceptive: reward for being close to zero
            total += optimal_sum - 1 - partition_sum

    return total


def integer_parity_blocks(
    x: np.ndarray,
    block_size: int = 4,
    cardinality: int = 4
) -> float:
    """
    Integer Parity Blocks function

    Rewards blocks where the sum of values has a specific parity.
    Tests ability to model complex non-linear relationships.

    Args:
        x: Integer solution vector
        block_size: Size of each block
        cardinality: Number of possible values

    Returns:
        Sum of block contributions based on parity

    Example:
        >>> x = np.array([0, 1, 2, 1, 3, 3, 3, 3])  # sums: 4 (even), 12 (even)
        >>> integer_parity_blocks(x, block_size=4, cardinality=4)
    """
    if x.ndim == 2:
        x = x.flatten()

    n = len(x)
    max_val = cardinality - 1
    total = 0.0

    for i in range(0, n, block_size):
        block = x[i:i + block_size]
        block_sum = np.sum(block)

        # Reward even parity
        if block_sum % 2 == 0:
            total += block_sum + max_val
        else:
            total += block_sum

    return total


def integer_hierarchical_blocks(
    vector: np.ndarray,
    cardinality: int = 4
) -> float:
    """
    Hierarchical Blocks function for integer representation

    Similar to HIFF but for integer-valued variables.
    Rewards consistent blocks at multiple levels.

    Args:
        vector: Integer solution vector (length should be power of 2)
        cardinality: Number of possible values

    Returns:
        Hierarchical fitness value

    Note:
        Problem size should be a power of 2 (e.g., 8, 16, 32)
    """
    if vector.ndim == 2:
        vector = vector.flatten()

    n = len(vector)
    max_val = cardinality - 1

    # Base level contribution
    total = float(np.sum(vector))

    # Check hierarchical consistency
    level_values = vector.copy()
    level_size = 2

    while level_size <= n:
        new_level_values = []

        for i in range(0, n, level_size):
            block = level_values[i:i + level_size]

            # Check if all values in block are the same
            if len(np.unique(block)) == 1:
                # Consistent block - add bonus
                total += level_size * max_val
                new_level_values.append(block[0])
            else:
                # Inconsistent - mark as mixed
                new_level_values.append(-1)

        level_values = np.array(new_level_values)
        level_size *= 2
        n = len(level_values)

    return total


# ============================================================================
# Factory Functions for EDA Integration
# ============================================================================

def create_integer_onemax_function(cardinality: int = 4) -> Callable:
    """Create an integer OneMax objective function for use with EDA"""
    def objective(x: np.ndarray) -> float:
        return integer_onemax(x, cardinality)
    return objective


def create_integer_max_blocks_function(k: int = 3, cardinality: int = 4) -> Callable:
    """Create an integer max blocks objective function for use with EDA"""
    def objective(x: np.ndarray) -> float:
        return integer_max_blocks(x, k, cardinality)
    return objective


def create_integer_multi_level_trap_function(k: int = 3, cardinality: int = 4) -> Callable:
    """Create an integer multi-level trap objective function for use with EDA"""
    def objective(x: np.ndarray) -> float:
        return integer_multi_level_trap(x, k, cardinality)
    return objective


def create_integer_dependency_chain_function(cardinality: int = 4) -> Callable:
    """Create an integer dependency chain objective function for use with EDA"""
    def objective(x: np.ndarray) -> float:
        return integer_dependency_chain(x, cardinality)
    return objective


def create_integer_categorical_match_function(
    target_pattern: Optional[np.ndarray] = None,
    cardinality: int = 4
) -> Callable:
    """Create an integer categorical match objective function for use with EDA"""
    def objective(x: np.ndarray) -> float:
        return integer_categorical_match(x, target_pattern, cardinality)
    return objective


def create_integer_hierarchical_function(cardinality: int = 4) -> Callable:
    """Create an integer hierarchical blocks objective function for use with EDA"""
    def objective(x: np.ndarray) -> float:
        return integer_hierarchical_blocks(x, cardinality)
    return objective


def create_integer_parity_function(block_size: int = 4, cardinality: int = 4) -> Callable:
    """Create an integer parity blocks objective function for use with EDA"""
    def objective(x: np.ndarray) -> float:
        return integer_parity_blocks(x, block_size, cardinality)
    return objective


# ============================================================================
# Integer NK Landscape (extends binary NK to integer domain)
# ============================================================================

class IntegerNKLandscape:
    """
    NK Landscape for integer-valued variables

    Extension of the binary NK landscape to support variables with
    cardinality > 2. Each variable can take values in [0, cardinality-1].

    Attributes:
        n_vars: Number of variables
        k: Number of neighbors per variable (epistasis level)
        cardinality: Number of possible values per variable
        structure: Factor structure defining dependencies
        tables: Lookup tables for fitness contributions
    """

    def __init__(
        self,
        n_vars: int,
        k: int,
        cardinality: int = 4,
        random_seed: Optional[int] = None
    ):
        """
        Initialize Integer NK Landscape

        Args:
            n_vars: Number of variables
            k: Number of neighbors (epistasis level)
            cardinality: Number of possible values per variable (default: 4)
            random_seed: Random seed for generating tables
        """
        self.n_vars = n_vars
        self.k = k
        self.cardinality = cardinality

        rng = np.random.default_rng(random_seed)

        # Create circular structure
        self.structure = self._create_circular_structure(n_vars, k)

        # Generate random tables for integer values
        self.tables = self._create_random_tables(rng)

    def _create_circular_structure(self, n_vars: int, k: int) -> List[np.ndarray]:
        """Create circular NK structure"""
        structure = []
        for i in range(n_vars):
            half_k = k // 2
            neighbors = list(range(i - half_k, i)) + list(range(i + 1, i + half_k + 1))
            neighbors = [(n if n >= 0 else n + n_vars) for n in neighbors]
            neighbors = [(n if n < n_vars else n - n_vars) for n in neighbors]
            factor = [i] + neighbors
            structure.append(np.array(factor, dtype=int))
        return structure

    def _create_random_tables(self, rng: np.random.Generator) -> List[np.ndarray]:
        """Create random lookup tables for each factor"""
        tables = []
        for factor in self.structure:
            # Table size = cardinality^(k+1) for each factor
            table_size = self.cardinality ** len(factor)
            table = rng.random(table_size)
            tables.append(table)
        return tables

    def evaluate(self, solution: np.ndarray) -> float:
        """
        Evaluate a single solution

        Args:
            solution: Integer solution vector with values in [0, cardinality-1]

        Returns:
            Fitness value (sum of factor contributions)
        """
        if solution.ndim == 2:
            solution = solution.flatten()

        total = 0.0

        for i, (factor, table) in enumerate(zip(self.structure, self.tables)):
            # Get values for variables in this factor
            factor_values = solution[factor]

            # Convert to index in the lookup table
            index = 0
            multiplier = 1
            for j in range(len(factor) - 1, -1, -1):
                val = int(factor_values[j])
                index += val * multiplier
                multiplier *= self.cardinality

            total += table[index]

        return total

    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """Evaluate a population of solutions"""
        if population.ndim == 1:
            return np.array([self.evaluate(population)])

        pop_size = population.shape[0]
        fitness = np.zeros(pop_size)
        for i in range(pop_size):
            fitness[i] = self.evaluate(population[i])

        return fitness


def create_integer_nk_objective_function(
    n_vars: int,
    k: int,
    cardinality: int = 4,
    random_seed: Optional[int] = None
) -> Callable:
    """
    Create an Integer NK landscape objective function for use with EDA

    Args:
        n_vars: Number of variables
        k: Epistasis level (number of neighbors)
        cardinality: Number of possible values per variable
        random_seed: Random seed for reproducibility

    Returns:
        Objective function compatible with EDA framework
    """
    nk_landscape = IntegerNKLandscape(n_vars, k, cardinality, random_seed)

    def objective(x: np.ndarray) -> float:
        if x.ndim == 1:
            return nk_landscape.evaluate(x)
        return nk_landscape.evaluate(x)

    # Attach landscape for access
    objective.nk_landscape = nk_landscape

    return objective
