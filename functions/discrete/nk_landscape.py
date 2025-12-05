"""
NK Landscape problem for optimization benchmarking

NK landscapes are a family of tunably rugged fitness landscapes used for
studying optimization in the presence of epistasis (variable interactions).

Based on MATEDA-2.0 NK landscape implementation
"""

import numpy as np
from typing import List, Tuple, Optional, Dict


def create_circular_nk_structure(n_vars: int, k: int) -> List[np.ndarray]:
    """
    Create circular NK landscape structure

    Each variable depends on its k/2 previous and k/2 subsequent neighbors
    in a circular (wrap-around) fashion.

    Args:
        n_vars: Number of variables
        k: Number of neighbors (0 < k < n_vars, should be even)

    Returns:
        List where each element i contains [i, neighbors...] indicating
        the variables that influence variable i

    Example:
        >>> structure = create_circular_nk_structure(5, 2)
        >>> # Variable 0 depends on itself and neighbors [4, 1]
    """
    structure = []

    for i in range(n_vars):
        # Calculate neighbors
        half_k = k // 2
        neighbors = list(range(i - half_k, i)) + list(range(i + 1, i + half_k + 1))

        # Handle wrap-around
        neighbors = [(n if n >= 0 else n + n_vars) for n in neighbors]
        neighbors = [(n if n < n_vars else n - n_vars) for n in neighbors]

        # Include the variable itself
        factor = [i] + neighbors
        structure.append(np.array(factor, dtype=int))

    return structure


def create_random_nk_tables(
    structure: List[np.ndarray],
    cardinality: np.ndarray,
    random_seed: Optional[int] = None
) -> List[np.ndarray]:
    """
    Create random lookup tables for NK landscape

    Args:
        structure: List of factor definitions (from create_circular_nk_structure)
        cardinality: Array of variable cardinalities (typically all 2 for binary)
        random_seed: Random seed for reproducibility

    Returns:
        List of random value tables, one for each factor

    Note:
        Each table assigns a random value to each possible configuration
        of the variables in that factor.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    tables = []

    for factor in structure:
        # Calculate table size: product of cardinalities
        table_size = 1
        for var_idx in factor:
            table_size *= int(cardinality[var_idx])

        # Generate random values for this factor
        table = np.random.random(table_size)
        tables.append(table)

    return tables


def evaluate_nk_landscape(
    solution: np.ndarray,
    structure: List[np.ndarray],
    tables: List[np.ndarray],
    cardinality: np.ndarray
) -> float:
    """
    Evaluate a solution on NK landscape

    Args:
        solution: Binary (or discrete) solution vector
        structure: List of factor definitions
        tables: List of value tables
        cardinality: Variable cardinalities

    Returns:
        Sum of values from all factors (to be maximized)
    """
    if solution.ndim == 2:
        solution = solution.flatten()

    total_value = 0.0

    for i, (factor, table) in enumerate(zip(structure, tables)):
        # Get values for variables in this factor
        factor_values = solution[factor]

        # Convert to index in the lookup table
        index = 0
        multiplier = 1
        for j in range(len(factor) - 1, -1, -1):
            var_idx = factor[j]
            val = int(factor_values[j])
            index += val * multiplier
            multiplier *= int(cardinality[var_idx])

        # Add value from table
        total_value += table[index]

    return total_value


class NKLandscape:
    """
    NK Landscape optimization problem

    Attributes:
        n_vars: Number of variables
        k: Number of neighbors per variable
        structure: Factor structure
        tables: Value tables for each factor
        cardinality: Variable cardinalities
    """

    def __init__(
        self,
        n_vars: int,
        k: int,
        cardinality: Optional[np.ndarray] = None,
        random_seed: Optional[int] = None
    ):
        """
        Initialize NK Landscape

        Args:
            n_vars: Number of variables
            k: Number of neighbors (epistasis level)
            cardinality: Variable cardinalities (default: all binary)
            random_seed: Random seed for generating tables
        """
        self.n_vars = n_vars
        self.k = k

        if cardinality is None:
            self.cardinality = 2 * np.ones(n_vars, dtype=int)
        else:
            self.cardinality = cardinality

        # Create circular structure
        self.structure = create_circular_nk_structure(n_vars, k)

        # Generate random tables
        self.tables = create_random_nk_tables(
            self.structure, self.cardinality, random_seed
        )

    def evaluate(self, solution: np.ndarray) -> float:
        """
        Evaluate a single solution

        Args:
            solution: Solution vector

        Returns:
            Fitness value (sum of factor values)
        """
        return evaluate_nk_landscape(
            solution, self.structure, self.tables, self.cardinality
        )

    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """
        Evaluate a population of solutions

        Args:
            population: 2D array of shape (pop_size, n_vars)

        Returns:
            1D array of fitness values
        """
        if population.ndim == 1:
            return np.array([self.evaluate(population)])

        pop_size = population.shape[0]
        fitness = np.zeros(pop_size)

        for i in range(pop_size):
            fitness[i] = self.evaluate(population[i])

        return fitness

    def save_instance(self, structure_file: str, tables_file: str):
        """
        Save NK landscape instance to files

        Args:
            structure_file: Path to save structure
            tables_file: Path to save tables
        """
        # Save structure
        np.savez(structure_file, structure=[s for s in self.structure])

        # Save tables
        np.savez(tables_file, tables=self.tables, cardinality=self.cardinality)

    @classmethod
    def load_instance(cls, structure_file: str, tables_file: str) -> 'NKLandscape':
        """
        Load NK landscape instance from files

        Args:
            structure_file: Path to structure file
            tables_file: Path to tables file

        Returns:
            NKLandscape instance
        """
        # Load files
        structure_data = np.load(structure_file, allow_pickle=True)
        tables_data = np.load(tables_file, allow_pickle=True)

        structure = structure_data['structure']
        tables = tables_data['tables']
        cardinality = tables_data['cardinality']

        # Create instance
        n_vars = len(structure)
        k = len(structure[0]) - 1  # Subtract the variable itself

        instance = cls.__new__(cls)
        instance.n_vars = n_vars
        instance.k = k
        instance.structure = list(structure)
        instance.tables = list(tables)
        instance.cardinality = cardinality

        return instance


def create_nk_objective_function(
    n_vars: int,
    k: int,
    random_seed: Optional[int] = None
):
    """
    Create an NK landscape objective function for use with EDA

    Args:
        n_vars: Number of variables
        k: Epistasis level (number of neighbors)
        random_seed: Random seed for reproducibility

    Returns:
        Objective function compatible with EDA framework

    Example:
        >>> nk_func = create_nk_objective_function(n_vars=50, k=4, random_seed=42)
        >>> population = np.random.randint(0, 2, (100, 50))
        >>> fitness = nk_func(population)
    """
    nk_landscape = NKLandscape(n_vars, k, random_seed=random_seed)

    def objective(population: np.ndarray) -> np.ndarray:
        return nk_landscape.evaluate_population(population)

    # Attach landscape for access to structure and tables
    objective.nk_landscape = nk_landscape

    return objective
