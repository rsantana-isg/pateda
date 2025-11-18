"""
Traveling Salesman Problem (TSP) for permutation-based EDAs

The TSP is a classic combinatorial optimization problem where the goal
is to find the shortest route visiting all cities exactly once.
"""

import numpy as np
from typing import Optional


class TSP:
    """
    Traveling Salesman Problem

    Given a distance matrix between cities, find the shortest Hamiltonian cycle.
    """

    def __init__(self, distance_matrix: np.ndarray):
        """
        Initialize TSP with a distance matrix.

        Args:
            distance_matrix: Symmetric matrix of distances between cities
                            Shape: (n_cities, n_cities)
        """
        self.distance_matrix = distance_matrix
        self.n_cities = distance_matrix.shape[0]

        # Verify symmetry
        if not np.allclose(distance_matrix, distance_matrix.T):
            raise ValueError("Distance matrix must be symmetric")

    def __call__(self, tour: np.ndarray) -> float:
        """
        Evaluate a tour (permutation).

        Args:
            tour: A permutation of cities (0-indexed or 1-indexed)

        Returns:
            Negative total distance (for maximization)
                    Lower distance = better fitness
        """
        tour = np.array(tour, dtype=int)

        # Convert to 0-indexed if needed
        if np.min(tour) == 1:
            tour = tour - 1

        # Calculate total distance
        total_distance = 0.0

        for i in range(len(tour)):
            from_city = tour[i]
            to_city = tour[(i + 1) % len(tour)]  # Wrap around to first city
            total_distance += self.distance_matrix[from_city, to_city]

        # Return negative distance (for maximization in EDA)
        return -total_distance

    def evaluate_distance(self, tour: np.ndarray) -> float:
        """
        Evaluate tour distance (positive value).

        Args:
            tour: A permutation of cities

        Returns:
            Total distance of the tour
        """
        return -self.__call__(tour)


def create_random_tsp(n_cities: int, seed: Optional[int] = None) -> TSP:
    """
    Create a random TSP instance with cities in 2D Euclidean space.

    Args:
        n_cities: Number of cities
        seed: Random seed for reproducibility

    Returns:
        TSP instance with Euclidean distance matrix

    Example:
        >>> tsp = create_random_tsp(10, seed=42)
        >>> tour = np.arange(10)
        >>> fitness = tsp(tour)
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate random city coordinates in [0, 100] x [0, 100]
    cities = np.random.rand(n_cities, 2) * 100

    # Calculate Euclidean distance matrix
    distance_matrix = np.zeros((n_cities, n_cities))

    for i in range(n_cities):
        for j in range(i + 1, n_cities):
            dist = np.linalg.norm(cities[i] - cities[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    return TSP(distance_matrix)


def create_tsp_from_coordinates(coordinates: np.ndarray) -> TSP:
    """
    Create TSP from city coordinates.

    Args:
        coordinates: Array of city coordinates, shape (n_cities, 2)

    Returns:
        TSP instance

    Example:
        >>> coords = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        >>> tsp = create_tsp_from_coordinates(coords)
    """
    n_cities = coordinates.shape[0]
    distance_matrix = np.zeros((n_cities, n_cities))

    for i in range(n_cities):
        for j in range(i + 1, n_cities):
            dist = np.linalg.norm(coordinates[i] - coordinates[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    return TSP(distance_matrix)
