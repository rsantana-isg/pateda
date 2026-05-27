"""
Ising Model evaluation functions

Equivalent to MATEDA's EvalIsing.m and LoadIsing.m
The Ising model is a spin glass optimization problem used to test EDAs.
"""

import numpy as np
from typing import Optional, Tuple
from pathlib import Path


def _default_ising_instances_dir() -> Path:
    """Return the packaged directory containing Ising benchmark instances."""
    return Path(__file__).resolve().parent.parent / "Ising_Instances"


def _parse_ising_instance_name(instance_name: str) -> Tuple[int, int]:
    """Parse ``SG_<n>_<inst>`` benchmark names."""
    parts = instance_name.replace(".txt", "").split("_")
    if len(parts) != 3 or parts[0] != "SG":
        raise ValueError(
            f"Invalid Ising instance name format: {instance_name}. Expected format: SG_<n>_<inst>"
        )
    return int(parts[1]), int(parts[2])


def load_ising(n: int, inst: int, instances_dir: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load an Ising model instance from file

    Equivalent to MATEDA's LoadIsing.m

    Args:
        n: Number of variables
        inst: Instance number
        instances_dir: Directory containing instance files. If None, uses default.

    Returns:
        Tuple of (lattice, inter):
            - lattice: Structure representing the J matrix (neighbors for each node)
                      Shape: (num_vars, max_neighbors + 1)
                      lattice[i, 0] = number of neighbors for variable i
                      lattice[i, 1:] = indices of neighbors (1-indexed)
            - inter: Interaction values between spins
                    Shape: (num_vars, max_neighbors)
                    inter[i, j] = interaction value between variable i and its j-th neighbor
    """
    if instances_dir is None:
        instances_dir = _default_ising_instances_dir()
    else:
        instances_dir = Path(instances_dir)

    instance_file = instances_dir / f"SG_{n}_{inst}.txt"

    if not instance_file.exists():
        raise FileNotFoundError(f"Ising instance file not found: {instance_file}")

    with open(instance_file, 'r') as fp:
        num_vars = int(fp.readline().strip())
        dim = int(fp.readline().strip())
        neigh = int(fp.readline().strip())
        width = int(fp.readline().strip())

        # Initialize lattice and inter
        neighbor = int(2**neigh * dim)
        lattice = np.zeros((num_vars, neighbor + 1), dtype=int)
        inter = np.zeros((num_vars, neighbor), dtype=float)

        if num_vars != n:
            raise ValueError(f"Instance SG_{n}_{inst} declares {num_vars} variables in the file")

        # Load the structures from file
        for i in range(num_vars):
            line = fp.readline().strip().split()
            lattice[i, 0] = int(line[0])

            if lattice[i, 0] > 0:
                n_neighbors = int(lattice[i, 0])
                for j in range(n_neighbors):
                    lattice[i, j + 1] = int(line[1 + j]) + 1

                for j in range(n_neighbors):
                    inter[i, j] = float(line[1 + n_neighbors + j])

    return lattice, inter


def load_ising_benchmark_instance(
    instance_name: str,
    instances_dir: Optional[str] = None,
) -> Tuple[int, np.ndarray, np.ndarray, object]:
    """
    Load a packaged Ising benchmark instance by name.

    Returns ``(n_vars, lattice, inter, optimal_fitness)``.
    """
    n_vars, inst = _parse_ising_instance_name(instance_name)
    lattice, inter = load_ising(n_vars, inst, instances_dir)

    optimal_fitness = "Unknown"
    if instance_name.replace(".txt", "") == "SG_100_1":
        optimal_fitness = 130
    elif instance_name.replace(".txt", "") == "SG_100_2":
        optimal_fitness = 136
    elif instance_name.replace(".txt", "") == "SG_100_3":
        optimal_fitness = 136
    elif instance_name.replace(".txt", "") == "SG_100_4":
        optimal_fitness = 130

    return n_vars, lattice, inter, optimal_fitness


def build_ising_interaction_matrix(lattice: np.ndarray) -> np.ndarray:
    """Build a binary interaction matrix from the Ising lattice neighborhood structure."""
    n_vars = lattice.shape[0]
    interaction_matrix = np.eye(n_vars, dtype=int)

    for i in range(n_vars):
        for j in range(1, int(lattice[i, 0]) + 1):
            neighbor_idx = int(lattice[i, j]) - 1
            interaction_matrix[i, neighbor_idx] = 1
            interaction_matrix[neighbor_idx, i] = 1

    return interaction_matrix


def eval_ising(ind: np.ndarray, lattice: np.ndarray, inter: np.ndarray) -> float:
    """
    Evaluate the Ising model for a binary vector configuration

    Equivalent to MATEDA's EvalIsing.m

    The Ising model computes the energy of a spin configuration. The goal is typically
    to minimize the energy (or maximize the negative energy).

    Args:
        ind: Binary individual/solution (vector of 0s and 1s representing spin configuration)
        lattice: Lattice structure from load_ising (neighbor information)
        inter: Interaction values from load_ising

    Returns:
        Energy value (negative of the sum of matching spin interactions)

    Note:
        The function returns the negative of the raw score, so minimizing this
        value corresponds to finding the optimal spin configuration.
    """
    r = 0.0
    n_vars = lattice.shape[0]

    for i in range(n_vars):
        if lattice[i, 0] > 0:
            for j in range(1, int(lattice[i, 0]) + 1):
                neighbor_idx = int(lattice[i, j]) - 1  # Convert to 0-indexed

                # Only count each interaction once (i < neighbor)
                if i < neighbor_idx:
                    # Check if spins match: 1 if match, -1 if differ
                    auxr = 2 * int(ind[i] == ind[neighbor_idx]) - 1
                    r += auxr * inter[i, j - 1]

    # Return negative (MATLAB version returns -1*r)
    return -r


def create_ising_objective_function(n: int, inst: int, instances_dir: str = None):
    """
    Create an objective function for the Ising model

    Args:
        n: Number of variables
        inst: Instance number
        instances_dir: Directory containing instance files

    Returns:
        Objective function that takes a solution and returns fitness value

    Example:
        >>> ising_func = create_ising_objective_function(16, 1)
        >>> solution = np.random.randint(0, 2, 16)
        >>> fitness = ising_func(solution)
    """
    lattice, inter = load_ising(n, inst, instances_dir)

    def objective_function(solution: np.ndarray) -> float:
        return eval_ising(solution, lattice, inter)

    return objective_function
