"""
Ising Model evaluation functions

Equivalent to MATEDA's EvalIsing.m and LoadIsing.m
The Ising model is a spin glass optimization problem used to test EDAs.
"""

import numpy as np
from typing import Tuple
from pathlib import Path


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
        instances_dir = Path(__file__).parent.parent.parent.parent / "functions" / "ising-model"
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

        # Load the structures from file
        for i in range(num_vars):
            lattice[i, 0] = int(fp.readline().strip())

            if lattice[i, 0] > 0:
                # Read neighbor indices (convert from 0-indexed to 1-indexed as in MATLAB)
                for j in range(1, lattice[i, 0] + 1):
                    lattice[i, j] = int(fp.readline().strip()) + 1

                # Read interaction values
                for j in range(lattice[i, 0]):
                    inter[i, j] = float(fp.readline().strip())

    return lattice, inter


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
