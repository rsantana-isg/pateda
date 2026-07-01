"""
Permutation Flowshop Scheduling Problem (PFSP) for permutation-based EDAs

The PFSP is a classic scheduling problem where a set of n jobs must be processed
on a set of m machines in the exact same order. The goal is to find a permutation
of the jobs that minimizes a given objective function, typically the makespan
(total completion time) or the total flowtime.

Reference:
    [1] J. Ceberio, E. Irurozki, A. Mendiburu, J.A Lozano: A Distance-based
        Ranking Model Estimation of Distribution Algorithm for the Flowshop
        Scheduling Problem. IEEE TEVC, 2014.
"""

import numpy as np
from typing import Optional


class PFSP:
    """
    Permutation Flowshop Scheduling Problem

    Given a matrix of processing times where P[i,j] is the time it takes
    for job i to be processed on machine j, find a permutation of jobs
    that minimizes the given objective.
    """

    def __init__(
        self, 
        processing_times: np.ndarray, 
        objective: str = "makespan", 
        minimize: bool = True
    ):
        """
        Initialize PFSP with a processing times matrix.

        Args:
            processing_times: Matrix of shape (n_jobs, n_machines).
                              processing_times[i, j] is the processing time of job i on machine j.
            objective: "makespan" (maximum completion time) or "flowtime" (sum of completion times).
            minimize: If True, minimize the objective (EDAs will maximize the negative value).

        Raises:
            ValueError: If objective is not recognized.
        """
        self.processing_times = processing_times
        self.n_jobs, self.n_machines = processing_times.shape
        self.minimize = minimize
        
        objective = objective.lower()
        if objective not in ["makespan", "flowtime"]:
            raise ValueError("Objective must be 'makespan' or 'flowtime'")
        self.objective = objective

    def __call__(self, permutation: np.ndarray) -> float:
        """
        Evaluate a permutation of jobs.

        Args:
            permutation: A permutation of jobs (0-indexed or 1-indexed).

        Returns:
            Fitness value (Negative makespan/flowtime if minimize=True, so that
            the EDA naturally maximizes the fitness).
        """
        perm = np.array(permutation, dtype=int)

        
        if np.min(perm) == 1:
            perm = perm - 1

        
        # C[i, m] = tiempo de completado del i-esimo trabajo (en orden perm)
        # en la maquina m. Recurrencia estandar del flowshop:
        #   C[0, 0] = P[perm[0], 0]
        #   C[i, 0] = C[i-1, 0] + P[perm[i], 0]        (maquina 0: sin solapamiento)
        #   C[0, m] = C[0, m-1] + P[perm[0], m]         (primer trabajo: secuencial)
        #   C[i, m] = max(C[i-1, m], C[i, m-1]) + P[perm[i], m]
        n = len(perm)
        C = np.zeros((n, self.n_machines))

        C[0, 0] = self.processing_times[perm[0], 0]
        for m in range(1, self.n_machines):
            C[0, m] = C[0, m - 1] + self.processing_times[perm[0], m]
        for i in range(1, n):
            C[i, 0] = C[i - 1, 0] + self.processing_times[perm[i], 0]
            for m in range(1, self.n_machines):
                C[i, m] = max(C[i - 1, m], C[i, m - 1]) + self.processing_times[perm[i], m]

        total_flowtime = float(C[:, -1].sum())
        makespan = float(C[-1, -1])

        value = makespan if self.objective == "makespan" else total_flowtime

        return -value if self.minimize else value

    def evaluate_makespan(self, permutation: np.ndarray) -> float:
        """Evaluate the exact makespan of a permutation (positive value)."""
        perm = np.array(permutation, dtype=int)
        if np.min(perm) == 1:
            perm -= 1
            
        n = len(perm)
        C = np.zeros((n, self.n_machines))
        C[0, 0] = self.processing_times[perm[0], 0]
        for m in range(1, self.n_machines):
            C[0, m] = C[0, m - 1] + self.processing_times[perm[0], m]
        for i in range(1, n):
            C[i, 0] = C[i - 1, 0] + self.processing_times[perm[i], 0]
            for m in range(1, self.n_machines):
                C[i, m] = max(C[i - 1, m], C[i, m - 1]) + self.processing_times[perm[i], m]
        return float(C[-1, -1])

    def evaluate_flowtime(self, permutation: np.ndarray) -> float:
        """Evaluate the exact total flowtime of a permutation (positive value)."""
        perm = np.array(permutation, dtype=int)
        if np.min(perm) == 1:
            perm -= 1
            
        n = len(perm)
        C = np.zeros((n, self.n_machines))
        C[0, 0] = self.processing_times[perm[0], 0]
        for m in range(1, self.n_machines):
            C[0, m] = C[0, m - 1] + self.processing_times[perm[0], m]
        for i in range(1, n):
            C[i, 0] = C[i - 1, 0] + self.processing_times[perm[i], 0]
            for m in range(1, self.n_machines):
                C[i, m] = max(C[i - 1, m], C[i, m - 1]) + self.processing_times[perm[i], m]
        return float(C[:, -1].sum())


def create_random_pfsp(
    n_jobs: int, 
    n_machines: int, 
    min_time: int = 1, 
    max_time: int = 99, 
    seed: Optional[int] = None
) -> PFSP:
    """
    Create a random PFSP instance (Taillard-style generation).

    Args:
        n_jobs: Number of jobs.
        n_machines: Number of machines.
        min_time: Minimum processing time (default 1).
        max_time: Maximum processing time (default 99).
        seed: Random seed for reproducibility.

    Returns:
        PFSP instance with random processing times.
    """
    if seed is not None:
        np.random.seed(seed)

    # Taillard instances typically draw uniform random integers [1, 99]
    processing_times = np.random.randint(min_time, max_time + 1, size=(n_jobs, n_machines))
    
    return PFSP(processing_times)


def load_taillard_instance(processing_times_matrix: np.ndarray) -> PFSP:
    """
    Load a PFSP instance from a Taillard format matrix.

    Note: Taillard instances are sometimes provided as (n_machines, n_jobs)
    in raw text files. Ensure the matrix passed here is already transposed 
    to (n_jobs, n_machines) if necessary.

    Args:
        processing_times_matrix: Matrix of shape (n_jobs, n_machines)

    Returns:
        PFSP instance
    """
    return PFSP(processing_times_matrix)