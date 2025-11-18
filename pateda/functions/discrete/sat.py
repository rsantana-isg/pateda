"""
SAT Problem Functions

This module provides functions for evaluating and loading SAT problems.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import os


class SATInstance:
    """
    Represents a SAT problem instance.

    A SAT instance consists of one or more formulas, where each formula
    is a set of clauses in conjunctive normal form (CNF).
    """

    def __init__(self):
        """Initialize an empty SAT instance."""
        self.formulas = []  # List of formulas
        self.n_vars = 0
        self.n_objectives = 0

    def add_formula(self, clauses: List[Tuple]):
        """
        Add a formula to the instance.

        Parameters
        ----------
        clauses : list of tuples
            Each clause is a tuple of 6 elements:
            (var1, var2, var3, neg1, neg2, neg3)
            where var_i are variable indices (1-based) and
            neg_i are negation flags (0 = negated, 1 = not negated)
        """
        self.formulas.append(clauses)
        self.n_objectives = len(self.formulas)

    def evaluate(self, solution: np.ndarray) -> np.ndarray:
        """
        Evaluate a solution on all formulas.

        Parameters
        ----------
        solution : np.ndarray
            Binary vector of variable assignments (0 or 1)

        Returns
        -------
        np.ndarray
            Vector of satisfied clause counts for each formula
        """
        if solution.ndim == 1:
            return self._evaluate_single(solution)
        else:
            return np.array([self._evaluate_single(sol) for sol in solution])

    def _evaluate_single(self, solution: np.ndarray) -> np.ndarray:
        """
        Evaluate a single solution.

        Parameters
        ----------
        solution : np.ndarray
            Binary vector of variable assignments

        Returns
        -------
        np.ndarray
            Vector of satisfied clause counts for each formula
        """
        results = np.zeros(self.n_objectives)

        for obj_idx, formula in enumerate(self.formulas):
            satisfied = 0
            for clause in formula:
                var1, var2, var3, neg1, neg2, neg3 = clause

                # Get variable values (convert from 1-based to 0-based indexing)
                val1 = solution[var1 - 1]
                val2 = solution[var2 - 1]
                val3 = solution[var3 - 1]

                # Apply negations (XOR with negation flag)
                lit1 = val1 if neg1 == 1 else (1 - val1)
                lit2 = val2 if neg2 == 1 else (1 - val2)
                lit3 = val3 if neg3 == 1 else (1 - val3)

                # Clause is satisfied if at least one literal is true
                if lit1 or lit2 or lit3:
                    satisfied += 1

            results[obj_idx] = satisfied

        return results


def evaluate_sat(solution: np.ndarray, sat_instance: SATInstance) -> np.ndarray:
    """
    Evaluate a solution or population on a SAT instance.

    This is a wrapper function that can be used as a fitness function.

    Parameters
    ----------
    solution : np.ndarray
        Binary vector or matrix (pop_size, n_vars)
    sat_instance : SATInstance
        The SAT problem instance

    Returns
    -------
    np.ndarray
        Objective values (number of satisfied clauses per formula)
    """
    return sat_instance.evaluate(solution)


def load_random_3sat(
    n_vars: int,
    n_clauses: int,
    n_objectives: int = 1,
    seed: Optional[int] = None
) -> SATInstance:
    """
    Generate random 3-SAT formulas.

    Parameters
    ----------
    n_vars : int
        Number of variables
    n_clauses : int
        Number of clauses per formula
    n_objectives : int
        Number of formulas (objectives)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    SATInstance
        The generated SAT instance
    """
    if seed is not None:
        np.random.seed(seed)

    instance = SATInstance()
    instance.n_vars = n_vars

    for _ in range(n_objectives):
        clauses = []
        for _ in range(n_clauses):
            # Randomly select 3 distinct variables
            vars_selected = np.random.choice(n_vars, 3, replace=False) + 1  # 1-based

            # Randomly decide if each literal is negated
            negations = np.random.randint(0, 2, 3)

            clause = (
                int(vars_selected[0]), int(vars_selected[1]), int(vars_selected[2]),
                int(negations[0]), int(negations[1]), int(negations[2])
            )
            clauses.append(clause)

        instance.add_formula(clauses)

    return instance


def make_random_formulas(
    n_vars: int,
    n_clauses: int,
    n_objectives: int = 1,
    seed: Optional[int] = None
) -> SATInstance:
    """
    Create random 3-SAT formulas.

    This is an alias for load_random_3sat for compatibility with MATLAB code.

    Parameters
    ----------
    n_vars : int
        Number of variables
    n_clauses : int
        Number of clauses per formula
    n_objectives : int
        Number of formulas (objectives)
    seed : int, optional
        Random seed

    Returns
    -------
    SATInstance
        The generated SAT instance
    """
    return load_random_3sat(n_vars, n_clauses, n_objectives, seed)


def make_var_dep_formulas(
    n_vars: int,
    n_clauses: int,
    dependency_prob: float = 0.5,
    n_objectives: int = 1,
    seed: Optional[int] = None
) -> SATInstance:
    """
    Create 3-SAT formulas with variable dependencies.

    Variables that appear together in clauses are more likely to appear
    in subsequent clauses, creating structure in the problem.

    Parameters
    ----------
    n_vars : int
        Number of variables
    n_clauses : int
        Number of clauses per formula
    dependency_prob : float
        Probability of reusing variables from previous clause
    n_objectives : int
        Number of formulas
    seed : int, optional
        Random seed

    Returns
    -------
    SATInstance
        The generated SAT instance
    """
    if seed is not None:
        np.random.seed(seed)

    instance = SATInstance()
    instance.n_vars = n_vars

    for _ in range(n_objectives):
        clauses = []
        prev_vars = None

        for _ in range(n_clauses):
            # Decide whether to reuse variables from previous clause
            if prev_vars is not None and np.random.random() < dependency_prob:
                # Keep one or two variables from previous clause
                n_keep = np.random.randint(1, 3)
                kept_vars = np.random.choice(prev_vars, n_keep, replace=False)
                new_vars = np.random.choice(
                    [v for v in range(n_vars) if v not in kept_vars],
                    3 - n_keep,
                    replace=False
                )
                vars_selected = np.concatenate([kept_vars, new_vars])
            else:
                # Select random variables
                vars_selected = np.random.choice(n_vars, 3, replace=False)

            vars_selected = vars_selected + 1  # Convert to 1-based
            prev_vars = vars_selected - 1

            # Randomly decide if each literal is negated
            negations = np.random.randint(0, 2, 3)

            clause = (
                int(vars_selected[0]), int(vars_selected[1]), int(vars_selected[2]),
                int(negations[0]), int(negations[1]), int(negations[2])
            )
            clauses.append(clause)

        instance.add_formula(clauses)

    return instance


def load_sat_from_file(filename: str) -> SATInstance:
    """
    Load a SAT instance from a file.

    Expected format:
    - First line: n_vars n_clauses n_objectives
    - Following lines: var1 var2 var3 neg1 neg2 neg3 (for each clause)
    - Objectives are separated by a line with "---"

    Parameters
    ----------
    filename : str
        Path to the file

    Returns
    -------
    SATInstance
        The loaded SAT instance
    """
    instance = SATInstance()

    with open(filename, 'r') as f:
        # Read header
        header = f.readline().strip().split()
        n_vars, n_clauses, n_objectives = map(int, header)
        instance.n_vars = n_vars

        current_formula = []
        for line in f:
            line = line.strip()
            if not line or line == '---':
                if current_formula:
                    instance.add_formula(current_formula)
                    current_formula = []
            else:
                parts = list(map(int, line.split()))
                if len(parts) == 6:
                    clause = tuple(parts)
                    current_formula.append(clause)

        # Add last formula
        if current_formula:
            instance.add_formula(current_formula)

    return instance
