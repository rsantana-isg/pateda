"""
HP Protein Folding evaluation functions

Equivalent to MATEDA's HP protein folding functions (EvaluateEnergy.m, EvalChain.m)

The HP model is a simplified protein folding model where amino acids are classified as
either Hydrophobic (H=0) or Polar (P=1). The goal is to fold the protein on a 2D lattice
to maximize contacts between hydrophobic residues.

References:
    R. Santana, P. Larrañaga, and J. A. Lozano (2004) Protein folding in 2-dimensional
    lattices with estimation of distribution algorithms. In Proceedings of the First
    International Symposium on Biological and Medical Data Analysis, Volume 3337 of
    Lecture Notes in Computer Science, pages 388-398, Barcelona, Spain, 2004. Springer Verlag.
"""

import numpy as np
from typing import Tuple


def create_fibonacci_hp_sequence(n: int) -> np.ndarray:
    """
    Create a Fibonacci-based HP protein sequence

    Equivalent to MATEDA's CreateFibbInitConf.m

    Args:
        n: Length of the sequence

    Returns:
        Binary sequence where 0=Hydrophobic (H), 1=Polar (P)

    Example:
        >>> create_fibonacci_hp_sequence(8)
        array([1, 0, 1, 1, 0, 1, 0, 1])
    """
    if n <= 0:
        return np.array([])
    elif n == 1:
        return np.array([1])

    si = [1]
    sj = [0]

    for i in range(2, n + 1):
        aux = sj.copy()
        sj = si + sj
        si = aux

    return np.array(sj)


def put_move_at_pos(pos: np.ndarray, position: int, move: int) -> np.ndarray:
    """
    Update lattice positions given a move

    Equivalent to MATEDA's PutMoveAtPos.m

    Args:
        pos: Current positions array (n_residues x 2)
        position: Position of the residue in the sequence (1-indexed)
        move: Move type (0=left/up, 1=forward, 2=right/down)

    Returns:
        Updated positions array

    Note:
        The function uses relative directions based on the previous move.
        - move=0: Turn left (perpendicular left to current direction)
        - move=1: Continue forward (same direction)
        - move=2: Turn right (perpendicular right to current direction)
    """
    if position < 3:
        return pos

    i = position - 1  # Convert to 0-indexed

    # Check if previous move was horizontal (y-coordinates are same)
    if pos[i - 1, 1] == pos[i - 2, 1]:
        if move == 0:  # UP MOVE (left turn)
            pos[i, 0] = pos[i - 1, 0]
            pos[i, 1] = pos[i - 1, 1] + (pos[i - 1, 0] - pos[i - 2, 0])
        elif move == 1:  # FORWARD MOVE
            pos[i, 0] = pos[i - 1, 0] + (pos[i - 1, 0] - pos[i - 2, 0])
            pos[i, 1] = pos[i - 1, 1]
        else:  # DOWN MOVE (right turn)
            pos[i, 0] = pos[i - 1, 0]
            pos[i, 1] = pos[i - 1, 1] - (pos[i - 1, 0] - pos[i - 2, 0])

    # Check if previous move was vertical (x-coordinates are same)
    if pos[i - 1, 0] == pos[i - 2, 0]:
        if move == 0:  # UP MOVE (left turn)
            pos[i, 1] = pos[i - 1, 1]
            pos[i, 0] = pos[i - 1, 0] - (pos[i - 1, 1] - pos[i - 2, 1])
        elif move == 1:  # FORWARD MOVE
            pos[i, 1] = pos[i - 1, 1] + (pos[i - 1, 1] - pos[i - 2, 1])
            pos[i, 0] = pos[i - 1, 0]
        else:  # DOWN MOVE (right turn)
            pos[i, 1] = pos[i - 1, 1]
            pos[i, 0] = pos[i - 1, 0] + (pos[i - 1, 1] - pos[i - 2, 1])

    return pos


def eval_chain(vector: np.ndarray, hp_sequence: np.ndarray) -> Tuple[int, int, np.ndarray]:
    """
    Evaluate a protein folding configuration

    Equivalent to MATEDA's EvalChain.m

    Given a chain of molecules in the HP model, calculates the number of collisions
    with neighboring same sign molecules, and the number of overlapping molecules.

    Args:
        vector: Sequence of moves (0=left, 1=forward, 2=right) for folding
        hp_sequence: HP sequence (0=Hydrophobic, 1=Polar)

    Returns:
        Tuple of (collisions, overlappings, pos):
            - collisions: Number of non-contiguous H residues that are neighbors in the lattice
            - overlappings: Number of residues that self-intersect in the lattice
            - pos: Position of the residues in 2D lattice (n_residues x 2)
    """
    size_chain = len(vector)
    collisions = 0
    overlappings = 0

    # Initialize positions
    pos = np.zeros((size_chain, 2), dtype=int)
    pos[0, :] = [0, 0]  # First molecule at origin
    pos[1, :] = [1, 0]  # Second molecule to the right

    # Fold the chain
    for i in range(2, size_chain):
        pos = put_move_at_pos(pos, i + 1, int(vector[i]))  # +1 for 1-indexing

        # Check for overlappings and collisions with all previous molecules (except the immediately previous one)
        for j in range(i - 1):
            # Check for overlapping (same position)
            if pos[i, 0] == pos[j, 0] and pos[i, 1] == pos[j, 1]:
                overlappings += 1
            # Check for collisions between hydrophobic residues (both are H=0)
            elif hp_sequence[i] == 0 and hp_sequence[j] == 0:
                # Check if they are adjacent (Manhattan distance = 1)
                if (pos[i, 0] == pos[j, 0] and pos[i, 1] == pos[j, 1] - 1) or \
                   (pos[i, 0] == pos[j, 0] + 1 and pos[i, 1] == pos[j, 1]) or \
                   (pos[i, 0] == pos[j, 0] and pos[i, 1] == pos[j, 1] + 1) or \
                   (pos[i, 0] == pos[j, 0] - 1 and pos[i, 1] == pos[j, 1]):
                    collisions += 1

    return collisions, overlappings, pos


def evaluate_hp_energy(vector: np.ndarray, hp_sequence: np.ndarray) -> float:
    """
    Evaluate the energy of an HP protein folding configuration

    Equivalent to MATEDA's EvaluateEnergy.m

    The energy function penalizes overlappings and rewards collisions between
    hydrophobic residues. Lower energy is better.

    Args:
        vector: Sequence of moves (0=left, 1=forward, 2=right) for folding
        hp_sequence: HP sequence (0=Hydrophobic, 1=Polar)

    Returns:
        Energy value (to be minimized)

    Note:
        - Negative collisions means favorable (more H-H contacts)
        - Overlappings are heavily penalized
        - The formula is: -collisions if no overlaps, else collisions*(overlappings+1)
    """
    collisions, overlappings, pos = eval_chain(vector, hp_sequence)

    if collisions < 0:
        energy = collisions * (overlappings + 1)
    else:
        energy = collisions / (overlappings + 1)

    return energy


def create_hp_objective_function(hp_sequence: np.ndarray):
    """
    Create an objective function for HP protein folding

    Args:
        hp_sequence: HP sequence (0=Hydrophobic, 1=Polar)

    Returns:
        Objective function that takes a folding solution and returns energy

    Example:
        >>> hp_seq = create_fibonacci_hp_sequence(20)
        >>> hp_func = create_hp_objective_function(hp_seq)
        >>> # Solution is a sequence of moves (0, 1, or 2) for positions 3 onwards
        >>> solution = np.random.randint(0, 3, 20)
        >>> energy = hp_func(solution)
    """
    def objective_function(solution: np.ndarray) -> float:
        return evaluate_hp_energy(solution, hp_sequence)

    return objective_function
