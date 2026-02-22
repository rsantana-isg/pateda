"""
Interaction Matrix Learning from Problem Structure

This module provides methods for constructing binary interaction matrices
from optimization problems whose structure is known a priori. A value of 1
in cell (i, j) of the returned matrix means that variables X_i and X_j
interact (i.e., they appear together in at least one structural constraint
or subfunction of the problem).

These interaction matrices can be passed directly to the restricted EDA
variants Tree-EDA_r, MN-FDA_r, and MN-FDAG_r to bias model learning
toward known structural relationships.

Available functions
-------------------
find_matrix_interactions_SAT(sat_instance)
    Interactions from a SAT instance: two variables interact if they
    appear together in at least one clause.

find_matrix_interactions_ising(ising_graph, n_vars)
    Interactions from an Ising model: two variables interact if there
    is a coupling term between them.

find_matrix_interactions_nk(n_vars, k_neighbors, neighbor_lists=None)
    Interactions from an NK-landscape: each variable interacts with its
    k epistatic neighbors.

find_matrix_interactions_additive_decomposable(subfunction_vars, n_vars)
    Interactions from an additively decomposable function: two variables
    interact if they appear in the same subfunction.

find_matrix_interactions_RNA_design(RNA_structure)
    Interactions from an RNA secondary structure in Dot-Bracket Notation:
    two variables interact if they are adjacent in the sequence (X_i and
    X_{i+1}) or if they represent a Watson-Crick base pair encoded by a
    matching pair of parentheses in the structure string.

References
----------
- Santana, R., Mendiburu, A., and Lozano, J.A. (2012). "Structural transfer
  using EDAs: An application to multi-marker tagging SNP selection."
  Proceedings of the 2012 IEEE Congress on Evolutionary Computation (CEC-2012),
  pages 3484-3491, Brisbane, Australia.
- Santana, R. (2017). "Gray-box optimization and factorized distribution
  algorithms: where two worlds collide." arXiv:1707.03093.
- Santana, R. (2005). "Estimation of distribution algorithms with Kikuchi
  approximations." Evolutionary Computation, 13(1):67-97.
"""

from typing import List, Optional, Sequence, Tuple
import numpy as np


def find_matrix_interactions_SAT(sat_instance) -> np.ndarray:
    """
    Compute a binary interaction matrix from a SAT instance.

    Two variables X_i and X_j are considered to interact if they appear
    together in at least one clause of any formula in the SAT instance.
    In 3-SAT, each clause involves exactly three variables, so all
    pairs within a clause are treated as interacting.

    The interaction matrix R is symmetric:
        R[i, j] = 1  iff  exists a clause containing both variable i and
                          variable j (using 0-based variable indices)
        R[i, j] = 0  otherwise
        R[i, i] = 0  (no self-interactions on the diagonal)

    Parameters
    ----------
    sat_instance : SATInstance
        A SAT problem instance from pateda.functions.discrete.sat.
        The instance must have:
        - sat_instance.n_vars: total number of variables (int)
        - sat_instance.formulas: list of formulas, each a list of clauses.
          Each clause is a tuple (var1, var2, var3, neg1, neg2, neg3)
          where var1, var2, var3 are 1-based variable indices.

    Returns
    -------
    np.ndarray
        Binary symmetric interaction matrix of shape (n_vars, n_vars).
        Entry (i, j) is 1 if variables X_i and X_j (0-based) appear in
        the same clause in any formula; 0 otherwise.
        Diagonal entries are always 0.

    Examples
    --------
    >>> from pateda.functions.discrete.sat import load_random_3sat
    >>> from pateda.learning.interaction_learning import find_matrix_interactions_SAT
    >>> sat = load_random_3sat(n_vars=10, n_clauses=20, seed=42)
    >>> R = find_matrix_interactions_SAT(sat)
    >>> R.shape
    (10, 10)
    >>> # The matrix is symmetric
    >>> np.array_equal(R, R.T)
    True
    >>> # Diagonal is always zero
    >>> np.all(np.diag(R) == 0)
    True
    """
    n_vars = sat_instance.n_vars
    interaction_matrix = np.zeros((n_vars, n_vars), dtype=int)

    for formula in sat_instance.formulas:
        for clause in formula:
            # Extract variable indices (1-based); ignore negation flags
            # Clause format: (var1, var2, var3, neg1, neg2, neg3)
            clause_vars_1based = [clause[0], clause[1], clause[2]]

            # Convert to 0-based indices
            clause_vars = [v - 1 for v in clause_vars_1based]

            # All pairs within the clause interact
            for idx_a in range(len(clause_vars)):
                for idx_b in range(idx_a + 1, len(clause_vars)):
                    i = clause_vars[idx_a]
                    j = clause_vars[idx_b]

                    # Validate indices (guard against malformed instances)
                    if 0 <= i < n_vars and 0 <= j < n_vars and i != j:
                        interaction_matrix[i, j] = 1
                        interaction_matrix[j, i] = 1

    return interaction_matrix


def find_matrix_interactions_ising(
    coupling_matrix: np.ndarray,
) -> np.ndarray:
    """
    Compute a binary interaction matrix from an Ising model.

    Two variables X_i and X_j interact if there is a non-zero coupling
    term J_{ij} in the Ising model. The coupling matrix may be asymmetric
    in storage, but interactions are treated symmetrically.

    The Ising energy function is:
        E(x) = -sum_{i<j} J_{ij} * s_i * s_j - sum_i h_i * s_i

    where s_i in {-1, +1} (or equivalently {0, 1}).

    Parameters
    ----------
    coupling_matrix : np.ndarray
        Square matrix of shape (n_vars, n_vars) containing Ising coupling
        strengths J_{ij}. Entry (i, j) != 0 indicates an interaction.

    Returns
    -------
    np.ndarray
        Binary symmetric interaction matrix of shape (n_vars, n_vars).
        Entry (i, j) is 1 if coupling_matrix[i, j] != 0 or
        coupling_matrix[j, i] != 0; 0 otherwise.
        Diagonal entries are always 0.

    Examples
    --------
    >>> J = np.array([[0, 1, 0], [1, 0, -1], [0, -1, 0]], dtype=float)
    >>> R = find_matrix_interactions_ising(J)
    >>> R
    array([[0, 1, 0],
           [1, 0, 1],
           [0, 1, 0]])
    """
    coupling_matrix = np.asarray(coupling_matrix)
    n_vars = coupling_matrix.shape[0]

    # Symmetrize: (i, j) interacts if either J_{ij} or J_{ji} is non-zero
    nonzero = (np.abs(coupling_matrix) + np.abs(coupling_matrix.T)) > 0
    interaction_matrix = nonzero.astype(int)

    # Remove diagonal
    np.fill_diagonal(interaction_matrix, 0)

    return interaction_matrix


def find_matrix_interactions_nk(
    n_vars: int,
    k_neighbors: int,
    neighbor_lists: Optional[List[List[int]]] = None,
) -> np.ndarray:
    """
    Compute a binary interaction matrix from an NK-landscape.

    In an NK-landscape, each variable X_i interacts with k other variables,
    called its epistatic neighbors. If neighbor_lists is provided, the exact
    neighborhood structure is used. Otherwise, a nearest-neighbor (adjacent)
    topology is assumed (common in standard NK-landscape benchmarks).

    Parameters
    ----------
    n_vars : int
        Number of variables.
    k_neighbors : int
        Number of epistatic neighbors per variable.
    neighbor_lists : list of list of int, optional
        Explicit neighborhood structure. neighbor_lists[i] is the list of
        0-based indices of the k neighbors of variable i (not including i
        itself). If None, adjacent wrapping neighbors are used.

    Returns
    -------
    np.ndarray
        Binary symmetric interaction matrix of shape (n_vars, n_vars).
        Entry (i, j) is 1 if j is in the neighborhood of i or vice versa.
        Diagonal entries are always 0.

    Examples
    --------
    >>> R = find_matrix_interactions_nk(n_vars=5, k_neighbors=2)
    >>> # Each variable interacts with its 2 adjacent neighbors (circular)
    >>> R[0, 1], R[0, 4]
    (1, 1)
    """
    interaction_matrix = np.zeros((n_vars, n_vars), dtype=int)

    if neighbor_lists is not None:
        # Use provided neighborhood structure
        for i, neighbors in enumerate(neighbor_lists):
            for j in neighbors:
                if 0 <= j < n_vars and i != j:
                    interaction_matrix[i, j] = 1
                    interaction_matrix[j, i] = 1
    else:
        # Default: adjacent circular neighborhood
        for i in range(n_vars):
            for offset in range(1, k_neighbors + 1):
                j = (i + offset) % n_vars
                interaction_matrix[i, j] = 1
                interaction_matrix[j, i] = 1

    return interaction_matrix


def find_matrix_interactions_additive_decomposable(
    subfunction_vars: List[Sequence[int]],
    n_vars: int,
) -> np.ndarray:
    """
    Compute a binary interaction matrix from an additively decomposable function.

    An additively decomposable function (ADF) has the form:
        f(x) = sum_k f_k(x_{S_k})

    where each subfunction f_k is defined on a subset S_k of the variables.
    Two variables X_i and X_j are considered to interact if they appear in
    the same subfunction definition set S_k.

    Parameters
    ----------
    subfunction_vars : list of sequences of int
        List of variable index sets, one per subfunction. Each element is
        a sequence of 0-based variable indices involved in that subfunction.
    n_vars : int
        Total number of variables.

    Returns
    -------
    np.ndarray
        Binary symmetric interaction matrix of shape (n_vars, n_vars).
        Entry (i, j) is 1 if i and j appear together in any subfunction;
        0 otherwise. Diagonal entries are always 0.

    Examples
    --------
    >>> # Deceptive-3 with 6 variables: two groups of 3
    >>> subf = [[0, 1, 2], [3, 4, 5]]
    >>> R = find_matrix_interactions_additive_decomposable(subf, n_vars=6)
    >>> R[0, 1], R[0, 3]
    (1, 0)
    """
    interaction_matrix = np.zeros((n_vars, n_vars), dtype=int)

    for var_set in subfunction_vars:
        var_list = list(var_set)
        for idx_a in range(len(var_list)):
            for idx_b in range(idx_a + 1, len(var_list)):
                i = var_list[idx_a]
                j = var_list[idx_b]
                if 0 <= i < n_vars and 0 <= j < n_vars and i != j:
                    interaction_matrix[i, j] = 1
                    interaction_matrix[j, i] = 1

    return interaction_matrix


def find_matrix_interactions_RNA_design(RNA_structure: str) -> np.ndarray:
    """
    Compute a binary interaction matrix from an RNA secondary structure.

    Two types of interactions are encoded:

    1. **Sequence adjacency**: variables X_i and X_{i+1} always interact
       because consecutive nucleotides in the RNA backbone are covalently
       bonded and their identities jointly determine local folding stability.

    2. **Base-pair interactions**: variables X_i and X_j interact if they
       are paired in the secondary structure, i.e., position i carries '('
       and position j carries the matching ')' in the Dot-Bracket string.
       These represent Watson-Crick (or wobble) hydrogen-bond interactions.

    Positions carrying '.' (unpaired bases) do not contribute base-pair
    interactions but still interact with their sequence neighbours.

    The Dot-Bracket format uses a standard stack-based matching rule:
    each '(' is paired with the closest unmatched ')' to its right.
    Pseudoknots (crossing pairs) are not supported by the standard notation
    and are not handled here.

    The returned matrix R is binary and symmetric:
        R[i, j] = 1  if |i - j| == 1  (adjacent in sequence)
        R[i, j] = 1  if positions i and j are a base pair  (matched parens)
        R[i, j] = 0  otherwise
        R[i, i] = 0  (diagonal is always zero)

    Parameters
    ----------
    RNA_structure : str
        Dot-Bracket Notation string of length n (the number of nucleotide
        positions / variables). Valid characters are '(', ')', and '.'.

    Returns
    -------
    np.ndarray
        Binary symmetric interaction matrix of shape (n, n) where n is
        ``len(RNA_structure)``.

    Raises
    ------
    ValueError
        If the Dot-Bracket string contains unmatched parentheses.

    Examples
    --------
    >>> R = find_matrix_interactions_RNA_design("((....))")
    >>> # Sequence adjacency: (0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7)
    >>> # Base pairs:          (0,7), (1,6)
    >>> R[0, 7], R[1, 6]   # base pairs
    (1, 1)
    >>> R[0, 1], R[6, 7]   # sequence neighbours
    (1, 1)
    >>> R[0, 5]             # no interaction
    0
    >>> import numpy as np
    >>> np.array_equal(R, R.T)   # symmetric
    True
    >>> np.all(np.diag(R) == 0)  # diagonal is zero
    True
    """
    n = len(RNA_structure)
    interaction_matrix = np.zeros((n, n), dtype=int)

    # ── 1. Sequence adjacency: X_i and X_{i+1} always interact ───────────────
    for i in range(n - 1):
        interaction_matrix[i, i + 1] = 1
        interaction_matrix[i + 1, i] = 1

    # ── 2. Base-pair interactions from Dot-Bracket notation ──────────────────
    # Use a stack to match opening '(' with closing ')'.
    stack: List[int] = []
    for i, symbol in enumerate(RNA_structure):
        if symbol == '(':
            stack.append(i)
        elif symbol == ')':
            if not stack:
                raise ValueError(
                    f"Unmatched ')' at position {i} in RNA structure: "
                    f"'{RNA_structure}'"
                )
            j = stack.pop()   # j is the matching '(' position
            # X_j and X_i are a base pair → they interact
            interaction_matrix[j, i] = 1
            interaction_matrix[i, j] = 1
        # '.' leaves no additional interaction beyond sequence adjacency

    if stack:
        raise ValueError(
            f"Unmatched '(' at position(s) {stack} in RNA structure: "
            f"'{RNA_structure}'"
        )

    return interaction_matrix
