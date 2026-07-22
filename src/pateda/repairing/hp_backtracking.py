"""
Backtracking repair for HP protein folding solutions

The sampling step of an EDA on the HP (hydrophobic-polar) protein model can
produce *self-intersecting* folds: two residues placed on the same lattice
site.  Following Santana, Karshenas, Bielza & Larrañaga (2011), who apply the
backtracking repair of Chen, Li & Hu (2009), this operator turns each sampled
move sequence into a valid *self-avoiding walk*.

A solution uses the relative-move encoding: variable ``x_i in {0,1,2}`` is the
move (left / forward / right) of residue ``i`` relative to the previous two.
The lattice positions depend only on the moves, so the repair needs only the
population (not the HP labels).  For each solution it rebuilds the walk residue
by residue, keeping the sampled move whenever it does not create a collision and
otherwise trying the alternative moves; when no move fits, it backtracks to the
previous residue and tries a different move there.  The sampled move is always
tried first, so a solution is changed as little as possible.  If the search
exceeds a node budget (a pathological, rare case in 2-D) the original solution
is kept unchanged --- the HP fitness still penalizes any residual overlap.

Implemented as a :class:`~pateda.core.components.RepairingMethod`, so it plugs
into the EDA exactly like the other repairing operators and runs on the sampled
population before evaluation.

References
----------
- Santana, R., Karshenas, H., Bielza, C., & Larrañaga, P. (2011). "Regularized
  k-order Markov models in EDAs." GECCO 2011, pp. 593-600.
- Chen, B., Li, L., & Hu, J. (2009). "An improved backtracking method for EDAs."
  ICROS-SICE International Joint Conference 2009, pp. 4669-4674.
"""

from typing import Any, Optional
import numpy as np

from pateda.core.components import RepairingMethod
from pateda.functions.discrete_non_binary.problems.hp_protein import put_move_at_pos


def repair_hp_self_avoiding(
    moves: np.ndarray, max_nodes: Optional[int] = None
) -> Optional[np.ndarray]:
    """
    Turn a single HP move sequence into a self-avoiding walk by backtracking.

    Parameters
    ----------
    moves : np.ndarray of shape (n,)
        Relative moves in ``{0, 1, 2}``; the first two are placement-free.
    max_nodes : int, optional
        Backtracking node budget.  ``None`` uses ``200 * n``.

    Returns
    -------
    np.ndarray or None
        A self-avoiding move sequence close to ``moves``, or ``None`` if the
        budget was exhausted without success.
    """
    moves = np.asarray(moves, dtype=int)
    n = moves.shape[0]
    if n <= 2:
        return moves.copy()
    budget = int(max_nodes) if max_nodes is not None else 200 * n

    pos = np.zeros((n, 2), dtype=int)
    pos[1] = [1, 0]                                  # first two residues fixed
    new_moves = moves.copy()
    occupied = {(0, 0), (1, 0)}
    tried = [set() for _ in range(n)]

    i = 2
    nodes = 0
    while i < n:
        nodes += 1
        if nodes > budget:
            return None                             # give up (keep original)

        # Try the sampled move first, then the remaining alternatives.
        orig = int(moves[i])
        candidates = [orig] + [m for m in (0, 1, 2) if m != orig]
        placed = False
        for m in candidates:
            if m in tried[i]:
                continue
            tried[i].add(m)
            put_move_at_pos(pos, i + 1, m)          # 1-indexed position
            site = (int(pos[i, 0]), int(pos[i, 1]))
            if site not in occupied:
                occupied.add(site)
                new_moves[i] = m
                placed = True
                break

        if placed:
            i += 1
        else:
            tried[i] = set()                        # reset for a future visit
            i -= 1
            if i < 2:
                return None                         # cannot be repaired
            occupied.discard((int(pos[i, 0]), int(pos[i, 1])))

    return new_moves


class HPBacktrackingRepair(RepairingMethod):
    """
    Repair HP folds into self-avoiding walks by backtracking.

    See the module docstring.  Solutions that cannot be repaired within the node
    budget are left unchanged (and remain penalized by the HP fitness).
    """

    def __init__(self, max_nodes: Optional[int] = None):
        """
        Args:
            max_nodes: Backtracking node budget per solution (``None`` -> ``200*n``).
        """
        self.max_nodes = max_nodes

    def repair(
        self,
        population: np.ndarray,
        cardinality: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        """
        Repair every solution of the population into a self-avoiding walk.

        Args:
            population: Population of move sequences ``(pop_size, n)``.
            cardinality: Variable cardinalities (unused; moves are in ``{0,1,2}``).
            **params: ``max_nodes`` overrides the instance budget.

        Returns:
            The repaired population (same shape).
        """
        max_nodes = params.get("max_nodes", self.max_nodes)
        repaired = np.asarray(population, dtype=int).copy()
        for j in range(repaired.shape[0]):
            fixed = repair_hp_self_avoiding(repaired[j], max_nodes=max_nodes)
            if fixed is not None:
                repaired[j] = fixed
        return repaired
