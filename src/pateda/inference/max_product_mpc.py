"""
Exact most-probable-configuration (MPC / MAP) for MN-FDA factorized models via
max-product on a junction tree.

Background (docs/MPC/):
  * Meilă, *Max Propagation and Sampling in a Junction Tree* (STAT 535 L10):
    the exact MAP configuration is obtained by a **two-pass** max-propagation ---
    an inward ``CollectEvidence`` with ``MaxAbsorb`` (max-marginalise each clique
    down to its separator) leaves the root clique holding ``p*``; a backward pass
    (indicator potentials / ``DistributeEvidence``) back-tracks the argmax.  This
    is generalized dynamic programming / Viterbi.
  * Kschischang et al. max-product on a factor graph is exact only when the graph
    has no loops; the single forward-pass greedy argmax used earlier is therefore
    an *approximation*.

Implementation.  We realise the two-pass junction-tree max-product as
**max-product variable elimination** (VE): eliminating a variable multiplies the
factors that contain it and max-marginalises it out (the inward pass), recording,
for every eliminated variable, the argmax as a function of the remaining
variables of its cluster (the back-tracking indices).  A final reverse pass over
the elimination order reconstructs ``x*``.  VE with an elimination order *is* the
junction-tree algorithm --- the order induces the cluster (junction) tree, and
each elimination step is one ``MaxAbsorb``.  Its cost is exponential only in the
induced width (treewidth); for the bounded / sparse MN-FDA structures it is
cheap and exact.

MN-FDA models are acyclic --- their clique/junction structure is a **forest**
(possibly several disjoint trees).  Consequently max-product is *exact* and the
induced width never exceeds ``max_clique_size - 1``, so exact MPC is always
tractable; variable elimination handles the several trees automatically (each
disconnected component is an independent sub-problem whose optimum multiplies
into ``p*``, and back-tracking recovers each component independently).  The
``max_table_size`` cap below is therefore only a safety net.

Everything is done in log-space to avoid underflow.
"""

from typing import List, Tuple
from collections import defaultdict
import numpy as np

from pateda.learning.utils.conversions import find_acc_card, num_convert_card


class _Factor:
    """A factor over a *sorted* list of variable indices; ``log`` holds the
    log-potential with axes in ``vars`` order."""

    __slots__ = ("vars", "log")

    def __init__(self, variables: List[int], log_values: np.ndarray):
        self.vars = list(variables)
        self.log = log_values


def _make_factor(var_order: List[int], values_in_order: np.ndarray) -> _Factor:
    """Canonicalise a factor to sorted variable order (transposing its axes)."""
    order = list(np.argsort(var_order))
    sorted_vars = [var_order[i] for i in order]
    return _Factor(sorted_vars, np.transpose(values_in_order, axes=order))


def _factor_from_clique_table(clique_vars, table, cardinality) -> _Factor:
    """Build a log-factor from a plain ``(clique_vars, table)`` pair.

    ``table`` is a joint potential over ``clique_vars`` stored in C-order (the
    convention used by :class:`MAPInference`): the first variable is the
    slowest-varying axis.  The table may be flat or already shaped; it is
    reshaped to the clique's cardinalities.
    """
    cv = [int(v) for v in clique_vars]
    shape = tuple(int(cardinality[v]) for v in cv)
    T = np.asarray(table, dtype=float).reshape(shape)
    return _make_factor(cv, np.log(np.maximum(T, 1e-300)))


def _factor_from_clique(row, table, cardinality) -> _Factor:
    """Build a log-factor over a clique's variables from its MN-FDA table.

    The clique table is ``p(new)`` (root) or ``p(new | overlap)``; as a function
    of ``(overlap, new)`` it is a factor over all clique variables, and the
    product of all clique factors equals the joint (each variable is "new" once,
    conditioned on its overlap = its parents).  Entries are read with the exact
    same mixed-radix indexing the sampler uses, so there is no ambiguity.
    """
    n_overlap = int(row[0])
    n_new = int(row[1])
    overlap_vars = row[2:2 + n_overlap].astype(int)
    new_vars = row[2 + n_overlap:2 + n_overlap + n_new].astype(int)
    clique_vars = [int(v) for v in overlap_vars] + [int(v) for v in new_vars]

    new_acc = find_acc_card(n_new, cardinality[new_vars])
    overlap_acc = find_acc_card(n_overlap, cardinality[overlap_vars]) if n_overlap else None
    tbl = np.asarray(table)

    shape = tuple(int(cardinality[v]) for v in clique_vars)
    T = np.empty(shape, dtype=float)
    for idx in np.ndindex(*shape):
        nvals = np.asarray(idx[n_overlap:], dtype=int)
        ni = num_convert_card(nvals, n_new, new_acc)
        if n_overlap == 0:
            T[idx] = tbl.ravel()[ni]
        else:
            ovals = np.asarray(idx[:n_overlap], dtype=int)
            oi = num_convert_card(ovals, n_overlap, overlap_acc)
            T[idx] = tbl[oi, ni]

    return _make_factor(clique_vars, np.log(np.maximum(T, 1e-300)))


def _multiply(factors: List[_Factor], cardinality) -> _Factor:
    """Product of factors in log-space (broadcast-add over the union of vars)."""
    if len(factors) == 1:
        return factors[0]
    union = sorted(set().union(*[set(f.vars) for f in factors]))
    pos = {v: i for i, v in enumerate(union)}
    out = np.zeros([int(cardinality[v]) for v in union], dtype=float)
    for f in factors:
        bshape = [1] * len(union)
        for i, v in enumerate(f.vars):
            bshape[pos[v]] = f.log.shape[i]
        out = out + f.log.reshape(bshape)      # both var lists sorted -> axes align
    return _Factor(union, out)


def _max_eliminate(factor: _Factor, var: int) -> Tuple[_Factor, List[int], np.ndarray]:
    """Max-marginalise ``var`` out of ``factor`` (one ``MaxAbsorb``).

    Returns the reduced factor, the remaining (sorted) variables, and the argmax
    of ``var`` as a function of those remaining variables (the back-track table).
    """
    axis = factor.vars.index(var)
    reduced = np.max(factor.log, axis=axis)
    argmax = np.argmax(factor.log, axis=axis)
    rem_vars = [v for v in factor.vars if v != var]
    return _Factor(rem_vars, reduced), rem_vars, argmax


def _min_fill_order(factors: List[_Factor]) -> List[int]:
    """Min-fill elimination order over the interaction (moral) graph."""
    adj = defaultdict(set)
    allv = set()
    for f in factors:
        vs = f.vars
        allv.update(vs)
        for a in vs:
            for b in vs:
                if a != b:
                    adj[a].add(b)
    remaining = set(allv)
    order = []
    while remaining:
        best_v, best_fill = None, None
        for v in remaining:
            nb = [u for u in adj[v] if u in remaining]
            fill = 0
            for i in range(len(nb)):
                ai = nb[i]
                for j in range(i + 1, len(nb)):
                    if nb[j] not in adj[ai]:
                        fill += 1
            if best_fill is None or fill < best_fill:
                best_v, best_fill = v, fill
        nb = [u for u in adj[best_v] if u in remaining]
        for a in nb:                                   # add fill edges
            for b in nb:
                if a != b:
                    adj[a].add(b)
        order.append(best_v)
        remaining.discard(best_v)
    return order


def _min_degree_order(factors: List[_Factor]) -> List[int]:
    """Min-degree elimination order (cheaper to compute than min-fill)."""
    adj = defaultdict(set)
    allv = set()
    for f in factors:
        vs = f.vars
        allv.update(vs)
        for a in vs:
            for b in vs:
                if a != b:
                    adj[a].add(b)
    remaining = set(allv)
    order = []
    while remaining:
        v = min(remaining, key=lambda u: len(adj[u] & remaining))
        nb = [u for u in adj[v] if u in remaining]
        for a in nb:
            for b in nb:
                if a != b:
                    adj[a].add(b)
        order.append(v)
        remaining.discard(v)
    return order


_ORDERINGS = {"min_fill": _min_fill_order, "min_degree": _min_degree_order}


def max_product_mpc(
    structure: np.ndarray,
    tables: List[np.ndarray],
    cardinality: np.ndarray,
    order_method: str = "min_degree",
    max_table_size: int = 1 << 24,
) -> Tuple[np.ndarray, float]:
    """Exact most-probable configuration of an MN-FDA factorized model.

    Args:
        structure: FactorizedModel structure rows
            ``[n_overlap, n_new, overlap..., new...]``.
        tables: per-clique probability tables.
        cardinality: variable cardinalities (length = n_vars).
        order_method: elimination order heuristic, ``"min_fill"`` (default,
            usually lowest treewidth) or ``"min_degree"`` (cheaper ordering).
        max_table_size: cap on the number of entries of any intermediate cluster
            table; exceeding it raises ``MPCIntractable`` (exact MAP would need a
            table larger than this — the model's treewidth is too high).

    Returns:
        ``(x_star, log_pstar)`` — the MAP configuration and its model log-prob.
    """
    cardinality = np.asarray(cardinality)
    factors = [_factor_from_clique(structure[c], tables[c], cardinality)
               for c in range(structure.shape[0])]
    return _eliminate_and_backtrack(
        factors, len(cardinality), cardinality, order_method, max_table_size)


def max_product_mpc_cliques(
    cliques,
    tables: List[np.ndarray],
    cardinality: np.ndarray,
    order_method: str = "min_degree",
    max_table_size: int = 1 << 24,
) -> Tuple[np.ndarray, float]:
    """Exact MPC from a plain ``(cliques, tables)`` representation.

    Each factor is a joint potential ``tables[c]`` over the variables
    ``cliques[c]`` stored in C-order (the :class:`MAPInference` convention).
    Same exact junction-tree max-product as :func:`max_product_mpc`.

    Returns ``(x_star, log_pstar)``.
    """
    cardinality = np.asarray(cardinality)
    factors = [_factor_from_clique_table(cliques[c], tables[c], cardinality)
               for c in range(len(cliques))]
    return _eliminate_and_backtrack(
        factors, len(cardinality), cardinality, order_method, max_table_size)


def _eliminate_and_backtrack(factors, n_vars, cardinality, order_method,
                             max_table_size):
    """Max-product variable elimination (inward pass) + argmax back-tracking
    (outward pass) — the two-pass junction-tree max-propagation."""
    order = _ORDERINGS[order_method](factors)

    active = list(factors)
    back = {}                                # var -> (remaining_vars, argmax_table)
    for v in order:
        relevant = [f for f in active if v in f.vars]
        if not relevant:
            continue
        active = [f for f in active if v not in f.vars]
        prod = _multiply(relevant, cardinality)
        if prod.log.size > max_table_size:
            raise MPCIntractable(
                f"exact MPC intractable: a cluster table would need "
                f"{prod.log.size} entries (> max_table_size={max_table_size}); "
                f"the model treewidth is too high for exact max-product.")
        reduced, rem_vars, argmax = _max_eliminate(prod, v)
        back[v] = (rem_vars, argmax)
        active.append(reduced)

    log_pstar = float(sum(np.asarray(f.log).sum() for f in active))

    # Backward pass: reconstruct x* in reverse elimination order.
    assign = {}
    for v in reversed(order):
        if v not in back:
            assign[v] = 0
            continue
        rem_vars, argmax = back[v]
        if argmax.ndim == 0:
            assign[v] = int(argmax)
        else:
            idx = tuple(assign[u] for u in rem_vars)
            assign[v] = int(argmax[idx])

    x_star = np.array([assign.get(i, 0) for i in range(n_vars)], dtype=int)
    return x_star, log_pstar


class MPCIntractable(RuntimeError):
    """Raised when exact max-product MPC would exceed the treewidth/table cap."""
