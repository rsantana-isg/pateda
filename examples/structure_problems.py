"""
Small combinatorial problems with an explicit interaction structure.

Shared by the network-crossover and substructural-search demos.  Each builder
returns ``(fitness_func, linkage_graph, optimum_or_None, label)`` where
``linkage_graph`` is the symmetric 0/1 variable-interaction matrix that the
structure-exploiting operators can use directly (the "known structure" case).
For Ising / UBQP / SAT the structure *depends on the randomly generated
instance*, exactly the setting the task targets.

All problems are binary and framed as maximization.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Additive: concatenated deceptive trap (known block structure)
# ---------------------------------------------------------------------------

def make_trap(n_blocks=5, k=4):
    """Concatenated trap-k: each block scores k if all-ones, else (k-1 - ones).

    Fully deceptive: single-variable moves point away from the block optimum, so
    the block structure must be respected.  The linkage graph is block-diagonal.
    """
    n = n_blocks * k

    def fitness(x):
        x = np.asarray(x)
        total = 0.0
        for b in range(0, n, k):
            u = int(x[b:b + k].sum())
            total += float(k) if u == k else float(k - 1 - u)
        return total

    G = np.zeros((n, n), dtype=int)
    for b in range(0, n, k):
        idx = np.arange(b, b + k)
        G[np.ix_(idx, idx)] = 1
    np.fill_diagonal(G, 0)
    return fitness, G, float(n_blocks * k), f"trap-{k} x{n_blocks} (n={n})"


# ---------------------------------------------------------------------------
# Ising 2D spin glass on an L x L grid (structure = grid adjacency)
# ---------------------------------------------------------------------------

def make_ising(L=5, seed=0):
    """2D +/-1 Ising spin glass on an L x L periodic grid.

    Maximize -energy = sum_<ij> J_ij s_i s_j with s in {-1,+1} (0->-1, 1->+1).
    The structure is the grid's nearest-neighbor graph, specific to the coupling
    signs of this instance.
    """
    rng = np.random.default_rng(seed)
    n = L * L
    edges = []
    G = np.zeros((n, n), dtype=int)
    for r in range(L):
        for c in range(L):
            i = r * L + c
            for dr, dc in ((0, 1), (1, 0)):           # right / down (periodic)
                j = ((r + dr) % L) * L + ((c + dc) % L)
                if i != j and G[i, j] == 0:
                    J = 1.0 if rng.random() < 0.5 else -1.0
                    edges.append((i, j, J))
                    G[i, j] = G[j, i] = 1
    edges_arr = np.array([(i, j) for i, j, _ in edges])
    J_arr = np.array([J for _, _, J in edges])

    def fitness(x):
        s = 2 * np.asarray(x) - 1                     # {0,1} -> {-1,+1}
        return float(np.sum(J_arr * s[edges_arr[:, 0]] * s[edges_arr[:, 1]]))

    return fitness, G, None, f"Ising 2D {L}x{L} spin glass (n={n})"


# ---------------------------------------------------------------------------
# UBQP: unconstrained binary quadratic (structure = nonzero Q entries)
# ---------------------------------------------------------------------------

def make_ubqp(n=24, density=0.15, seed=0):
    """Maximize x^T Q x for a sparse symmetric Q with +/- integer weights.

    The interaction graph is the sparsity pattern of Q, specific to the instance.
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((n, n))
    G = np.zeros((n, n), dtype=int)
    for i in range(n):
        Q[i, i] = rng.integers(-5, 6)
        for j in range(i + 1, n):
            if rng.random() < density:
                w = float(rng.integers(-5, 6))
                Q[i, j] = Q[j, i] = w
                G[i, j] = G[j, i] = 1

    def fitness(x):
        x = np.asarray(x, dtype=float)
        return float(x @ Q @ x)

    return fitness, G, None, f"UBQP (n={n}, density={density})"


# ---------------------------------------------------------------------------
# SAT: random 3-CNF (structure = variables sharing a clause)
# ---------------------------------------------------------------------------

def make_sat(n=24, ratio=4.0, seed=0):
    """Random 3-SAT with m = ratio*n clauses; maximize satisfied-clause count.

    The interaction graph joins variables that co-occur in a clause, specific to
    the instance.
    """
    rng = np.random.default_rng(seed)
    m = int(ratio * n)
    clauses = []
    G = np.zeros((n, n), dtype=int)
    for _ in range(m):
        vars_ = rng.choice(n, size=3, replace=False)
        signs = rng.integers(0, 2, size=3)            # 1 = positive literal
        clauses.append((vars_, signs))
        for a in range(3):
            for b in range(a + 1, 3):
                G[vars_[a], vars_[b]] = G[vars_[b], vars_[a]] = 1

    def fitness(x):
        x = np.asarray(x)
        sat = 0
        for vars_, signs in clauses:
            if np.any(x[vars_] == signs):
                sat += 1
        return float(sat)

    return fitness, G, float(m), f"3-SAT (n={n}, m={m})"
