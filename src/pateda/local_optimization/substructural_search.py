"""
Substructural neighborhood search — structure-guided local search for EDAs

Standard local searchers flip one variable at a time.  Substructural
neighborhood search (Lima, Pelikan, Sastry, Butz, Goldberg & Lobo, 2006)
instead moves over the joint values of whole *substructures* -- groups of
tightly-linked variables identified by the learned probabilistic model.  By
optimizing a linkage group as a unit it can cross the "valleys" that trap a
single-variable hill climber on problems with strong dependencies (deceptive
traps, spin glasses, ...), providing the intensification that complements the
global model-based search of the EDA.

The substructures come from :func:`pateda.knowledge_extraction.model_to_substructures`,
so the *same* searcher works for every discrete model class in pateda -- for a
Bayesian network the substructure of a variable is ``{X_i} ∪ parents(X_i)`` (or
children, or both; the paper's three neighborhoods), and for a factorized /
Markov-network model it is a clique -- and also with a linkage graph known a
priori from the problem structure (an additive function's blocks, an Ising
lattice, a UBQP weight matrix, a SAT clause graph).

For each substructure ``S`` the searcher enumerates all joint value
combinations of the variables in ``S`` (``∏_{i∈S} card_i`` of them) and moves to
the best-improving combination, using the *actual* fitness function to accept a
move -- the robust acceptance variant of the paper (the alternative surrogate
variant needs a fitness model that pateda's Bayesian networks do not store).

Consistency with the rest of pateda:

- It is a :class:`~pateda.core.components.LocalOptMethod`, subclassing
  :class:`~pateda.local_optimization.budgeted_search.BudgetedLocalSearch`, so it
  shares the *same* intensity controls as every other memetic local search:
  ``subset_fraction`` (fraction of the sampled population optimized) and
  ``evaluation_budget`` (total evaluations shared among the selected solutions).
- The EDA passes the current model to ``optimize`` (as ``model=...``); the model
  learned in the previous generation -- the one that produced this population --
  is exactly the structure to exploit.  A fixed ``linkage_graph`` or explicit
  ``substructures`` can be given instead when the structure is known a priori.

Relationship to partial sampling: :class:`~pateda.sampling.partial.SamplePartialFDA`
also isolates a subset of variables, but it *resamples* them from the model's
probabilities given the rest; substructural search *searches* their joint values
by evaluating the objective.  Same structural decomposition, different operation
(sample vs. optimize), mirroring how network crossover *swaps* a substructure.

References
----------
- Lima, C. F., Pelikan, M., Sastry, K., Butz, M., Goldberg, D. E., & Lobo, F. G.
  (2006). "Substructural Neighborhoods for Local Search in the Bayesian
  Optimization Algorithm." PPSN IX / MEDAL Report 2006007.
- Sastry, K., & Goldberg, D. E. (2004). "Let's get ready to rumble: Crossover
  versus mutation head to head." GECCO 2004 (substructural mutation).
"""

from itertools import product
from typing import Any, List, Optional, Tuple
import numpy as np

from pateda.local_optimization.budgeted_search import (
    BudgetedLocalSearch,
    _BudgetEvaluator,
)
from pateda.knowledge_extraction.model_structure import model_to_substructures


class SubstructuralLocalSearch(BudgetedLocalSearch):
    """
    Substructural neighborhood hill climber.

    Optimizes each solution by moving over the joint values of the substructures
    induced by the learned model (or a supplied linkage graph).  Shares the
    subset / evaluation-budget interface of
    :class:`~pateda.local_optimization.budgeted_search.BudgetedLocalSearch`.
    """

    def __init__(
        self,
        neighborhood: str = "both",
        linkage_graph: Optional[np.ndarray] = None,
        substructures: Optional[List[np.ndarray]] = None,
        max_substructure_size: int = 6,
        max_combinations: int = 4096,
        acceptance: str = "best",
        subset_fraction: float = 1.0,
        evaluation_budget: Optional[int] = None,
        per_solution_budget: int = 300,
        subset_selection: str = "best",
        seed: Optional[int] = None,
    ):
        """
        Args:
            neighborhood: Substructure definition passed to
                :func:`~pateda.knowledge_extraction.model_to_substructures`:
                ``"parental"``, ``"children"``, ``"both"`` (default),
                ``"neighborhood"`` or ``"clique"``.
            linkage_graph: Optional fixed interaction matrix (problem-known
                structure); used when no model is provided at call time.
            substructures: Optional explicit list of variable-index groups,
                bypassing extraction entirely.
            max_substructure_size: Skip substructures with more variables than
                this (bounds the enumeration cost).
            max_combinations: Skip substructures whose joint value count
                ``∏ card_i`` exceeds this (a second, cardinality-aware bound).
            acceptance: ``"best"`` moves to the best-improving combination in the
                substructure; ``"first"`` moves to the first improving one.
            subset_fraction, evaluation_budget, per_solution_budget,
            subset_selection, seed: See
                :class:`~pateda.local_optimization.budgeted_search.BudgetedLocalSearch`.
        """
        super().__init__(
            subset_fraction=subset_fraction,
            evaluation_budget=evaluation_budget,
            per_solution_budget=per_solution_budget,
            subset_selection=subset_selection,
            seed=seed,
        )
        if acceptance not in ("best", "first"):
            raise ValueError(f"acceptance must be 'best' or 'first', got {acceptance!r}")
        self.neighborhood = neighborhood
        self.linkage_graph = None if linkage_graph is None else np.asarray(linkage_graph)
        self.substructures = substructures
        self.max_substructure_size = max_substructure_size
        self.max_combinations = max_combinations
        self.acceptance = acceptance
        self._substructures: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Structure resolution (once per optimize call)
    # ------------------------------------------------------------------
    def _resolve_substructures(self, n_vars, cardinality, params) -> List[np.ndarray]:
        if self.substructures is not None:
            groups = [np.asarray(s, dtype=int) for s in self.substructures]
        else:
            source = self.linkage_graph
            if source is None:
                source = params.get("model", None)
            if source is None:
                # No structure available: fall back to singletons (coordinate
                # descent), so the search still runs but per single variable.
                return [np.array([i], dtype=int) for i in range(n_vars)]
            groups = model_to_substructures(
                source, n_vars, mode=self.neighborhood,
                max_size=self.max_substructure_size,
            )
        # Cardinality-aware combination cap.
        card = np.asarray(cardinality).astype(int).reshape(n_vars)
        kept = []
        for s in groups:
            s = np.asarray(s, dtype=int)
            if len(s) == 0 or len(s) > self.max_substructure_size:
                continue
            if int(np.prod(card[s])) > self.max_combinations:
                continue
            kept.append(s)
        # Guarantee coverage of every variable (add singletons if needed).
        covered = set(int(v) for s in kept for v in s)
        for i in range(n_vars):
            if i not in covered:
                kept.append(np.array([i], dtype=int))
        return kept

    def optimize(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        fitness_func,
        cardinality: np.ndarray,
        **params: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resolve the substructures from the model / graph, then run the shared
        subset + evaluation-budget optimization loop of the base class."""
        n_vars = np.asarray(population).shape[1]
        self._substructures = self._resolve_substructures(n_vars, cardinality, params)
        return super().optimize(population, fitness, fitness_func, cardinality, **params)

    # ------------------------------------------------------------------
    # Per-solution substructural hill climb
    # ------------------------------------------------------------------
    def _optimize_one(
        self,
        x: np.ndarray,
        fx: float,
        evaluator: _BudgetEvaluator,
        cardinality: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, float]:
        best_x = x.copy()
        best_f = float(fx)
        card = np.asarray(cardinality).astype(int)
        substructures = self._substructures

        # Repeat sweeps over the substructures until a full sweep yields no
        # improvement or the budget is spent.
        improved = True
        while improved and not evaluator.exhausted:
            improved = False
            order = rng.permutation(len(substructures))
            for idx in order:
                if evaluator.exhausted:
                    break
                s = substructures[idx]
                current = tuple(int(best_x[v]) for v in s)
                move_combo = None
                move_f = best_f
                # Enumerate all joint value combinations of the substructure.
                for combo in product(*(range(int(card[v])) for v in s)):
                    if evaluator.exhausted:
                        break
                    if combo == current:
                        continue           # current values -> already best_f
                    cand = best_x.copy()
                    cand[s] = combo
                    f = evaluator(cand)
                    if f > move_f:
                        move_f, move_combo = f, combo
                        if self.acceptance == "first":
                            break          # first-improvement: take it now
                if move_combo is not None and move_f > best_f:
                    best_x[s] = move_combo
                    best_f = move_f
                    improved = True

        return best_x, best_f
