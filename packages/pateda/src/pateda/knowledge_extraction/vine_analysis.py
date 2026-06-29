"""
Analysis of the vine-copula models learned by continuous EDAs.

A *vine copula* factorizes an ``n``-dimensional dependence structure into a
sequence of ``n-1`` nested trees ``T_1, ..., T_{n-1}`` of bivariate
*pair-copulas* (Bedford & Cooke 2002; Aas et al. 2009).  The first tree ``T_1``
captures the strongest *unconditional* pairwise dependencies (the variable
order/structure is selected to maximize the Kendall's tau of its edges); the
higher trees capture *conditional* dependencies.  Each edge carries a
pair-copula *family* (Gaussian, Clayton, Gumbel, Frank, t, independence, ...)
and a *dependence parameter*, summarised by Kendall's tau.

This module extracts, **only when the structure and/or families are learned
during the search**, the information needed to analyse those models:

* the vine **structure** (per-tree edges with conditioned / conditioning sets);
* the **first-tree interaction network** (strongest unconditional dependencies),
  which can be combined with Gaussian- and Bayesian-network-derived networks;
* the pair-copula **family composition** (which dependence types are selected);
* the **Kendall's tau** statistics per tree (strength of dependence and how it
  decays with the tree level);
* the effective **truncation level** (model complexity);
* the evolution of all of the above across generations.

Backend: `pyvinecopulib` (the library used by ``pateda.learning.vine_copula``).

References
----------
* T. Bedford, R. M. Cooke, "Vines — a new graphical model for dependent random
  variables", Annals of Statistics, 30(4):1031-1068, 2002.
* K. Aas, C. Czado, A. Frigessi, H. Bakken, "Pair-copula constructions of
  multiple dependence", Insurance: Mathematics and Economics, 44(2):182-198,
  2009.
* D. Carrera, R. Santana, J. A. Lozano, "Vine copula classifiers for the mind
  reading problem", Progress in Artificial Intelligence, 5:289-305, 2016.
* (sand-dunes paper) "Detection of sand dunes on Mars using a regular
  vine-based classification approach", Knowledge-Based Systems, 2019.
* R. Santana et al., "Network measures for information extraction in
  evolutionary algorithms", IJCIS 6(6), 2013 (network analysis of the
  first-tree dependence graph).

Author: Roberto Santana (roberto.santana@ehu.eus)
"""

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Access the underlying pyvinecopulib model
# ---------------------------------------------------------------------------

def get_vine_model(model: Any):
    """Return the underlying ``pyvinecopulib.Vinecop`` from a learned model.

    Accepts the dict produced by ``pateda.learning.vine_copula`` (key
    ``'vine_model'``) or the ``Vinecop`` object directly.  Returns ``None`` if
    no vine model is present.
    """
    if model is None:
        return None
    if isinstance(model, dict):
        for key in ("vine_model", "vinecop", "model"):
            if key in model and model[key] is not None:
                return model[key]
        return None
    # An object that quacks like a Vinecop.
    if hasattr(model, "get_pair_copula") and hasattr(model, "structure"):
        return model
    if hasattr(model, "parameters") and isinstance(model.parameters, dict):
        return get_vine_model(model.parameters)
    return None


def _family_name(family) -> str:
    """Human-readable copula family name."""
    s = str(family)
    return s.split(".")[-1] if "." in s else s


# ---------------------------------------------------------------------------
# Structure extraction
# ---------------------------------------------------------------------------

def vine_structure(model: Any) -> Dict[str, Any]:
    """
    Extract the full structure of a learned vine copula.

    Decodes, for every (tree, edge), the *conditioned* variable pair, the
    *conditioning* set, the pair-copula family, Kendall's tau, the dependence
    parameters and the rotation.

    Returns
    -------
    dict with keys
        ``n_vars``, ``order``, ``trunc_lvl``, ``n_trees``, and ``edges`` — a list
        of dicts ``{tree, edge, conditioned, conditioning, family, tau,
        parameters, rotation}`` (1-based variable indices, matching the model).
    """
    vine = get_vine_model(model)
    if vine is None:
        raise ValueError("No vine-copula model found in the supplied object")

    d = int(vine.dim)
    order = list(vine.order)
    structure = vine.structure
    trunc = int(vine.trunc_lvl)
    families = vine.families
    taus = vine.taus

    edges: List[Dict[str, Any]] = []
    for t in range(trunc):
        for e in range(d - 1 - t):
            # Conditioned pair = (order[e], struct_array(t, e)); conditioning set
            # = {struct_array(t', e) : t' < t}.  (Verified against Vinecop.str().)
            a = int(order[e])
            b = int(structure.struct_array(t, e))
            conditioned = tuple(sorted((a, b)))
            conditioning = tuple(sorted(int(structure.struct_array(tp, e))
                                        for tp in range(t)))
            try:
                bicop = vine.get_pair_copula(t, e)
                params = np.asarray(bicop.parameters, dtype=float).ravel().tolist()
                rotation = int(bicop.rotation)
            except Exception:
                params, rotation = [], 0
            edges.append({
                "tree": t + 1,
                "edge": e + 1,
                "conditioned": conditioned,
                "conditioning": conditioning,
                "family": _family_name(families[t][e]),
                "tau": float(taus[t][e]),
                "parameters": params,
                "rotation": rotation,
            })

    return {
        "n_vars": d,
        "order": order,
        "trunc_lvl": trunc,
        "n_trees": d - 1,
        "edges": edges,
    }


def first_tree_network(model: Any, tau_threshold: float = 0.0) -> Dict[str, Any]:
    """
    Build the interaction network of the first vine tree ``T_1``.

    ``T_1`` holds the *unconditional* pairwise dependencies the vine considers
    most important; its edges (optionally filtered by ``|tau| >= tau_threshold``,
    and excluding independence copulas) form an undirected interaction network
    directly comparable with the Gaussian and Bayesian-network networks.

    Returns ``adjacency`` (0/1), ``weights`` (signed Kendall's tau), the
    ``edge_list`` and ``directed = False``.
    """
    structure = vine_structure(model)
    n = structure["n_vars"]
    adjacency = np.zeros((n, n), dtype=int)
    weights = np.zeros((n, n))
    edge_list = []
    for edge in structure["edges"]:
        if edge["tree"] != 1:
            continue
        if edge["family"].lower() in ("independence", "indep"):
            continue
        if abs(edge["tau"]) < tau_threshold:
            continue
        i, j = edge["conditioned"]
        i0, j0 = i - 1, j - 1  # 0-based for the adjacency matrix
        adjacency[i0, j0] = adjacency[j0, i0] = 1
        weights[i0, j0] = weights[j0, i0] = edge["tau"]
        edge_list.append((i0, j0, edge["tau"], edge["family"]))
    return {
        "adjacency": adjacency,
        "weights": weights,
        "edge_list": edge_list,
        "n_edges": len(edge_list),
        "directed": False,
    }


# ---------------------------------------------------------------------------
# Family and dependence statistics
# ---------------------------------------------------------------------------

def family_composition(model: Any, include_independence: bool = True) -> Dict[str, Any]:
    """
    Count the pair-copula families used in the learned vine.

    Returns the overall family counts/frequencies and the per-tree counts.  The
    family composition reveals which *types* of dependence (e.g. tail-dependent
    Clayton/Gumbel vs. symmetric Gaussian) the EDA has selected — meaningful
    only when families are selected during the search.
    """
    structure = vine_structure(model)
    overall = Counter()
    per_tree: Dict[int, Counter] = {}
    for edge in structure["edges"]:
        fam = edge["family"]
        if not include_independence and fam.lower() in ("independence", "indep"):
            continue
        overall[fam] += 1
        per_tree.setdefault(edge["tree"], Counter())[fam] += 1
    total = sum(overall.values())
    frequencies = {fam: c / total for fam, c in overall.items()} if total else {}
    return {
        "counts": dict(overall),
        "frequencies": frequencies,
        "per_tree": {t: dict(c) for t, c in per_tree.items()},
        "n_non_independence": sum(c for f, c in overall.items()
                                  if f.lower() not in ("independence", "indep")),
        "total_pair_copulas": total,
    }


def tau_by_tree(model: Any) -> Dict[str, Any]:
    """
    Kendall's tau statistics per vine tree.

    Returns, for each tree, the list of ``|tau|`` values and their mean/max, plus
    the overall mean ``|tau|``.  In a well-fitted vine the dependence strength
    typically *decreases* with the tree level (higher trees model weaker,
    conditional dependencies).
    """
    structure = vine_structure(model)
    by_tree: Dict[int, List[float]] = {}
    for edge in structure["edges"]:
        by_tree.setdefault(edge["tree"], []).append(abs(edge["tau"]))
    trees = sorted(by_tree)
    mean_abs_tau = {t: float(np.mean(by_tree[t])) for t in trees}
    max_abs_tau = {t: float(np.max(by_tree[t])) for t in trees}
    all_abs = [v for t in trees for v in by_tree[t]]
    return {
        "abs_tau_by_tree": {t: by_tree[t] for t in trees},
        "mean_abs_tau_by_tree": mean_abs_tau,
        "max_abs_tau_by_tree": max_abs_tau,
        "overall_mean_abs_tau": float(np.mean(all_abs)) if all_abs else 0.0,
    }


def effective_truncation(model: Any, tau_threshold: float = 1e-6) -> int:
    """
    Effective truncation level: the index of the last tree containing a
    non-independence pair-copula (a measure of model complexity; Carrera et al.
    2016, vine truncation).
    """
    structure = vine_structure(model)
    last = 0
    for edge in structure["edges"]:
        non_indep = edge["family"].lower() not in ("independence", "indep")
        if non_indep and abs(edge["tau"]) >= tau_threshold:
            last = max(last, edge["tree"])
    return last


def tau_matrix(model: Any) -> np.ndarray:
    """Symmetric matrix of the *unconditional* (first-tree) Kendall's taus."""
    net = first_tree_network(model)
    return net["weights"]


# ---------------------------------------------------------------------------
# Aggregate per-model summary and evolution
# ---------------------------------------------------------------------------

def analyze_vine(model: Any) -> Dict[str, Any]:
    """One-call summary of a learned vine: structure, first-tree network,
    family composition, tau-by-tree and effective truncation."""
    structure = vine_structure(model)
    return {
        "structure": structure,
        "first_tree_network": first_tree_network(model),
        "family_composition": family_composition(model),
        "tau_by_tree": tau_by_tree(model),
        "effective_truncation": effective_truncation(model),
        "n_vars": structure["n_vars"],
        "trunc_lvl": structure["trunc_lvl"],
    }


def vine_evolution(models: Sequence[Any]) -> Dict[str, Any]:
    """
    Analyse a sequence of learned vines (one per generation).

    Returns
    -------
    dict with keys
        ``first_tree_adjacencies`` : list of 0/1 matrices (feed to
        ``network_measures.compute_measures_evolution``).
        ``per_generation`` : list of ``analyze_vine`` summaries.
        ``series`` : per-generation arrays for ``first_tree_edges``,
        ``effective_truncation``, ``overall_mean_abs_tau`` and
        ``n_non_independence``.
        ``family_frequency_series`` : dict ``family -> per-generation frequency``.
    """
    first_tree_adjacencies = []
    per_generation = []
    first_tree_edges, eff_trunc, mean_tau, n_non_indep = [], [], [], []
    all_families = set()

    for model in models:
        try:
            summary = analyze_vine(model)
            per_generation.append(summary)
            first_tree_adjacencies.append(summary["first_tree_network"]["adjacency"])
            first_tree_edges.append(summary["first_tree_network"]["n_edges"])
            eff_trunc.append(summary["effective_truncation"])
            mean_tau.append(summary["tau_by_tree"]["overall_mean_abs_tau"])
            n_non_indep.append(summary["family_composition"]["n_non_independence"])
            all_families.update(summary["family_composition"]["frequencies"])
        except Exception:
            per_generation.append(None)
            first_tree_adjacencies.append(np.zeros((0, 0), dtype=int))
            first_tree_edges.append(0)
            eff_trunc.append(0)
            mean_tau.append(0.0)
            n_non_indep.append(0)

    family_frequency_series: Dict[str, List[float]] = {f: [] for f in all_families}
    for summary in per_generation:
        freqs = summary["family_composition"]["frequencies"] if summary else {}
        for fam in all_families:
            family_frequency_series[fam].append(freqs.get(fam, 0.0))

    return {
        "first_tree_adjacencies": first_tree_adjacencies,
        "per_generation": per_generation,
        "series": {
            "first_tree_edges": np.array(first_tree_edges),
            "effective_truncation": np.array(eff_trunc),
            "overall_mean_abs_tau": np.array(mean_tau),
            "n_non_independence": np.array(n_non_indep),
        },
        "family_frequency_series": {f: np.array(v)
                                    for f, v in family_frequency_series.items()},
        "n_generations": len(models),
    }
