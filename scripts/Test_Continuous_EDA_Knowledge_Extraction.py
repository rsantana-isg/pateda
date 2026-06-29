"""
Knowledge extraction from the models learned by *continuous* EDAs.

This script analyses how the probabilistic models learned by continuous EDAs
evolve during the search, focusing on the structural information that can be
extracted from them:

  * **Gaussian EDAs** — the interaction network is read from the inverse
    covariance (precision) matrix of the learned multivariate Gaussian, the
    continuous analogue of the Bayesian network of discrete EDAs
    (``pateda.knowledge_extraction.gaussian_networks``).
  * **Vine-copula EDAs** — when the vine structure and/or the pair-copula
    families are learned during the search, the script extracts the first-tree
    interaction network, the pair-copula family composition, the Kendall's-tau
    statistics per tree and the effective truncation level
    (``pateda.knowledge_extraction.vine_analysis``).

Both kinds of network are then analysed with the generic network measures
(``pateda.knowledge_extraction.network_measures``) and **combined/compared** with
each other and with the known interaction structure of the problem.

Problem
-------
Negative Rosenbrock, whose terms couple *consecutive* variables, so the optimum
induces a chain interaction structure ``x_i — x_{i+1}`` that the learned models
should progressively reveal.

EDAs analysed
-------------
  * Gaussian EDA           (full covariance)            -> Gaussian network
  * Vine-copula EDA (auto) (R-vine + family selection)  -> full vine analysis
  * C-vine EDA             (C-vine, fixed family)        -> vine structure

References
----------
  * Sundaramoorthy et al., "Sparse Inverse Covariance Estimation for Causal
    Inference in Process Data Analytics", IEEE TCST 30(3), 2022.
  * Friedman, Hastie, Tibshirani, "Sparse inverse covariance estimation with the
    graphical lasso", Biostatistics 9(3), 2008.
  * Carrera, Santana, Lozano, "Vine copula classifiers for the mind reading
    problem", Progress in Artificial Intelligence 5, 2016.
  * Aas et al., "Pair-copula constructions of multiple dependence", 2009.
  * Santana et al., "Network measures for information extraction in evolutionary
    algorithms", IJCIS 6(6), 2013.

Usage
-----
    python scripts/Test_Continuous_EDA_Knowledge_Extraction.py
    python scripts/Test_Continuous_EDA_Knowledge_Extraction.py --n-vars 10 --pop-size 600 --n-gen 20
    python scripts/Test_Continuous_EDA_Knowledge_Extraction.py --quick
    python scripts/Test_Continuous_EDA_Knowledge_Extraction.py --out-dir /tmp/cke
"""

import argparse
import csv
import os
import warnings

import numpy as np

from pateda import GaussianEDA, VineEDA, CVineEDA
from pateda.core.components import CacheConfig
from pateda.functions.continuous.benchmarks import rosenbrock

import pateda.knowledge_extraction as ke
from pateda.knowledge_extraction.network_measures import (
    compute_measures_evolution, SCALAR_MEASURE_KEYS,
)

GAUSSIAN_MEASURES = ["n_edges", "density", "clustering_mean", "max_modularity",
                     "characteristic_path_length", "max_clique_size"]


def neg_rosenbrock(x):
    """Maximised objective (EDAs maximise fitness)."""
    return -rosenbrock(x)


def known_chain_structure(n_vars):
    """Known interaction graph of Rosenbrock: consecutive variables interact."""
    adj = np.zeros((n_vars, n_vars), dtype=int)
    for i in range(n_vars - 1):
        adj[i, i + 1] = adj[i + 1, i] = 1
    return adj


# ---------------------------------------------------------------------------
# Gaussian EDA analysis
# ---------------------------------------------------------------------------

def analyse_gaussian_eda(n_vars, pop_size, n_gen, sel_ratio, seed, out_dir, bounds):
    print("\n[Gaussian EDA] running ...")
    eda = GaussianEDA(n_vars=n_vars, bounds=bounds, fitness_func=neg_rosenbrock,
                      pop_size=pop_size, n_gen=n_gen, selection_ratio=sel_ratio,
                      random_seed=seed)
    stats, cache = eda._eda.run(cache_config=CacheConfig(cache_models=True), verbose=False)
    print(f"  best fitness = {float(stats.best_fitness_overall):.4f}   "
          f"models cached = {len(cache.models)}")

    # Gaussian interaction network from the precision matrix, per generation.
    gevo = ke.gaussian_network_evolution(cache.models, method="partial_correlation",
                                         threshold=0.2)
    measures = compute_measures_evolution(gevo["adjacencies"], n_vars=n_vars)

    # Figures.
    ke.plot_gaussian_parameter_evolution(
        gevo, save_path=os.path.join(out_dir, "Gaussian_parameter_evolution.png"),
        title="Gaussian EDA: variance & partial-correlation evolution")
    ke.plot_measures_evolution(
        measures, measures=GAUSSIAN_MEASURES,
        save_path=os.path.join(out_dir, "Gaussian_network_measures.png"),
        title="Gaussian EDA: network-measure evolution")
    ke.plot_precision_heatmap(
        gevo["partial_correlations"][-1],
        save_path=os.path.join(out_dir, "Gaussian_partial_correlation_final.png"),
        title="Gaussian EDA: final partial-correlation matrix")
    final_net = ke.gaussian_interaction_network(cache.models[-1],
                                                method="partial_correlation", threshold=0.2)
    ke.plot_partial_correlation_network(
        final_net, save_path=os.path.join(out_dir, "Gaussian_network_final.png"),
        title="Gaussian EDA: final interaction network")

    # Combine / compare with the known (Bayesian-network-like) chain structure.
    known = known_chain_structure(n_vars)
    ke.plot_network_comparison(
        final_net["adjacency"], known, labels=("Gaussian", "known chain"),
        save_path=os.path.join(out_dir, "Gaussian_vs_known.png"),
        title="Gaussian network vs known structure")
    cmp = ke.compare_networks(final_net["adjacency"], known)
    print(f"  Gaussian-vs-known: Jaccard={cmp['jaccard']:.2f}  "
          f"common={cmp['common_edges']}")
    # Causal orientation (undirected GGM -> directed graph), comparable to a BN.
    oriented = ke.orient_edges_likelihood_score(final_net["adjacency"],
                                                final_net["covariance"])

    _save_measures_csv(os.path.join(out_dir, "Gaussian_measures.csv"), measures)
    return {"label": "Gaussian-EDA", "best": float(stats.best_fitness_overall),
            "measures": measures, "final_adjacency": final_net["adjacency"],
            "oriented": oriented, "jaccard_known": cmp["jaccard"]}


# ---------------------------------------------------------------------------
# Vine copula EDA analysis
# ---------------------------------------------------------------------------

def analyse_vine_eda(label, eda, n_vars, out_dir, with_family_analysis):
    stats, cache = eda.run(verbose=False, cache_models=True)
    best = float(stats.best_fitness[-1]) if hasattr(stats, "best_fitness") else float("nan")
    print(f"  best fitness = {best:.4f}   models cached = {len(cache.models)}")

    vevo = ke.vine_evolution(cache.models)
    first_tree_adjs = vevo["first_tree_adjacencies"]
    measures = compute_measures_evolution(first_tree_adjs, n_vars=n_vars)

    ke.plot_vine_evolution(
        vevo, save_path=os.path.join(out_dir, f"{label}_vine_evolution.png"),
        title=f"{label}: vine structure / family / tau evolution")
    ke.plot_vine_first_tree(
        cache.models[-1], save_path=os.path.join(out_dir, f"{label}_first_tree_final.png"),
        title=f"{label}: final first-tree dependence network")
    ke.plot_tau_by_tree(
        cache.models[-1], save_path=os.path.join(out_dir, f"{label}_tau_by_tree.png"),
        title=f"{label}: mean |tau| per tree (final)")
    if with_family_analysis:
        ke.plot_family_composition(
            cache.models[-1],
            save_path=os.path.join(out_dir, f"{label}_family_composition.png"),
            title=f"{label}: pair-copula family composition (final)")
    ke.plot_measures_evolution(
        measures, measures=GAUSSIAN_MEASURES,
        save_path=os.path.join(out_dir, f"{label}_network_measures.png"),
        title=f"{label}: first-tree network-measure evolution")

    final_summary = ke.analyze_vine(cache.models[-1])
    print(f"  final: first-tree edges={final_summary['first_tree_network']['n_edges']}  "
          f"effective truncation={final_summary['effective_truncation']}  "
          f"families={final_summary['family_composition']['counts']}")

    _save_measures_csv(os.path.join(out_dir, f"{label}_measures.csv"), measures)
    return {"label": label, "best": best, "measures": measures,
            "final_adjacency": final_summary["first_tree_network"]["adjacency"]}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_measures_csv(path, evolution):
    series = evolution["series"]
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["generation"] + SCALAR_MEASURE_KEYS)
        for g in range(evolution["n_generations"]):
            writer.writerow([g] + [f"{series[k][g]:.6g}" for k in SCALAR_MEASURE_KEYS])


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Knowledge extraction from continuous-EDA models "
                    "(Gaussian networks and vine copulas).")
    parser.add_argument("--n-vars", type=int, default=8)
    parser.add_argument("--pop-size", type=int, default=400)
    parser.add_argument("--n-gen", type=int, default=12)
    parser.add_argument("--sel-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out-dir", type=str,
                        default=os.path.join("results", "continuous_knowledge_extraction"))
    parser.add_argument("--quick", action="store_true",
                        help="fast preset (small problem, few generations)")
    args = parser.parse_args(argv)

    if args.quick:
        args.n_vars, args.pop_size, args.n_gen = 6, 200, 5

    os.makedirs(args.out_dir, exist_ok=True)
    warnings.filterwarnings("ignore")
    bounds = (-2.0, 2.0)

    print("=" * 84)
    print("Knowledge extraction from continuous-EDA models (negative Rosenbrock)")
    print(f"  n_vars={args.n_vars}  pop_size={args.pop_size}  n_gen={args.n_gen}  "
          f"seed={args.seed}")
    print(f"  results -> {os.path.abspath(args.out_dir)}")
    print("=" * 84)

    results = []

    # --- Gaussian EDA ---
    try:
        results.append(analyse_gaussian_eda(
            args.n_vars, args.pop_size, args.n_gen, args.sel_ratio,
            args.seed, args.out_dir, bounds))
    except Exception as exc:
        import traceback; print(f"  ERROR (Gaussian): {exc}"); traceback.print_exc()

    # --- Vine-copula EDA (auto structure + family selection) ---
    try:
        print("\n[Vine-EDA (auto R-vine + family selection)] running ...")
        v = VineEDA(n_vars=args.n_vars, bounds=bounds, fitness_func=neg_rosenbrock,
                    pop_size=args.pop_size, n_gen=args.n_gen,
                    selection_ratio=args.sel_ratio, random_seed=args.seed)
        results.append(analyse_vine_eda("Vine-EDA", v, args.n_vars, args.out_dir,
                                        with_family_analysis=True))
    except Exception as exc:
        import traceback; print(f"  ERROR (Vine-EDA): {exc}"); traceback.print_exc()

    # --- C-vine EDA (structure learned, fixed Gaussian family) ---
    try:
        print("\n[C-vine EDA (C-vine structure, Gaussian family)] running ...")
        cv = CVineEDA(n_vars=args.n_vars, bounds=bounds, fitness_func=neg_rosenbrock,
                      pop_size=args.pop_size, n_gen=args.n_gen,
                      selection_ratio=args.sel_ratio, copula_family="gaussian",
                      random_seed=args.seed)
        results.append(analyse_vine_eda("CVine-EDA", cv, args.n_vars, args.out_dir,
                                        with_family_analysis=False))
    except Exception as exc:
        import traceback; print(f"  ERROR (C-vine): {exc}"); traceback.print_exc()

    # --- Cross-model combination: Gaussian vs Vine first-tree vs known ---
    try:
        gauss = next((r for r in results if r["label"] == "Gaussian-EDA"), None)
        vine = next((r for r in results if r["label"] == "Vine-EDA"), None)
        if gauss is not None and vine is not None:
            print("\n[Combining Gaussian and Vine networks]")
            ke.plot_network_comparison(
                gauss["final_adjacency"], vine["final_adjacency"],
                labels=("Gaussian", "Vine first-tree"),
                save_path=os.path.join(args.out_dir, "Gaussian_vs_Vine.png"),
                title="Gaussian network vs Vine first-tree network")
            cmp = ke.compare_networks(gauss["final_adjacency"], vine["final_adjacency"])
            known = known_chain_structure(args.n_vars)
            combined = ke.combine_networks(gauss["final_adjacency"],
                                           vine["final_adjacency"], mode="union")
            cmp_known = ke.compare_networks(combined, known)
            print(f"  Gaussian vs Vine: Jaccard={cmp['jaccard']:.2f} "
                  f"common={cmp['common_edges']}")
            print(f"  combined(union) vs known chain: Jaccard={cmp_known['jaccard']:.2f} "
                  f"common={cmp_known['common_edges']}")
    except Exception as exc:
        import traceback; print(f"  ERROR (combination): {exc}"); traceback.print_exc()

    # --- Summary CSV ---
    summary_path = os.path.join(args.out_dir, "summary.csv")
    with open(summary_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["EDA", "best_fitness"] + SCALAR_MEASURE_KEYS)
        for r in results:
            last = r["measures"]["series"]
            writer.writerow([r["label"], f"{r['best']:.4f}"]
                            + [f"{last[k][-1]:.6g}" for k in SCALAR_MEASURE_KEYS])

    print("\n" + "=" * 84)
    print(f"Done.  Summary -> {os.path.abspath(summary_path)}")
    print("Per-generation measures (CSV) and figures are in the same folder.")
    print("=" * 84)


if __name__ == "__main__":
    main()
