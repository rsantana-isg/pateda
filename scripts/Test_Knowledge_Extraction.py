"""
Test / demonstration of the knowledge-extraction tools in
``pateda.knowledge_extraction``.

The script runs several classes of EDAs on a problem with a *known* interaction
structure (the non-overlapping Deceptive3 function, whose variables interact in
consecutive blocks of three), saving the structural and parameter information of
the probabilistic model learned at every generation.  It then uses the
network-measures and visualization tools of ``pateda.knowledge_extraction`` to
analyse how the learned *structures* evolve, following:

  * Santana et al., "Network measures for information extraction in evolutionary
    algorithms", IJCIS 6(6), 2013.
  * Santana et al., "Mining probabilistic models learned by EDAs ...",
    GECCO-2009.

EDA classes exercised
---------------------
  * Factorization-based EDA  : MN-FDA            (FactorizedModel)
  * Bayesian-network EDAs    : EBNA, BOA, AffEDA (BayesianNetworkModel,
                               three *different* BN-learning strategies)
  * Tree-based EDA           : Tree-EDA          (tree-structured model)

Outputs (written to the results directory)
------------------------------------------
  * ``<EDA>_measures.csv``        : per-generation scalar network measures.
  * ``<EDA>_adjacencies.npz``     : per-generation adjacency matrices (structure).
  * ``<EDA>_*.png``               : evolution / frequency / degree / motif /
                                    snapshot / betweenness figures.
  * ``bn_learners_comparison.png``: comparison of the three BN-learning EDAs.
  * ``summary.csv``               : final-generation measures for every EDA.

Usage
-----
    python scripts/Test_Knowledge_Extraction.py
    python scripts/Test_Knowledge_Extraction.py --n-vars 18 --pop-size 700 --n-gen 15
    python scripts/Test_Knowledge_Extraction.py --quick
    python scripts/Test_Knowledge_Extraction.py --out-dir /tmp/ke_results
"""

import argparse
import csv
import os
import sys
import warnings

import numpy as np

from pateda import EBNA, BOA, AffEDA, TreeEDA, MNFDA
from pateda.core.components import CacheConfig
from pateda.functions.discrete.deceptive import deceptive3

from pateda.knowledge_extraction import (
    compute_measures_evolution,
    plot_measures_evolution,
    plot_edge_frequency_matrix,
    plot_degree_distribution,
    plot_motif_evolution,
    plot_network_snapshots,
    plot_betweenness_two_approaches,
    compare_measures_grid,
    SCALAR_MEASURE_KEYS,
)


# Measures highlighted in the figures / comparison (a representative subset of
# the descriptors defined in the Network-measures paper).
KEY_MEASURES = [
    "n_edges", "density", "clustering_mean", "characteristic_path_length",
    "max_modularity", "motif_number_z3", "max_clique_size", "dagdif",
]


def build_eda_configs():
    """Return the list of (label, family, class, extra_kwargs) to run."""
    return [
        # Factorization-based EDA.
        ("MN-FDA", "Factorization", MNFDA, dict(max_clique_size=3)),
        # Bayesian-network EDAs with three different structure learners.
        ("EBNA",   "Bayesian-net", EBNA,   dict(alpha=0.1)),
        ("BOA",    "Bayesian-net", BOA,    dict()),
        ("AffEDA", "Bayesian-net", AffEDA, dict(max_clique_size=4, alpha=0.1)),
        # Tree-based EDA.
        ("Tree-EDA", "Tree", TreeEDA, dict()),
    ]


def run_eda(label, eda_cls, extra_kwargs, n_vars, pop_size, n_gen, sel_ratio, seed):
    """Run one EDA caching the per-generation models; return (stats, models)."""
    alg = eda_cls(
        n_vars=n_vars,
        cardinality=2,
        fitness_func=deceptive3,
        pop_size=pop_size,
        n_gen=n_gen,
        selection_ratio=sel_ratio,
        elitism=True,
        random_seed=seed,
        **extra_kwargs,
    )
    # The plug-and-play wrappers delegate to an inner EDA; request model caching.
    stats, cache = alg._eda.run(
        cache_config=CacheConfig(cache_models=True), verbose=False
    )
    return stats, cache.models


def save_measures_csv(path, evolution):
    """Write the per-generation scalar measures to a CSV file."""
    series = evolution["series"]
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["generation"] + SCALAR_MEASURE_KEYS)
        for g in range(evolution["n_generations"]):
            writer.writerow([g] + [f"{series[k][g]:.6g}" for k in SCALAR_MEASURE_KEYS])


def save_adjacencies(path, evolution):
    """Save the per-generation adjacency matrices (structure)."""
    arrays = {f"gen_{g}": np.asarray(a)
              for g, a in enumerate(evolution["adjacencies"])}
    np.savez_compressed(path, **arrays)


def analyse_eda(label, models, n_vars, out_dir, quick):
    """Compute measures and render all figures for a single EDA run."""
    evolution = compute_measures_evolution(
        models, n_vars=n_vars, include_motifs_z4=not quick
    )
    series = evolution["series"]
    adjacencies = evolution["adjacencies"]

    # Persist structural + measure information.
    save_measures_csv(os.path.join(out_dir, f"{label}_measures.csv"), evolution)
    save_adjacencies(os.path.join(out_dir, f"{label}_adjacencies.npz"), evolution)

    # Figures.
    plot_measures_evolution(
        evolution, measures=KEY_MEASURES,
        save_path=os.path.join(out_dir, f"{label}_measures_evolution.png"),
        title=f"{label}: network-measure evolution",
    )
    plot_edge_frequency_matrix(
        adjacencies,
        save_path=os.path.join(out_dir, f"{label}_edge_frequency.png"),
        title=f"{label}: arc-frequency matrix",
    )
    plot_degree_distribution(
        adjacencies,
        save_path=os.path.join(out_dir, f"{label}_degree_distribution.png"),
        title=f"{label}: average degree distribution",
    )
    plot_motif_evolution(
        adjacencies,
        save_path=os.path.join(out_dir, f"{label}_motif_evolution.png"),
        title=f"{label}: triad (Z=3) motif evolution",
    )
    plot_network_snapshots(
        adjacencies,
        save_path=os.path.join(out_dir, f"{label}_network_snapshots.png"),
        title=f"{label}: learned network at selected generations",
    )
    plot_betweenness_two_approaches(
        evolution,
        save_path=os.path.join(out_dir, f"{label}_betweenness.png"),
        title=f"{label}: betweenness (vertex vs generation approach)",
    )

    # Console summary line.
    last = {k: series[k][-1] for k in KEY_MEASURES}
    print(f"  [{label:9s}] final: "
          + "  ".join(f"{k}={last[k]:.3g}" for k in KEY_MEASURES))
    return evolution


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Knowledge-extraction demo: network analysis of EDA models.")
    parser.add_argument("--n-vars", type=int, default=15,
                        help="number of variables (multiple of 3; default 15)")
    parser.add_argument("--pop-size", type=int, default=500, help="population size")
    parser.add_argument("--n-gen", type=int, default=12, help="number of generations")
    parser.add_argument("--sel-ratio", type=float, default=0.5, help="selection ratio")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--out-dir", type=str,
                        default=os.path.join("results", "knowledge_extraction"),
                        help="directory for the saved CSV / NPZ / PNG outputs")
    parser.add_argument("--quick", action="store_true",
                        help="fast preset (small problem, few generations)")
    args = parser.parse_args(argv)

    if args.quick:
        args.n_vars, args.pop_size, args.n_gen = 9, 150, 5

    if args.n_vars % 3 != 0:
        args.n_vars += 3 - (args.n_vars % 3)

    os.makedirs(args.out_dir, exist_ok=True)
    warnings.filterwarnings("ignore")  # silence benign numpy/nx divide warnings

    print("=" * 80)
    print("Knowledge extraction from EDA probabilistic models (Deceptive3)")
    print(f"  n_vars={args.n_vars} (blocks of 3)  pop_size={args.pop_size}  "
          f"n_gen={args.n_gen}  seed={args.seed}")
    print(f"  results -> {os.path.abspath(args.out_dir)}")
    print("=" * 80)

    evolutions = {}
    families = {}
    final_rows = []

    for label, family, cls, kwargs in build_eda_configs():
        try:
            print(f"\nRunning {family} EDA: {label} ...")
            stats, models = run_eda(
                label, cls, kwargs, args.n_vars, args.pop_size,
                args.n_gen, args.sel_ratio, args.seed,
            )
            print(f"  best fitness = {float(stats.best_fitness_overall):.4f}  "
                  f"(optimum = {args.n_vars // 3})   models cached = {len(models)}")
            evolution = analyse_eda(label, models, args.n_vars, args.out_dir, args.quick)
            evolutions[label] = evolution
            families[label] = family
            last = evolution["series"]
            final_rows.append(
                [label, family, f"{float(stats.best_fitness_overall):.4f}"]
                + [f"{last[k][-1]:.6g}" for k in SCALAR_MEASURE_KEYS]
            )
        except Exception as exc:  # keep going if one EDA fails
            import traceback
            print(f"  ERROR running {label}: {exc}")
            traceback.print_exc()

    # ---- Comparison of the Bayesian-network learning strategies ------------
    bn_labels = [l for l, fam in families.items() if fam == "Bayesian-net"]
    if len(bn_labels) >= 2:
        print(f"\nComparing BN-learning EDAs: {', '.join(bn_labels)}")
        bn_evolutions = {l: evolutions[l] for l in bn_labels}
        compare_measures_grid(
            bn_evolutions, measures=KEY_MEASURES,
            save_path=os.path.join(args.out_dir, "bn_learners_comparison.png"),
            title="Bayesian-network EDAs: different structure-learning strategies",
        )

    # ---- Global summary table ---------------------------------------------
    summary_path = os.path.join(args.out_dir, "summary.csv")
    with open(summary_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["EDA", "family", "best_fitness"] + SCALAR_MEASURE_KEYS)
        writer.writerows(final_rows)

    print("\n" + "=" * 80)
    print("Done.  Final-generation network measures written to:")
    print(f"  {os.path.abspath(summary_path)}")
    print("Figures and per-generation structures (CSV/NPZ) are in the same folder.")
    print("=" * 80)


if __name__ == "__main__":
    main()
