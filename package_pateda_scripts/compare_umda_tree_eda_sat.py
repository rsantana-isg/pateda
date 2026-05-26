"""
Comparison of UMDA, Tree-EDA, and Tree-EDA_r on SAT Instances

This script compares the performance of three EDA variants on a set of
random 3-SAT instances:

  - UMDA: Univariate Marginal Distribution Algorithm. Assumes variables are
    independent. Computes a marginal probability for each variable.

  - Tree-EDA: Learns a tree-structured probabilistic model using the maximum
    weight spanning tree over the full pairwise mutual information matrix.

  - Tree-EDA_r: Variant of Tree-EDA that restricts mutual information
    computation to pairs of variables that are known to interact according
    to the SAT clause structure (i.e., variables that appear together in at
    least one clause). The interaction matrix is computed via
    find_matrix_interactions_SAT().

Results reported per algorithm and SAT instance:
  - Best fitness (number of satisfied clauses) per run
  - Mean best fitness across independent runs
  - Mean number of generations until best solution found
  - Mean number of function evaluations

Usage::

    python scripts/compare_umda_tree_eda_sat.py

The script is self-contained and requires no arguments.

References
----------
- Santana, R., Mendiburu, A., and Lozano, J.A. (2012). "Structural transfer
  using EDAs: An application to multi-marker tagging SNP selection."
  Proceedings of the 2012 IEEE Congress on Evolutionary Computation.
- Santana, R. (2017). "Gray-box optimization and factorized distribution
  algorithms: where two worlds collide." arXiv:1707.03093.
"""

import sys
import os
from pathlib import Path

# Allow running directly from any directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
#sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import Dict, List, Optional, Tuple

from pateda.core.eda import EDA, EDAComponents
from pateda.learning import LearnUMDA, LearnTreeModel, LearnTreeModelR
from pateda.learning import find_matrix_interactions_SAT
from pateda.sampling import SampleFDA
from pateda.seeding import RandomInit
from pateda.selection import TruncationSelection
from pateda.replacement import ElitistReplacement
from pateda.stop_conditions import MaxGenerations
from pateda.functions.discrete.sat import load_random_3sat, SATInstance


# ─────────────────────────────────────────────────────────────────────────────
# Experiment configuration
# ─────────────────────────────────────────────────────────────────────────────

# SAT instance parameters
SAT_CONFIGS = [
    {"n_vars": 20, "n_clauses": 85,  "label": "20v-85c"},
    {"n_vars": 30, "n_clauses": 128, "label": "30v-128c"},
    {"n_vars": 50, "n_clauses": 213, "label": "50v-213c"},
]

# Number of distinct SAT instances per configuration
N_INSTANCES = 5

# Number of independent EDA runs per instance
N_RUNS = 1

# EDA parameters
POP_SIZE = 200
N_SELECTED = 20          # truncation selection keeps top 20 %
MAX_GENERATIONS = 50
N_ELITE = 5               # elitist replacement: keep best 5 individuals
RANDOM_SEED_BASE = 1000   # seeds are RANDOM_SEED_BASE + instance_idx * 100 + run_idx


# ─────────────────────────────────────────────────────────────────────────────
# Helper: fitness function factory
# ─────────────────────────────────────────────────────────────────────────────

def make_sat_fitness(sat_instance: SATInstance):
    """Return a scalar fitness function (maximise satisfied clauses)."""
    n_clauses = len(sat_instance.formulas[0])

    def fitness_func(solution: np.ndarray) -> float:
        result = sat_instance.evaluate(solution)
        return float(result[0])   # single objective: first formula

    return fitness_func


# ─────────────────────────────────────────────────────────────────────────────
# Single EDA run
# ─────────────────────────────────────────────────────────────────────────────

def run_eda(
    algorithm_name: str,
    sat_instance: SATInstance,
    n_vars: int,
    interaction_matrix: Optional[np.ndarray],
    seed: int,
) -> Dict:
    """
    Run one EDA on a SAT instance and return summary statistics.

    Parameters
    ----------
    algorithm_name : str
        One of 'UMDA', 'Tree-EDA', 'Tree-EDA_r'.
    sat_instance : SATInstance
        The SAT problem to solve.
    n_vars : int
        Number of Boolean variables.
    interaction_matrix : np.ndarray or None
        Binary interaction matrix (required for Tree-EDA_r; ignored otherwise).
    seed : int
        Random seed for this run.

    Returns
    -------
    dict with keys:
        - best_fitness: float (best number of satisfied clauses reached)
        - generation_found: int (generation at which best was found)
        - n_evals: int (total fitness evaluations)
        - fitness_curve: list of float (best fitness per generation)
    """
    cardinality = np.full(n_vars, 2)
    fitness_func = make_sat_fitness(sat_instance)

    # Select learner based on algorithm name
    if algorithm_name == "UMDA":
        learner = LearnUMDA(alpha=0.0)
    elif algorithm_name == "Tree-EDA":
        learner = LearnTreeModel(alpha=0.0)
    elif algorithm_name == "Tree-EDA_r":
        if interaction_matrix is None:
            raise ValueError("interaction_matrix required for Tree-EDA_r")
        learner = LearnTreeModelR(interaction_matrix=interaction_matrix, alpha=0.0)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    components = EDAComponents(
        seeding=RandomInit(),
        learning=learner,
        sampling=SampleFDA(n_samples=POP_SIZE),
        selection=TruncationSelection(n_select=N_SELECTED),
        replacement=ElitistReplacement(n_elite=N_ELITE),
        stop_condition=MaxGenerations(max_gen=MAX_GENERATIONS),
    )

    eda = EDA(
        pop_size=POP_SIZE,
        n_vars=n_vars,
        cardinality=cardinality,
        fitness_func=fitness_func,
        components=components,
        random_seed=seed,
    )

    stats, _ = eda.run(verbose=False)

    return {
        "best_fitness": stats.best_fitness_overall,
        "generation_found": stats.generation_found,
        "n_evals": len(stats.best_fitness) * POP_SIZE,
        "fitness_curve": stats.best_fitness,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Experiment loop
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(config: Dict, instance_idx: int) -> Dict:
    """
    Run UMDA, Tree-EDA, and Tree-EDA_r on one SAT instance with N_RUNS
    independent repetitions each.

    Returns a dict with per-algorithm aggregated statistics.
    """
    n_vars = config["n_vars"]
    n_clauses = config["n_clauses"]

    # Generate the SAT instance deterministically from instance index
    instance_seed = RANDOM_SEED_BASE + instance_idx * 1000
    sat_instance = load_random_3sat(
        n_vars=n_vars,
        n_clauses=n_clauses,
        n_objectives=1,
        seed=instance_seed,
    )

    # Build interaction matrix from SAT structure
    interaction_matrix = find_matrix_interactions_SAT(sat_instance)
    n_interacting_pairs = int(np.sum(np.triu(interaction_matrix, k=1)))
    n_total_pairs = n_vars * (n_vars - 1) // 2
    coverage = n_interacting_pairs / n_total_pairs * 100.0

    print(
        f"  Instance {instance_idx+1}: {n_vars}v, {n_clauses}c | "
        f"interaction pairs: {n_interacting_pairs}/{n_total_pairs} "
        f"({coverage:.1f}%)"
    )

    algorithms = ["UMDA", "Tree-EDA", "Tree-EDA_r"]
    results = {alg: [] for alg in algorithms}

    for run_idx in range(N_RUNS):
        run_seed = RANDOM_SEED_BASE + instance_idx * 1000 + run_idx + 1
        for alg in algorithms:
            r = run_eda(
                algorithm_name=alg,
                sat_instance=sat_instance,
                n_vars=n_vars,
                interaction_matrix=interaction_matrix if alg == "Tree-EDA_r" else None,
                seed=run_seed,
            )
            results[alg].append(r)

    # Aggregate results per algorithm
    aggregated = {}
    for alg in algorithms:
        runs = results[alg]
        best_fitness_list = [r["best_fitness"] for r in runs]
        gen_found_list = [r["generation_found"] for r in runs if r["generation_found"] is not None]
        n_evals_list = [r["n_evals"] for r in runs]

        aggregated[alg] = {
            "mean_best_fitness": float(np.mean(best_fitness_list)),
            "std_best_fitness": float(np.std(best_fitness_list)),
            "max_best_fitness": float(np.max(best_fitness_list)),
            "mean_generation_found": float(np.mean(gen_found_list)) if gen_found_list else float("nan"),
            "mean_n_evals": float(np.mean(n_evals_list)),
            "raw_best_fitness": best_fitness_list,
        }

    aggregated["n_clauses"] = n_clauses
    aggregated["n_vars"] = n_vars
    aggregated["n_interacting_pairs"] = n_interacting_pairs
    aggregated["n_total_pairs"] = n_total_pairs

    return aggregated


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_results_table(config_label: str, all_instance_results: List[Dict]) -> None:
    """Print a formatted results table for one configuration."""
    algorithms = ["UMDA", "Tree-EDA", "Tree-EDA_r"]

    print()
    print("=" * 78)
    print(f"  Configuration: {config_label}")
    print("=" * 78)

    # Header
    header = f"  {'Instance':<12}" + "".join(
        f"  {alg:<20}" for alg in algorithms
    )
    print(header)
    sub_header = f"  {'':12}" + "".join(
        f"  {'mean±std (best)':20}" for _ in algorithms
    )
    print(sub_header)
    print("-" * 78)

    # Per-instance rows
    for i, inst_res in enumerate(all_instance_results):
        n_clauses = inst_res["n_clauses"]
        row = f"  Instance {i+1:<4}"
        for alg in algorithms:
            res = inst_res[alg]
            mean = res["mean_best_fitness"]
            std = res["std_best_fitness"]
            best = res["max_best_fitness"]
            pct = mean / n_clauses * 100.0
            cell = f"{mean:.1f}±{std:.1f} ({pct:.1f}%)"
            row += f"  {cell:<20}"
        print(row)

    print("-" * 78)

    # Aggregate across instances
    agg_row = f"  {'MEAN across':12}"
    agg_row2 = f"  {'all instances':12}"
    for alg in algorithms:
        all_means = [inst[alg]["mean_best_fitness"] for inst in all_instance_results]
        all_clauses = [inst["n_clauses"] for inst in all_instance_results]
        grand_mean = np.mean(all_means)
        grand_pct = np.mean(
            [m / c * 100 for m, c in zip(all_means, all_clauses)]
        )
        cell = f"{grand_mean:.1f} ({grand_pct:.1f}%)"
        agg_row += f"  {cell:<20}"

    print(agg_row)
    print()

    # Generation and evals table
    print(f"  Mean generation of best solution found:")
    gen_header = f"  {'Instance':<12}" + "".join(
        f"  {alg:<20}" for alg in algorithms
    )
    print(gen_header)
    print("-" * 78)
    for i, inst_res in enumerate(all_instance_results):
        row = f"  Instance {i+1:<4}"
        for alg in algorithms:
            gen = inst_res[alg]["mean_generation_found"]
            evals = inst_res[alg]["mean_n_evals"]
            cell = f"gen {gen:.1f} / {evals:.0f} ev"
            row += f"  {cell:<20}"
        print(row)
    print()


def print_summary(all_configs_results: Dict) -> None:
    """Print a high-level summary comparing the three algorithms."""
    print("=" * 78)
    print("  SUMMARY: Tree-EDA_r vs UMDA and Tree-EDA")
    print("=" * 78)
    print()
    print(
        "  For each configuration, Tree-EDA_r restricts mutual-information\n"
        "  computation to pairs of variables that appear in the same clause.\n"
        "  This exploits the known SAT structure to guide model building.\n"
    )

    for config_label, all_instance_results in all_configs_results.items():
        n_vars = all_instance_results[0]["n_vars"]
        n_clauses = all_instance_results[0]["n_clauses"]
        n_pairs = all_instance_results[0]["n_total_pairs"]
        n_int = all_instance_results[0]["n_interacting_pairs"]

        print(f"  Config {config_label}  ({n_vars} vars, {n_clauses} clauses)")
        print(
            f"    Interaction pairs used by Tree-EDA_r: "
            f"{n_int}/{n_pairs} ({n_int/n_pairs*100:.1f}%)"
        )

        for alg in ["UMDA", "Tree-EDA", "Tree-EDA_r"]:
            grand_pct = np.mean(
                [
                    inst[alg]["mean_best_fitness"] / inst["n_clauses"] * 100
                    for inst in all_instance_results
                ]
            )
            print(f"    {alg:<14}: {grand_pct:.2f}% clauses satisfied (mean)")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("  UMDA vs Tree-EDA vs Tree-EDA_r on Random 3-SAT Instances")
    print("=" * 78)
    print(f"  Population size:      {POP_SIZE}")
    print(f"  Selected individuals: {N_SELECTED}")
    print(f"  Max generations:      {MAX_GENERATIONS}")
    print(f"  Elite individuals:    {N_ELITE}")
    print(f"  Runs per instance:    {N_RUNS}")
    print(f"  Instances per config: {N_INSTANCES}")
    print()

    all_configs_results = {}

    for config in SAT_CONFIGS:
        label = config["label"]
        print(f"Configuration: {label}")
        instance_results = []

        for inst_idx in range(N_INSTANCES):
            result = run_experiment(config, inst_idx)
            instance_results.append(result)

        all_configs_results[label] = instance_results
        print_results_table(label, instance_results)

    print_summary(all_configs_results)


if __name__ == "__main__":
    main()
