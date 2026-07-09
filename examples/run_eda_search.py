"""
Run EDA Search - Benchmark Function Exploration with File Output
================================================================

This program runs an EDA algorithm on a benchmark function for a given number
of repetitions and produces two output files:

1. ``<func>_<n>_structure.dat``   – Problem structure as a list of edges of
   the symmetric interaction matrix.  The first line contains the number of
   edges; each subsequent line is ``i j`` with ``i < j`` (0-based indices).

2. ``<func>_<n>_<eda>_samples.dat`` – The ``SAMP`` best solutions found
   across all repetitions, sorted by fitness (descending).  Each line is::

       <fitness>  <x_0> <x_1> ... <x_{n-1}>

Usage
-----
::

    python run_eda_search.py <obj_func> <n> <eda> <pop_size> <n_gen> \\
                              <n_reps> <SAMP> [seed]

Arguments
---------
obj_func : str
    Objective function name.  Supported values:

    OneMax, KDeceptive<k> (e.g. KDeceptive3), Deceptive3,
    Deceptive3Overlap, HIFF, FHTrap1, HardDecep5,
    Polytree3, Polytree3Overlap, Polytree5,
    FC2, FC3, FC4, FC5,
    DecepMarta3, DecepMarta3New, Decep3MH,
    TwoPeaksDecep3, DecepVenturini
n : int
    Number of binary variables.
eda : str
    EDA name.  Supported: UMDA, TreeEDA, EBNA, MOA, MN-FDA, MN-FDAG,
    MK-EDA1, MK-EDA2, MK-EDA3, MT-EDA2, MT-EDA3
pop_size : int
    Population size.
n_gen : int
    Maximum number of generations per run.
n_reps : int
    Number of independent repetitions.
SAMP : int
    Number of best solutions to store in the samples file.
seed : int (optional)
    Base random seed (default 0).  Each repetition uses seed + rep_index.

Examples
--------
::

    python run_eda_search.py KDeceptive3 30 UMDA 200 100 10 20
    python run_eda_search.py HIFF 32 TreeEDA 300 150 5 10 42
"""

import math
import sys
from typing import List, Sequence, Tuple

import numpy as np

from pateda.core.eda import EDA, EDAComponents
from pateda.learning.ebna import LearnEBNA
from pateda.learning.fda import LearnFDA
from pateda.learning.markov import LearnMarkovChain
from pateda.learning.mixture_trees import LearnMixtureTrees
from pateda.learning.mnfda import LearnMNFDA
from pateda.learning.mnfdag import LearnMNFDAG
from pateda.learning.moa import LearnMOA
from pateda.learning.tree import LearnTreeModel
from pateda.learning.umda import LearnUMDA
from pateda.replacement import ElitistReplacement
from pateda.sampling.bayesian_network import SampleBayesianNetwork
from pateda.sampling.fda import SampleFDA
from pateda.sampling.gibbs import SampleGibbs
from pateda.sampling.markov import SampleMarkovChain
from pateda.sampling.mixture_trees import SampleMixtureTrees
from pateda.seeding import RandomInit
from pateda.selection import TruncationSelection
from pateda.stop_conditions import MaxGenerations

from pateda.functions.discrete_binary.toy_functions.additive_decomposable import (
    k_deceptive,
    decep3,
    decep_marta3,
    decep_marta3_new,
    decep3_mh,
    two_peaks_decep3,
    decep_venturini,
    hard_decep5,
    hiff,
    fhtrap1,
    first_polytree3_ochoa,
    first_polytree5_ochoa,
    fc2,
    fc3,
    fc4,
    fc5,
)
from pateda.learning.interaction_learning import (
    find_matrix_interactions_additive_decomposable,
)


# ---------------------------------------------------------------------------
# Problem registry
# ---------------------------------------------------------------------------

def _subfunctions_k_deceptive(k: int, n_vars: int) -> List[List[int]]:
    return [list(range(i, min(i + k, n_vars))) for i in range(0, n_vars, k)]


def _subfunctions_decep3_overlap(n_vars: int) -> List[List[int]]:
    step = 2
    return [list(range(i, i + 3)) for i in range(0, n_vars - 2, step)]


def _subfunctions_hiff(n_vars: int) -> List[List[int]]:
    """Hierarchical If-and-only-If subfunction variable sets at every level."""
    subfs: List[List[int]] = []
    level_size = 2
    while level_size <= n_vars:
        for start in range(0, n_vars, level_size):
            block = list(range(start, min(start + level_size, n_vars)))
            if len(block) > 1:
                subfs.append(block)
        level_size *= 2
    return subfs


def _subfunctions_fhtrap1(n_vars: int) -> List[List[int]]:
    """FHTrap1 (hierarchical trap, power-of-3 tree) subfunction variable sets."""
    subfs: List[List[int]] = []
    # Level 1: triplets of adjacent variables
    for start in range(0, n_vars, 3):
        block = list(range(start, min(start + 3, n_vars)))
        if len(block) == 3:
            subfs.append(block)
    # Higher levels: 3^2=9, 3^3=27, ... (power-of-3 hierarchy)
    level_size = 3 ** 2
    while level_size <= n_vars:
        for start in range(0, n_vars, level_size):
            block = list(range(start, min(start + level_size, n_vars)))
            if len(block) > 1:
                subfs.append(block)
        level_size *= 3
    return subfs


def get_function_subfunction_vars(obj_func: str, n_vars: int) -> List[List[int]]:
    """
    Return the list of subfunction variable-index sets for *obj_func*.

    Each element is a list of 0-based variable indices that appear in the
    same subfunction (and therefore interact with each other).

    For functions with no variable interactions (e.g. OneMax) an empty
    list is returned.
    """
    if obj_func == "OneMax":
        return []

    if obj_func.startswith("KDeceptive"):
        k = int(obj_func[len("KDeceptive"):])
        return _subfunctions_k_deceptive(k, n_vars)

    if obj_func == "Deceptive3":
        return _subfunctions_k_deceptive(3, n_vars)

    if obj_func == "Deceptive3Overlap":
        return _subfunctions_decep3_overlap(n_vars)

    if obj_func == "HIFF":
        return _subfunctions_hiff(n_vars)

    if obj_func == "FHTrap1":
        return _subfunctions_fhtrap1(n_vars)

    if obj_func == "HardDecep5":
        return _subfunctions_k_deceptive(5, n_vars)

    if obj_func in ("Polytree3", "Polytree3Overlap"):
        step = 1 if obj_func == "Polytree3Overlap" else 3
        return [list(range(i, min(i + 3, n_vars)))
                for i in range(0, n_vars - 2, step)]

    if obj_func == "Polytree5":
        return _subfunctions_k_deceptive(5, n_vars)

    if obj_func in ("FC2", "FC3", "FC4", "FC5"):
        return _subfunctions_k_deceptive(5, n_vars)

    if obj_func in ("DecepMarta3", "DecepMarta3New", "Decep3MH"):
        return _subfunctions_k_deceptive(3, n_vars)

    if obj_func == "TwoPeaksDecep3":
        return _subfunctions_k_deceptive(3, n_vars)

    if obj_func == "DecepVenturini":
        return _subfunctions_k_deceptive(3, n_vars)

    raise ValueError(f"Unknown objective function: '{obj_func}'")


def get_interaction_matrix(obj_func: str, n_vars: int) -> np.ndarray:
    """Return the ``n_vars × n_vars`` symmetric binary interaction matrix."""
    subfs = get_function_subfunction_vars(obj_func, n_vars)
    if not subfs:
        return np.zeros((n_vars, n_vars), dtype=int)
    return find_matrix_interactions_additive_decomposable(subfs, n_vars)


# ---------------------------------------------------------------------------
# Fitness-function registry
# ---------------------------------------------------------------------------

def _wrap(func):
    """Wrap a single-individual fitness function for EDA (accepts 1-D vector)."""
    def wrapped(x: np.ndarray):
        if x.ndim == 1:
            return float(func(x))
        return np.array([float(func(ind)) for ind in x])
    return wrapped


def parse_problem(obj_func: str, n: int):
    """
    Return ``(fitness_func, n_vars, optimal_fitness)`` for *obj_func*.

    The returned *fitness_func* accepts a 1-D numpy array (single individual)
    and returns a scalar float.
    """
    n_vars = n

    if obj_func == "OneMax":
        return _wrap(lambda x: float(np.sum(x))), n_vars, float(n_vars)

    if obj_func.startswith("KDeceptive"):
        k = int(obj_func[len("KDeceptive"):])
        if n_vars % k != 0:
            raise ValueError(f"n must be a multiple of {k} for {obj_func}")
        return _wrap(lambda x, _k=k: k_deceptive(x, k=_k)), n_vars, float(n_vars)

    if obj_func == "Deceptive3":
        if n_vars % 3 != 0:
            raise ValueError("n must be a multiple of 3 for Deceptive3")
        # Optimal per block is 1.0 (all-ones); there are n_vars // 3 blocks.
        return _wrap(lambda x: decep3(x, overlap=False)), n_vars, float(n_vars // 3)

    if obj_func == "Deceptive3Overlap":
        return _wrap(lambda x: decep3(x, overlap=True)), n_vars, None

    if obj_func == "HIFF":
        if n_vars <= 0 or (n_vars & (n_vars - 1)) != 0:
            raise ValueError("n must be a power of 2 for HIFF")
        opt = float(n_vars * (1 + int(math.log2(n_vars))))
        return _wrap(hiff), n_vars, opt

    if obj_func == "FHTrap1":
        return _wrap(fhtrap1), n_vars, float(n_vars)

    if obj_func == "HardDecep5":
        if n_vars % 5 != 0:
            raise ValueError("n must be a multiple of 5 for HardDecep5")
        return _wrap(hard_decep5), n_vars, float(n_vars)

    if obj_func == "Polytree3":
        return _wrap(lambda x: first_polytree3_ochoa(x, overlap=False)), n_vars, None

    if obj_func == "Polytree3Overlap":
        return _wrap(lambda x: first_polytree3_ochoa(x, overlap=True)), n_vars, None

    if obj_func == "Polytree5":
        return _wrap(first_polytree5_ochoa), n_vars, None

    if obj_func == "FC2":
        return _wrap(fc2), n_vars, None

    if obj_func == "FC3":
        return _wrap(fc3), n_vars, None

    if obj_func == "FC4":
        return _wrap(fc4), n_vars, None

    if obj_func == "FC5":
        return _wrap(fc5), n_vars, None

    if obj_func == "DecepMarta3":
        return _wrap(decep_marta3), n_vars, None

    if obj_func == "DecepMarta3New":
        return _wrap(decep_marta3_new), n_vars, None

    if obj_func == "Decep3MH":
        return _wrap(decep3_mh), n_vars, None

    if obj_func == "TwoPeaksDecep3":
        return _wrap(two_peaks_decep3), n_vars, None

    if obj_func == "DecepVenturini":
        return _wrap(decep_venturini), n_vars, None

    raise ValueError(f"Unknown objective function: '{obj_func}'")


# ---------------------------------------------------------------------------
# EDA builder
# ---------------------------------------------------------------------------

def build_eda(
    alg: str,
    fitness_func,
    n_vars: int,
    pop_size: int,
    max_generations: int,
    random_seed: int = None,
) -> EDA:
    """Assemble and return a configured :class:`~pateda.core.eda.EDA` instance."""
    cardinality = np.full(n_vars, 2)

    if alg == "UMDA":
        learning = LearnUMDA(alpha=1.0)
        sampling = SampleFDA(n_samples=pop_size)

    elif alg == "TreeEDA":
        learning = LearnTreeModel(alpha=0.1)
        sampling = SampleFDA(n_samples=pop_size)

    elif alg == "EBNA":
        learning = LearnEBNA(max_parents=3, score_metric="bic")
        sampling = SampleBayesianNetwork(n_samples=pop_size)

    elif alg == "MOA":
        learning = LearnMOA(k_neighbors=5, threshold_factor=1.5)
        sampling = SampleGibbs(n_samples=pop_size, IT=4, temperature=1.0)

    elif alg == "MN-FDA":
        learning = LearnMNFDA(max_clique_size=3, threshold=0.05, return_factorized=True)
        sampling = SampleFDA(n_samples=pop_size)

    elif alg == "MN-FDAG":
        learning = LearnMNFDAG(max_clique_size=5, alpha=0.01, return_factorized=True)
        sampling = SampleFDA(n_samples=pop_size)

    elif alg == "MK-EDA1":
        learning = LearnMarkovChain(k=1, alpha=0.1)
        sampling = SampleMarkovChain(n_samples=pop_size)

    elif alg == "MK-EDA2":
        learning = LearnMarkovChain(k=2, alpha=0.1)
        sampling = SampleMarkovChain(n_samples=pop_size)

    elif alg == "MK-EDA3":
        learning = LearnMarkovChain(k=3, alpha=0.1)
        sampling = SampleMarkovChain(n_samples=pop_size)

    elif alg == "MT-EDA2":
        learning = LearnMixtureTrees(
            n_components=2, component_learning="tree", alpha=0.1,
            weight_learning="uniform", random_seed=random_seed,
        )
        sampling = SampleMixtureTrees(n_samples=pop_size)

    elif alg == "MT-EDA3":
        learning = LearnMixtureTrees(
            n_components=3, component_learning="tree", alpha=0.1,
            weight_learning="uniform", random_seed=random_seed,
        )
        sampling = SampleMixtureTrees(n_samples=pop_size)

    else:
        raise ValueError(
            f"Unknown EDA: '{alg}'.  Supported: UMDA, TreeEDA, EBNA, MOA, "
            "MN-FDA, MN-FDAG, MK-EDA1, MK-EDA2, MK-EDA3, MT-EDA2, MT-EDA3"
        )

    components = EDAComponents(
        seeding=RandomInit(),
        selection=TruncationSelection(ratio=0.5),
        learning=learning,
        sampling=sampling,
        replacement=ElitistReplacement(),
        stop_condition=MaxGenerations(max_gen=max_generations),
    )

    return EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        fitness_func=fitness_func,
        cardinality=cardinality,
        components=components,
        random_seed=random_seed,
    )


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def save_structure_file(interaction_matrix: np.ndarray, filepath: str) -> None:
    """
    Write the problem structure to *filepath*.

    Format::

        <n_edges>
        <i> <j>
        ...

    where only the upper triangle (``i < j``) is listed and indices are
    0-based.
    """
    n_vars = interaction_matrix.shape[0]
    edges: List[Tuple[int, int]] = []
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if interaction_matrix[i, j]:
                edges.append((i, j))

    with open(filepath, "w") as fh:
        fh.write(f"{len(edges)}\n")
        for i, j in edges:
            fh.write(f"{i} {j}\n")


def save_samples_file(
    solutions: List[np.ndarray],
    fitnesses: List[float],
    filepath: str,
) -> None:
    """
    Write sampled solutions to *filepath*.

    Each line has the format::

        <fitness>  <x_0> <x_1> ... <x_{n-1}>

    Solutions are written in descending order of fitness.
    """
    pairs = sorted(zip(fitnesses, solutions), key=lambda p: p[0], reverse=True)
    with open(filepath, "w") as fh:
        for fit, sol in pairs:
            var_str = " ".join(str(int(v)) for v in sol)
            fh.write(f"{fit} {var_str}\n")


# ---------------------------------------------------------------------------
# Main search loop
# ---------------------------------------------------------------------------

def run_eda_search(
    obj_func: str,
    n_vars: int,
    eda_name: str,
    pop_size: int,
    n_gen: int,
    n_reps: int,
    samp: int,
    base_seed: int = 0,
    verbose: bool = True,
    output_dir: str = ".",
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Run *eda_name* on *obj_func* for *n_reps* repetitions and write the
    two result files.

    Parameters
    ----------
    obj_func : str
        Objective function name (see module docstring for supported names).
    n_vars : int
        Number of binary variables.
    eda_name : str
        EDA algorithm name.
    pop_size : int
        Population size.
    n_gen : int
        Maximum generations per run.
    n_reps : int
        Number of independent repetitions.
    samp : int
        Number of best solutions to store in the samples file.
    base_seed : int
        Base random seed; repetition *r* uses seed ``base_seed + r``.
    verbose : bool
        Print per-repetition progress.
    output_dir : str
        Directory in which the output files are written (default: current dir).

    Returns
    -------
    best_solutions : list of np.ndarray
        The *samp* best solutions found (or fewer if fewer were encountered).
    best_fitnesses : list of float
        Corresponding fitness values.
    """
    import os

    fitness_func, _, _ = parse_problem(obj_func, n_vars)

    # ── Structure file ────────────────────────────────────────────────────
    interaction_matrix = get_interaction_matrix(obj_func, n_vars)
    struct_filename = os.path.join(output_dir, f"{obj_func}_{n_vars}_structure.dat")
    save_structure_file(interaction_matrix, struct_filename)
    if verbose:
        n_edges = int(np.sum(interaction_matrix)) // 2
        print(f"Structure saved to '{struct_filename}'  ({n_edges} edges)")

    # ── Run EDA multiple times ────────────────────────────────────────────
    all_solutions: List[np.ndarray] = []
    all_fitnesses: List[float] = []

    for rep in range(n_reps):
        seed = base_seed + rep
        eda = build_eda(eda_name, fitness_func, n_vars, pop_size, n_gen, seed)
        stats, _ = eda.run(verbose=False)

        best_sol = stats.best_individual
        best_fit = float(stats.best_fitness_overall)

        if verbose:
            print(f"  Rep {rep + 1}/{n_reps}: best fitness = {best_fit:.6f}")

        if best_sol is not None:
            all_solutions.append(best_sol.copy())
            all_fitnesses.append(best_fit)

    # Keep the *samp* best solutions across all repetitions
    if all_fitnesses:
        order = np.argsort(all_fitnesses)[::-1]
        top_n = min(samp, len(order))
        top_solutions = [all_solutions[i] for i in order[:top_n]]
        top_fitnesses = [all_fitnesses[i] for i in order[:top_n]]
    else:
        top_solutions = []
        top_fitnesses = []

    # ── Samples file ──────────────────────────────────────────────────────
    samples_filename = os.path.join(output_dir, f"{obj_func}_{n_vars}_{eda_name}_samples.dat")
    save_samples_file(top_solutions, top_fitnesses, samples_filename)
    if verbose:
        print(f"Samples saved to  '{samples_filename}'  "
              f"({len(top_solutions)} solutions)")

    return top_solutions, top_fitnesses


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Command-line entry point."""
    usage = (
        "Usage: python run_eda_search.py <obj_func> <n> <eda> <pop_size> "
        "<n_gen> <n_reps> <SAMP> [seed]\n"
        "\n"
        "Supported obj_func values:\n"
        "  OneMax, KDeceptive3, KDeceptive5, Deceptive3, Deceptive3Overlap,\n"
        "  HIFF, FHTrap1, HardDecep5, Polytree3, Polytree3Overlap, Polytree5,\n"
        "  FC2, FC3, FC4, FC5, DecepMarta3, DecepMarta3New, Decep3MH,\n"
        "  TwoPeaksDecep3, DecepVenturini\n"
        "\n"
        "Supported eda values:\n"
        "  UMDA, TreeEDA, EBNA, MOA, MN-FDA, MN-FDAG,\n"
        "  MK-EDA1, MK-EDA2, MK-EDA3, MT-EDA2, MT-EDA3\n"
        "\n"
        "Example:\n"
        "  python run_eda_search.py KDeceptive3 30 UMDA 200 100 10 20\n"
    )

    if len(sys.argv) < 8:
        print(usage)
        sys.exit(1)

    obj_func = sys.argv[1]
    n_vars = int(sys.argv[2])
    eda_name = sys.argv[3]
    pop_size = int(sys.argv[4])
    n_gen = int(sys.argv[5])
    n_reps = int(sys.argv[6])
    samp = int(sys.argv[7])
    base_seed = int(sys.argv[8]) if len(sys.argv) > 8 else 0

    print("=" * 70)
    print("EDA Search")
    print("=" * 70)
    print(f"  Function   : {obj_func} (n={n_vars})")
    print(f"  EDA        : {eda_name}")
    print(f"  Pop size   : {pop_size}")
    print(f"  Generations: {n_gen}")
    print(f"  Repetitions: {n_reps}")
    print(f"  SAMP       : {samp}")
    print(f"  Base seed  : {base_seed}")
    print("=" * 70)

    try:
        solutions, fitnesses = run_eda_search(
            obj_func=obj_func,
            n_vars=n_vars,
            eda_name=eda_name,
            pop_size=pop_size,
            n_gen=n_gen,
            n_reps=n_reps,
            samp=samp,
            base_seed=base_seed,
            verbose=True,
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    if fitnesses:
        print(f"\nBest fitness across all runs: {max(fitnesses):.6f}")
        print(f"Mean best fitness           : {np.mean(fitnesses):.6f}")
    print("Done.")


if __name__ == "__main__":
    main()
