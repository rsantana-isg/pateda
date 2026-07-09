"""Discrete non-binary combinatorial optimization problems.

Problems whose natural encoding uses integer / categorical variables with
cardinality greater than two: the HP lattice protein folding model and the
graph problems whose variables are colors / partition indices (graph coloring,
clique covering).
"""

from pateda.functions.discrete_non_binary.problems.hp_protein import (
    create_fibonacci_hp_sequence,
    eval_chain,
    evaluate_hp_energy,
    create_hp_objective_function,
)
from pateda.functions.discrete_non_binary.problems.graph_coloring import (
    GraphColoringInstance,
    eval_graph_coloring,
    create_graph_coloring_objective_function,
)
from pateda.functions.discrete_non_binary.problems.clique_covering import (
    CliqueCoveringInstance,
    eval_clique_covering,
    create_clique_covering_objective_function,
)
from pateda.functions.discrete_non_binary.problems.braid import (
    TAU,
    TAU_ICO,
    fibonacci_anyon_generators,
    su2_from_axis_angle,
    icosahedral_group,
    default_inverse_index,
    braid_matrix,
    braid_error,
    effective_length,
    braid_fitness,
    BraidProblem,
    make_fibonacci_braid_problem,
    make_icosahedral_benchmark_problem,
    load_icosahedral_targets,
    load_anyon_generators,
    save_braid_instances,
    create_braid_objective_function,
)

__all__ = [
    "create_fibonacci_hp_sequence",
    "eval_chain",
    "evaluate_hp_energy",
    "create_hp_objective_function",
    "GraphColoringInstance",
    "eval_graph_coloring",
    "create_graph_coloring_objective_function",
    "CliqueCoveringInstance",
    "eval_clique_covering",
    "create_clique_covering_objective_function",
    # Braid quantum-gate approximation
    "TAU",
    "TAU_ICO",
    "fibonacci_anyon_generators",
    "su2_from_axis_angle",
    "icosahedral_group",
    "default_inverse_index",
    "braid_matrix",
    "braid_error",
    "effective_length",
    "braid_fitness",
    "BraidProblem",
    "make_fibonacci_braid_problem",
    "make_icosahedral_benchmark_problem",
    "load_icosahedral_targets",
    "load_anyon_generators",
    "save_braid_instances",
    "create_braid_objective_function",
]
