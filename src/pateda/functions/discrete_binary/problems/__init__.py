"""Discrete binary combinatorial optimization problems.

Problems with real-world combinatorial structure, encoded with a binary vector
and (where applicable) loaded from packaged instance files: NK landscapes, the
Ising spin-glass model, MAX-SAT, UBQP, the binary graph problems (max-cut,
maximum clique, maximum independent set, dominating set), and the number
problems from Baluja & Davies (1997) (equal products, linear equations).
"""

from pateda.functions.discrete_binary.problems.nk_landscape import (
    NKLandscape,
    create_nk_objective_function,
    create_circular_nk_structure,
    create_random_nk_tables,
    evaluate_nk_landscape,
)
from pateda.functions.discrete_binary.problems.ising import (
    load_ising,
    eval_ising,
    create_ising_objective_function,
    load_ising_benchmark_instance,
    build_ising_interaction_matrix,
)
from pateda.functions.discrete_binary.problems.sat import (
    SATInstance,
    evaluate_sat,
    load_random_3sat,
    make_random_formulas,
    make_var_dep_formulas,
    load_sat_from_file,
    load_sat_benchmark_instance,
    build_sat_interaction_matrix,
)
from pateda.functions.discrete_binary.problems.ubqp import (
    UBQPInstance,
    evaluate_ubqp,
    load_ubqp_instance,
    load_ubqp_benchmark_instance,
    generate_random_ubqp,
    save_ubqp_instance,
    create_max_cut_ubqp,
    create_set_packing_ubqp,
    build_ubqp_interaction_matrix,
)
from pateda.functions.discrete_binary.problems.max_cut import (
    MaxCutInstance,
    eval_max_cut,
    create_max_cut_objective_function,
)
from pateda.functions.discrete_binary.problems.max_clique import (
    MaxCliqueInstance,
    eval_max_clique,
    create_max_clique_objective_function,
)
from pateda.functions.discrete_binary.problems.max_independent_set import (
    MaxIndependentSetInstance,
    eval_max_independent_set,
    create_max_independent_set_objective_function,
)
from pateda.functions.discrete_binary.problems.dominating_set import (
    DominatingSetInstance,
    eval_dominating_set,
    create_dominating_set_objective_function,
)
from pateda.functions.discrete_binary.problems.equal_products import (
    EqualProductsInstance,
    generate_random_equal_products,
    eval_equal_products,
    create_equal_products_objective_function,
)
from pateda.functions.discrete_binary.problems.linear_equations import (
    LinearEquationsInstance,
    generate_random_linear_equations,
    decode_binary_to_integers,
    eval_linear_equations_real,
    eval_linear_equations,
    create_linear_equations_objective_function,
)

__all__ = [
    "NKLandscape",
    "create_nk_objective_function",
    "create_circular_nk_structure",
    "create_random_nk_tables",
    "evaluate_nk_landscape",
    "load_ising",
    "eval_ising",
    "create_ising_objective_function",
    "load_ising_benchmark_instance",
    "build_ising_interaction_matrix",
    "SATInstance",
    "evaluate_sat",
    "load_random_3sat",
    "make_random_formulas",
    "make_var_dep_formulas",
    "load_sat_from_file",
    "load_sat_benchmark_instance",
    "build_sat_interaction_matrix",
    "UBQPInstance",
    "evaluate_ubqp",
    "load_ubqp_instance",
    "load_ubqp_benchmark_instance",
    "generate_random_ubqp",
    "save_ubqp_instance",
    "create_max_cut_ubqp",
    "create_set_packing_ubqp",
    "build_ubqp_interaction_matrix",
    "MaxCutInstance",
    "eval_max_cut",
    "create_max_cut_objective_function",
    "MaxCliqueInstance",
    "eval_max_clique",
    "create_max_clique_objective_function",
    "MaxIndependentSetInstance",
    "eval_max_independent_set",
    "create_max_independent_set_objective_function",
    "DominatingSetInstance",
    "eval_dominating_set",
    "create_dominating_set_objective_function",
    "EqualProductsInstance",
    "generate_random_equal_products",
    "eval_equal_products",
    "create_equal_products_objective_function",
    "LinearEquationsInstance",
    "generate_random_linear_equations",
    "decode_binary_to_integers",
    "eval_linear_equations_real",
    "eval_linear_equations",
    "create_linear_equations_objective_function",
]
