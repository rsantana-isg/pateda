"""Discrete optimization test functions"""

from pateda.functions.discrete.onemax import onemax
from pateda.functions.discrete.deceptive import deceptive3
from pateda.functions.discrete.ising import (
    load_ising,
    eval_ising,
    create_ising_objective_function
)
from pateda.functions.discrete.hp_protein import (
    create_fibonacci_hp_sequence,
    eval_chain,
    evaluate_hp_energy,
    create_hp_objective_function
)
from pateda.functions.discrete.trap import (
    trap_n,
    trap_partition,
    create_trap_objective_function
)
from pateda.functions.discrete.nk_landscape import (
    NKLandscape,
    create_nk_objective_function,
    create_circular_nk_structure,
    create_random_nk_tables,
    evaluate_nk_landscape
)
from pateda.functions.discrete.sat import (
    SATInstance,
    evaluate_sat,
    load_random_3sat,
    make_random_formulas,
    make_var_dep_formulas,
    load_sat_from_file
)
from pateda.functions.discrete.ubqp import (
    UBQPInstance,
    evaluate_ubqp,
    load_ubqp_instance,
    generate_random_ubqp,
    save_ubqp_instance,
    create_max_cut_ubqp,
    create_set_packing_ubqp
)
from pateda.functions.discrete.additive_decomposable import (
    # K-Deceptive functions
    k_deceptive,
    gen_k_decep,
    gen_k_decep_overlap,
    # Deceptive-3 variants
    decep3,
    decep_marta3,
    decep_marta3_new,
    decep3_mh,
    two_peaks_decep3,
    decep_venturini,
    # Hard deceptive-5
    hard_decep5,
    # Hierarchical functions
    hiff,
    fhtrap1,
    # Polytree functions
    first_polytree3_ochoa,
    first_polytree5_ochoa,
    # Cuban functions
    fc2,
    fc3,
    fc4,
    fc5,
    # Factory functions
    create_k_deceptive_function,
    create_hiff_function,
    create_decep3_function,
    create_polytree3_function,
)

__all__ = [
    "onemax",
    "deceptive3",
    "load_ising",
    "eval_ising",
    "create_ising_objective_function",
    "create_fibonacci_hp_sequence",
    "eval_chain",
    "evaluate_hp_energy",
    "create_hp_objective_function",
    "trap_n",
    "trap_partition",
    "create_trap_objective_function",
    "NKLandscape",
    "create_nk_objective_function",
    "create_circular_nk_structure",
    "create_random_nk_tables",
    "evaluate_nk_landscape",
    "SATInstance",
    "evaluate_sat",
    "load_random_3sat",
    "make_random_formulas",
    "make_var_dep_formulas",
    "load_sat_from_file",
    "UBQPInstance",
    "evaluate_ubqp",
    "load_ubqp_instance",
    "generate_random_ubqp",
    "save_ubqp_instance",
    "create_max_cut_ubqp",
    "create_set_packing_ubqp",
    # Additive decomposable functions
    "k_deceptive",
    "gen_k_decep",
    "gen_k_decep_overlap",
    "decep3",
    "decep_marta3",
    "decep_marta3_new",
    "decep3_mh",
    "two_peaks_decep3",
    "decep_venturini",
    "hard_decep5",
    "hiff",
    "fhtrap1",
    "first_polytree3_ochoa",
    "first_polytree5_ochoa",
    "fc2",
    "fc3",
    "fc4",
    "fc5",
    "create_k_deceptive_function",
    "create_hiff_function",
    "create_decep3_function",
    "create_polytree3_function",
]
