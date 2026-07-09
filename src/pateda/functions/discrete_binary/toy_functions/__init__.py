"""Discrete binary toy functions.

Classic pseudo-boolean benchmark functions evaluated directly on a binary
string: OneMax, deceptive/trap families, additively decomposable functions,
contiguous-block functions, and the bit-pattern benchmarks from Baluja &
Davies (1997) (four/six/continuous peaks, plateau, checkerboard, tree-max,
summation cancellation).
"""

from pateda.functions.discrete_binary.toy_functions.onemax import onemax
from pateda.functions.discrete_binary.toy_functions.deceptive import deceptive3
from pateda.functions.discrete_binary.toy_functions.trap import (
    trap_n,
    trap_partition,
    create_trap_objective_function,
)
from pateda.functions.discrete_binary.toy_functions.contiguous_block import (
    contiguous_block,
    contiguous_block_with_penalty,
    create_contiguous_block_function,
)
from pateda.functions.discrete_binary.toy_functions.additive_decomposable import (
    k_deceptive,
    gen_k_decep,
    gen_k_decep_overlap,
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
    create_k_deceptive_function,
    create_hiff_function,
    create_decep3_function,
    create_polytree3_function,
)
from pateda.functions.discrete_binary.toy_functions.tree_max import (
    TreeMaxInstance,
    generate_random_tree_max,
    eval_tree_max,
    create_tree_max_objective_function,
)
from pateda.functions.discrete_binary.toy_functions.four_peaks import (
    head,
    tail,
    four_peaks,
    create_four_peaks_objective_function,
)
from pateda.functions.discrete_binary.toy_functions.six_peaks import (
    six_peaks,
    create_six_peaks_objective_function,
)
from pateda.functions.discrete_binary.toy_functions.continuous_peaks import (
    max_run,
    continuous_peaks,
    create_continuous_peaks_objective_function,
)
from pateda.functions.discrete_binary.toy_functions.plateau import (
    plateau,
    create_plateau_objective_function,
)
from pateda.functions.discrete_binary.toy_functions.checkerboard import (
    checkerboard,
    create_checkerboard_objective_function,
)
from pateda.functions.discrete_binary.toy_functions.summation_cancellation import (
    decode_binary_to_real,
    summation_cancellation_real,
    summation_cancellation,
    create_summation_cancellation_objective_function,
)

__all__ = [
    "onemax",
    "deceptive3",
    "trap_n",
    "trap_partition",
    "create_trap_objective_function",
    "contiguous_block",
    "contiguous_block_with_penalty",
    "create_contiguous_block_function",
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
    "TreeMaxInstance",
    "generate_random_tree_max",
    "eval_tree_max",
    "create_tree_max_objective_function",
    "head",
    "tail",
    "four_peaks",
    "create_four_peaks_objective_function",
    "six_peaks",
    "create_six_peaks_objective_function",
    "max_run",
    "continuous_peaks",
    "create_continuous_peaks_objective_function",
    "plateau",
    "create_plateau_objective_function",
    "checkerboard",
    "create_checkerboard_objective_function",
    "decode_binary_to_real",
    "summation_cancellation_real",
    "summation_cancellation",
    "create_summation_cancellation_objective_function",
]
