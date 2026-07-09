"""Discrete non-binary toy functions.

Benchmark functions defined over integer / categorical (cardinality > 2)
representations: the integer-representation function family used to test
non-binary EDAs.
"""

from pateda.functions.discrete_non_binary.toy_functions.integer_functions import (
    integer_onemax,
    integer_leading_blocks,
    integer_max_blocks,
    integer_categorical_match,
    integer_plateau_search,
    integer_dependency_chain,
    integer_multi_level_trap,
    integer_parity_blocks,
    integer_hierarchical_blocks,
    IntegerNKLandscape,
    create_integer_onemax_function,
    create_integer_max_blocks_function,
    create_integer_multi_level_trap_function,
    create_integer_dependency_chain_function,
    create_integer_categorical_match_function,
    create_integer_hierarchical_function,
    create_integer_parity_function,
    create_integer_nk_objective_function,
)

__all__ = [
    "integer_onemax",
    "integer_leading_blocks",
    "integer_max_blocks",
    "integer_categorical_match",
    "integer_plateau_search",
    "integer_dependency_chain",
    "integer_multi_level_trap",
    "integer_parity_blocks",
    "integer_hierarchical_blocks",
    "IntegerNKLandscape",
    "create_integer_onemax_function",
    "create_integer_max_blocks_function",
    "create_integer_multi_level_trap_function",
    "create_integer_dependency_chain_function",
    "create_integer_categorical_match_function",
    "create_integer_hierarchical_function",
    "create_integer_parity_function",
    "create_integer_nk_objective_function",
]
