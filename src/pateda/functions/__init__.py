"""Test functions for optimization.

Functions are organized by representation and by kind:

``discrete_binary/``      binary-string functions
    ``toy_functions/``    pseudo-boolean academic benchmarks
    ``problems/``         binary combinatorial optimization problems
``discrete_non_binary/``  integer / categorical functions
    ``toy_functions/``    integer benchmark functions
    ``problems/``         non-binary combinatorial problems
``continuous/``           real-valued functions
    ``toy_functions/``    real-valued benchmark functions
    ``problems/``         continuous problems

``graph_utils`` holds shared readers for graph instance files and is used by
both the binary graph problems and the non-binary graph problems.
"""

from typing import Callable

from pateda.functions import discrete_binary
from pateda.functions import discrete_non_binary
from pateda.functions import continuous
from pateda.functions import graph_utils

# Commonly used callables re-exported at the top level for convenience.
from pateda.functions.discrete_binary.toy_functions.onemax import onemax
from pateda.functions.discrete_binary.toy_functions.deceptive import deceptive3
from pateda.functions.discrete_binary.toy_functions.contiguous_block import (
    contiguous_block,
    contiguous_block_with_penalty,
    create_contiguous_block_function,
)


def get_function(name: str) -> Callable:
    """
    Get a simple scalar test function by name.

    Args:
        name: Function name (case-insensitive). MATEDA aliases are supported.

    Returns:
        Function callable.

    Raises:
        ValueError: If function name is unknown.
    """
    functions = {
        "onemax": onemax,
        "sum": onemax,  # MATEDA alias
        "deceptive3": deceptive3,
        "evalfuncdec3": deceptive3,  # MATEDA alias
        "contiguous_block": contiguous_block,
    }

    if name.lower() in functions:
        return functions[name.lower()]
    else:
        raise ValueError(f"Unknown function: {name}")


__all__ = [
    "discrete_binary",
    "discrete_non_binary",
    "continuous",
    "graph_utils",
    "onemax",
    "deceptive3",
    "contiguous_block",
    "contiguous_block_with_penalty",
    "create_contiguous_block_function",
    "get_function",
]
