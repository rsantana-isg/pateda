"""Test functions for optimization"""

from pateda.functions.discrete.onemax import onemax
from pateda.functions.discrete.deceptive import deceptive3
from pateda.functions.discrete.contiguous_block import (
    contiguous_block,
    contiguous_block_with_penalty,
    create_contiguous_block_function,
)
from typing import Callable


def get_function(name: str) -> Callable:
    """
    Get function by name

    Args:
        name: Function name

    Returns:
        Function callable

    Raises:
        ValueError: If function name is unknown
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
    "onemax",
    "deceptive3",
    "contiguous_block",
    "contiguous_block_with_penalty",
    "create_contiguous_block_function",
    "get_function",
]
