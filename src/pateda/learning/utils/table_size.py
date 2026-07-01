"""Utilities for constraining joint table sizes during model learning."""

from typing import Iterable, List
import numpy as np


def joint_table_size(cardinality: np.ndarray, variables: Iterable[int]) -> int:
    """Return the number of cells in the joint table of ``variables``."""
    vars_arr = np.asarray(list(variables), dtype=int)
    if len(vars_arr) == 0:
        return 1
    return int(np.prod(cardinality[vars_arr].astype(int)))


def split_variables_by_table_limit(
    variables: Iterable[int], cardinality: np.ndarray, max_table_size: int
) -> List[np.ndarray]:
    """
    Split variables into chunks whose joint table sizes do not exceed max_table_size.

    Variables with cardinality larger than max_table_size are kept as singleton
    chunks because no further decomposition is possible.
    """
    vars_arr = np.asarray(list(variables), dtype=int)
    if len(vars_arr) == 0:
        return []

    chunks: List[np.ndarray] = []
    current: List[int] = []
    current_size = 1

    for var in vars_arr:
        var_card = int(cardinality[int(var)])
        if len(current) == 0:
            current = [int(var)]
            current_size = var_card
            continue

        if current_size * var_card <= max_table_size:
            current.append(int(var))
            current_size *= var_card
            continue

        chunks.append(np.array(current, dtype=int))
        current = [int(var)]
        current_size = var_card

    if len(current) > 0:
        chunks.append(np.array(current, dtype=int))

    return chunks
