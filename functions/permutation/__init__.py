"""Permutation-based optimization problems"""

from pateda.functions.permutation.tsp import (
    TSP,
    create_random_tsp,
    create_tsp_from_coordinates,
)
from pateda.functions.permutation.qap import (
    QAP,
    create_random_qap,
    create_uniform_qap,
    create_grid_qap,
    load_qaplib_instance,
)
from pateda.functions.permutation.lop import (
    LOP,
    create_random_lop,
    create_tournament_lop,
    create_triangular_lop,
    create_sparse_lop,
    load_lolib_instance,
    feedback_arc_set_to_lop,
)

__all__ = [
    # TSP
    "TSP",
    "create_random_tsp",
    "create_tsp_from_coordinates",
    # QAP
    "QAP",
    "create_random_qap",
    "create_uniform_qap",
    "create_grid_qap",
    "load_qaplib_instance",
    # LOP
    "LOP",
    "create_random_lop",
    "create_tournament_lop",
    "create_triangular_lop",
    "create_sparse_lop",
    "load_lolib_instance",
    "feedback_arc_set_to_lop",
]
