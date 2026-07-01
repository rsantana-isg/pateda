"""Permutation-based optimization problems.

Classical permutation problems used to benchmark permutation EDAs:
Traveling Salesman (TSP), Quadratic Assignment (QAP) and Linear Ordering (LOP).
The Permutation Flowshop Scheduling Problem (PFSP) is planned (see ROADMAP).
"""

from perm_pateda.functions.tsp import (
    TSP,
    create_random_tsp,
    create_tsp_from_coordinates,
)
from perm_pateda.functions.qap import (
    QAP,
    create_random_qap,
    create_uniform_qap,
    create_grid_qap,
    load_qaplib_instance,
)
from perm_pateda.functions.lop import (
    LOP,
    create_random_lop,
    create_tournament_lop,
    create_triangular_lop,
    create_sparse_lop,
    load_lolib_instance,
    feedback_arc_set_to_lop,
)

from perm_pateda.functions.pfsp import(
    PFSP,
    create_random_pfsp,
    load_taillard_instance,

)

from perm_pateda.functions.mis import(
    MIS,
    create_random_mis,
    create_mis_from_edges,
)

from perm_pateda.functions.maxcut import(
    MaxCut,
    create_random_max_cut,
    create_max_cut_from_edges,
)

from perm_pateda.functions.mvc import(
    MVC,
    create_random_mvc,
    create_mvc_from_edges,

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
    #PFSP
    "PFSP",
    "create_random_pfsp",
    "load_taillard_instance",
    #MIS
    "MIS",
    "create_random_mis",
    "create_mis_from_edges",
    #MaxCut
    "MaxCut",
    "create_random_max_cut",
    "create_max_cut_from_edges",
    #MVC
    "MVC",
    "create_random_mvc",
    "create_mvc_from_edges",


]
