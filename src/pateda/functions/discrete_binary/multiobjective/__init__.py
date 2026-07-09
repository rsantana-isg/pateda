"""Discrete binary multi-objective benchmark problems.

Multi-objective (vector-valued) functions over a binary representation. Each
callable returns an objective *vector* to be maximised, matching pateda's
``maximize=True`` convention and the MOEA-D / Pareto / indicator drivers in
:mod:`pateda.multiobjective`.

Simple pseudo-boolean MO functions:

* ``mo_onemax_zeromax`` -- diversity-only front (``f1 + f2 = n``).
* ``make_mo_deceptive`` -- two conflicting deceptive (trap) objectives.
* ``make_mubqp`` -- multi-objective UBQP with tunable objective correlation.

Configurable instance families (models + objective functions, with generator
scripts in :mod:`pateda.functions.instance_generators`):

* ``mnm`` -- truncated-Walsh / Markov-network model (CAEPIA 2015).
* ``mnk_landscape`` -- multi-objective NK landscapes (possibly heterogeneous).
* ``mubqp`` -- multi-objective UBQP, including the hard-instance construction
  from order-5 building blocks ("On the Design of Hard mUBQP Instances").
"""

from pateda.functions.discrete_binary.multiobjective.pseudoboolean import (
    mo_onemax_zeromax,
    make_mo_onemax_zeromax,
    make_mo_deceptive,
    make_mubqp,
    mo_pareto_front_onemax_zeromax,
)
from pateda.functions.discrete_binary.multiobjective.mnm import (
    MNMModel,
    MNMInstance,
    create_mnm_objective_function,
    generate_mnm,
)
from pateda.functions.discrete_binary.multiobjective.mnk_landscape import (
    NKObjective,
    MNKLandscape,
    create_random_nk_neighbourhoods,
    create_mnk_objective_function,
    generate_mnk,
)
from pateda.functions.discrete_binary.multiobjective.mubqp import (
    MUBQPInstance,
    create_mubqp_objective_function,
    generate_mubqp,
    create_artificial_mubqp,
    enumerate_order5_chunks,
    chunk_deception,
    chunk_pair_metrics,
    chunk_pair_hardness,
    select_hard_chunk_pairs,
    create_mubqp_from_chunk,
    create_heavy_mubqp_from_chunks,
)

__all__ = [
    # simple pseudo-boolean MO
    "mo_onemax_zeromax",
    "make_mo_onemax_zeromax",
    "make_mo_deceptive",
    "make_mubqp",
    "mo_pareto_front_onemax_zeromax",
    # mNM model
    "MNMModel",
    "MNMInstance",
    "create_mnm_objective_function",
    "generate_mnm",
    # MNK landscape
    "NKObjective",
    "MNKLandscape",
    "create_random_nk_neighbourhoods",
    "create_mnk_objective_function",
    "generate_mnk",
    # mUBQP (incl. hard instances)
    "MUBQPInstance",
    "create_mubqp_objective_function",
    "generate_mubqp",
    "create_artificial_mubqp",
    "enumerate_order5_chunks",
    "chunk_deception",
    "chunk_pair_metrics",
    "chunk_pair_hardness",
    "select_hard_chunk_pairs",
    "create_mubqp_from_chunk",
    "create_heavy_mubqp_from_chunks",
]
