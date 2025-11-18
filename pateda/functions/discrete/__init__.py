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

__all__ = [
    "onemax",
    "deceptive3",
    "load_ising",
    "eval_ising",
    "create_ising_objective_function",
    "create_fibonacci_hp_sequence",
    "eval_chain",
    "evaluate_hp_energy",
    "create_hp_objective_function"
]
