"""Discrete non-binary optimization test functions.

Split into three groups:

* :mod:`~pateda.functions.discrete_non_binary.toy_functions` -- integer /
  categorical benchmark functions.
* :mod:`~pateda.functions.discrete_non_binary.problems` -- combinatorial
  problems with non-binary encodings (HP protein, graph coloring, clique
  covering, braid quantum-gate approximation).
* :mod:`~pateda.functions.discrete_non_binary.multiobjective` -- multi-objective
  non-binary benchmarks (bi-objective braid: error vs. length).
"""

from pateda.functions.discrete_non_binary import toy_functions
from pateda.functions.discrete_non_binary import problems
from pateda.functions.discrete_non_binary import multiobjective
from pateda.functions.discrete_non_binary.toy_functions import *  # noqa: F401,F403
from pateda.functions.discrete_non_binary.problems import *  # noqa: F401,F403
from pateda.functions.discrete_non_binary.multiobjective import *  # noqa: F401,F403

from pateda.functions.discrete_non_binary.toy_functions import __all__ as _toy_all
from pateda.functions.discrete_non_binary.problems import __all__ as _prob_all
from pateda.functions.discrete_non_binary.multiobjective import __all__ as _mo_all

__all__ = (["toy_functions", "problems", "multiobjective"]
           + list(_toy_all) + list(_prob_all) + list(_mo_all))
