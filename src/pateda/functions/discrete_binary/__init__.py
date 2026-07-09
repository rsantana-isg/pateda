"""Discrete binary optimization test functions.

Split into two groups:

* :mod:`~pateda.functions.discrete_binary.toy_functions` -- pseudo-boolean
  academic benchmarks (OneMax, trap/deceptive families, contiguous blocks, ...).
* :mod:`~pateda.functions.discrete_binary.problems` -- combinatorial
  optimization problems (NK, Ising, SAT, UBQP, binary graph problems, ...).
* :mod:`~pateda.functions.discrete_binary.multiobjective` -- multi-objective
  binary benchmarks (mo-onemax/zeromax, mo-deceptive, mUBQP).
"""

from pateda.functions.discrete_binary import toy_functions
from pateda.functions.discrete_binary import problems
from pateda.functions.discrete_binary import multiobjective
from pateda.functions.discrete_binary.toy_functions import *  # noqa: F401,F403
from pateda.functions.discrete_binary.problems import *  # noqa: F401,F403
from pateda.functions.discrete_binary.multiobjective import *  # noqa: F401,F403

from pateda.functions.discrete_binary.toy_functions import __all__ as _toy_all
from pateda.functions.discrete_binary.problems import __all__ as _prob_all
from pateda.functions.discrete_binary.multiobjective import __all__ as _mo_all

__all__ = (["toy_functions", "problems", "multiobjective"]
           + list(_toy_all) + list(_prob_all) + list(_mo_all))
