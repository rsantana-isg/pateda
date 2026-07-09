"""Continuous optimization test functions.

Split into two groups:

* :mod:`~pateda.functions.continuous.toy_functions` -- real-valued benchmark
  functions (sphere, Rastrigin, ...).
* :mod:`~pateda.functions.continuous.problems` -- continuous problems (AB
  off-lattice protein model).
"""

from pateda.functions.continuous import toy_functions
from pateda.functions.continuous import problems
from pateda.functions.continuous.toy_functions import *  # noqa: F401,F403
from pateda.functions.continuous.problems import *  # noqa: F401,F403

from pateda.functions.continuous.toy_functions import __all__ as _toy_all
from pateda.functions.continuous.problems import __all__ as _prob_all

__all__ = ["toy_functions", "problems"] + list(_toy_all) + list(_prob_all)
