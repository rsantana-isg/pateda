"""
Multi-objective optimisation toolkit for pateda.

This subpackage provides the model-agnostic building blocks that support the
three classical multi-objective paradigms on top of pateda's probabilistic
models:

* **Pareto-based** -- dominance utilities (:mod:`.dominance`), crowding distance
  (:mod:`.crowding`) and an external :class:`.ParetoArchive`.  Combined with the
  Pareto / crowding selection methods in :mod:`pateda.selection`, any pateda
  model becomes a Pareto-based MOEDA.
* **Indicator / metric-based** -- quality indicators (:mod:`.indicators`:
  hypervolume, hypervolume contributions, additive epsilon / IBEA fitness, IGD)
  that both evaluate and drive the search (via
  :class:`pateda.selection.IndicatorBasedSelection`).
* **Decomposition-based** -- weight-vector designs (:mod:`.weights`),
  scalarising functions (:mod:`.scalarization`) and the :class:`.MOEAD` driver
  that reuses any pateda learning/sampling component.
"""

from pateda.multiobjective.dominance import (
    pareto_dominates,
    find_pareto_set,
    pareto_ranking,
    non_dominated_front,
)
from pateda.multiobjective.crowding import crowding_distance
from pateda.multiobjective.archive import ParetoArchive
from pateda.multiobjective.weights import (
    uniform_weights,
    das_dennis_weights,
    generate_weights,
    weight_neighbourhoods,
)
from pateda.multiobjective.scalarization import (
    scalarize,
    weighted_sum,
    tchebycheff,
    pbi,
    SCALARIZATIONS,
)
from pateda.multiobjective.indicators import (
    reference_point_from,
    hypervolume,
    hypervolume_contributions,
    additive_epsilon_matrix,
    ibea_fitness,
    igd,
)
from pateda.multiobjective.moead import MOEAD, MOEADResult

__all__ = [
    # dominance
    "pareto_dominates",
    "find_pareto_set",
    "pareto_ranking",
    "non_dominated_front",
    "crowding_distance",
    "ParetoArchive",
    # weights / scalarization
    "uniform_weights",
    "das_dennis_weights",
    "generate_weights",
    "weight_neighbourhoods",
    "scalarize",
    "weighted_sum",
    "tchebycheff",
    "pbi",
    "SCALARIZATIONS",
    # indicators
    "reference_point_from",
    "hypervolume",
    "hypervolume_contributions",
    "additive_epsilon_matrix",
    "ibea_fitness",
    "igd",
    # decomposition driver
    "MOEAD",
    "MOEADResult",
]
