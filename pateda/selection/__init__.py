"""Selection methods"""

from pateda.selection.truncation import TruncationSelection
from pateda.selection.tournament import TournamentSelection
from pateda.selection.proportional import ProportionalSelection
from pateda.selection.ranking import RankingSelection
from pateda.selection.sus import StochasticUniversalSampling
from pateda.selection.boltzmann import BoltzmannSelection
from pateda.selection.non_dominated import NonDominatedSelection
from pateda.selection.pareto_front import ParetoFrontSelection

__all__ = [
    "TruncationSelection",
    "TournamentSelection",
    "ProportionalSelection",
    "RankingSelection",
    "StochasticUniversalSampling",
    "BoltzmannSelection",
    "NonDominatedSelection",
    "ParetoFrontSelection",
]
