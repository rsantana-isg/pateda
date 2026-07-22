"""Replacement methods"""

from pateda.replacement.generational import GenerationalReplacement
from pateda.replacement.elitist import ElitistReplacement, RTRReplacement
from pateda.replacement.niching import (
    DeterministicCrowdingReplacement,
    RestrictedTournamentReplacement,
    ClusteringReplacement,
)

__all__ = [
    "GenerationalReplacement",
    "ElitistReplacement",
    "RTRReplacement",
    "DeterministicCrowdingReplacement",
    "RestrictedTournamentReplacement",
    "ClusteringReplacement",
]
