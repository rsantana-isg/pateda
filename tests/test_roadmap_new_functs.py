"""
Tests for the functionalities added from ROADMAP_New_Functs.md:

* T1 probabilistic seeding
* T2 niching replacement (deterministic crowding, RTR, clustering)
* T3 stop conditions (no-improvement, convergence, composite)
* T5 multi-objective indicator tracker
* T7 structural transfer learning
"""

import numpy as np
import pytest

from pateda.core.eda import EDA
from pateda.core.components import EDAComponents, CacheConfig
from pateda.seeding import RandomInit, ProbabilisticInit
from pateda.selection.truncation import TruncationSelection
from pateda.selection.crowding import CrowdingDistanceSelection
from pateda.learning.umda import LearnUMDA
from pateda.sampling.fda import SampleFDA
from pateda.replacement import (
    ElitistReplacement,
    DeterministicCrowdingReplacement,
    RestrictedTournamentReplacement,
    ClusteringReplacement,
)
from pateda.stop_conditions import (
    MaxGenerations,
    NoImprovement,
    PopulationConvergence,
    CompositeStop,
)


def onemax(x):
    return float(np.sum(x))


# --------------------------------------------------------------------------- T1
class TestProbabilisticInit:
    def test_binary_bias(self):
        n = 30
        seeder = ProbabilisticInit([np.array([0.1, 0.9])] * n)
        pop = seeder.seed(n, 500, np.full(n, 2), rng=np.random.default_rng(0))
        assert pop.shape == (500, n)
        assert set(np.unique(pop)).issubset({0, 1})
        # mean fraction of ones close to 0.9
        assert abs(pop.mean() - 0.9) < 0.05

    def test_non_binary(self):
        card = np.array([2, 3, 4])
        probs = [np.array([0.5, 0.5]),
                 np.array([0.0, 0.0, 1.0]),   # variable 1 always value 2
                 np.array([0.25, 0.25, 0.25, 0.25])]
        seeder = ProbabilisticInit(probs)
        pop = seeder.seed(3, 400, card, rng=np.random.default_rng(1))
        assert np.all(pop[:, 1] == 2)
        assert pop[:, 2].max() <= 3

    def test_uniform_fallback(self):
        seeder = ProbabilisticInit()
        pop = seeder.seed(5, 100, np.full(5, 3), rng=np.random.default_rng(2))
        assert pop.max() <= 2 and pop.min() >= 0

    def test_bad_distribution_raises(self):
        seeder = ProbabilisticInit([np.array([1.0, 1.0, 1.0])])  # len 3 for card 2
        with pytest.raises(ValueError):
            seeder.seed(1, 10, np.array([2]), rng=np.random.default_rng(0))


# --------------------------------------------------------------------------- T2
def _run_with_replacement(repl, n=20, gens=12):
    comp = EDAComponents(
        seeding=RandomInit(), selection=TruncationSelection(ratio=0.5),
        learning=LearnUMDA(alpha=1.0), sampling=SampleFDA(n_samples=200),
        replacement=repl, stop_condition=MaxGenerations(gens))
    stats, _ = EDA(200, n, onemax, np.full(n, 2), comp, random_seed=1).run(verbose=False)
    return stats


class TestNichingReplacement:
    def test_deterministic_crowding_runs_and_keeps_size(self):
        repl = DeterministicCrowdingReplacement()
        pop = np.random.default_rng(0).integers(0, 2, (50, 10))
        fit = pop.sum(axis=1).reshape(-1, 1).astype(float)
        new = np.random.default_rng(1).integers(0, 2, (50, 10))
        nfit = new.sum(axis=1).reshape(-1, 1).astype(float)
        out_pop, out_fit = repl.replace(pop, fit, new, nfit,
                                        rng=np.random.default_rng(2))
        assert out_pop.shape == pop.shape

    def test_all_three_optimize_onemax(self):
        for repl in (DeterministicCrowdingReplacement(),
                     RestrictedTournamentReplacement(window_size=10),
                     ClusteringReplacement(n_clusters=4)):
            stats = _run_with_replacement(repl)
            assert stats.best_fitness_overall >= 18  # near-optimal on 20-bit OneMax

    def test_clustering_preserves_population_size(self):
        repl = ClusteringReplacement(n_clusters=3)
        rng = np.random.default_rng(0)
        pop = rng.integers(0, 2, (60, 12))
        fit = pop.sum(axis=1).reshape(-1, 1).astype(float)
        new = rng.integers(0, 2, (60, 12))
        nfit = new.sum(axis=1).reshape(-1, 1).astype(float)
        out_pop, out_fit = repl.replace(pop, fit, new, nfit, rng=rng)
        assert out_pop.shape[0] == 60
        assert out_fit.shape[0] == 60


# --------------------------------------------------------------------------- T3
class TestStopConditions:
    def test_no_improvement_triggers(self):
        cond = NoImprovement(k=3, epsilon=0.5)
        pop = np.zeros((10, 5), dtype=int)
        fit = np.full((10, 1), 5.0)  # constant best -> stagnation
        stops = [cond.should_stop(g, pop, fit) for g in range(6)]
        assert stops[-1] is True
        assert stops[0] is False

    def test_no_improvement_resets_on_gain(self):
        cond = NoImprovement(k=2, epsilon=0.0)
        pop = np.zeros((4, 3), dtype=int)
        assert cond.should_stop(0, pop, np.array([[1.0]])) is False
        assert cond.should_stop(1, pop, np.array([[1.0]])) is False  # stall 1
        # improvement resets the stall counter
        assert cond.should_stop(2, pop, np.array([[5.0]])) is False
        cond.reset()
        assert cond._best is None

    def test_population_convergence(self):
        cond = PopulationConvergence(tol=1e-6, patience=1)
        diverse = np.random.default_rng(0).integers(0, 2, (100, 20))
        converged = np.ones((100, 20), dtype=int)
        assert cond.should_stop(0, diverse, np.zeros((100, 1))) is False
        assert cond.should_stop(1, converged, np.zeros((100, 1))) is True

    def test_composite_any_all(self):
        never = MaxGenerations(10**9)
        always = MaxGenerations(0)
        assert CompositeStop([never, always], mode="any").should_stop(
            0, np.zeros((2, 2), dtype=int), np.zeros((2, 1))) is True
        assert CompositeStop([never, always], mode="all").should_stop(
            0, np.zeros((2, 2), dtype=int), np.zeros((2, 1))) is False


# --------------------------------------------------------------------------- T5
class TestMultiObjectiveTracker:
    def test_hypervolume_recorded(self):
        from pateda.statistics import MultiObjectiveTracker

        n = 16

        def mo(x):
            return np.array([float(np.sum(x)), float(np.sum(1 - x)) + 0.01 * float(x[0])])

        comp = EDAComponents(
            seeding=RandomInit(), selection=CrowdingDistanceSelection(n_select=100),
            learning=LearnUMDA(alpha=1.0), sampling=SampleFDA(n_samples=200),
            replacement=ElitistReplacement(n_elite=1), stop_condition=MaxGenerations(6),
            statistics=MultiObjectiveTracker())
        stats, _ = EDA(200, n, mo, np.full(n, 2), comp, random_seed=1).run(verbose=False)
        hv = stats.custom["hypervolume"]
        # one indicator value per executed generation
        assert len(hv) == len(stats.best_fitness)
        assert len(hv) >= 6
        assert all(np.isfinite(v) for v in hv)


# --------------------------------------------------------------------------- T7
class TestTransferLearning:
    def _cached_run(self):
        from pateda.learning.tree import LearnTreeModel
        n = 15
        from pateda.functions import deceptive3
        comp = EDAComponents(
            seeding=RandomInit(), selection=TruncationSelection(ratio=0.5),
            learning=LearnTreeModel(alpha=1.0), sampling=SampleFDA(n_samples=200),
            replacement=ElitistReplacement(n_elite=1), stop_condition=MaxGenerations(6))
        _, cache = EDA(200, n, deceptive3, np.full(n, 2), comp, random_seed=1).run(
            cache_config=CacheConfig(cache_populations=True, cache_fitness=True,
                                     cache_models=True), verbose=False)
        return cache

    def test_aggregate_and_interaction_matrix(self):
        from pateda.knowledge_extraction.transfer import (
            aggregate_edge_frequencies, structure_to_interaction_matrix)
        cache = self._cached_run()
        freq = aggregate_edge_frequencies(cache.models)
        assert freq.shape == (15, 15)
        assert np.allclose(freq, freq.T)
        assert np.all(np.diag(freq) == 0)
        M = structure_to_interaction_matrix(freq, threshold=0.3)
        assert set(np.unique(M)).issubset({0, 1})
        assert np.all(M == M.T)

    def test_persist_roundtrip(self, tmp_path):
        from pateda.knowledge_extraction.transfer import TransferredStructure
        cache = self._cached_run()
        ts = TransferredStructure.from_models(cache.models, source="deceptive3")
        p = str(tmp_path / "ts.npz")
        ts.save(p)
        ts2 = TransferredStructure.load(p)
        assert ts2.n_vars == ts.n_vars
        assert np.allclose(ts2.edge_frequencies, ts.edge_frequencies)

    def test_warm_start_population(self):
        from pateda.knowledge_extraction.transfer import warm_start_population
        cache = self._cached_run()
        ws = warm_start_population(cache.populations, cache.fitness_values,
                                   pop_size=150, fraction=0.4,
                                   rng=np.random.default_rng(0))
        assert ws.shape == (150, 15)
