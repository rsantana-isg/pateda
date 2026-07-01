"""
Plug-and-play permutation EDA wrappers.

All algorithms work on permutations of range(n_vars).

Usage example::

    from pateda import EHMEDA
    import numpy as np

    def lop_proxy(perm):
        return float(np.sum(np.array(perm) * np.arange(len(perm))))

    alg = EHMEDA(n_vars=12, fitness_func=lop_proxy,
                 pop_size=100, n_gen=50, random_seed=42)
    stats, cache = alg.run()
    print("Best:", stats.best_fitness_overall)

Notes
-----
Permutation EDAs use a self-contained run loop because several learners and
samplers use only ``__call__`` (not ``.learn()`` / ``.sample()``) or require
custom ``sample_size`` parameters that the core EDA class does not pass.
"""

from typing import Callable, Optional, Tuple
import numpy as np

from pateda.core.eda import Statistics, Cache
from perm_pateda.seeding.permutation_init import PermutationInit

from perm_pateda.learning.histogram import LearnEHM, LearnNHM
from perm_pateda.sampling.histogram import SampleEHM, SampleNHM

from perm_pateda.learning.mallows import (
    LearnMallowsKendall,
    LearnMallowsCayley,
    LearnMallowsUlam,
    LearnGeneralizedMallowsKendall,
    LearnGeneralizedMallowsCayley,
)
from perm_pateda.sampling.mallows import (
    SampleMallowsKendall,
    SampleMallowsCayley,
    SampleMallowsUlam,
    SampleGeneralizedMallowsKendall,
    SampleGeneralizedMallowsCayley,
)

from perm_pateda.learning.plackett_luce import LearnPlackettLuce
from perm_pateda.sampling.plackett_luce import SamplePlackettLuce

from perm_pateda.learning.mixture_plackett_luce import LearnPlackettLuceMixture
from perm_pateda.sampling.mixture_plackett_luce import SamplePlackettLuceMixture

from perm_pateda.learning.hamming_kmm import LearnHammingKMM
from perm_pateda.sampling.hamming_kmm import SampleHammingKMM

def _dummy_cardinality(n_vars: int) -> np.ndarray:
    """Return a placeholder cardinality array for permutations."""
    return np.arange(n_vars)


class _PermEDA:
    """
    Self-contained run loop for permutation EDAs.

    Subclasses must provide ``_learner`` and ``_sampler`` attributes that
    expose ``learn(gen, n_vars, cardinality, pop, fit)`` and
    ``sample(n_vars, model, cardinality, pop, fit, sample_size, rng)``
    (or ``__call__`` equivalents bridged in this class).
    """

    def __init__(
        self,
        n_vars: int,
        fitness_func: Callable,
        pop_size: int = 100,
        n_gen: int = 50,
        selection_ratio: float = 0.5,
        elitism: bool = True,
        random_seed: Optional[int] = None,
    ):
        self.n_vars = n_vars
        self.fitness_func = fitness_func
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.selection_ratio = selection_ratio
        self.elitism = elitism
        self.rng = np.random.default_rng(random_seed)
        self._card = _dummy_cardinality(n_vars)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate(self, population: np.ndarray) -> np.ndarray:
        return np.array([self.fitness_func(population[i]) for i in range(len(population))])

    def _learn(self, gen: int, population: np.ndarray, fitness: np.ndarray):
        """Call the learner.  Subclasses may override."""
        learner = self._learner  # type: ignore[attr-defined]
        if hasattr(learner, 'learn'):
            return learner.learn(gen, self.n_vars, self._card, population, fitness)
        # fall back to __call__
        return learner(
            generation=gen,
            n_vars=self.n_vars,
            cardinality=self._card,
            selected_pop=population,
            selected_fitness=fitness,
        )

    def _sample(self, model, current_pop: np.ndarray = None) -> np.ndarray:
        """Call the sampler.  Subclasses may override."""
        sampler = self._sampler  # type: ignore[attr-defined]
        # Provide a dummy non-empty population so that samplers that call
        # np.min(population) to detect indexing style do not crash on
        # a zero-size array.  We supply a single identity permutation.
        if current_pop is None or (hasattr(current_pop, 'size') and current_pop.size == 0):
            dummy_pop = np.arange(self.n_vars, dtype=int).reshape(1, self.n_vars)
        else:
            dummy_pop = current_pop
        dummy_fit = np.array([0.0])

        if hasattr(sampler, 'sample'):
            return sampler.sample(
                n_vars=self.n_vars,
                model=model,
                cardinality=self._card,
                population=dummy_pop,
                fitness=dummy_fit,
                sample_size=self.pop_size,
                rng=self.rng,
            )
        # fall back to __call__
        return sampler(
            n_vars=self.n_vars,
            model=model,
            cardinality=self._card,
            population=dummy_pop,
            fitness=dummy_fit,
            sample_size=self.pop_size,
            rng=self.rng,
        )

    # ------------------------------------------------------------------
    # Public run() method
    # ------------------------------------------------------------------

    def run(self, verbose: bool = False) -> Tuple[Statistics, Cache]:
        """
        Execute the permutation EDA.

        Returns
        -------
        stats : Statistics
        cache : Cache
        """
        stats = Statistics()
        cache = Cache()
        n_select = max(2, int(self.pop_size * self.selection_ratio))

        # Initial population
        seeder = PermutationInit()
        population = seeder.seed(self.n_vars, self.pop_size, self._card, rng=self.rng)
        fitness = self._evaluate(population)

        best_idx = int(np.argmax(fitness))
        best_ind = population[best_idx].copy()
        best_fit = float(fitness[best_idx])

        for gen in range(self.n_gen):
            stats.update(gen, population, fitness.reshape(-1, 1))

            if verbose:
                print(f"Gen {gen}: best={stats.best_fitness[-1]:.4f} "
                      f"mean={stats.mean_fitness[-1]:.4f}")

            # Truncation selection
            sorted_idx = np.argsort(fitness)[::-1]
            selected = population[sorted_idx[:n_select]]
            sel_fit = fitness[sorted_idx[:n_select]]

            # Learn model
            model = self._learn(gen, selected, sel_fit)

            # Sample new population
            new_pop = self._sample(model, current_pop=population)
            new_fit = self._evaluate(new_pop)

            # Elitism: keep global best
            if self.elitism:
                new_best_idx = int(np.argmax(new_fit))
                if best_fit > new_fit[new_best_idx]:
                    worst_idx = int(np.argmin(new_fit))
                    new_pop[worst_idx] = best_ind.copy()
                    new_fit[worst_idx] = best_fit
                else:
                    best_fit = float(new_fit[new_best_idx])
                    best_ind = new_pop[new_best_idx].copy()

            population = new_pop
            fitness = new_fit

        # Final generation
        stats.update(self.n_gen, population, fitness.reshape(-1, 1))
        return stats, cache


# ---------------------------------------------------------------------------
# EHMEDA
# ---------------------------------------------------------------------------

class EHMEDA(_PermEDA):
    """
    Edge Histogram Model EDA (EHM-EDA).

    Learns an edge histogram matrix that captures transition probabilities
    between consecutive items in permutations.

    Parameters
    ----------
    n_vars : int
        Permutation length (items are 0 .. n_vars-1).
    fitness_func : callable
        Function mapping a permutation (1-D int array) to a scalar fitness
        (higher is better).
    pop_size : int
        Population size.
    n_gen : int
        Number of generations.
    selection_ratio : float
        Fraction of population selected for model learning.
    elitism : bool
        Whether to preserve the best solution across generations.
    random_seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_vars: int,
        fitness_func: Callable,
        pop_size: int = 100,
        n_gen: int = 50,
        selection_ratio: float = 0.5,
        elitism: bool = True,
        random_seed: Optional[int] = None,
    ):
        super().__init__(n_vars, fitness_func, pop_size, n_gen, selection_ratio, elitism, random_seed)
        self._learner = LearnEHM()
        self._sampler = SampleEHM()


# ---------------------------------------------------------------------------
# NHMEDA
# ---------------------------------------------------------------------------

class NHMEDA(_PermEDA):
    """
    Node Histogram Model EDA (NHM-EDA).

    Learns a node histogram matrix recording how often each item appears
    at each position, then samples valid permutations sequentially.

    Parameters
    ----------
    n_vars, fitness_func, pop_size, n_gen, selection_ratio, elitism,
    random_seed : see EHMEDA.
    """

    def __init__(
        self,
        n_vars: int,
        fitness_func: Callable,
        pop_size: int = 100,
        n_gen: int = 50,
        selection_ratio: float = 0.5,
        elitism: bool = True,
        random_seed: Optional[int] = None,
    ):
        super().__init__(n_vars, fitness_func, pop_size, n_gen, selection_ratio, elitism, random_seed)
        self._learner = LearnNHM()
        self._sampler = SampleNHM()


# ---------------------------------------------------------------------------
# MallowsKendallEDA
# ---------------------------------------------------------------------------

class MallowsKendallEDA(_PermEDA):
    """
    Mallows Model EDA with Kendall distance.

    Fits a Mallows distribution (consensus ranking + spread parameter theta)
    using the Kendall tau distance.

    Parameters
    ----------
    n_vars, fitness_func, pop_size, n_gen, selection_ratio, elitism,
    random_seed : see EHMEDA.
    """

    def __init__(
        self,
        n_vars: int,
        fitness_func: Callable,
        pop_size: int = 100,
        n_gen: int = 50,
        selection_ratio: float = 0.5,
        elitism: bool = True,
        random_seed: Optional[int] = None,
    ):
        super().__init__(n_vars, fitness_func, pop_size, n_gen, selection_ratio, elitism, random_seed)
        self._learner = LearnMallowsKendall()
        self._sampler = SampleMallowsKendall()


# ---------------------------------------------------------------------------
# MallowsCayleyEDA
# ---------------------------------------------------------------------------

class MallowsCayleyEDA(_PermEDA):
    """
    Mallows Model EDA with Cayley distance.

    Fits a Mallows distribution using the Cayley (transposition) distance.

    Parameters
    ----------
    n_vars, fitness_func, pop_size, n_gen, selection_ratio, elitism,
    random_seed : see EHMEDA.
    """

    def __init__(
        self,
        n_vars: int,
        fitness_func: Callable,
        pop_size: int = 100,
        n_gen: int = 50,
        selection_ratio: float = 0.5,
        elitism: bool = True,
        random_seed: Optional[int] = None,
    ):
        super().__init__(n_vars, fitness_func, pop_size, n_gen, selection_ratio, elitism, random_seed)
        self._learner = LearnMallowsCayley()
        self._sampler = SampleMallowsCayley()


# ---------------------------------------------------------------------------
# GMallowsKendallEDA
# ---------------------------------------------------------------------------

class GMallowsKendallEDA(_PermEDA):
    """
    Generalized Mallows Model EDA with Kendall distance.

    Uses position-dependent spread parameters (one theta per position)
    instead of a single global theta.

    Parameters
    ----------
    n_vars, fitness_func, pop_size, n_gen, selection_ratio, elitism,
    random_seed : see EHMEDA.
    """

    def __init__(
        self,
        n_vars: int,
        fitness_func: Callable,
        pop_size: int = 100,
        n_gen: int = 50,
        selection_ratio: float = 0.5,
        elitism: bool = True,
        random_seed: Optional[int] = None,
    ):
        super().__init__(n_vars, fitness_func, pop_size, n_gen, selection_ratio, elitism, random_seed)
        self._learner = LearnGeneralizedMallowsKendall()
        self._sampler = SampleGeneralizedMallowsKendall()


# ---------------------------------------------------------------------------
# GMallowsCayleyEDA
# ---------------------------------------------------------------------------

class GMallowsCayleyEDA(_PermEDA):
    """
    Generalized Mallows Model EDA with Cayley distance.

    Uses position-dependent spread parameters with the Cayley distance.

    Parameters
    ----------
    n_vars, fitness_func, pop_size, n_gen, selection_ratio, elitism,
    random_seed : see EHMEDA.
    """

    def __init__(
        self,
        n_vars: int,
        fitness_func: Callable,
        pop_size: int = 100,
        n_gen: int = 50,
        selection_ratio: float = 0.5,
        elitism: bool = True,
        random_seed: Optional[int] = None,
    ):
        super().__init__(n_vars, fitness_func, pop_size, n_gen, selection_ratio, elitism, random_seed)
        self._learner = LearnGeneralizedMallowsCayley()
        self._sampler = SampleGeneralizedMallowsCayley()


# ---------------------------------------------------------------------------
# PlackettLuceEDA
# ---------------------------------------------------------------------------

class PlackettLuceEDA(_PermEDA):
    """
    Plackett-Luce Model EDA.

    Learns a single Plackett-Luce distribution over permutations via the
    MM (Minorization-Maximization) algorithm. Each item i is assigned a
    weight w[i] representing its relative preference; permutations are
    sampled sequentially with probability proportional to remaining weights
    (Gumbel-max trick for efficiency).

    Parameters
    ----------
    n_vars, fitness_func, pop_size, n_gen, selection_ratio, elitism,
    random_seed : see EHMEDA.
    """

    def __init__(
        self,
        n_vars: int,
        fitness_func: Callable,
        pop_size: int = 100,
        n_gen: int = 50,
        selection_ratio: float = 0.5,
        elitism: bool = True,
        random_seed: Optional[int] = None,
    ):
        super().__init__(n_vars, fitness_func, pop_size, n_gen, selection_ratio, elitism, random_seed)
        self._learner = LearnPlackettLuce()
        self._sampler = SamplePlackettLuce()


# ---------------------------------------------------------------------------
# PlackettLuceMixtureEDA
# ---------------------------------------------------------------------------

class PlackettLuceMixtureEDA(_PermEDA):
    """
    Mixture of Plackett-Luce Models EDA.

    Learns a mixture of K Plackett-Luce models to capture heterogeneous 
    preferences in the population using Spectral EM.

    Parameters
    ----------
    n_vars, fitness_func, pop_size, n_gen, selection_ratio, elitism,
    random_seed : see EHMEDA, n_components
    """

    def __init__(
        self,
        n_vars: int,
        fitness_func: Callable,
        pop_size: int = 100,
        n_gen: int = 50,
        selection_ratio: float = 0.5,
        elitism: bool = True,
        random_seed: Optional[int] = None,
        n_components: int = 2,  # ¡Parámetro exclusivo de nuestro algoritmo!
    ):
        super().__init__(n_vars, fitness_func, pop_size, n_gen, selection_ratio, elitism, random_seed)
        self.n_components = n_components
        self._learner = LearnPlackettLuceMixture(
            n_components=n_components,
            random_state=random_seed,
        )
        self._sampler = SamplePlackettLuceMixture()



# ---------------------------------------------------------------------------
# HammingKMMEDA
# ---------------------------------------------------------------------------

class HammingKMMEDA(_PermEDA):
    def __init__(
        self,
        n_vars: int,
        fitness_func: Callable,
        pop_size: int = 100,
        n_gen: int = 50,
        selection_ratio: float = 0.5,
        elitism: bool = True,
        random_seed: Optional[int] = None,
        expected_dist_start: Optional[float] = None,
        expected_dist_end: float = 0.25,
        gamma: float = 5.14,
    ):
        super().__init__(n_vars, fitness_func, pop_size, n_gen,
                         selection_ratio, elitism, random_seed)
        self._learner = LearnHammingKMM(
            expected_dist_start=expected_dist_start,
            expected_dist_end=expected_dist_end,
            gamma=gamma,
            n_gen=n_gen,
        )
        self._sampler = SampleHammingKMM()


# ---------------------------------------------------------------------------
# MallowsUlamEDA
# ---------------------------------------------------------------------------

class MallowsUlamEDA(_PermEDA):

    def __init__(
        self,
        n_vars: int,
        fitness_func: Callable,
        pop_size: int = 100,
        n_gen: int = 50,
        selection_ratio: float = 0.5,
        elitism: bool = True,
        random_seed: Optional[int] = None,
        burn_in: int = 1000,
        step_size: int = 100,
    ):
        super().__init__(
            n_vars, fitness_func, pop_size, n_gen, selection_ratio, elitism, random_seed
        )
        self._learner = LearnMallowsUlam()
        self._sampler = SampleMallowsUlam(burn_in=burn_in, step_size=step_size)