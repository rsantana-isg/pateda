"""
Plug-and-play continuous EDA wrappers.

Each class assembles the appropriate pateda components and exposes a clean
constructor + run() interface.

Usage example::

    from pateda import GaussianUMDA
    import numpy as np

    def sphere(x):
        return -np.sum(x**2)   # maximise negated sphere

    alg = GaussianUMDA(n_vars=10, bounds=(-5, 5), fitness_func=sphere,
                       pop_size=200, n_gen=100, random_seed=42)
    stats, cache = alg.run()
    print("Best:", stats.best_fitness_overall)
"""

from typing import Callable, Optional, Tuple, Union
import numpy as np

from pateda.core.eda import EDA, Statistics, Cache
from pateda.core.components import EDAComponents
from pateda.seeding.random_init import RandomInit
from pateda.selection.truncation import TruncationSelection
from pateda.replacement.elitist import ElitistReplacement
from pateda.stop_conditions.max_generations import MaxGenerations

from pateda.learning.basic_gaussian import LearnGaussianUnivariate, LearnGaussianFull
from pateda.learning.mixture_gaussian import LearnMixtureGaussian
from pateda.sampling.basic_gaussian import SampleGaussianUnivariate, SampleGaussianFull
from pateda.sampling.mixture_gaussian import SampleMixtureGaussian

from pateda.algorithms.base import _BaseEDA, _GMRFLearner, _GMRFSampler


def _to_bounds(bounds, n_vars: int) -> np.ndarray:
    """
    Convert bounds specification to np.ndarray of shape (2, n_vars).

    Accepts:
    - tuple (min_val, max_val) — same bounds for all variables
    - np.ndarray of shape (2, n_vars) — per-variable bounds (row0=min, row1=max)
    """
    if isinstance(bounds, tuple):
        lo, hi = bounds
        out = np.zeros((2, n_vars))
        out[0, :] = lo
        out[1, :] = hi
        return out
    bounds_arr = np.asarray(bounds, dtype=float)
    if bounds_arr.shape == (2, n_vars):
        return bounds_arr
    raise ValueError(
        f"bounds must be a (min, max) tuple or np.ndarray of shape (2, {n_vars}), "
        f"got shape {bounds_arr.shape}"
    )


def _make_components(
    learner,
    sampler,
    pop_size: int,
    selection_ratio: float,
    n_gen: int,
    elitism: bool,
) -> EDAComponents:
    """Assemble standard EDA components."""
    return EDAComponents(
        seeding=RandomInit(),
        selection=TruncationSelection(ratio=selection_ratio),
        learning=learner,
        sampling=sampler,
        stop_condition=MaxGenerations(n_gen),
        replacement=ElitistReplacement(n_elite=1) if elitism else None,
    )


# ---------------------------------------------------------------------------
# GaussianUMDA
# ---------------------------------------------------------------------------

class GaussianUMDA(_BaseEDA):
    """
    Gaussian UMDA — univariate Gaussian EDA for continuous search spaces.

    Models each variable independently with a Gaussian distribution (mean
    and standard deviation estimated from the selected population).

    Parameters
    ----------
    n_vars : int
        Number of continuous variables.
    bounds : tuple (lo, hi) or np.ndarray of shape (2, n_vars)
        Search-space bounds.  A single tuple applies the same bounds to all
        variables; a 2-D array allows per-variable bounds.
    fitness_func : callable
        Function mapping an individual (1-D array) to a scalar fitness value
        (higher is better).
    pop_size : int
        Population size.
    n_gen : int
        Number of generations.
    selection_ratio : float
        Fraction of the population used for learning.
    elitism : bool
        Whether to preserve the best individual across generations.
    random_seed : int or None
        Seed for the random number generator.
    """

    def __init__(
        self,
        n_vars: int,
        bounds: Union[Tuple[float, float], np.ndarray],
        fitness_func: Callable,
        pop_size: int = 100,
        n_gen: int = 50,
        selection_ratio: float = 0.5,
        elitism: bool = True,
        random_seed: Optional[int] = None,
    ):
        card = _to_bounds(bounds, n_vars)
        learner = LearnGaussianUnivariate()
        sampler = SampleGaussianUnivariate(n_samples=pop_size)
        components = _make_components(learner, sampler, pop_size, selection_ratio, n_gen, elitism)
        eda = EDA(pop_size, n_vars, fitness_func, card, components, random_seed=random_seed)
        super().__init__(eda)


# ---------------------------------------------------------------------------
# GaussianEDA
# ---------------------------------------------------------------------------

class GaussianEDA(_BaseEDA):
    """
    Full multivariate Gaussian EDA.

    Learns a full covariance matrix over all variables, capturing all
    pairwise linear correlations.

    Parameters
    ----------
    n_vars, bounds, fitness_func, pop_size, n_gen, selection_ratio,
    elitism, random_seed : see GaussianUMDA.
    """

    def __init__(
        self,
        n_vars: int,
        bounds: Union[Tuple[float, float], np.ndarray],
        fitness_func: Callable,
        pop_size: int = 100,
        n_gen: int = 50,
        selection_ratio: float = 0.5,
        elitism: bool = True,
        random_seed: Optional[int] = None,
    ):
        card = _to_bounds(bounds, n_vars)
        learner = LearnGaussianFull()
        sampler = SampleGaussianFull(n_samples=pop_size)
        components = _make_components(learner, sampler, pop_size, selection_ratio, n_gen, elitism)
        eda = EDA(pop_size, n_vars, fitness_func, card, components, random_seed=random_seed)
        super().__init__(eda)


# ---------------------------------------------------------------------------
# MixtureGaussianEDA
# ---------------------------------------------------------------------------

class MixtureGaussianEDA(_BaseEDA):
    """
    Mixture of Gaussians EDA.

    Clusters the selected population into ``n_components`` groups and fits a
    Gaussian model to each cluster, enabling modelling of multimodal landscapes.

    Parameters
    ----------
    n_vars, bounds, fitness_func, pop_size, n_gen, selection_ratio,
    elitism, random_seed : see GaussianUMDA.
    n_components : int
        Number of Gaussian mixture components.
    """

    def __init__(
        self,
        n_vars: int,
        bounds: Union[Tuple[float, float], np.ndarray],
        fitness_func: Callable,
        pop_size: int = 100,
        n_gen: int = 50,
        selection_ratio: float = 0.5,
        elitism: bool = True,
        n_components: int = 3,
        random_seed: Optional[int] = None,
    ):
        card = _to_bounds(bounds, n_vars)
        learner = LearnMixtureGaussian(n_clusters=n_components)
        sampler = SampleMixtureGaussian(n_samples=pop_size)
        components = _make_components(learner, sampler, pop_size, selection_ratio, n_gen, elitism)
        eda = EDA(pop_size, n_vars, fitness_func, card, components, random_seed=random_seed)
        super().__init__(eda)


# ---------------------------------------------------------------------------
# GMRFEDA — self-contained run loop (function-based API)
# ---------------------------------------------------------------------------

class GMRFEDA:
    """
    Gaussian Markov Random Field EDA (GMRF-EDA).

    Learns variable dependencies via regularized regression and clusters
    variables into disjoint cliques using affinity propagation.  A
    multivariate Gaussian is then fitted to each clique.

    Uses a self-contained run loop because the underlying API is function-based
    rather than class-based.

    Parameters
    ----------
    n_vars, bounds, fitness_func, pop_size, n_gen, selection_ratio,
    elitism, random_seed : see GaussianUMDA.
    regularization : str
        Regularization type: 'lasso' | 'elasticnet' | 'lars' | 'lassolars'.
    """

    def __init__(
        self,
        n_vars: int,
        bounds: Union[Tuple[float, float], np.ndarray],
        fitness_func: Callable,
        pop_size: int = 100,
        n_gen: int = 50,
        selection_ratio: float = 0.5,
        elitism: bool = True,
        regularization: str = 'lasso',
        random_seed: Optional[int] = None,
    ):
        self.n_vars = n_vars
        self.bounds = _to_bounds(bounds, n_vars)
        self.fitness_func = fitness_func
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.selection_ratio = selection_ratio
        self.elitism = elitism
        self.regularization = regularization
        self.rng = np.random.default_rng(random_seed)

    def _evaluate(self, population: np.ndarray) -> np.ndarray:
        fitness = np.array([self.fitness_func(population[i]) for i in range(len(population))])
        return fitness

    def run(self, verbose: bool = False) -> Tuple[Statistics, Cache]:
        from pateda.learning.gmrf_eda import learn_gmrf_eda
        from pateda.sampling.gmrf_eda import sample_gmrf_eda

        stats = Statistics()
        cache = Cache()
        n_select = max(1, int(self.pop_size * self.selection_ratio))

        # Initial population
        seeder = RandomInit()
        population = seeder.seed(self.n_vars, self.pop_size, self.bounds, rng=self.rng)
        fitness = self._evaluate(population)

        best_idx = int(np.argmax(fitness))
        best_ind = population[best_idx].copy()
        best_fit = float(fitness[best_idx])

        for gen in range(self.n_gen):
            stats.update(gen, population, fitness.reshape(-1, 1))

            if verbose:
                print(f"Gen {gen}: best={stats.best_fitness[-1]:.4f} "
                      f"mean={stats.mean_fitness[-1]:.4f}")

            # Selection
            sorted_idx = np.argsort(fitness)[::-1]
            selected = population[sorted_idx[:n_select]]
            sel_fit = fitness[sorted_idx[:n_select]]

            # Learning
            model = learn_gmrf_eda(
                selected, sel_fit,
                {'regularization': self.regularization}
            )

            # Sampling
            new_pop = sample_gmrf_eda(model, self.pop_size, bounds=self.bounds)
            new_fit = self._evaluate(new_pop)

            # Elitism
            if self.elitism:
                worst_new_idx = int(np.argmin(new_fit))
                new_best_idx = int(np.argmax(new_fit))
                if best_fit > new_fit[new_best_idx]:
                    new_pop[worst_new_idx] = best_ind.copy()
                    new_fit[worst_new_idx] = best_fit
                else:
                    best_fit = float(new_fit[new_best_idx])
                    best_ind = new_pop[new_best_idx].copy()

            population = new_pop
            fitness = new_fit

        # Final generation stats
        stats.update(self.n_gen, population, fitness.reshape(-1, 1))
        return stats, cache


# ---------------------------------------------------------------------------
# Vine copula EDAs — self-contained run loop (pyvinecopulib optional dependency)
# ---------------------------------------------------------------------------

# Copula families recognized by ``CVineEDA`` / ``RVineEDA``.  These mirror the
# index-based encoding used by ``enhanced_edas/copula_models.py``.
_VINE_COPULA_FAMILIES = (
    'gaussian', 'gumbel', 'frank', 'joe', 'indep',
    'clayton', 'bb1', 'bb6', 'bb7', 'bb8', 'tll',
)


def _require_pyvinecopulib():
    """Raise an informative ImportError when pyvinecopulib is missing."""
    try:
        import pyvinecopulib  # noqa: F401
    except ImportError:
        raise ImportError(
            "Vine copula EDAs require pyvinecopulib. "
            "Install it with: pip install pyvinecopulib"
        )


class _BaseVineEDA:
    """
    Shared run-loop machinery for vine copula EDAs.

    Subclasses override ``_learn_model`` to choose the structure (auto,
    C-vine, R-vine) and family configuration.
    """

    def __init__(
        self,
        n_vars: int,
        bounds: Union[Tuple[float, float], np.ndarray],
        fitness_func: Callable,
        pop_size: int = 100,
        n_gen: int = 50,
        selection_ratio: float = 0.5,
        elitism: bool = True,
        random_seed: Optional[int] = None,
        selection_weighting: str = "uniform",
        weighting_beta: float = 1.0,
    ):
        _require_pyvinecopulib()
        self.n_vars = n_vars
        self.bounds = _to_bounds(bounds, n_vars)
        self.fitness_func = fitness_func
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.selection_ratio = selection_ratio
        self.elitism = elitism
        self.rng = np.random.default_rng(random_seed)
        self.selection_weighting = selection_weighting
        self.weighting_beta = weighting_beta

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    def _learn_model(self, selected: np.ndarray, sel_fit: np.ndarray,
                     p: Optional[np.ndarray] = None):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate(self, population: np.ndarray) -> np.ndarray:
        return np.array([self.fitness_func(population[i]) for i in range(len(population))])

    def run(self, verbose: bool = False, cache_models: bool = False) -> Tuple[Statistics, Cache]:
        """Run the vine-copula EDA.

        Parameters
        ----------
        verbose : bool
            Print per-generation progress.
        cache_models : bool
            When True, the learned vine model of every generation is stored in
            the returned ``cache.models`` (used by the knowledge-extraction
            tools in :mod:`pateda.knowledge_extraction`).
        """
        from pateda.sampling.vine_copula import sample_vine_copula
        from pateda.selection.weighting import compute_selection_probabilities

        stats = Statistics()
        cache = Cache()
        n_select = max(1, int(self.pop_size * self.selection_ratio))

        seeder = RandomInit()
        population = seeder.seed(self.n_vars, self.pop_size, self.bounds, rng=self.rng)
        fitness = self._evaluate(population)

        best_idx = int(np.argmax(fitness))
        best_ind = population[best_idx].copy()
        best_fit = float(fitness[best_idx])

        for gen in range(self.n_gen):
            stats.update(gen, population, fitness.reshape(-1, 1))

            if verbose:
                print(f"Gen {gen}: best={stats.best_fitness[-1]:.4f} "
                      f"mean={stats.mean_fitness[-1]:.4f}")

            sorted_idx = np.argsort(fitness)[::-1]
            selected = population[sorted_idx[:n_select]]
            sel_fit = fitness[sorted_idx[:n_select]]

            # Customized selection: compute per-individual probability vector p
            p = compute_selection_probabilities(
                sel_fit,
                mode=self.selection_weighting,
                beta=self.weighting_beta,
            )

            model = self._learn_model(selected, sel_fit, p)
            if cache_models:
                cache.models.append(model)
            new_pop = sample_vine_copula(model, self.pop_size, bounds=self.bounds, rng=self.rng)
            new_fit = self._evaluate(new_pop)

            if self.elitism:
                worst_new_idx = int(np.argmin(new_fit))
                new_best_idx = int(np.argmax(new_fit))
                if best_fit > new_fit[new_best_idx]:
                    new_pop[worst_new_idx] = best_ind.copy()
                    new_fit[worst_new_idx] = best_fit
                else:
                    best_fit = float(new_fit[new_best_idx])
                    best_ind = new_pop[new_best_idx].copy()

            population = new_pop
            fitness = new_fit

        stats.update(self.n_gen, population, fitness.reshape(-1, 1))
        return stats, cache


def _copula_family_index(name: str) -> int:
    """Map a family name to its index in COPULA_FAMILIES."""
    name = str(name).lower()
    if name not in _VINE_COPULA_FAMILIES:
        raise ValueError(
            f"Unknown copula family '{name}'. "
            f"Choose one of: {', '.join(_VINE_COPULA_FAMILIES)}"
        )
    return _VINE_COPULA_FAMILIES.index(name)


class VineEDA(_BaseVineEDA):
    """
    Auto-selecting vine copula EDA (R-vine + full family set).

    Soto et al. (2011).  Models the joint distribution with a vine copula in
    which both the vine structure (R-vine) and the bivariate copula family
    of every pair are selected automatically from the data using BIC.

    Requires the optional ``pyvinecopulib`` package::

        pip install pyvinecopulib

    Parameters
    ----------
    n_vars, bounds, fitness_func, pop_size, n_gen, selection_ratio,
    elitism, random_seed : see GaussianUMDA.
    truncation_level : int or None, default None
        Truncate the vine after this many trees.  ``None`` lets the library
        pick the truncation level automatically.
    """

    def __init__(
        self,
        n_vars: int,
        bounds: Union[Tuple[float, float], np.ndarray],
        fitness_func: Callable,
        pop_size: int = 100,
        n_gen: int = 50,
        selection_ratio: float = 0.5,
        elitism: bool = True,
        truncation_level: Optional[int] = None,
        random_seed: Optional[int] = None,
        selection_weighting: str = "uniform",
        weighting_beta: float = 1.0,
    ):
        super().__init__(
            n_vars, bounds, fitness_func, pop_size, n_gen,
            selection_ratio, elitism, random_seed,
            selection_weighting, weighting_beta,
        )
        self.truncation_level = truncation_level

    def _learn_model(self, selected: np.ndarray, sel_fit: np.ndarray,
                     p: Optional[np.ndarray] = None):
        from pateda.learning.vine_copula import learn_vine_copula_auto
        params = {'truncation_level': self.truncation_level}
        if p is not None:
            params['p'] = p
        return learn_vine_copula_auto(selected, sel_fit, params)


class CVineEDA(_BaseVineEDA):
    """
    Canonical vine (C-vine) EDA with a single, user-chosen copula family.

    The C-vine structure has a star topology in each tree: one variable is
    the root and every other variable depends on it (conditionally on the
    previous trees).  All bivariate pair-copulas in the vine use the same
    family.

    Requires the optional ``pyvinecopulib`` package.

    Parameters
    ----------
    n_vars, bounds, fitness_func, pop_size, n_gen, selection_ratio,
    elitism, random_seed : see GaussianUMDA.
    copula_family : str, default 'gaussian'
        Bivariate copula family used at every pair.  One of: gaussian,
        gumbel, frank, joe, indep, clayton, bb1, bb6, bb7, bb8, tll.
    truncation_level : int or None, default None
        Truncate the vine after this many trees (``None`` = no truncation).
    """

    def __init__(
        self,
        n_vars: int,
        bounds: Union[Tuple[float, float], np.ndarray],
        fitness_func: Callable,
        pop_size: int = 100,
        n_gen: int = 50,
        selection_ratio: float = 0.5,
        elitism: bool = True,
        copula_family: str = 'gaussian',
        truncation_level: Optional[int] = None,
        random_seed: Optional[int] = None,
        selection_weighting: str = "uniform",
        weighting_beta: float = 1.0,
    ):
        super().__init__(
            n_vars, bounds, fitness_func, pop_size, n_gen,
            selection_ratio, elitism, random_seed,
            selection_weighting, weighting_beta,
        )
        self.copula_family = copula_family
        self.copula_family_idx = _copula_family_index(copula_family)
        self.truncation_level = truncation_level

    def _learn_model(self, selected: np.ndarray, sel_fit: np.ndarray,
                     p: Optional[np.ndarray] = None):
        from pateda.learning.vine_copula import learn_vine_copula_cvine
        params = {
            'copula_family': self.copula_family_idx,
            'truncation_level': self.truncation_level,
            'select_families': False,
        }
        if p is not None:
            params['p'] = p
        return learn_vine_copula_cvine(selected, sel_fit, params)


class RVineEDA(_BaseVineEDA):
    """
    Regular vine (R-vine) EDA with a single, user-chosen copula family.

    The R-vine structure is selected automatically from the data, but every
    bivariate pair-copula is forced to a single family.  Useful for
    isolating the effect of the pair-copula family from the structure.

    Requires the optional ``pyvinecopulib`` package.

    Parameters
    ----------
    n_vars, bounds, fitness_func, pop_size, n_gen, selection_ratio,
    elitism, random_seed : see GaussianUMDA.
    copula_family : str, default 'gaussian'
        Bivariate copula family used at every pair.  One of: gaussian,
        gumbel, frank, joe, indep, clayton, bb1, bb6, bb7, bb8, tll.
    truncation_level : int or None, default None
        Truncate the vine after this many trees (``None`` = no truncation).
    """

    def __init__(
        self,
        n_vars: int,
        bounds: Union[Tuple[float, float], np.ndarray],
        fitness_func: Callable,
        pop_size: int = 100,
        n_gen: int = 50,
        selection_ratio: float = 0.5,
        elitism: bool = True,
        copula_family: str = 'gaussian',
        truncation_level: Optional[int] = None,
        random_seed: Optional[int] = None,
        selection_weighting: str = "uniform",
        weighting_beta: float = 1.0,
    ):
        super().__init__(
            n_vars, bounds, fitness_func, pop_size, n_gen,
            selection_ratio, elitism, random_seed,
            selection_weighting, weighting_beta,
        )
        self.copula_family = copula_family
        self.copula_family_idx = _copula_family_index(copula_family)
        self.truncation_level = truncation_level

    def _learn_model(self, selected: np.ndarray, sel_fit: np.ndarray,
                     p: Optional[np.ndarray] = None):
        from pateda.learning.vine_copula import learn_vine_copula_dvine
        params = {
            'copula_family': self.copula_family_idx,
            'truncation_level': self.truncation_level,
            'select_families': False,
        }
        if p is not None:
            params['p'] = p
        return learn_vine_copula_dvine(selected, sel_fit, params)
