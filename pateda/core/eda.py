"""
Main EDA execution framework

This module implements the core EDA algorithm, equivalent to MATEDA's RunEDA.m
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import warnings

from pateda.core.components import (
    EDAComponents,
    CacheConfig,
    StopCondition,
    StatisticsMethod,
)
from pateda.core.models import Model


@dataclass
class Statistics:
    """Container for EDA execution statistics"""

    # Per-generation statistics
    best_fitness: List[float] = field(default_factory=list)
    mean_fitness: List[float] = field(default_factory=list)
    std_fitness: List[float] = field(default_factory=list)
    worst_fitness: List[float] = field(default_factory=list)

    # Best solution found
    best_individual: Optional[np.ndarray] = None
    best_fitness_overall: Optional[float] = None
    generation_found: Optional[int] = None

    # Custom statistics from StatisticsMethod
    custom: Dict[str, List[Any]] = field(default_factory=dict)

    def update(
        self, generation: int, population: np.ndarray, fitness: np.ndarray
    ) -> None:
        """Update statistics for current generation"""
        self.best_fitness.append(float(np.max(fitness)))
        self.mean_fitness.append(float(np.mean(fitness)))
        self.std_fitness.append(float(np.std(fitness)))
        self.worst_fitness.append(float(np.min(fitness)))

        # Update overall best
        gen_best_idx = np.argmax(fitness)
        gen_best_fitness = fitness[gen_best_idx]

        if (
            self.best_fitness_overall is None
            or gen_best_fitness > self.best_fitness_overall
        ):
            self.best_fitness_overall = float(gen_best_fitness)
            self.best_individual = population[gen_best_idx].copy()
            self.generation_found = generation


@dataclass
class Cache:
    """Container for cached EDA execution data"""

    populations: List[np.ndarray] = field(default_factory=list)
    fitness_values: List[np.ndarray] = field(default_factory=list)
    models: List[Model] = field(default_factory=list)
    statistics: List[Dict[str, Any]] = field(default_factory=list)
    selected_populations: List[np.ndarray] = field(default_factory=list)


class EDA:
    """
    Main EDA (Estimation of Distribution Algorithm) executor

    This class implements the core EDA algorithm, equivalent to MATEDA's RunEDA.m.
    It orchestrates all EDA components to perform evolutionary optimization.

    Example:
        >>> from pateda import EDA, EDAComponents
        >>> from pateda.seeding import RandomInit
        >>> from pateda.learning import LearnFDA
        >>> # ... configure components
        >>> eda = EDA(pop_size=300, n_vars=30, fitness_func=my_func,
        ...           cardinality=np.full(30, 2), components=components)
        >>> stats, cache = eda.run()
    """

    def __init__(
        self,
        pop_size: int,
        n_vars: int,
        fitness_func: Union[Callable, str],
        cardinality: Union[np.ndarray, List],
        components: EDAComponents,
    ):
        """
        Initialize EDA

        Args:
            pop_size: Population size
            n_vars: Number of variables
            fitness_func: Fitness evaluation function or function name
            cardinality: Variable cardinalities (discrete) or ranges (continuous)
                        For discrete: 1D array of cardinalities
                        For continuous: 2D array [min_values, max_values]
            components: EDA components configuration

        Raises:
            ValueError: If components are invalid or incompatible
        """
        self.pop_size = pop_size
        self.n_vars = n_vars
        self.cardinality = np.array(cardinality)
        self.components = components

        # Validate components
        self.components.validate()

        # Handle fitness function
        if isinstance(fitness_func, str):
            # Import from functions module
            from pateda.functions import get_function

            self.fitness_func = get_function(fitness_func)
        else:
            self.fitness_func = fitness_func

        # State variables
        self.generation = 0
        self.population: Optional[np.ndarray] = None
        self.fitness: Optional[np.ndarray] = None
        self.model: Optional[Model] = None

    def evaluate_fitness(self, population: np.ndarray) -> np.ndarray:
        """
        Evaluate fitness for a population

        Args:
            population: Population to evaluate

        Returns:
            Fitness values for each individual
        """
        n_individuals = population.shape[0]
        fitness = np.zeros(n_individuals)

        for i in range(n_individuals):
            fitness[i] = self.fitness_func(population[i])

        return fitness

    def run(
        self,
        cache_config: Optional[Union[CacheConfig, List]] = None,
        verbose: bool = True,
    ) -> Tuple[Statistics, Cache]:
        """
        Execute the EDA

        Args:
            cache_config: Configuration for caching data
                         Can be CacheConfig object or MATEDA-style list
            verbose: Print progress information

        Returns:
            Tuple of (statistics, cache) containing execution results

        Raises:
            RuntimeError: If execution fails
        """
        # Setup cache
        if cache_config is None:
            cache_config = CacheConfig()
        elif isinstance(cache_config, list):
            cache_config = CacheConfig.from_list(cache_config)

        cache = Cache()
        statistics = Statistics()

        # Reset stopping condition
        self.components.stop_condition.reset()

        # Main evolution loop
        self.generation = 0
        continue_evolution = True

        if verbose:
            print(f"Starting EDA execution: {self.pop_size} individuals, "
                  f"{self.n_vars} variables")

        while continue_evolution:
            if self.generation == 0:
                # Initial population
                if verbose:
                    print(f"Generation {self.generation}: Seeding initial population")

                self.population = self.components.seeding.seed(
                    self.n_vars,
                    self.pop_size,
                    self.cardinality,
                    **self.components.seeding_params,
                )

                # Repair if needed
                if self.components.repairing is not None:
                    self.population = self.components.repairing.repair(
                        self.population,
                        self.cardinality,
                        **self.components.repairing_params,
                    )

                # Evaluate fitness
                self.fitness = self.evaluate_fitness(self.population)

                # Local optimization if requested
                if self.components.local_opt is not None:
                    self.population, self.fitness = self.components.local_opt.optimize(
                        self.population,
                        self.fitness,
                        self.fitness_func,
                        self.cardinality,
                        **self.components.local_opt_params,
                    )

            else:
                # Subsequent generations: sample from model
                if verbose:
                    print(f"Generation {self.generation}: Sampling from model")

                new_pop = self.components.sampling.sample(
                    self.n_vars,
                    self.model,
                    self.cardinality,
                    self.population,
                    self.fitness,
                    **self.components.sampling_params,
                )

                # Repair if needed
                if self.components.repairing is not None:
                    new_pop = self.components.repairing.repair(
                        new_pop, self.cardinality, **self.components.repairing_params
                    )

                # Evaluate fitness
                new_fitness = self.evaluate_fitness(new_pop)

                # Local optimization if requested
                if self.components.local_opt is not None:
                    new_pop, new_fitness = self.components.local_opt.optimize(
                        new_pop,
                        new_fitness,
                        self.fitness_func,
                        self.cardinality,
                        **self.components.local_opt_params,
                    )

                # Replacement
                if self.components.replacement is not None:
                    self.population, self.fitness = self.components.replacement.replace(
                        self.population,
                        self.fitness,
                        new_pop,
                        new_fitness,
                        **self.components.replacement_params,
                    )
                else:
                    # No replacement: just use new population
                    self.population = new_pop
                    self.fitness = new_fitness

            # Update statistics
            statistics.update(self.generation, self.population, self.fitness)

            if verbose:
                print(
                    f"  Best: {statistics.best_fitness[-1]:.6f}, "
                    f"Mean: {statistics.mean_fitness[-1]:.6f}, "
                    f"Std: {statistics.std_fitness[-1]:.6f}"
                )

            # Custom statistics
            if self.components.statistics is not None:
                custom_stats = self.components.statistics.collect(
                    self.generation,
                    self.population,
                    self.fitness,
                    self.model,
                    **self.components.statistics_params,
                )
                for key, value in custom_stats.items():
                    if key not in statistics.custom:
                        statistics.custom[key] = []
                    statistics.custom[key].append(value)

            # Selection
            selected_pop, selected_fitness = self.components.selection.select(
                self.population,
                self.fitness,
                **self.components.selection_params,
            )

            if verbose:
                print(f"  Selected {len(selected_pop)} individuals for learning")

            # Learning
            self.model = self.components.learning.learn(
                self.generation,
                self.n_vars,
                self.cardinality,
                selected_pop,
                selected_fitness,
                **self.components.learning_params,
            )

            # Cache data if requested
            if cache_config.cache_populations:
                cache.populations.append(self.population.copy())
            if cache_config.cache_fitness:
                cache.fitness_values.append(self.fitness.copy())
            if cache_config.cache_models:
                cache.models.append(self.model)
            if cache_config.cache_statistics and self.components.statistics is not None:
                cache.statistics.append(custom_stats.copy())
            if cache_config.cache_selections:
                cache.selected_populations.append(selected_pop.copy())

            # Check stopping condition
            continue_evolution = not self.components.stop_condition.should_stop(
                self.generation,
                self.population,
                self.fitness,
                **self.components.stop_params,
            )

            self.generation += 1

        if verbose:
            print(f"\nEDA completed after {self.generation} generations")
            print(f"Best fitness found: {statistics.best_fitness_overall:.6f}")
            print(f"  at generation {statistics.generation_found}")

        return statistics, cache

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "EDA":
        """
        Create EDA from configuration dictionary

        Args:
            config: Configuration dictionary with keys:
                   - pop_size, n_vars, fitness_func, cardinality
                   - components: dict mapping component types to configs

        Returns:
            Configured EDA instance
        """
        # TODO: Implement configuration-based initialization
        raise NotImplementedError("from_config not yet implemented")
