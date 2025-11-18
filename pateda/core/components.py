"""
Component interfaces for EDAs

This module defines the abstract base classes for all EDA components,
replacing MATLAB's eval()-based dynamic dispatch with a type-safe
component architecture.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np

from pateda.core.models import Model


class SeedingMethod(ABC):
    """Abstract base class for population seeding methods"""

    @abstractmethod
    def seed(
        self,
        n_vars: int,
        pop_size: int,
        cardinality: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        """
        Generate initial population

        Args:
            n_vars: Number of variables
            pop_size: Population size
            cardinality: Variable cardinalities (discrete) or ranges (continuous)
            **params: Additional method-specific parameters

        Returns:
            Initial population as (pop_size, n_vars) array
        """
        pass


class LearningMethod(ABC):
    """Abstract base class for probabilistic model learning methods"""

    @abstractmethod
    def learn(
        self,
        generation: int,
        n_vars: int,
        cardinality: np.ndarray,
        population: np.ndarray,
        fitness: np.ndarray,
        **params: Any,
    ) -> Model:
        """
        Learn probabilistic model from population

        Args:
            generation: Current generation number
            n_vars: Number of variables
            cardinality: Variable cardinalities or ranges
            population: Selected population to learn from
            fitness: Fitness values (may be used by some methods)
            **params: Additional method-specific parameters

        Returns:
            Learned probabilistic model
        """
        pass


class SamplingMethod(ABC):
    """Abstract base class for sampling methods"""

    @abstractmethod
    def sample(
        self,
        n_vars: int,
        model: Model,
        cardinality: np.ndarray,
        aux_pop: Optional[np.ndarray] = None,
        aux_fitness: Optional[np.ndarray] = None,
        **params: Any,
    ) -> np.ndarray:
        """
        Sample new population from probabilistic model

        Args:
            n_vars: Number of variables
            model: Probabilistic model to sample from
            cardinality: Variable cardinalities or ranges
            aux_pop: Auxiliary population (for partial sampling/resampling)
            aux_fitness: Auxiliary fitness values
            **params: Additional method-specific parameters

        Returns:
            Sampled population as (n_samples, n_vars) array
        """
        pass


class SelectionMethod(ABC):
    """Abstract base class for selection methods"""

    @abstractmethod
    def select(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        n_select: Optional[int] = None,
        **params: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select individuals from population

        Args:
            population: Population to select from
            fitness: Fitness values
            n_select: Number of individuals to select (None = use method default)
            **params: Additional method-specific parameters

        Returns:
            Tuple of (selected_population, selected_fitness)
        """
        pass


class ReplacementMethod(ABC):
    """Abstract base class for replacement strategies"""

    @abstractmethod
    def replace(
        self,
        old_pop: np.ndarray,
        old_fitness: np.ndarray,
        new_pop: np.ndarray,
        new_fitness: np.ndarray,
        **params: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Combine old and new populations

        Args:
            old_pop: Previous generation population
            old_fitness: Previous generation fitness
            new_pop: Newly sampled population
            new_fitness: New population fitness
            **params: Additional method-specific parameters

        Returns:
            Tuple of (combined_population, combined_fitness)
        """
        pass


class StopCondition(ABC):
    """Abstract base class for stopping conditions"""

    @abstractmethod
    def should_stop(
        self,
        generation: int,
        population: np.ndarray,
        fitness: np.ndarray,
        **params: Any,
    ) -> bool:
        """
        Check if evolution should stop

        Args:
            generation: Current generation number
            population: Current population
            fitness: Current fitness values
            **params: Additional method-specific parameters

        Returns:
            True if evolution should stop, False otherwise
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the stopping condition state"""
        pass


class LocalOptMethod(ABC):
    """Abstract base class for local optimization methods"""

    @abstractmethod
    def optimize(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        fitness_func: Callable,
        cardinality: np.ndarray,
        **params: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply local optimization to population

        Args:
            population: Population to optimize
            fitness: Current fitness values
            fitness_func: Fitness evaluation function
            cardinality: Variable cardinalities or ranges
            **params: Additional method-specific parameters

        Returns:
            Tuple of (optimized_population, optimized_fitness)
        """
        pass


class RepairingMethod(ABC):
    """Abstract base class for constraint repair methods"""

    @abstractmethod
    def repair(
        self,
        population: np.ndarray,
        cardinality: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        """
        Repair invalid solutions

        Args:
            population: Population to repair
            cardinality: Variable cardinalities or ranges
            **params: Additional method-specific parameters

        Returns:
            Repaired population
        """
        pass


class StatisticsMethod(ABC):
    """Abstract base class for statistics collection methods"""

    @abstractmethod
    def collect(
        self,
        generation: int,
        population: np.ndarray,
        fitness: np.ndarray,
        model: Optional[Model] = None,
        **params: Any,
    ) -> Dict[str, Any]:
        """
        Collect statistics for current generation

        Args:
            generation: Current generation number
            population: Current population
            fitness: Current fitness values
            model: Current probabilistic model (if available)
            **params: Additional method-specific parameters

        Returns:
            Dictionary of statistics
        """
        pass


@dataclass
class EDAComponents:
    """
    Container for all EDA components

    This replaces MATLAB's edaparams cell array with a typed structure.
    """

    # Required components
    seeding: Optional[SeedingMethod] = None
    learning: Optional[LearningMethod] = None
    sampling: Optional[SamplingMethod] = None
    selection: Optional[SelectionMethod] = None
    stop_condition: Optional[StopCondition] = None

    # Optional components with defaults
    replacement: Optional[ReplacementMethod] = None
    local_opt: Optional[LocalOptMethod] = None
    repairing: Optional[RepairingMethod] = None
    statistics: Optional[StatisticsMethod] = None

    # Component parameters (method-specific)
    seeding_params: Dict[str, Any] = field(default_factory=dict)
    learning_params: Dict[str, Any] = field(default_factory=dict)
    sampling_params: Dict[str, Any] = field(default_factory=dict)
    selection_params: Dict[str, Any] = field(default_factory=dict)
    replacement_params: Dict[str, Any] = field(default_factory=dict)
    stop_params: Dict[str, Any] = field(default_factory=dict)
    local_opt_params: Dict[str, Any] = field(default_factory=dict)
    repairing_params: Dict[str, Any] = field(default_factory=dict)
    statistics_params: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate that required components are present"""
        required = ["seeding", "learning", "sampling", "selection", "stop_condition"]
        missing = [name for name in required if getattr(self, name) is None]
        if missing:
            raise ValueError(f"Missing required EDA components: {missing}")


@dataclass
class CacheConfig:
    """
    Configuration for caching EDA execution data

    Equivalent to MATEDA's cache parameter [c1, c2, c3, c4, c5]
    """

    cache_populations: bool = False  # Cache all populations
    cache_fitness: bool = False  # Cache all fitness evaluations
    cache_models: bool = False  # Cache probabilistic models
    cache_statistics: bool = False  # Cache statistics
    cache_selections: bool = False  # Cache selected populations

    @classmethod
    def from_list(cls, cache_list: list) -> "CacheConfig":
        """Create from MATEDA-style list [c1, c2, c3, c4, c5]"""
        if len(cache_list) != 5:
            raise ValueError("cache_list must have exactly 5 elements")
        return cls(
            cache_populations=bool(cache_list[0]),
            cache_fitness=bool(cache_list[1]),
            cache_models=bool(cache_list[2]),
            cache_statistics=bool(cache_list[3]),
            cache_selections=bool(cache_list[4]),
        )

    def to_list(self) -> list:
        """Convert to MATEDA-style list"""
        return [
            int(self.cache_populations),
            int(self.cache_fitness),
            int(self.cache_models),
            int(self.cache_statistics),
            int(self.cache_selections),
        ]
