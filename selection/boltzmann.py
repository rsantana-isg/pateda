"""
Boltzmann selection

Equivalent to MATEDA's Boltzmann selection methods
"""

from typing import Any, Optional, Tuple
import numpy as np

from pateda.core.components import SelectionMethod


class BoltzmannSelection(SelectionMethod):
    """
    Boltzmann selection

    Selection probabilities follow a Boltzmann distribution based on fitness.
    Temperature parameter controls selection pressure, which can be
    adapted over generations (simulated annealing).
    """

    def __init__(
        self,
        n_select: Optional[int] = None,
        ratio: float = 0.5,
        temperature: float = 1.0,
        temperature_schedule: Optional[callable] = None,
        replacement: bool = True,
    ):
        """
        Initialize Boltzmann selection

        Args:
            n_select: Number of individuals to select (None = use ratio)
            ratio: Fraction of population to select (used if n_select is None)
            temperature: Initial temperature parameter (controls selection pressure)
                        Higher temperature = more random selection
                        Lower temperature = stronger selection pressure
            temperature_schedule: Optional function(generation) -> temperature
                                 for adaptive temperature
            replacement: Whether to allow selecting same individual multiple times
        """
        self.n_select = n_select
        self.ratio = ratio
        self.temperature = temperature
        self.temperature_schedule = temperature_schedule
        self.replacement = replacement

    def select(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        n_select: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        **params: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select individuals using Boltzmann selection

        Args:
            population: Population to select from (pop_size, n_vars)
            fitness: Fitness values (pop_size,) or (pop_size, n_objectives)
                    For multi-objective, uses mean fitness across objectives
            n_select: Number to select (overrides instance n_select)
            rng: Random number generator (None = create default generator)
            **params: Additional parameters
                     - ratio: Override instance ratio
                     - temperature: Override instance temperature
                     - generation: Current generation (for temperature scheduling)
                     - replacement: Override instance replacement

        Returns:
            Tuple of (selected_population, selected_fitness)
        """
        if rng is None:
            rng = np.random.default_rng()

        pop_size = population.shape[0]

        # Determine number to select
        if n_select is None:
            n_select = self.n_select

        if n_select is None:
            ratio = params.get("ratio", self.ratio)
            n_select = max(1, int(pop_size * ratio))

        replacement = params.get("replacement", self.replacement)

        # Ensure we don't select more than available without replacement
        if not replacement:
            n_select = min(n_select, pop_size)

        # Handle multi-objective fitness by taking mean
        if fitness.ndim == 2 and fitness.shape[1] > 1:
            fitness_for_selection = np.mean(fitness, axis=1)
        elif fitness.ndim == 2:
            fitness_for_selection = fitness[:, 0]
        else:
            fitness_for_selection = fitness

        # Get temperature
        temperature = params.get("temperature", self.temperature)

        # Apply temperature schedule if provided
        if self.temperature_schedule is not None and "generation" in params:
            temperature = self.temperature_schedule(params["generation"])

        # Avoid division by zero
        if temperature <= 0:
            raise ValueError("Temperature must be positive")

        # Normalize fitness to avoid numerical issues
        # Subtract mean to center around zero
        normalized_fitness = fitness_for_selection - np.mean(fitness_for_selection)

        # Calculate Boltzmann probabilities: P(i) ∝ exp(f(i) / T)
        try:
            # Use log-sum-exp trick for numerical stability
            log_probs = normalized_fitness / temperature
            log_probs = log_probs - np.max(log_probs)  # Subtract max for stability
            probabilities = np.exp(log_probs)
            probabilities = probabilities / np.sum(probabilities)
        except (OverflowError, RuntimeWarning):
            # If still numerical issues, fall back to truncation
            print("Warning: Numerical issues in Boltzmann selection, using truncation")
            sorted_indices = np.argsort(fitness_for_selection)[::-1]
            selected_indices = sorted_indices[:n_select]
            selected_pop = population[selected_indices]
            selected_fitness = fitness[selected_indices]
            return selected_pop, selected_fitness

        # Select individuals
        selected_indices = rng.choice(
            pop_size, size=n_select, replace=replacement, p=probabilities
        )

        selected_pop = population[selected_indices]
        selected_fitness = fitness[selected_indices]

        return selected_pop, selected_fitness
