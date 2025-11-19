"""
Truncation selection

Truncation selection is the most common selection method used in Estimation of
Distribution Algorithms (EDAs). It selects the top N individuals based solely
on their fitness values, creating strong selection pressure toward the best solutions.

Selection in EDAs:
Selection is a critical component of EDAs that determines which individuals are
used to learn the probabilistic model. Unlike traditional genetic algorithms where
selection is used for mating, in EDAs selection determines the "training set" for
model learning.

Truncation Selection Algorithm:
1. Sort population by fitness (descending for maximization)
2. Select the top N% (typically 50%) of individuals
3. These selected individuals are used to learn the probability distribution
4. Discard the remaining individuals

Selection Pressure:
Truncation selection provides deterministic, strong selection pressure:
- Selection pressure = ratio of best to worst selected individual's fitness
- No randomness in who gets selected (unlike tournament, proportional)
- Sharp cutoff between selected and non-selected
- Typically selects top 30-50% of population

Advantages:
- Simple and deterministic
- Clear interpretation: "learn from the best"
- No randomness in selection (reproducible)
- Works well with elitism (always keeps best solutions)
- Computationally efficient (just sort and select)

Disadvantages:
- Very strong selection pressure can reduce diversity
- Sharp cutoff may discard useful building blocks in slightly worse solutions
- No chance for lower-fitness individuals (unlike tournament)
- Can lead to premature convergence if selection ratio too aggressive

Selection Ratio Guidelines:
- ratio = 0.5 (50%): Standard, balanced exploration-exploitation
- ratio = 0.3 (30%): Stronger selection, faster convergence, higher risk of premature convergence
- ratio = 0.7 (70%): Weaker selection, maintains more diversity, slower convergence
- Optimal ratio depends on problem difficulty and population size

Relationship to Model Learning:
The selected individuals form the "dataset" for learning the probabilistic model:
- Larger selection ratio → More data for learning → More reliable model
- Smaller selection ratio → Less data → Higher variance in model
- Trade-off: selection pressure vs. model quality

Multi-objective Optimization:
For multi-objective problems, truncation selection typically:
- Uses scalar fitness (e.g., mean across objectives)
- Alternative: non-dominated sorting (Pareto-based selection)
- Can use reference point methods or decomposition approaches

Comparison with Other Selection Methods:
- Tournament: Stochastic, adjustable pressure via tournament size
- Proportional (Roulette): Fitness-proportional probability, can be weak pressure
- Ranking: Based on rank rather than raw fitness values
- Boltzmann: Temperature-controlled selection pressure

Use in EDA Framework:
Truncation selection is used between evaluation and learning:
1. Evaluate: Compute fitness of all individuals
2. Select: Use truncation to select promising subset
3. Learn: Learn probabilistic model from selected individuals
4. Sample: Generate new population from model

Equivalent to MATEDA's truncation_selection.m

References:
- Mühlenbein, H., & Paass, G. (1996). "From recombination of genes to the
  estimation of distributions I. Binary parameters." PPSN IV, pp. 178-187.
- Pelikan, M., Goldberg, D.E., & Lobo, F. (2002). "A survey of optimization by
  building and using probabilistic models." Computational Optimization and
  Applications, 21(1):5-20.
- MATEDA-2.0 User Guide, Section 5.6: "Selection of promising solutions"
"""

from typing import Any, Optional, Tuple
import numpy as np

from pateda.core.components import SelectionMethod


class TruncationSelection(SelectionMethod):
    """
    Truncation selection: Select top N individuals by fitness

    This is the most common selection method for EDAs, selecting the best
    individuals based on their fitness values.
    """

    def __init__(self, ratio: float = 0.5, n_select: Optional[int] = None):
        """
        Initialize truncation selection

        Args:
            ratio: Fraction of population to select (used if n_select is None)
            n_select: Exact number of individuals to select (overrides ratio)
        """
        self.ratio = ratio
        self.n_select = n_select

    def select(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        n_select: Optional[int] = None,
        **params: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select top individuals by fitness

        Args:
            population: Population to select from (pop_size, n_vars)
            fitness: Fitness values (pop_size,) or (pop_size, n_objectives)
                    For multi-objective, uses mean fitness across objectives
            n_select: Number to select (overrides instance n_select)
            **params: Additional parameters
                     - ratio: Override instance ratio

        Returns:
            Tuple of (selected_population, selected_fitness)
        """
        pop_size = population.shape[0]

        # Determine number to select
        if n_select is None:
            n_select = self.n_select

        if n_select is None:
            ratio = params.get("ratio", self.ratio)
            n_select = max(1, int(pop_size * ratio))

        # Ensure we don't select more than available
        n_select = min(n_select, pop_size)

        # Handle multi-objective fitness by taking mean
        if fitness.ndim == 2 and fitness.shape[1] > 1:
            fitness_for_selection = np.mean(fitness, axis=1)
        elif fitness.ndim == 2:
            fitness_for_selection = fitness[:, 0]
        else:
            fitness_for_selection = fitness

        # Get indices of top individuals (sorted by fitness, descending)
        sorted_indices = np.argsort(fitness_for_selection)[::-1]
        selected_indices = sorted_indices[:n_select]

        # Select individuals
        selected_pop = population[selected_indices]
        selected_fitness = fitness[selected_indices]

        return selected_pop, selected_fitness
