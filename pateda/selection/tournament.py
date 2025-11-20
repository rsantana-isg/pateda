"""
Tournament selection

Tournament selection is a popular stochastic selection method that provides
adjustable selection pressure through the tournament size parameter. It's widely
used in both traditional genetic algorithms and EDAs.

Tournament Selection Algorithm:
1. Repeat n_select times:
   a. Randomly select k individuals from population (tournament)
   b. Choose the individual with best fitness from the tournament
   c. Add winner to selected set
2. Return selected individuals

Tournament Size and Selection Pressure:
The tournament size k controls selection pressure:
- k = 2: Moderate selection pressure (most common)
- k = 1: Random selection (no pressure)
- k = pop_size: Equivalent to selecting best individual repeatedly
- Larger k → Stronger selection pressure
- Typical range: k ∈ {2, 3, 4, 5, 7}

Selection Pressure Analysis:
For a population sorted by fitness (1 = best, N = worst), the probability that
individual i wins a tournament of size k is approximately:
    P(i wins) ∝ (N - i + 1)^k - (N - i)^k

This shows that larger k exponentially increases advantage of better individuals.

Advantages:
- Simple to implement and understand
- Adjustable selection pressure via tournament size
- Works well without requiring fitness scaling
- Maintains diversity better than truncation
- No issues with negative fitness values
- Parallelizable (tournaments are independent)

Disadvantages:
- Stochastic (less reproducible than truncation)
- May select same individual multiple times (with replacement)
- Weaker individuals have non-zero selection probability
- Requires more random number generation than truncation

Replacement vs. Non-replacement:
With replacement (default):
- Same individual can be selected multiple times
- Consistent selection pressure across all tournaments
- Can select more individuals than population size

Without replacement:
- Each individual selected at most once
- Ensures diversity in selected set
- Cannot select more than population size individuals
- Selection pressure decreases as available pool shrinks

Comparison with Truncation Selection:
Tournament:
- Stochastic: different runs give different selections
- Gradual selection pressure (not sharp cutoff)
- Lower-ranked individuals have small but non-zero probability
- Better diversity preservation

Truncation:
- Deterministic: always selects same individuals
- Sharp cutoff between selected and non-selected
- Zero probability for individuals below threshold
- Stronger selection pressure

When to Use Tournament vs. Truncation:
Use Tournament when:
- Want to maintain more population diversity
- Need adjustable selection pressure (tuning k)
- Fitness landscape is deceptive or has local optima
- Running many independent trials (stochastic nature helps)

Use Truncation when:
- Want reproducible, deterministic selection
- Prefer simplicity and computational efficiency
- Population is large enough that diversity isn't a concern
- Strong selection pressure is beneficial

Use in EDAs:
While truncation is most common in EDAs, tournament selection offers advantages:
- Better diversity in selected individuals → more robust model learning
- Stochastic nature can help escape local optima
- Adjustable pressure allows adaptive selection strategies
- Can be combined with elitism for best of both worlds

Multi-objective Optimization:
For multi-objective problems, tournament selection can use:
- Scalar fitness (e.g., mean across objectives)
- Pareto dominance: tournament winner is non-dominated individual
- Random objective: select random objective for each tournament
- Crowding distance: prefer individuals in less crowded regions

Adaptive Tournament Size:
Some EDAs use adaptive tournament size:
- Start with small k (weak pressure) for exploration
- Increase k over time (stronger pressure) for exploitation
- Can base k on population diversity or convergence metrics

Equivalent to MATEDA's tournament selection methods

References:
- Goldberg, D.E., & Deb, K. (1991). "A comparative analysis of selection schemes
  used in genetic algorithms." Foundations of Genetic Algorithms, pp. 69-93.
- Miller, B.L., & Goldberg, D.E. (1995). "Genetic algorithms, tournament selection,
  and the effects of noise." Complex Systems, 9(3):193-212.
- Larrañaga, P., & Lozano, J.A. (Eds.). (2002). "Estimation of Distribution
  Algorithms: A New Tool for Evolutionary Computation." Kluwer Academic.
- MATEDA-2.0 User Guide, Section 5.6: "Selection of promising solutions"
"""

from typing import Any, Optional, Tuple
import numpy as np

from pateda.core.components import SelectionMethod


class TournamentSelection(SelectionMethod):
    """
    Tournament selection: Select individuals by running tournaments

    In each tournament, k individuals are randomly selected and the best
    one is chosen. This process is repeated until n_select individuals
    are selected.
    """

    def __init__(
        self,
        tournament_size: int = 2,
        n_select: Optional[int] = None,
        ratio: float = 0.5,
        replacement: bool = True,
    ):
        """
        Initialize tournament selection

        Args:
            tournament_size: Number of individuals per tournament
            n_select: Number of individuals to select (None = use ratio)
            ratio: Fraction of population to select (used if n_select is None)
            replacement: Whether to allow selecting same individual multiple times
        """
        self.tournament_size = tournament_size
        self.n_select = n_select
        self.ratio = ratio
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
        Select individuals using tournament selection

        Args:
            population: Population to select from (pop_size, n_vars)
            fitness: Fitness values (pop_size,) or (pop_size, n_objectives)
                    For multi-objective, uses mean fitness across objectives
            n_select: Number to select (overrides instance n_select)
            rng: Random number generator (None = create default generator)
            **params: Additional parameters
                     - tournament_size: Override instance tournament_size
                     - ratio: Override instance ratio
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

        tournament_size = params.get("tournament_size", self.tournament_size)
        replacement = params.get("replacement", self.replacement)

        # Ensure tournament size is valid
        tournament_size = min(tournament_size, pop_size)

        # Handle multi-objective fitness by taking mean
        if fitness.ndim == 2 and fitness.shape[1] > 1:
            fitness_for_selection = np.mean(fitness, axis=1)
        elif fitness.ndim == 2:
            fitness_for_selection = fitness[:, 0]
        else:
            fitness_for_selection = fitness

        selected_indices = []

        if replacement:
            # With replacement: can select same individual multiple times
            for _ in range(n_select):
                # Randomly select tournament participants
                tournament_indices = rng.choice(
                    pop_size, size=tournament_size, replace=False
                )

                # Get best individual from tournament
                tournament_fitness = fitness_for_selection[tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                selected_indices.append(winner_idx)
        else:
            # Without replacement: each individual can only be selected once
            available = set(range(pop_size))

            for _ in range(min(n_select, pop_size)):
                if len(available) < tournament_size:
                    # Not enough individuals left for full tournament
                    tournament_indices = list(available)
                else:
                    tournament_indices = rng.choice(
                        list(available), size=tournament_size, replace=False
                    )

                # Get best individual from tournament
                tournament_fitness = fitness_for_selection[tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                selected_indices.append(winner_idx)
                available.remove(winner_idx)

        selected_indices = np.array(selected_indices)
        selected_pop = population[selected_indices]
        selected_fitness = fitness[selected_indices]

        return selected_pop, selected_fitness
