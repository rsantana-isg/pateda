"""
Two-Point Crossover for EDAs

Implements the classic two-point crossover operator in the EDA framework,
split into learning and sampling phases.

Two-Point Crossover:
--------------------
A genetic algorithm recombination operator that selects two random crossover
points and exchanges the middle segment between two parent solutions.

For parents P1 and P2 with crossover points at positions i and j (i < j):
- Offspring 1: P1[0:i] + P2[i:j] + P1[j:end]
- Offspring 2: P2[0:i] + P1[i:j] + P2[j:end]

EDA Implementation:
-------------------
Following the non-probabilistic model paradigm from MATEDA-2.0 Section 4.5,
this operator is split into two phases:

1. Learning Phase (LearnTwoPointCrossover):
   - Randomly selects N/2 parent pairs from the selected population
   - For each pair, randomly selects two crossover points (point1 < point2)
   - Stores this information in a model

2. Sampling Phase (SampleTwoPointCrossover):
   - Applies the two-point crossover using the learned parameters
   - Creates two offspring per parent pair (N/2 pairs → N offspring)
   - Optionally applies mutation operator to offspring

Advantages:
-----------
- Preserves more building blocks than one-point crossover
- Can recombine genes that are far apart in the genome
- Well-suited for problems with multiple independent sub-problems

Equivalent to MATEDA's LearnTwoPointCrossover.m and SampleTwoPointCrossover.m

References:
-----------
- De Jong, K. A., & Spears, W. M. (1992). "A formal analysis of the role of
  multi-point crossover in genetic algorithms." Annals of Mathematics and AI.
- MATEDA-2.0 User Guide, Section 4.5: "Non probabilistic models"
- Last MATLAB version: 12/21/2020. Roberto Santana (roberto.santana@ehu.es)
"""

from typing import Any, Optional, Callable
import numpy as np

from pateda.core.components import LearningMethod, SamplingMethod
from pateda.core.models import Model


class LearnTwoPointCrossover(LearningMethod):
    """
    Learn two-point crossover model from selected population

    Creates a crossover plan by randomly selecting parent pairs and
    two crossover points for each pair.
    """

    def __init__(self, n_offspring: int):
        """
        Initialize two-point crossover learning

        Args:
            n_offspring: Number of offspring to generate (must be even)
        """
        if n_offspring % 2 != 0:
            raise ValueError(f"n_offspring must be even, got {n_offspring}")
        self.n_offspring = n_offspring

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
        Learn two-point crossover model from population

        Args:
            generation: Current generation number
            n_vars: Number of variables
            cardinality: Variable cardinalities (not used)
            population: Selected population to sample parents from
            fitness: Fitness values (not used)
            **params: Additional parameters
                     - n_offspring: Override instance n_offspring

        Returns:
            Model containing mating pool and crossover points
            - structure: Mating pool matrix (n_pairs, 2) with parent indices
            - parameters: Points matrix (n_pairs, 2) with crossover points
        """
        n_offspring = params.get("n_offspring", self.n_offspring)
        if n_offspring % 2 != 0:
            raise ValueError(f"n_offspring must be even, got {n_offspring}")

        n_sel = population.shape[0]  # Number of selected individuals
        n_pairs = n_offspring // 2   # Number of parent pairs

        # Randomly select parent pairs
        # Each row: [parent1_idx, parent2_idx]
        # They may coincide but this is highly unlikely and causes no harm
        mating_pool = np.random.randint(0, n_sel, size=(n_pairs, 2))

        # Randomly select two crossover points for each pair
        # Ensure point1 < point2
        # Point1: random position in [0, n_vars)
        points = np.zeros((n_pairs, 2), dtype=int)
        points[:, 0] = np.random.randint(0, n_vars, size=n_pairs)

        # Point2: random position in (point1, n_vars]
        # For each pair, select from remaining positions after point1
        for i in range(n_pairs):
            remaining = n_vars - points[i, 0]
            if remaining > 0:
                points[i, 1] = points[i, 0] + np.random.randint(1, remaining + 1)
            else:
                points[i, 1] = points[i, 0]

        # Create and return model
        model = Model(
            structure=mating_pool,
            parameters=points,
            metadata={
                "generation": generation,
                "model_type": "TwoPointCrossover",
                "n_offspring": n_offspring,
            },
        )

        return model


class SampleTwoPointCrossover(SamplingMethod):
    """
    Sample new population using two-point crossover

    Applies two-point crossover according to the learned model and
    optionally applies a mutation operator to the offspring.
    """

    def __init__(
        self,
        n_samples: int,
        mutation_fn: Optional[Callable] = None,
        mutation_params: Optional[dict] = None,
    ):
        """
        Initialize two-point crossover sampling

        Args:
            n_samples: Number of samples to generate (must be even)
            mutation_fn: Optional mutation function to apply after crossover
                        Signature: mutation_fn(n_vars, cardinality, population, params) -> population
            mutation_params: Parameters to pass to mutation function
        """
        if n_samples % 2 != 0:
            raise ValueError(f"n_samples must be even, got {n_samples}")
        self.n_samples = n_samples
        self.mutation_fn = mutation_fn
        self.mutation_params = mutation_params or {}

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
        Sample new population using two-point crossover

        Args:
            n_vars: Number of variables
            model: TwoPointCrossover model from learning phase
            cardinality: Variable cardinalities
            aux_pop: Auxiliary population to select parents from (required)
            aux_fitness: Auxiliary fitness (not used)
            **params: Additional parameters
                     - n_samples: Override instance n_samples
                     - mutation_fn: Override instance mutation function
                     - mutation_params: Override instance mutation parameters

        Returns:
            New population generated by two-point crossover (n_samples, n_vars)
        """
        if aux_pop is None:
            raise ValueError("aux_pop is required for two-point crossover sampling")

        n_samples = params.get("n_samples", self.n_samples)
        if n_samples % 2 != 0:
            raise ValueError(f"n_samples must be even, got {n_samples}")

        # Extract model components
        mating_pool = model.structure  # (n_pairs, 2)
        points = model.parameters      # (n_pairs, 2)

        n_pairs = n_samples // 2
        new_pop = np.zeros((n_samples, n_vars), dtype=aux_pop.dtype)

        # Generate two offspring from each parent pair
        for i in range(n_pairs):
            parent1_idx = mating_pool[i, 0]
            parent2_idx = mating_pool[i, 1]
            point1 = points[i, 0]
            point2 = points[i, 1]

            parent1 = aux_pop[parent1_idx]
            parent2 = aux_pop[parent2_idx]

            # First offspring: P1[0:p1] + P2[p1:p2] + P1[p2:end]
            new_pop[i, :point1] = parent1[:point1]
            new_pop[i, point1:point2] = parent2[point1:point2]
            new_pop[i, point2:] = parent1[point2:]

            # Second offspring: P2[0:p1] + P1[p1:p2] + P2[p2:end]
            offspring2_idx = i + n_pairs
            new_pop[offspring2_idx, :point1] = parent2[:point1]
            new_pop[offspring2_idx, point1:point2] = parent1[point1:point2]
            new_pop[offspring2_idx, point2:] = parent2[point2:]

        # Apply mutation if specified
        mutation_fn = params.get("mutation_fn", self.mutation_fn)
        mutation_params = params.get("mutation_params", self.mutation_params)

        if mutation_fn is not None:
            new_pop = mutation_fn(n_vars, cardinality, new_pop, mutation_params)

        return new_pop
