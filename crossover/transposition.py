"""
Transposition Operator for EDAs

Implements a genetic algorithm transposition operator in the EDA framework,
split into learning and sampling phases.

Transposition Operator:
-----------------------
A variation operator that rearranges the genome by moving a segment to a
different location. The operator works as follows:

1. Randomly choose a transposition length len ∈ [1, n/2]
2. Randomly choose a starting location loc ∈ [0, n)
3. Randomly choose an offset o ∈ [1, n-len]
4. Exchange the segment [loc:loc+len] with [loc+o:loc+o+len]
   using toroidal wrapping (genome is treated as circular)

This operator is particularly effective for problems with symmetry, as it can
move building blocks to different positions while preserving their internal
structure.

EDA Implementation:
-------------------
Following the non-probabilistic model paradigm from MATEDA-2.0 Section 4.5,
this operator is split into two phases:

1. Learning Phase (LearnTransposition):
   - For each of N offspring to generate:
     * Randomly selects a parent from the selected population
     * Randomly determines transposition length (1 to n/2)
     * Randomly determines starting location (0 to n-1)
     * Randomly determines offset
   - Stores all parameters in a model

2. Sampling Phase (SampleTransposition):
   - For each offspring, applies transposition to selected parent
   - Uses toroidal indexing for wrap-around
   - Optionally applies mutation operator to offspring

Toroidal Wrapping:
------------------
The genome is treated as circular (toroid), so indices wrap around:
- If loc + len > n, wraps to beginning
- This allows transpositions that cross the boundary

Applications:
-------------
- Problems with circular or periodic structure
- Symmetric optimization problems
- Reordering building blocks to find better combinations

Equivalent to MATEDA's LearnTransposition.m and SampleTransposition.m

References:
-----------
- Santana, R., McKay, B., & Lozano, M. (2013). "Symmetry in evolutionary and
  estimation of distribution algorithms." IEEE CEC 2013.
- MATEDA-2.0 User Guide, Section 4.5: "Non probabilistic models"
- Last MATLAB version: 12/21/2020. Roberto Santana (roberto.santana@ehu.es)
"""

from typing import Any, Optional, Callable
import numpy as np

from pateda.core.components import LearningMethod, SamplingMethod
from pateda.core.models import Model


class LearnTransposition(LearningMethod):
    """
    Learn transposition model from selected population

    Creates a transposition plan by randomly selecting parameters
    (parent, length, location, offset) for each offspring.
    """

    def __init__(self, n_offspring: int):
        """
        Initialize transposition learning

        Args:
            n_offspring: Number of offspring to generate
        """
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
        Learn transposition model from population

        Args:
            generation: Current generation number
            n_vars: Number of variables
            cardinality: Variable cardinalities (not used)
            population: Selected population to sample parents from
            fitness: Fitness values (not used)
            **params: Additional parameters
                     - n_offspring: Override instance n_offspring

        Returns:
            Model containing transposition parameters
            - structure: Dictionary with transposition parameters
              * individuals: Parent indices for each offspring (n_offspring,)
              * lengths: Transposition lengths (n_offspring,)
              * locations: Starting locations (n_offspring,)
              * offsets: Offsets for transposition (n_offspring,)
        """
        n_offspring = params.get("n_offspring", self.n_offspring)
        n_sel = population.shape[0]  # Number of selected individuals

        # Randomly select parent for each offspring
        trans_individuals = np.random.randint(0, n_sel, size=n_offspring)

        # Randomly select transposition length: [1, n_vars/2]
        lengths = np.random.randint(1, n_vars // 2 + 1, size=n_offspring)

        # Randomly select starting location: [0, n_vars)
        locations = np.random.randint(0, n_vars, size=n_offspring)

        # Randomly select offset: [1, n_vars - length]
        # Offset determines how far the transposed segment moves
        diff_len_nvars = n_vars - lengths
        offsets = np.random.randint(1, diff_len_nvars + 1, size=n_offspring)

        # Create and return model
        # Store parameters as dictionary in structure for clarity
        model = Model(
            structure={
                "individuals": trans_individuals,
                "lengths": lengths,
                "locations": locations,
                "offsets": offsets,
            },
            parameters=None,  # All parameters stored in structure
            metadata={
                "generation": generation,
                "model_type": "Transposition",
                "n_offspring": n_offspring,
            },
        )

        return model


class SampleTransposition(SamplingMethod):
    """
    Sample new population using transposition operator

    Applies transposition according to the learned model and
    optionally applies a mutation operator to the offspring.
    """

    def __init__(
        self,
        n_samples: int,
        mutation_fn: Optional[Callable] = None,
        mutation_params: Optional[dict] = None,
    ):
        """
        Initialize transposition sampling

        Args:
            n_samples: Number of samples to generate
            mutation_fn: Optional mutation function to apply after transposition
                        Signature: mutation_fn(n_vars, cardinality, population, params) -> population
            mutation_params: Parameters to pass to mutation function
        """
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
        Sample new population using transposition

        Args:
            n_vars: Number of variables
            model: Transposition model from learning phase
            cardinality: Variable cardinalities
            aux_pop: Auxiliary population to select parents from (required)
            aux_fitness: Auxiliary fitness (not used)
            **params: Additional parameters
                     - n_samples: Override instance n_samples
                     - mutation_fn: Override instance mutation function
                     - mutation_params: Override instance mutation parameters

        Returns:
            New population generated by transposition (n_samples, n_vars)
        """
        if aux_pop is None:
            raise ValueError("aux_pop is required for transposition sampling")

        n_samples = params.get("n_samples", self.n_samples)

        # Extract model components
        trans_params = model.structure
        trans_individuals = trans_params["individuals"]
        lengths = trans_params["lengths"]
        locations = trans_params["locations"]
        offsets = trans_params["offsets"]

        new_pop = np.zeros((n_samples, n_vars), dtype=aux_pop.dtype)

        # Create toroidal structure by duplicating the population
        # This simplifies wrap-around indexing
        double_pop = np.concatenate([aux_pop, aux_pop], axis=1)

        # Apply transposition to each offspring
        for i in range(n_samples):
            ind_idx = trans_individuals[i]
            loc = locations[i]
            length = lengths[i]
            offset = offsets[i]

            # Start with copy of parent
            individual = aux_pop[ind_idx].copy()

            # Apply transposition using toroidal wrapping
            # Exchange segment at loc with segment at loc+offset
            for j in range(length):
                # Position in the individual (with wrapping)
                if loc + j >= n_vars:
                    pos = loc + j - n_vars
                else:
                    pos = loc + j

                # Source position in doubled array (toroidal indexing)
                # The source comes from loc+offset+j in the toroidal structure
                individual[pos] = double_pop[ind_idx, loc + offset + j]

            new_pop[i] = individual

        # Apply mutation if specified
        mutation_fn = params.get("mutation_fn", self.mutation_fn)
        mutation_params = params.get("mutation_params", self.mutation_params)

        if mutation_fn is not None:
            new_pop = mutation_fn(n_vars, cardinality, new_pop, mutation_params)

        return new_pop
