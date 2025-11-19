"""
Block Crossover for EDAs

Implements a block-based crossover operator that exchanges complete blocks
of variables defined by equivalence classes (e.g., symmetry groups).

Block Crossover (Symmetric Blind Block Crossover):
---------------------------------------------------
A genetic algorithm recombination operator designed for problems with
symmetry or grouped variables. Instead of exchanging individual genes,
it exchanges complete blocks of variables.

Key Concepts:
- Variables are grouped into equivalence classes (blocks)
- Each class typically represents a symmetric component or related variables
- For each block, randomly decide whether to take it from parent 1 or parent 2
- Creates two complementary offspring from each parent pair

Example:
If we have 4 blocks and the crossover mask is [0, 1, 0, 1]:
- Offspring 1: [P1_block0, P2_block1, P1_block2, P2_block3]
- Offspring 2: [P2_block0, P1_block1, P2_block2, P1_block3]

EDA Implementation:
-------------------
Following the non-probabilistic model paradigm from MATEDA-2.0 Section 4.5,
this operator is split into two phases:

1. Learning Phase (LearnBlockCrossover):
   - Randomly selects N/2 parent pairs from the selected population
   - For each pair, generates a random binary mask for each equivalence class
   - Stores mating pool and crossover masks in a model

2. Sampling Phase (SampleBlockCrossover):
   - Applies block crossover using the learned parameters
   - For each equivalence class, swaps complete blocks based on mask
   - Creates two offspring per parent pair (N/2 pairs → N offspring)
   - Optionally applies bit-flip mutation to offspring

Advantages:
-----------
- Preserves symmetry and structural properties
- Effective for problems with groupings or modular structure
- Better than uniform crossover for maintaining building blocks
- Well-suited for grid-based problems (TSP, knapsack variants, etc.)

Equivalent to MATEDA's LearnSymmetricBlindBlockCrossover.m and
SampleSymmetricBlindBlockCrossover.m

References:
-----------
- Santana, R., McKay, B., & Lozano, M. (2013). "Symmetry in evolutionary and
  estimation of distribution algorithms." IEEE CEC 2013.
- MATEDA-2.0 User Guide, Section 4.5: "Non probabilistic models"
- Last MATLAB version: 12/21/2020. Roberto Santana (roberto.santana@ehu.es)
"""

from typing import Any, Optional
import numpy as np

from pateda.core.components import LearningMethod, SamplingMethod
from pateda.core.models import Model


class LearnBlockCrossover(LearningMethod):
    """
    Learn block crossover model from selected population

    Creates a crossover plan by randomly selecting parent pairs and
    binary masks for each equivalence class.
    """

    def __init__(self, n_offspring: int, symmetry_index: np.ndarray):
        """
        Initialize block crossover learning

        Args:
            n_offspring: Number of offspring to generate (must be even)
            symmetry_index: Matrix defining equivalence classes (n_classes, class_size)
                           Each row contains the variable indices for one class
        """
        if n_offspring % 2 != 0:
            raise ValueError(f"n_offspring must be even, got {n_offspring}")
        self.n_offspring = n_offspring
        self.symmetry_index = symmetry_index
        self.n_classes = symmetry_index.shape[0]
        self.class_size = symmetry_index.shape[1]

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
        Learn block crossover model from population

        Args:
            generation: Current generation number
            n_vars: Number of variables
            cardinality: Variable cardinalities (not used)
            population: Selected population to sample parents from
            fitness: Fitness values (not used)
            **params: Additional parameters
                     - n_offspring: Override instance n_offspring
                     - symmetry_index: Override instance symmetry_index

        Returns:
            Model containing mating pool and crossover masks
            - structure: Mating pool matrix (n_pairs, 2) with parent indices
            - parameters: Dictionary with:
              * masks: Binary masks (n_pairs, n_classes) for block selection
              * symmetry_index: The symmetry index used
        """
        n_offspring = params.get("n_offspring", self.n_offspring)
        symmetry_index = params.get("symmetry_index", self.symmetry_index)

        if n_offspring % 2 != 0:
            raise ValueError(f"n_offspring must be even, got {n_offspring}")

        n_sel = population.shape[0]  # Number of selected individuals
        n_pairs = n_offspring // 2   # Number of parent pairs
        n_classes = symmetry_index.shape[0]

        # Generate random binary crossover masks for each pair and each class
        # 0 means take block from first parent, 1 means take from second parent
        crossover_masks = np.random.randint(0, 2, size=(n_pairs, n_classes))

        # Randomly select parent pairs
        # Each row: [parent1_idx, parent2_idx]
        mating_pool = np.random.randint(0, n_sel, size=(n_pairs, 2))

        # Create and return model
        model = Model(
            structure=mating_pool,
            parameters={
                "masks": crossover_masks,
                "symmetry_index": symmetry_index,
            },
            metadata={
                "generation": generation,
                "model_type": "BlockCrossover",
                "n_offspring": n_offspring,
                "n_classes": n_classes,
            },
        )

        return model


class SampleBlockCrossover(SamplingMethod):
    """
    Sample new population using block crossover

    Applies block crossover according to the learned model and
    optionally applies bit-flip mutation to the offspring.
    """

    def __init__(
        self,
        n_samples: int,
        symmetry_index: np.ndarray,
        mutation_prob: float = 0.0,
    ):
        """
        Initialize block crossover sampling

        Args:
            n_samples: Number of samples to generate (must be even)
            symmetry_index: Matrix defining equivalence classes (n_classes, class_size)
            mutation_prob: Probability of bit-flip mutation for each variable
        """
        if n_samples % 2 != 0:
            raise ValueError(f"n_samples must be even, got {n_samples}")
        self.n_samples = n_samples
        self.symmetry_index = symmetry_index
        self.mutation_prob = mutation_prob

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
        Sample new population using block crossover

        Args:
            n_vars: Number of variables
            model: BlockCrossover model from learning phase
            cardinality: Variable cardinalities
            aux_pop: Auxiliary population to select parents from (required)
            aux_fitness: Auxiliary fitness (not used)
            **params: Additional parameters
                     - n_samples: Override instance n_samples
                     - symmetry_index: Override instance symmetry_index
                     - mutation_prob: Override instance mutation probability

        Returns:
            New population generated by block crossover (n_samples, n_vars)
        """
        if aux_pop is None:
            raise ValueError("aux_pop is required for block crossover sampling")

        n_samples = params.get("n_samples", self.n_samples)
        symmetry_index = params.get("symmetry_index", self.symmetry_index)
        mutation_prob = params.get("mutation_prob", self.mutation_prob)

        if n_samples % 2 != 0:
            raise ValueError(f"n_samples must be even, got {n_samples}")

        # Extract model components
        mating_pool = model.structure  # (n_pairs, 2)
        masks = model.parameters["masks"]  # (n_pairs, n_classes)
        # Use symmetry_index from model if available, otherwise use instance
        model_symmetry = model.parameters.get("symmetry_index", symmetry_index)

        n_pairs = n_samples // 2
        n_classes = model_symmetry.shape[0]
        new_pop = np.zeros((n_samples, n_vars), dtype=aux_pop.dtype)

        # Apply block crossover for each equivalence class
        # This is done class by class, not pair by pair, for efficiency
        for class_idx in range(n_classes):
            # Get variable indices for this equivalence class
            var_indices = model_symmetry[class_idx, :]

            # Find pairs where mask is 0 (take from first parent)
            pairs_mask_0 = np.where(masks[:, class_idx] == 0)[0]
            # Find pairs where mask is 1 (take from second parent)
            pairs_mask_1 = np.where(masks[:, class_idx] == 1)[0]

            # For offspring 1 (indices 0 to n_pairs-1):
            # - If mask=0: take from parent1
            # - If mask=1: take from parent2
            if len(pairs_mask_0) > 0:
                parent1_indices = mating_pool[pairs_mask_0, 0]
                new_pop[pairs_mask_0[:, None], var_indices] = aux_pop[
                    parent1_indices[:, None], var_indices
                ]

            if len(pairs_mask_1) > 0:
                parent2_indices = mating_pool[pairs_mask_1, 1]
                new_pop[pairs_mask_1[:, None], var_indices] = aux_pop[
                    parent2_indices[:, None], var_indices
                ]

            # For offspring 2 (indices n_pairs to n_samples-1):
            # - If mask=0: take from parent2
            # - If mask=1: take from parent1
            offspring2_start = n_pairs
            if len(pairs_mask_0) > 0:
                parent2_indices = mating_pool[pairs_mask_0, 1]
                offspring2_indices = pairs_mask_0 + offspring2_start
                new_pop[offspring2_indices[:, None], var_indices] = aux_pop[
                    parent2_indices[:, None], var_indices
                ]

            if len(pairs_mask_1) > 0:
                parent1_indices = mating_pool[pairs_mask_1, 0]
                offspring2_indices = pairs_mask_1 + offspring2_start
                new_pop[offspring2_indices[:, None], var_indices] = aux_pop[
                    parent1_indices[:, None], var_indices
                ]

        # Apply bit-flip mutation if specified
        if mutation_prob > 0:
            # Generate mutation mask
            mutation_mask = np.random.rand(n_samples, n_vars) < mutation_prob
            # Flip bits where mask is True (assumes binary variables)
            new_pop[mutation_mask] = 1 - new_pop[mutation_mask]

        return new_pop
