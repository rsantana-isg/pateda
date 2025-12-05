"""
Local optimization for the contiguous block problem

This module implements a local search strategy specifically designed for
the contiguous block problem, where the goal is to arrange k ones in a
binary vector to form a contiguous block.

The strategy iteratively tries to move ones closer together to form
a contiguous sequence.
"""

from typing import Any, Callable, Tuple
import numpy as np

from pateda.core.components import LocalOptMethod


class ContiguousBlockOptimizer(LocalOptMethod):
    """
    Local optimization for contiguous block problem.

    This optimizer uses a greedy strategy to move scattered ones into a
    contiguous block. It iteratively finds isolated ones (or small groups)
    and tries to move them adjacent to existing blocks.

    The algorithm:
    1. Find all positions with ones
    2. Identify the largest contiguous block
    3. Try to move isolated ones to be adjacent to the main block
    4. Accept moves that improve or maintain fitness

    Attributes:
        max_iterations: Maximum number of improvement iterations per solution
        aggressive: If True, also try swapping ones with zeros to improve contiguity
    """

    def __init__(self, max_iterations: int = 10, aggressive: bool = True):
        """
        Initialize contiguous block optimizer.

        Args:
            max_iterations: Maximum iterations of local improvement per solution
            aggressive: If True, try more aggressive moves (swaps)
        """
        self.max_iterations = max_iterations
        self.aggressive = aggressive

    def optimize(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        fitness_func: Callable,
        cardinality: np.ndarray,
        **params: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply local optimization to improve contiguous block structure.

        Args:
            population: Population to optimize (n_individuals, n_vars)
            fitness: Current fitness values (n_individuals,) or (n_individuals, 1)
            fitness_func: Fitness evaluation function
            cardinality: Variable cardinalities (not used for binary)
            **params: Additional parameters
                max_iterations: Override default max_iterations
                aggressive: Override default aggressive setting

        Returns:
            Tuple of (optimized_population, optimized_fitness)
        """
        n_individuals, n_vars = population.shape
        max_iterations = params.get('max_iterations', self.max_iterations)
        aggressive = params.get('aggressive', self.aggressive)

        new_pop = population.copy()

        # Handle both 1D and 2D fitness arrays
        if fitness.ndim == 2:
            new_fitness = fitness[:, 0].copy()
            fitness_is_2d = True
        else:
            new_fitness = fitness.copy()
            fitness_is_2d = False

        for i in range(n_individuals):
            improved = True
            iteration = 0

            while improved and iteration < max_iterations:
                improved = False
                iteration += 1
                current_x = new_pop[i, :].copy()
                current_fitness = new_fitness[i]

                # Find positions of ones
                ones_positions = np.where(current_x == 1)[0]

                if len(ones_positions) <= 1:
                    # Already optimal (0 or 1 one)
                    break

                # Find the largest contiguous block
                block_start, block_end = self._find_largest_block(current_x)
                block_size = block_end - block_start + 1

                if block_size == len(ones_positions):
                    # Already a perfect contiguous block
                    break

                # Try to extend the block by moving isolated ones
                for pos in ones_positions:
                    if block_start <= pos <= block_end:
                        continue  # Already in the main block

                    # Try to move this one to be adjacent to the block
                    candidate_x = current_x.copy()

                    # Find nearest available position (zero) near the block
                    if pos < block_start:
                        # One is before the block, try to move it to block_start - 1
                        target_pos = block_start - 1
                        if target_pos >= 0 and candidate_x[target_pos] == 0:
                            # Swap: move the one to target_pos
                            candidate_x[pos] = 0
                            candidate_x[target_pos] = 1

                            # Evaluate
                            candidate_fitness = fitness_func(candidate_x)

                            if candidate_fitness >= current_fitness:
                                current_x = candidate_x
                                current_fitness = candidate_fitness
                                improved = True
                                # Update block boundaries
                                block_start = target_pos
                                break

                    elif pos > block_end:
                        # One is after the block, try to move it to block_end + 1
                        target_pos = block_end + 1
                        if target_pos < n_vars and candidate_x[target_pos] == 0:
                            # Swap: move the one to target_pos
                            candidate_x[pos] = 0
                            candidate_x[target_pos] = 1

                            # Evaluate
                            candidate_fitness = fitness_func(candidate_x)

                            if candidate_fitness >= current_fitness:
                                current_x = candidate_x
                                current_fitness = candidate_fitness
                                improved = True
                                # Update block boundaries
                                block_end = target_pos
                                break

                # More aggressive strategy: try random swaps of ones and zeros
                if aggressive and not improved:
                    # Try a few random swaps
                    for _ in range(min(5, len(ones_positions))):
                        candidate_x = current_x.copy()

                        # Pick a random one outside the main block
                        outside_ones = [p for p in ones_positions if not (block_start <= p <= block_end)]
                        if not outside_ones:
                            break

                        swap_one_pos = np.random.choice(outside_ones)

                        # Find a zero near the block to swap with
                        zeros_near_block = []
                        if block_start > 0 and candidate_x[block_start - 1] == 0:
                            zeros_near_block.append(block_start - 1)
                        if block_end < n_vars - 1 and candidate_x[block_end + 1] == 0:
                            zeros_near_block.append(block_end + 1)

                        if zeros_near_block:
                            swap_zero_pos = np.random.choice(zeros_near_block)

                            # Perform swap
                            candidate_x[swap_one_pos] = 0
                            candidate_x[swap_zero_pos] = 1

                            # Evaluate
                            candidate_fitness = fitness_func(candidate_x)

                            if candidate_fitness >= current_fitness:
                                current_x = candidate_x
                                current_fitness = candidate_fitness
                                improved = True
                                break

                # Update solution if improved
                if improved:
                    new_pop[i, :] = current_x
                    new_fitness[i] = current_fitness

        # Convert back to 2D if needed
        if fitness_is_2d:
            new_fitness = new_fitness.reshape(-1, 1)

        return new_pop, new_fitness

    def _find_largest_block(self, x: np.ndarray) -> Tuple[int, int]:
        """
        Find the start and end positions of the largest contiguous block of ones.

        Args:
            x: Binary vector

        Returns:
            Tuple of (start_position, end_position) of the largest block
            Returns (0, -1) if no ones are found
        """
        if np.sum(x) == 0:
            return (0, -1)

        max_length = 0
        max_start = 0
        max_end = -1

        current_length = 0
        current_start = 0

        for i, bit in enumerate(x):
            if bit == 1:
                if current_length == 0:
                    current_start = i
                current_length += 1
            else:
                if current_length > max_length:
                    max_length = current_length
                    max_start = current_start
                    max_end = i - 1
                current_length = 0

        # Check if the last block is the longest
        if current_length > max_length:
            max_start = current_start
            max_end = len(x) - 1

        return (max_start, max_end)

    def __repr__(self) -> str:
        return f"ContiguousBlockOptimizer(max_iterations={self.max_iterations}, aggressive={self.aggressive})"
