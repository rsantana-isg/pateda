"""
Diffusion-by-Deblending EDA (DbD-EDA) Examples

This module demonstrates the four variants of DbD-EDA:
- DbD-CS: Current population to Selected population
- DbD-CD: Current population to Distance-matched selected
- DbD-UC: Univariate current to Current population
- DbD-US: Univariate current to Selected population

Based on the paper: "Learning search distributions in estimation of distribution
algorithms with minimalist diffusion models"
"""

import numpy as np
import time
from typing import Dict, Any, Optional, Callable
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from learning.dbd import (
    learn_dbd,
    find_closest_neighbors,
    sample_univariate_gaussian
)
from sampling.dbd import sample_dbd, sample_dbd_from_univariate


class DbDEDA:
    """
    Base class for Diffusion-by-Deblending EDAs with restart mechanism.
    """

    def __init__(
        self,
        variant: str = 'CS',
        pop_size: int = 200,
        selection_ratio: float = 0.3,
        dbd_params: Optional[Dict[str, Any]] = None,
        restart_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize DbD-EDA.

        Parameters
        ----------
        variant : str
            Variant to use: 'CS', 'CD', 'UC', or 'US'
        pop_size : int
            Population size
        selection_ratio : float
            Ratio of population to select
        dbd_params : dict, optional
            Parameters for DbD model:
            - 'num_alpha_samples': alpha samples per pair (default: 10)
            - 'hidden_dims': hidden layer dimensions (default: [64, 64])
            - 'epochs': training epochs (default: 50)
            - 'batch_size': batch size (default: 32)
            - 'learning_rate': learning rate (default: 1e-3)
            - 'num_iterations': sampling iterations (default: 10)
        restart_params : dict, optional
            Parameters for restart mechanism:
            - 'trigger_no_improvement': generations without improvement (default: 5)
            - 'diversity_threshold': minimum diversity threshold (default: 1e-6)
            - 'keep_best': number of best solutions to keep (default: 2)
        """
        self.variant = variant.upper()
        assert self.variant in ['CS', 'CD', 'UC', 'US'], \
            f"Variant must be one of ['CS', 'CD', 'UC', 'US'], got {variant}"

        self.pop_size = pop_size
        self.selection_size = int(pop_size * selection_ratio)

        # DbD parameters
        if dbd_params is None:
            dbd_params = {}
        self.dbd_params = {
            'num_alpha_samples': dbd_params.get('num_alpha_samples', 10),
            'hidden_dims': dbd_params.get('hidden_dims', [64, 64]),
            'epochs': dbd_params.get('epochs', 50),
            'batch_size': dbd_params.get('batch_size', 32),
            'learning_rate': dbd_params.get('learning_rate', 1e-3),
            'normalize': True
        }
        self.num_iterations = dbd_params.get('num_iterations', 10)

        # Restart parameters
        if restart_params is None:
            restart_params = {}
        self.trigger_no_improvement = restart_params.get('trigger_no_improvement', 5)
        self.diversity_threshold = restart_params.get('diversity_threshold', 1e-6)
        self.keep_best = restart_params.get('keep_best', 2)

        # Tracking variables
        self.generations_without_improvement = 0
        self.best_fitness = np.inf
        self.best_solution = None

    def _prepare_distributions(
        self,
        population: np.ndarray,
        selected_population: np.ndarray
    ) -> tuple:
        """
        Prepare p0 and p1 distributions based on variant.

        Parameters
        ----------
        population : np.ndarray
            Current population
        selected_population : np.ndarray
            Selected population

        Returns
        -------
        p0, p1 : tuple of np.ndarray
            Source and target distributions
        """
        p_size = len(population)
        sel_p_size = len(selected_population)
        to_take = p_size * 2  # Number of samples for training

        if self.variant == 'CS':
            # DbD-CS: Current to Selected
            p0_indices = np.random.randint(0, p_size, size=to_take)
            p0 = population[p0_indices]

            p1_indices = np.random.randint(0, sel_p_size, size=to_take)
            p1 = selected_population[p1_indices]

        elif self.variant == 'CD':
            # DbD-CD: Current to Distance-matched selected
            p0_indices = np.random.randint(0, p_size, size=to_take)
            p0 = population[p0_indices]

            # Find closest neighbors in selected population
            p1 = find_closest_neighbors(p0, selected_population)

        elif self.variant == 'UC':
            # DbD-UC: Univariate current to Current
            p0 = sample_univariate_gaussian(population, to_take)

            p1_indices = np.random.randint(0, p_size, size=to_take)
            p1 = population[p1_indices]

        elif self.variant == 'US':
            # DbD-US: Univariate current to Selected
            p0 = sample_univariate_gaussian(population, to_take)

            p1_indices = np.random.randint(0, sel_p_size, size=to_take)
            p1 = selected_population[p1_indices]

        return p0, p1

    def _get_sampling_p0(
        self,
        population: np.ndarray,
        selected_population: np.ndarray
    ) -> np.ndarray:
        """
        Get p0 for sampling based on variant.

        Parameters
        ----------
        population : np.ndarray
            Current population
        selected_population : np.ndarray
            Selected population

        Returns
        -------
        p0 : np.ndarray
            Source distribution for sampling
        """
        if self.variant in ['CS', 'CD']:
            # Start from selected population
            return selected_population
        else:  # UC or US
            # Start from univariate approximation of selected
            return sample_univariate_gaussian(selected_population, self.pop_size)

    def _check_restart_condition(
        self,
        selected_fitness: np.ndarray,
        current_best: float
    ) -> bool:
        """
        Check if restart should be triggered.

        Parameters
        ----------
        selected_fitness : np.ndarray
            Fitness values of selected solutions
        current_best : float
            Current best fitness

        Returns
        -------
        should_restart : bool
            True if restart should be triggered
        """
        # Check diversity (std of selected fitness excluding best few)
        sel_diversity = np.std(selected_fitness[self.keep_best:])

        # Check if diversity is too low or no improvement for too long
        return (sel_diversity < self.diversity_threshold or
                self.generations_without_improvement >= self.trigger_no_improvement)

    def optimize(
        self,
        fitness_function: Callable,
        n_vars: int,
        bounds: np.ndarray,
        n_generations: int,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run DbD-EDA optimization.

        Parameters
        ----------
        fitness_function : callable
            Function to minimize
        n_vars : int
            Number of variables
        bounds : np.ndarray
            Array of shape (2, n_vars) with [min, max] bounds
        n_generations : int
            Number of generations
        verbose : bool
            Whether to print progress

        Returns
        -------
        results : dict
            Optimization results containing:
            - 'best_fitness': best fitness found
            - 'best_solution': best solution found
            - 'fitness_history': fitness over generations
            - 'learning_times': time spent learning
            - 'sampling_times': time spent sampling
        """
        # Initialize population
        population = np.random.uniform(bounds[0], bounds[1], (self.pop_size, n_vars))

        # Tracking
        fitness_history = []
        learning_times = []
        sampling_times = []

        for gen in range(n_generations):
            # Evaluate
            fitness = np.array([fitness_function(ind) for ind in population])

            # Track best
            best_idx = np.argmin(fitness)
            gen_best_fitness = fitness[best_idx]
            fitness_history.append(gen_best_fitness)

            if verbose:
                print(f"Generation {gen+1:3d}: Best fitness = {gen_best_fitness:.6e}, "
                      f"No improvement = {self.generations_without_improvement}")

            # Update global best
            if gen_best_fitness < self.best_fitness:
                self.best_fitness = gen_best_fitness
                self.best_solution = population[best_idx].copy()
                self.generations_without_improvement = 0
            else:
                self.generations_without_improvement += 1

            # Select best individuals
            idx = np.argsort(fitness)[:self.selection_size]
            selected_population = population[idx]
            selected_fitness = fitness[idx]

            # Check restart condition
            if self._check_restart_condition(selected_fitness, gen_best_fitness):
                if verbose:
                    print(f"  -> Restart triggered at generation {gen+1}")

                # Reinitialize population
                population = np.random.uniform(bounds[0], bounds[1], (self.pop_size, n_vars))

                # Keep best solutions
                population[:self.keep_best] = selected_population[:self.keep_best]

                # Reset counter
                self.generations_without_improvement = 0

                # Re-evaluate after restart
                continue

            # Prepare distributions based on variant
            p0_train, p1_train = self._prepare_distributions(population, selected_population)

            # Learn model
            start_time = time.time()
            model = learn_dbd(p0_train, p1_train, params=self.dbd_params)
            learning_time = time.time() - start_time
            learning_times.append(learning_time)

            # Get p0 for sampling
            p0_sample = self._get_sampling_p0(population, selected_population)

            # Sample new population
            start_time = time.time()
            new_population = sample_dbd(
                model,
                p0_sample,
                self.pop_size,
                bounds=bounds,
                params={'num_iterations': self.num_iterations}
            )
            sampling_time = time.time() - start_time
            sampling_times.append(sampling_time)

            # Replace population
            population = new_population

        # Final evaluation
        final_fitness = np.array([fitness_function(ind) for ind in population])
        final_best_idx = np.argmin(final_fitness)
        if final_fitness[final_best_idx] < self.best_fitness:
            self.best_fitness = final_fitness[final_best_idx]
            self.best_solution = population[final_best_idx]

        return {
            'best_fitness': self.best_fitness,
            'best_solution': self.best_solution,
            'fitness_history': fitness_history,
            'learning_times': learning_times,
            'sampling_times': sampling_times
        }


# ==================== Test Functions ====================

def sphere_function(x):
    """Simple sphere function: f(x) = sum(x_i^2)"""
    return np.sum(x**2)


def rosenbrock_function(x):
    """Rosenbrock function"""
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


def rastrigin_function(x):
    """Rastrigin function"""
    n = len(x)
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def ackley_function(x):
    """Ackley function"""
    n = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + 20 + np.e


# ==================== Examples ====================

def test_single_variant(variant='CS', n_vars=10, n_generations=30):
    """Test a single DbD-EDA variant."""
    print(f"\n{'='*80}")
    print(f"Testing DbD-{variant} on Sphere Function")
    print(f"{'='*80}\n")

    bounds = np.array([[-5.0] * n_vars, [5.0] * n_vars])

    eda = DbDEDA(
        variant=variant,
        pop_size=200,
        selection_ratio=0.3,
        dbd_params={
            'num_alpha_samples': 10,
            'hidden_dims': [64, 64],
            'epochs': 30,
            'batch_size': 32,
            'learning_rate': 1e-3,
            'num_iterations': 10
        },
        restart_params={
            'trigger_no_improvement': 5,
            'diversity_threshold': 1e-6,
            'keep_best': 2
        }
    )

    result = eda.optimize(
        sphere_function,
        n_vars,
        bounds,
        n_generations,
        verbose=True
    )

    print(f"\n{'='*80}")
    print(f"RESULTS for DbD-{variant}")
    print(f"{'='*80}")
    print(f"Initial best fitness: {result['fitness_history'][0]:.6e}")
    print(f"Final best fitness:   {result['best_fitness']:.6e}")
    print(f"Improvement:          {result['fitness_history'][0] / max(result['best_fitness'], 1e-10):.2f}x")
    print(f"Avg learning time:    {np.mean(result['learning_times']):.3f}s")
    print(f"Avg sampling time:    {np.mean(result['sampling_times']):.3f}s")
    print(f"{'='*80}\n")

    return result


def compare_all_variants(n_vars=10, n_generations=25):
    """Compare all four DbD-EDA variants."""
    print(f"\n{'#'*80}")
    print(f"# COMPARING ALL DbD-EDA VARIANTS")
    print(f"{'#'*80}\n")

    variants = ['CS', 'CD', 'UC', 'US']
    results = {}

    for variant in variants:
        result = test_single_variant(variant, n_vars, n_generations)
        results[variant] = result

    # Print comparison
    print(f"\n{'='*80}")
    print(f"VARIANT COMPARISON")
    print(f"{'='*80}")
    print(f"{'Variant':<10} {'Initial':<15} {'Final':<15} {'Improvement':<12} {'Avg Time/Gen'}")
    print(f"{'-'*80}")
    for variant in variants:
        r = results[variant]
        initial = r['fitness_history'][0]
        final = r['best_fitness']
        improvement = initial / max(final, 1e-10)
        avg_time = np.mean(r['learning_times']) + np.mean(r['sampling_times'])
        print(f"DbD-{variant:<6} {initial:<15.6e} {final:<15.6e} {improvement:<12.2f} {avg_time:.3f}s")
    print(f"{'='*80}\n")


def test_benchmark_functions():
    """Test DbD-CS on multiple benchmark functions."""
    print(f"\n{'#'*80}")
    print(f"# TESTING DbD-CS ON BENCHMARK FUNCTIONS")
    print(f"{'#'*80}\n")

    n_vars = 10
    n_generations = 25

    benchmarks = [
        ('Sphere', sphere_function, np.array([[-5.0] * n_vars, [5.0] * n_vars])),
        ('Rosenbrock', rosenbrock_function, np.array([[-5.0] * n_vars, [5.0] * n_vars])),
        ('Rastrigin', rastrigin_function, np.array([[-5.12] * n_vars, [5.12] * n_vars])),
        ('Ackley', ackley_function, np.array([[-5.0] * n_vars, [5.0] * n_vars]))
    ]

    for name, func, bounds in benchmarks:
        print(f"\n{'='*80}")
        print(f"Testing on {name} Function")
        print(f"{'='*80}\n")

        eda = DbDEDA(variant='CS', pop_size=200, selection_ratio=0.3)
        result = eda.optimize(func, n_vars, bounds, n_generations, verbose=False)

        print(f"Initial: {result['fitness_history'][0]:.6e}")
        print(f"Final:   {result['best_fitness']:.6e}")
        print(f"Improvement: {result['fitness_history'][0] / max(result['best_fitness'], 1e-10):.2f}x")


def main():
    """Run all DbD-EDA tests and demonstrations."""
    print(f"\n{'#'*80}")
    print(f"# DIFFUSION-BY-DEBLENDING EDA (DbD-EDA) - COMPREHENSIVE TESTS")
    print(f"{'#'*80}")

    # Test 1: Single variant
    print("\n" + "="*80)
    print("TEST 1: DbD-CS Variant")
    print("="*80)
    test_single_variant('CS', n_vars=10, n_generations=30)

    # Test 2: Compare all variants
    print("\n" + "="*80)
    print("TEST 2: Compare All Variants")
    print("="*80)
    compare_all_variants(n_vars=10, n_generations=25)

    # Test 3: Benchmark functions
    print("\n" + "="*80)
    print("TEST 3: Benchmark Functions")
    print("="*80)
    test_benchmark_functions()

    print(f"\n{'#'*80}")
    print(f"# ALL TESTS COMPLETED SUCCESSFULLY!")
    print(f"{'#'*80}\n")


if __name__ == '__main__':
    main()
