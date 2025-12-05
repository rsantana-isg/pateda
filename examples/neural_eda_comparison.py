"""
Neural Network-Based EDA Comparison
====================================

This module provides a comprehensive comparison of four neural network-based
Estimation of Distribution Algorithms (EDAs):

1. **RBM-EDA**: Restricted Boltzmann Machine EDA
   - Uses softmax RBM for discrete variables or Gaussian RBM for continuous
   - Energy-based probabilistic model
   - Trained using contrastive divergence

2. **VAE-EDA**: Variational Autoencoder EDA
   - Encoder-decoder architecture with latent space
   - Probabilistic generation through reparameterization trick
   - Variants: Basic VAE, Extended VAE (E-VAE), Conditional Extended VAE (CE-VAE)

3. **Backdrive-EDA**: Back-driven Network Inversion EDA
   - Trains MLP to predict fitness from solutions
   - Generates new solutions by backpropagating high fitness values
   - Network inversion for solution generation

4. **GAN-EDA**: Generative Adversarial Network EDA
   - Generator-discriminator adversarial training
   - Generator learns to create realistic samples
   - Sampling from random latent vectors

Based on analysis in Santana (2017), neural models offer different trade-offs
compared to traditional probabilistic graphical models:
- Advantages: Flexible, efficient learning, GPU parallelization
- Disadvantages: Sampling complexity, parameter sensitivity, potential overfitting

This comparison helps identify which neural approach works best for different
types of optimization problems.
"""

import numpy as np
import time
from typing import Dict, Any, Optional, Callable, List
import warnings

# Neural network learning functions
from pateda.learning.rbm import learn_rbm
from pateda.learning.vae import learn_vae, learn_extended_vae, learn_conditional_extended_vae
from pateda.learning.backdrive import learn_backdrive
from pateda.learning.gan import learn_gan

# Neural network sampling functions
from pateda.sampling.rbm import sample_rbm
from pateda.sampling.vae import sample_vae, sample_extended_vae, sample_conditional_extended_vae
from pateda.sampling.backdrive import sample_backdrive
from pateda.sampling.gan import sample_gan


# ==================== Benchmark Functions ====================

def sphere_function(x):
    """
    Sphere function: f(x) = sum(x_i^2)
    Global minimum: f(0,...,0) = 0
    Domain: typically [-5, 5]^n
    """
    if x.ndim == 1:
        return np.sum(x**2)
    return np.sum(x**2, axis=1)


def rosenbrock_function(x):
    """
    Rosenbrock function: f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
    Global minimum: f(1,...,1) = 0
    Domain: typically [-5, 5]^n
    """
    if x.ndim == 1:
        return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    return np.sum(100 * (x[:, 1:] - x[:, :-1]**2)**2 + (1 - x[:, :-1])**2, axis=1)


def rastrigin_function(x):
    """
    Rastrigin function: f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))
    Global minimum: f(0,...,0) = 0
    Domain: typically [-5.12, 5.12]^n
    Highly multimodal
    """
    if x.ndim == 1:
        n = len(x)
        return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
    n = x.shape[1]
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=1)


def ackley_function(x):
    """
    Ackley function
    Global minimum: f(0,...,0) = 0
    Domain: typically [-5, 5]^n
    Highly multimodal
    """
    if x.ndim == 1:
        n = len(x)
        sum_sq = np.sum(x**2)
        sum_cos = np.sum(np.cos(2 * np.pi * x))
        return -20 * np.exp(-0.2 * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + 20 + np.e

    n = x.shape[1]
    sum_sq = np.sum(x**2, axis=1)
    sum_cos = np.sum(np.cos(2 * np.pi * x), axis=1)
    return -20 * np.exp(-0.2 * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + 20 + np.e


def griewank_function(x):
    """
    Griewank function
    Global minimum: f(0,...,0) = 0
    Domain: typically [-600, 600]^n
    """
    if x.ndim == 1:
        n = len(x)
        sum_sq = np.sum(x**2)
        prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, n+1))))
        return 1 + sum_sq / 4000 - prod_cos

    sum_sq = np.sum(x**2, axis=1)
    prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, x.shape[1]+1))), axis=1)
    return 1 + sum_sq / 4000 - prod_cos


# ==================== Base Neural EDA Class ====================

class NeuralEDA:
    """
    Base class for neural network-based EDAs with restart mechanism.
    """

    def __init__(
        self,
        method: str,
        pop_size: int = 100,
        selection_ratio: float = 0.3,
        neural_params: Optional[Dict[str, Any]] = None,
        restart_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Neural EDA.

        Parameters
        ----------
        method : str
            Neural method to use: 'rbm', 'vae', 'extended_vae', 'conditional_vae',
            'backdrive', or 'gan'
        pop_size : int
            Population size
        selection_ratio : float
            Ratio of population to select
        neural_params : dict, optional
            Parameters for neural network training
        restart_params : dict, optional
            Parameters for restart mechanism
        """
        self.method = method.lower()
        valid_methods = ['rbm', 'vae', 'extended_vae', 'conditional_vae', 'backdrive', 'gan']
        assert self.method in valid_methods, \
            f"Method must be one of {valid_methods}, got {method}"

        self.pop_size = pop_size
        self.selection_size = int(pop_size * selection_ratio)

        # Neural network parameters with reasonable defaults
        if neural_params is None:
            neural_params = {}

        self.neural_params = self._get_default_params(neural_params)

        # Restart parameters
        if restart_params is None:
            restart_params = {}
        self.trigger_no_improvement = restart_params.get('trigger_no_improvement', 10)
        self.diversity_threshold = restart_params.get('diversity_threshold', 1e-8)
        self.keep_best = restart_params.get('keep_best', 2)

        # Tracking variables
        self.generations_without_improvement = 0
        self.best_fitness = np.inf
        self.best_solution = None
        self.previous_model = None

    def _get_default_params(self, user_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get default parameters for the specific neural method."""
        defaults = {
            'rbm': {
                'n_hidden': 50,
                'epochs': 30,
                'batch_size': 16,
                'learning_rate': 0.01,
                'k_gibbs': 1
            },
            'vae': {
                'latent_dim': None,  # Will be set based on n_vars
                'hidden_dims': [64, 32],
                'epochs': 30,
                'batch_size': 16,
                'learning_rate': 0.001
            },
            'extended_vae': {
                'latent_dim': None,
                'hidden_dims': [64, 32],
                'epochs': 30,
                'batch_size': 16,
                'learning_rate': 0.001,
                'fitness_weight': 0.1
            },
            'conditional_vae': {
                'latent_dim': None,
                'hidden_dims': [64, 32],
                'epochs': 30,
                'batch_size': 16,
                'learning_rate': 0.001,
                'fitness_weight': 0.1
            },
            'backdrive': {
                'hidden_layers': [100, 100],
                'activation': 'tanh',
                'epochs': 50,
                'batch_size': 32,
                'learning_rate': 0.001,
                'backdrive_iterations': 100,
                'backdrive_lr': 0.1
            },
            'gan': {
                'latent_dim': None,
                'generator_dims': [32, 64],
                'discriminator_dims': [64, 32],
                'epochs': 50,
                'batch_size': 32,
                'learning_rate': 0.0002,
                'd_steps': 1
            }
        }

        # Get defaults for this method
        params = defaults.get(self.method, {}).copy()

        # Override with user params
        params.update(user_params)

        return params

    def _check_restart_condition(
        self,
        selected_fitness: np.ndarray,
        current_best: float
    ) -> bool:
        """Check if restart should be triggered."""
        # Check diversity
        if len(selected_fitness) > self.keep_best:
            sel_diversity = np.std(selected_fitness[self.keep_best:])
        else:
            sel_diversity = np.std(selected_fitness)

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
        Run Neural EDA optimization.

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
            Optimization results
        """
        # Set latent_dim based on n_vars if not specified
        if self.neural_params.get('latent_dim') is None:
            if self.method in ['vae', 'extended_vae', 'conditional_vae']:
                self.neural_params['latent_dim'] = max(2, n_vars // 2)
            elif self.method == 'gan':
                self.neural_params['latent_dim'] = max(2, n_vars // 2)

        # Initialize population
        population = np.random.uniform(bounds[0], bounds[1], (self.pop_size, n_vars))

        # Tracking
        fitness_history = []
        learning_times = []
        sampling_times = []
        n_evaluations = 0
        n_restarts = 0

        for gen in range(n_generations):
            # Evaluate
            fitness = fitness_function(population)
            n_evaluations += len(population)

            # Track best
            best_idx = np.argmin(fitness)
            gen_best_fitness = fitness[best_idx]
            fitness_history.append(gen_best_fitness)

            if verbose:
                print(f"Generation {gen+1:3d}: Best = {gen_best_fitness:.6e}, "
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

                population = np.random.uniform(bounds[0], bounds[1], (self.pop_size, n_vars))
                population[:self.keep_best] = selected_population[:self.keep_best]

                self.generations_without_improvement = 0
                n_restarts += 1
                continue

            # Learn model
            start_time = time.time()
            model = self._learn_model(
                gen, n_vars, bounds, selected_population, selected_fitness
            )
            learning_time = time.time() - start_time
            learning_times.append(learning_time)

            # Sample new population
            start_time = time.time()
            new_population = self._sample_model(
                model, n_vars, bounds, selected_fitness
            )
            sampling_time = time.time() - start_time
            sampling_times.append(sampling_time)

            # Replace population
            population = new_population

            # Store model for next generation
            self.previous_model = model

        # Final evaluation
        final_fitness = fitness_function(population)
        n_evaluations += len(population)
        final_best_idx = np.argmin(final_fitness)
        if final_fitness[final_best_idx] < self.best_fitness:
            self.best_fitness = final_fitness[final_best_idx]
            self.best_solution = population[final_best_idx]

        return {
            'best_fitness': self.best_fitness,
            'best_solution': self.best_solution,
            'fitness_history': fitness_history,
            'learning_times': learning_times,
            'sampling_times': sampling_times,
            'n_evaluations': n_evaluations,
            'n_restarts': n_restarts,
            'method': self.method
        }

    def _learn_model(
        self,
        generation: int,
        n_vars: int,
        bounds: np.ndarray,
        selected_population: np.ndarray,
        selected_fitness: np.ndarray
    ):
        """Learn model based on method."""
        if self.method == 'rbm':
            return learn_rbm(
                generation, n_vars, bounds, selected_population,
                selected_fitness, params=self.neural_params
            )

        elif self.method == 'vae':
            return learn_vae(
                selected_population, selected_fitness, params=self.neural_params
            )

        elif self.method == 'extended_vae':
            return learn_extended_vae(
                selected_population, selected_fitness, params=self.neural_params
            )

        elif self.method == 'conditional_vae':
            return learn_conditional_extended_vae(
                selected_population, selected_fitness, params=self.neural_params
            )

        elif self.method == 'backdrive':
            # Add previous model for transfer learning
            params = self.neural_params.copy()
            if self.previous_model is not None:
                params['previous_model'] = self.previous_model
            return learn_backdrive(
                generation, n_vars, bounds, selected_population,
                selected_fitness, params=params
            )

        elif self.method == 'gan':
            return learn_gan(
                generation, n_vars, bounds, selected_population,
                selected_fitness, params=self.neural_params
            )

    def _sample_model(
        self,
        model,
        n_vars: int,
        bounds: np.ndarray,
        selected_fitness: np.ndarray
    ) -> np.ndarray:
        """Sample from model based on method."""
        if self.method == 'rbm':
            return sample_rbm(
                model, n_samples=self.pop_size, bounds=bounds
            )

        elif self.method == 'vae':
            return sample_vae(
                model, n_samples=self.pop_size, bounds=bounds
            )

        elif self.method == 'extended_vae':
            return sample_extended_vae(
                model, n_samples=self.pop_size, bounds=bounds,
                params={'use_predictor': False}
            )

        elif self.method == 'conditional_vae':
            target_fitness = np.min(selected_fitness)
            return sample_conditional_extended_vae(
                model, n_samples=self.pop_size, bounds=bounds,
                params={'target_fitness': target_fitness, 'fitness_noise': 0.1}
            )

        elif self.method == 'backdrive':
            target_fitness = np.min(selected_fitness)
            return sample_backdrive(
                model, n_samples=self.pop_size, bounds=bounds,
                params={'target_fitness': target_fitness}
            )

        elif self.method == 'gan':
            return sample_gan(
                model, n_samples=self.pop_size, bounds=bounds
            )


# ==================== Testing and Comparison Functions ====================

def test_single_method(
    method: str,
    fitness_function: Callable,
    function_name: str,
    n_vars: int = 10,
    bounds: Optional[np.ndarray] = None,
    n_generations: int = 30,
    pop_size: int = 100,
    verbose: bool = True
) -> Dict[str, Any]:
    """Test a single neural EDA method on a benchmark function."""

    if bounds is None:
        bounds = np.array([[-5.0] * n_vars, [5.0] * n_vars])

    print(f"\n{'='*80}")
    print(f"Testing {method.upper()} on {function_name}")
    print(f"{'='*80}\n")

    # Suppress warnings during optimization
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        eda = NeuralEDA(
            method=method,
            pop_size=pop_size,
            selection_ratio=0.3,
            restart_params={
                'trigger_no_improvement': 10,
                'diversity_threshold': 1e-8,
                'keep_best': 2
            }
        )

        result = eda.optimize(
            fitness_function,
            n_vars,
            bounds,
            n_generations,
            verbose=verbose
        )

    if verbose:
        print(f"\n{'='*80}")
        print(f"RESULTS for {method.upper()}")
        print(f"{'='*80}")
        print(f"Initial best fitness: {result['fitness_history'][0]:.6e}")
        print(f"Final best fitness:   {result['best_fitness']:.6e}")
        print(f"Improvement:          {result['fitness_history'][0] / max(result['best_fitness'], 1e-10):.2f}x")
        print(f"Evaluations:          {result['n_evaluations']}")
        print(f"Restarts:             {result['n_restarts']}")
        print(f"Avg learning time:    {np.mean(result['learning_times']):.3f}s")
        print(f"Avg sampling time:    {np.mean(result['sampling_times']):.3f}s")
        print(f"{'='*80}\n")

    return result


def compare_all_methods(
    fitness_function: Callable,
    function_name: str,
    n_vars: int = 10,
    bounds: Optional[np.ndarray] = None,
    n_generations: int = 25,
    pop_size: int = 100,
    n_runs: int = 1
):
    """Compare all four neural EDA methods on a benchmark function."""

    if bounds is None:
        bounds = np.array([[-5.0] * n_vars, [5.0] * n_vars])

    print(f"\n{'#'*80}")
    print(f"# COMPARING NEURAL EDA METHODS ON {function_name}")
    print(f"# Variables: {n_vars}, Generations: {n_generations}, Runs: {n_runs}")
    print(f"{'#'*80}\n")

    methods = ['rbm', 'vae', 'backdrive', 'gan']
    all_results = {method: [] for method in methods}

    for run in range(n_runs):
        if n_runs > 1:
            print(f"\n{'='*80}")
            print(f"RUN {run + 1} / {n_runs}")
            print(f"{'='*80}")

        for method in methods:
            try:
                result = test_single_method(
                    method, fitness_function, function_name,
                    n_vars, bounds, n_generations, pop_size,
                    verbose=(n_runs == 1)
                )
                all_results[method].append(result)
            except Exception as e:
                print(f"Error running {method}: {e}")
                continue

    # Print comparison summary
    print(f"\n{'='*80}")
    print(f"COMPARISON SUMMARY - {function_name}")
    print(f"{'='*80}")
    print(f"{'Method':<15} {'Initial':<12} {'Final':<12} {'Improv':<8} {'Evals':<8} {'Time/Gen':<10}")
    print(f"{'-'*80}")

    for method in methods:
        if not all_results[method]:
            continue

        results = all_results[method]
        initial = np.mean([r['fitness_history'][0] for r in results])
        final = np.mean([r['best_fitness'] for r in results])
        improvement = initial / max(final, 1e-10)
        evals = np.mean([r['n_evaluations'] for r in results])
        avg_time = np.mean([
            np.mean(r['learning_times']) + np.mean(r['sampling_times'])
            for r in results
        ])

        print(f"{method.upper():<15} {initial:<12.6e} {final:<12.6e} "
              f"{improvement:<8.2f} {evals:<8.0f} {avg_time:<10.3f}s")

    print(f"{'='*80}\n")

    return all_results


def comprehensive_benchmark():
    """Run comprehensive benchmark on multiple test functions."""
    print(f"\n{'#'*80}")
    print(f"# COMPREHENSIVE NEURAL EDA BENCHMARK")
    print(f"{'#'*80}\n")

    benchmarks = [
        ('Sphere', sphere_function, np.array([[-5.0] * 10, [5.0] * 10])),
        ('Rosenbrock', rosenbrock_function, np.array([[-5.0] * 10, [5.0] * 10])),
        ('Rastrigin', rastrigin_function, np.array([[-5.12] * 10, [5.12] * 10])),
        ('Ackley', ackley_function, np.array([[-5.0] * 10, [5.0] * 10])),
    ]

    all_benchmark_results = {}

    for name, func, bounds in benchmarks:
        results = compare_all_methods(
            func, name, n_vars=10, bounds=bounds,
            n_generations=30, pop_size=100, n_runs=1
        )
        all_benchmark_results[name] = results

    # Print final summary
    print(f"\n{'#'*80}")
    print(f"# FINAL SUMMARY - ALL BENCHMARKS")
    print(f"{'#'*80}\n")

    methods = ['rbm', 'vae', 'backdrive', 'gan']

    print(f"{'Benchmark':<15} ", end='')
    for method in methods:
        print(f"{method.upper():<12} ", end='')
    print()
    print(f"{'-'*80}")

    for bench_name in all_benchmark_results.keys():
        print(f"{bench_name:<15} ", end='')
        for method in methods:
            results = all_benchmark_results[bench_name].get(method, [])
            if results:
                final = np.mean([r['best_fitness'] for r in results])
                print(f"{final:<12.2e} ", end='')
            else:
                print(f"{'N/A':<12} ", end='')
        print()

    print(f"{'='*80}\n")


def main():
    """Run all neural EDA tests and comparisons."""
    print(f"\n{'#'*80}")
    print(f"# NEURAL NETWORK-BASED EDA COMPARISON")
    print(f"# RBM, VAE, Backdrive, and GAN EDAs")
    print(f"{'#'*80}")

    # Test 1: Single method demonstration
    print("\n" + "="*80)
    print("TEST 1: Single Method Demonstration (VAE-EDA on Sphere)")
    print("="*80)
    test_single_method('vae', sphere_function, 'Sphere', n_vars=10, n_generations=25)

    # Test 2: Compare all methods on Sphere
    print("\n" + "="*80)
    print("TEST 2: Compare All Methods (Sphere Function)")
    print("="*80)
    compare_all_methods(sphere_function, 'Sphere', n_vars=10, n_generations=25)

    # Test 3: Comprehensive benchmark
    print("\n" + "="*80)
    print("TEST 3: Comprehensive Benchmark on Multiple Functions")
    print("="*80)
    comprehensive_benchmark()

    print(f"\n{'#'*80}")
    print(f"# ALL TESTS COMPLETED SUCCESSFULLY!")
    print(f"{'#'*80}\n")


if __name__ == '__main__':
    main()
