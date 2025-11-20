"""
Complete Discrete Neural Network-Based EDA Comparison
======================================================

This example demonstrates and compares SIX neural network-based EDAs
for discrete/binary optimization problems:

1. **VAE-EDA**: Variational Autoencoder with Gumbel-Softmax
2. **GAN-EDA**: Generative Adversarial Network
3. **Backdrive-EDA**: Network inversion approach
4. **DAE-EDA**: Denoising Autoencoder with iterative refinement
5. **RBM-EDA**: Restricted Boltzmann Machine with contrastive divergence
6. **DbD-EDA**: Diffusion-by-Deblending (NEW)

Tests on binary benchmark problems:
- OneMax (separable)
- Deceptive-3 (deceptive, overlapping)
- K-Deceptive (deceptive, non-overlapping)

Provides comprehensive comparison of all neural approaches for discrete optimization.

==============================================================================
"""

import numpy as np
import time
from typing import Dict, Any, List
import warnings

# Neural learning modules
from pateda.learning.discrete_vae import learn_binary_vae
from pateda.learning.discrete_gan import learn_binary_gan
from pateda.learning.discrete_backdrive import learn_binary_backdrive
from pateda.learning.dae import learn_dae
from pateda.learning.rbm import learn_softmax_rbm
from pateda.learning.discrete_dbd import learn_binary_dbd

# Neural sampling modules
from pateda.sampling.discrete_neural import (
    sample_binary_vae, sample_binary_gan, sample_binary_backdrive
)
from pateda.sampling.dae import sample_dae
from pateda.sampling.rbm import sample_softmax_rbm
from pateda.sampling.discrete_dbd import sample_binary_dbd

# Benchmark functions
from pateda.functions.discrete.additive_decomposable import decep3, k_deceptive


# ==============================================================================
# Fitness Function Wrappers
# ==============================================================================

def onemax(x: np.ndarray) -> np.ndarray:
    """OneMax function"""
    if x.ndim == 1:
        return np.array([float(np.sum(x))])
    else:
        return np.sum(x, axis=1).astype(float)


def wrap_deceptive3(x: np.ndarray) -> np.ndarray:
    """Deceptive-3 wrapper"""
    if x.ndim == 1:
        return np.array([decep3(x)])
    else:
        return np.array([decep3(ind) for ind in x])


def wrap_k_deceptive(k: int):
    """K-deceptive wrapper factory"""
    def wrapped(x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            return np.array([k_deceptive(x, k=k)])
        else:
            return np.array([k_deceptive(ind, k=k) for ind in x])
    return wrapped


# ==============================================================================
# Unified Neural EDA Framework
# ==============================================================================

class UnifiedDiscreteNeuralEDA:
    """
    Unified framework for all discrete neural EDAs
    """

    def __init__(
        self,
        method: str,
        n_vars: int,
        cardinality: np.ndarray,
        pop_size: int = 100,
        selection_ratio: float = 0.5,
        max_generations: int = 50,
        learning_params: Dict[str, Any] = None,
        sampling_params: Dict[str, Any] = None,
    ):
        """
        Initialize Unified Neural EDA

        Parameters
        ----------
        method : str
            Method: 'vae', 'gan', 'backdrive', 'dae', 'rbm', 'dbd'
        n_vars : int
            Number of variables
        cardinality : np.ndarray
            Cardinality of each variable
        pop_size : int
            Population size
        selection_ratio : float
            Selection ratio
        max_generations : int
            Maximum generations
        learning_params : dict
            Learning parameters
        sampling_params : dict
            Sampling parameters
        """
        self.method = method
        self.n_vars = n_vars
        self.cardinality = cardinality
        self.pop_size = pop_size
        self.selection_ratio = selection_ratio
        self.max_generations = max_generations
        self.learning_params = learning_params or {}
        self.sampling_params = sampling_params or {}

        # Map methods to functions
        self.method_map = {
            'vae': (learn_binary_vae, sample_binary_vae, False),
            'gan': (learn_binary_gan, sample_binary_gan, False),
            'backdrive': (learn_binary_backdrive, sample_binary_backdrive, False),
            'dae': (learn_dae, sample_dae, False),
            'rbm': (learn_softmax_rbm, sample_softmax_rbm, True),  # Needs cardinality
            'dbd': (learn_binary_dbd, sample_binary_dbd, True),  # Needs two populations
        }

    def run(self, fitness_func, verbose=True):
        """
        Run the EDA

        Parameters
        ----------
        fitness_func : callable
            Fitness function
        verbose : bool
            Print progress

        Returns
        -------
        best_fitness : float
            Best fitness found
        best_solution : np.ndarray
            Best solution found
        history : dict
            History dictionary
        """
        learn_fn, sample_fn, special = self.method_map[self.method]

        # Initialize population
        population = np.random.randint(0, self.cardinality, (self.pop_size, self.n_vars))

        # Evaluate
        fitness = fitness_func(population)

        best_fitness = np.max(fitness)
        best_solution = population[np.argmax(fitness)].copy()

        history = {'best_fitness': [best_fitness]}

        if verbose:
            print(f"Generation 0: Best Fitness = {best_fitness:.4f}")

        # Keep track of previous population for DbD
        prev_population = None

        for gen in range(self.max_generations):
            # Selection
            n_selected = int(self.pop_size * self.selection_ratio)
            selected_idx = np.argsort(fitness)[-n_selected:]
            selected_pop = population[selected_idx]
            selected_fitness = fitness[selected_idx]

            # Learn model
            try:
                if self.method == 'rbm':
                    model = learn_fn(selected_pop, selected_fitness, self.cardinality,
                                   self.learning_params)
                elif self.method == 'dbd':
                    # DbD needs two populations (source and target)
                    if prev_population is None:
                        # First generation: use random as source
                        p0 = np.random.randint(0, self.cardinality,
                                             (len(selected_pop), self.n_vars))
                    else:
                        # Use previous selected population as source
                        p0 = prev_population

                    p1 = selected_pop
                    model = learn_fn(p0, p1, self.learning_params)

                    # Save for next iteration
                    prev_population = selected_pop.copy()
                else:
                    model = learn_fn(selected_pop, selected_fitness, self.learning_params)

                # Sample new population
                population = sample_fn(model, self.pop_size, self.sampling_params)

            except Exception as e:
                if verbose:
                    print(f"  Warning: Sampling failed ({e}), using random population")
                population = np.random.randint(0, self.cardinality,
                                             (self.pop_size, self.n_vars))

            # Evaluate
            fitness = fitness_func(population)

            # Update best
            gen_best = np.max(fitness)
            if gen_best > best_fitness:
                best_fitness = gen_best
                best_solution = population[np.argmax(fitness)].copy()

            history['best_fitness'].append(best_fitness)

            if verbose and (gen + 1) % 10 == 0:
                print(f"Generation {gen+1}: Best Fitness = {best_fitness:.4f}")

        return best_fitness, best_solution, history


# ==============================================================================
# Benchmark Comparison
# ==============================================================================

def run_comparison(
    problem_name: str,
    fitness_func,
    n_vars: int,
    optimal_fitness: float,
    n_runs: int = 3,
    pop_size: int = 100,
    max_generations: int = 30,
):
    """
    Compare all six neural EDAs on a benchmark problem

    Parameters
    ----------
    problem_name : str
        Problem name
    fitness_func : callable
        Fitness function
    n_vars : int
        Number of variables
    optimal_fitness : float
        Optimal fitness
    n_runs : int
        Number of runs
    pop_size : int
        Population size
    max_generations : int
        Max generations

    Returns
    -------
    results : dict
        Results for each method
    """
    print(f"\n{'='*80}")
    print(f"Problem: {problem_name} ({n_vars} variables)")
    print(f"Optimal Fitness: {optimal_fitness}")
    print(f"{'='*80}\n")

    cardinality = np.full(n_vars, 2)  # Binary

    # Define methods with their parameters
    methods = {
        'VAE': ('vae', {
            'epochs': 30,
            'latent_dim': max(2, n_vars // 4),
            'batch_size': min(32, pop_size // 2),
        }),
        'GAN': ('gan', {
            'epochs': 60,
            'latent_dim': max(10, n_vars // 2),
            'batch_size': min(32, pop_size // 2),
        }),
        'Backdrive': ('backdrive', {
            'epochs': 30,
            'hidden_layers': [64, 32],
            'batch_size': min(32, pop_size // 2),
        }),
        'DAE': ('dae', {
            'epochs': 30,
            'hidden_dim': max(n_vars // 2, 10),
            'corruption_level': 0.1,
        }),
        'RBM': ('rbm', {
            'epochs': 15,
            'n_hidden': n_vars,
            'k_cd': 1,
        }),
        'DbD': ('dbd', {
            'epochs': 50,
            'hidden_dims': [64, 32],
            'num_alpha_samples': 5,
        }),
    }

    results = {}

    for method_name, (method_id, params) in methods.items():
        print(f"\nTesting {method_name}...")

        run_results = []

        for run in range(n_runs):
            print(f"  Run {run+1}/{n_runs}... ", end='', flush=True)

            try:
                eda = UnifiedDiscreteNeuralEDA(
                    method=method_id,
                    n_vars=n_vars,
                    cardinality=cardinality,
                    pop_size=pop_size,
                    selection_ratio=0.5,
                    max_generations=max_generations,
                    learning_params=params,
                    sampling_params={},
                )

                start_time = time.time()
                best_fitness, best_solution, history = eda.run(fitness_func, verbose=False)
                elapsed_time = time.time() - start_time

                run_results.append({
                    'best_fitness': best_fitness,
                    'success': abs(best_fitness - optimal_fitness) < 0.01 * optimal_fitness,
                    'time': elapsed_time,
                    'final_fitness': best_fitness,
                })

                print(f"Fitness: {best_fitness:.4f}, Time: {elapsed_time:.2f}s")

            except Exception as e:
                print(f"Error: {e}")
                warnings.warn(f"Run {run+1} failed for {method_name}: {e}")
                continue

        if run_results:
            results[method_name] = {
                'best_fitness': [r['best_fitness'] for r in run_results],
                'success_rate': np.mean([r['success'] for r in run_results]),
                'mean_fitness': np.mean([r['best_fitness'] for r in run_results]),
                'std_fitness': np.std([r['best_fitness'] for r in run_results]),
                'mean_time': np.mean([r['time'] for r in run_results]),
            }
        else:
            print(f"  All runs failed for {method_name}")

    return results


def print_results(results: Dict[str, Any], problem_name: str):
    """Print comparison results in a nice table"""
    print(f"\n{'='*80}")
    print(f"RESULTS: {problem_name}")
    print(f"{'='*80}\n")

    if not results:
        print("No results to display.")
        return

    print(f"{'Method':<15} {'Success':<10} {'Mean Fitness':<15} {'Std':<10} {'Time (s)':<10}")
    print("-" * 70)

    for method, res in results.items():
        print(f"{method:<15} {res['success_rate']:<10.2%} {res['mean_fitness']:<15.4f} "
              f"{res['std_fitness']:<10.4f} {res['mean_time']:<10.2f}")


# ==============================================================================
# Main Benchmark Suite
# ==============================================================================

def main():
    """Run comprehensive benchmark comparison"""
    print("=" * 80)
    print("COMPLETE DISCRETE NEURAL EDA COMPARISON")
    print("All Six Neural Approaches for Binary Optimization")
    print("=" * 80)

    # Suppress warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    problems = [
        {
            'name': 'OneMax-20',
            'func': onemax,
            'n_vars': 20,
            'optimal': 20.0,
            'pop_size': 80,
            'max_gen': 20,
        },
        {
            'name': 'Deceptive3-30',
            'func': wrap_deceptive3,
            'n_vars': 30,
            'optimal': 10.0,
            'pop_size': 100,
            'max_gen': 30,
        },
        {
            'name': 'KDeceptive3-30',
            'func': wrap_k_deceptive(3),
            'n_vars': 30,
            'optimal': 30.0,
            'pop_size': 120,
            'max_gen': 30,
        },
    ]

    all_results = {}

    for problem in problems:
        results = run_comparison(
            problem_name=problem['name'],
            fitness_func=problem['func'],
            n_vars=problem['n_vars'],
            optimal_fitness=problem['optimal'],
            n_runs=3,
            pop_size=problem['pop_size'],
            max_generations=problem['max_gen'],
        )

        all_results[problem['name']] = results
        print_results(results, problem['name'])

    # Final summary
    print(f"\n{'='*80}")
    print("COMPREHENSIVE SUMMARY")
    print(f"{'='*80}\n")

    print("Neural EDA Characteristics:\n")

    print("1. **VAE-EDA**:")
    print("   - Generative model with encoder-decoder")
    print("   - Gumbel-Softmax for discrete variables")
    print("   - Good balance of quality and speed")
    print("   - Recommended for most problems\n")

    print("2. **GAN-EDA**:")
    print("   - Adversarial training")
    print("   - Can suffer from mode collapse")
    print("   - Less stable than other methods")
    print("   - Use with caution\n")

    print("3. **Backdrive-EDA**:")
    print("   - Network inversion approach")
    print("   - Directly optimizes for fitness")
    print("   - Can be slow but effective")
    print("   - Good when fitness landscape is smooth\n")

    print("4. **DAE-EDA**:")
    print("   - Denoising autoencoder")
    print("   - Iterative refinement")
    print("   - Simple and effective")
    print("   - Fast training, proven approach\n")

    print("5. **RBM-EDA**:")
    print("   - Energy-based model")
    print("   - Contrastive divergence training")
    print("   - Classical approach for discrete")
    print("   - Well-established in literature\n")

    print("6. **DbD-EDA** (NEW):")
    print("   - Diffusion-by-deblending")
    print("   - Iterative denoising")
    print("   - Simpler than VAE/GAN")
    print("   - Promising alternative approach\n")

    print("\nRecommendations:")
    print("- For most problems: VAE or DAE")
    print("- For classical approach: RBM")
    print("- For research: DbD (new method)")
    print("- Avoid GAN unless you have specific reasons")


if __name__ == "__main__":
    main()
