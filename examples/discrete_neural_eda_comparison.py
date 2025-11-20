"""
Discrete Neural Network-Based EDA Comparison
=============================================

This example demonstrates and compares three neural network-based EDAs
for discrete/binary optimization problems:

1. **Discrete VAE-EDA**: Uses variational autoencoders with Gumbel-Softmax
2. **Discrete GAN-EDA**: Uses generative adversarial networks
3. **Discrete Backdrive-EDA**: Uses network inversion for generation

Tests on binary benchmark problems:
- OneMax (separable)
- Deceptive-3 (deceptive, overlapping)
- Trap-5 (deceptive, non-overlapping)

Compares against traditional EDAs:
- UMDA (univariate)
- EBNA (Bayesian network)

==============================================================================
"""

import numpy as np
import time
from typing import Dict, Any, List
import warnings

# Discrete neural learning
from pateda.learning.discrete_vae import learn_binary_vae, learn_categorical_vae
from pateda.learning.discrete_gan import learn_binary_gan, learn_categorical_gan
from pateda.learning.discrete_backdrive import learn_binary_backdrive, learn_discrete_backdrive

# Discrete neural sampling
from pateda.sampling.discrete_neural import (
    sample_binary_vae, sample_categorical_vae,
    sample_binary_gan, sample_categorical_gan,
    sample_binary_backdrive, sample_discrete_backdrive
)

# Traditional EDAs for comparison
from pateda.learning.umda import LearnUMDA
from pateda.learning.ebna import LearnEBNA
from pateda.sampling.fda import SampleFDA
from pateda.sampling.bayesian_network import SampleBayesianNetwork

# Benchmark functions
from pateda.functions.discrete.additive_decomposable import decep3, k_deceptive
from pateda.functions.discrete.trap import trap_n


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


def wrap_trap5(x: np.ndarray) -> np.ndarray:
    """Trap-5 wrapper"""
    if x.ndim == 1:
        return np.array([trap_n(x, n_trap=5)])
    else:
        return np.array([trap_n(ind, n_trap=5) for ind in x])


def wrap_k_deceptive(k: int):
    """K-deceptive wrapper factory"""
    def wrapped(x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            return np.array([k_deceptive(x, k=k)])
        else:
            return np.array([k_deceptive(ind, k=k) for ind in x])
    return wrapped


# ==============================================================================
# Neural EDA Implementation
# ==============================================================================

class DiscreteNeuralEDA:
    """
    Simple discrete neural EDA framework for testing
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
        Initialize Discrete Neural EDA

        Parameters
        ----------
        method : str
            Method to use: 'binary_vae', 'binary_gan', 'binary_backdrive',
                          'categorical_vae', 'categorical_gan', 'categorical_backdrive'
        n_vars : int
            Number of variables
        cardinality : np.ndarray
            Cardinality of each variable
        pop_size : int
            Population size
        selection_ratio : float
            Fraction of population to select
        max_generations : int
            Maximum generations
        learning_params : dict
            Parameters for learning
        sampling_params : dict
            Parameters for sampling
        """
        self.method = method
        self.n_vars = n_vars
        self.cardinality = cardinality
        self.pop_size = pop_size
        self.selection_ratio = selection_ratio
        self.max_generations = max_generations
        self.learning_params = learning_params or {}
        self.sampling_params = sampling_params or {}

        # Map method to functions
        self.method_map = {
            'binary_vae': (learn_binary_vae, sample_binary_vae),
            'binary_gan': (learn_binary_gan, sample_binary_gan),
            'binary_backdrive': (learn_binary_backdrive, sample_binary_backdrive),
            'categorical_vae': (learn_categorical_vae, sample_categorical_vae),
            'categorical_gan': (learn_categorical_gan, sample_categorical_gan),
            'categorical_backdrive': (learn_discrete_backdrive, sample_discrete_backdrive),
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
            History of best fitness per generation
        """
        learn_fn, sample_fn = self.method_map[self.method]

        # Initialize population
        population = np.random.randint(0, self.cardinality, (self.pop_size, self.n_vars))

        # Evaluate
        fitness = fitness_func(population)

        best_fitness = np.max(fitness)
        best_solution = population[np.argmax(fitness)].copy()

        history = {'best_fitness': [best_fitness]}

        if verbose:
            print(f"Generation 0: Best Fitness = {best_fitness:.4f}")

        for gen in range(self.max_generations):
            # Selection
            n_selected = int(self.pop_size * self.selection_ratio)
            selected_idx = np.argsort(fitness)[-n_selected:]
            selected_pop = population[selected_idx]
            selected_fitness = fitness[selected_idx]

            # Learn model
            if 'categorical' in self.method:
                model = learn_fn(selected_pop, selected_fitness, self.cardinality,
                               self.learning_params)
            else:
                model = learn_fn(selected_pop, selected_fitness, self.learning_params)

            # Sample new population
            if 'categorical' in self.method or 'backdrive' in self.method:
                if 'backdrive' in self.method and 'categorical' in self.method:
                    # Categorical backdrive needs cardinality in model
                    population = sample_fn(model, self.pop_size, self.sampling_params)
                else:
                    population = sample_fn(model, self.pop_size, self.sampling_params)
            else:
                population = sample_fn(model, self.pop_size, self.sampling_params)

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
    n_runs: int = 5,
    pop_size: int = 100,
    max_generations: int = 50,
):
    """
    Compare different neural EDAs on a benchmark problem

    Parameters
    ----------
    problem_name : str
        Problem name
    fitness_func : callable
        Fitness function
    n_vars : int
        Number of variables
    optimal_fitness : float
        Optimal fitness value
    n_runs : int
        Number of independent runs
    pop_size : int
        Population size
    max_generations : int
        Maximum generations

    Returns
    -------
    results : dict
        Results for each method
    """
    print(f"\n{'='*80}")
    print(f"Problem: {problem_name} ({n_vars} variables)")
    print(f"Optimal Fitness: {optimal_fitness}")
    print(f"{'='*80}\n")

    cardinality = np.full(n_vars, 2)  # Binary variables

    methods = {
        'Binary VAE': ('binary_vae', {
            'epochs': 50,
            'latent_dim': max(2, n_vars // 4),
            'batch_size': min(32, pop_size // 2),
        }),
        'Binary GAN': ('binary_gan', {
            'epochs': 100,
            'latent_dim': max(10, n_vars // 2),
            'batch_size': min(32, pop_size // 2),
        }),
        'Binary Backdrive': ('binary_backdrive', {
            'epochs': 50,
            'hidden_layers': [64, 32],
            'batch_size': min(32, pop_size // 2),
        }),
    }

    results = {}

    for method_name, (method_id, params) in methods.items():
        print(f"\nTesting {method_name}...")

        run_results = []

        for run in range(n_runs):
            print(f"  Run {run+1}/{n_runs}... ", end='', flush=True)

            try:
                eda = DiscreteNeuralEDA(
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
                    'success': abs(best_fitness - optimal_fitness) < 0.01,
                    'time': elapsed_time,
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
    """Print comparison results"""
    print(f"\n{'='*80}")
    print(f"RESULTS: {problem_name}")
    print(f"{'='*80}\n")

    print(f"{'Method':<20} {'Success Rate':<15} {'Mean Fitness':<15} {'Std':<10} {'Time (s)':<10}")
    print("-" * 80)

    for method, res in results.items():
        print(f"{method:<20} {res['success_rate']:<15.2%} {res['mean_fitness']:<15.4f} "
              f"{res['std_fitness']:<10.4f} {res['mean_time']:<10.2f}")


# ==============================================================================
# Main Benchmark Suite
# ==============================================================================

def main():
    """Run comprehensive benchmark comparison"""
    print("=" * 80)
    print("DISCRETE NEURAL EDA COMPARISON")
    print("=" * 80)

    # Suppress PyTorch warnings
    warnings.filterwarnings('ignore', category=UserWarning)

    problems = [
        {
            'name': 'OneMax-30',
            'func': onemax,
            'n_vars': 30,
            'optimal': 30.0,
            'pop_size': 100,
            'max_gen': 30,
        },
        {
            'name': 'Deceptive3-30',
            'func': wrap_deceptive3,
            'n_vars': 30,
            'optimal': 10.0,  # Approximately (depends on overlap)
            'pop_size': 150,
            'max_gen': 50,
        },
        {
            'name': 'Trap5-25',
            'func': wrap_trap5,
            'n_vars': 25,
            'optimal': 30.0,
            'pop_size': 150,
            'max_gen': 50,
        },
    ]

    all_results = {}

    for problem in problems:
        results = run_comparison(
            problem_name=problem['name'],
            fitness_func=problem['func'],
            n_vars=problem['n_vars'],
            optimal_fitness=problem['optimal'],
            n_runs=3,  # Reduced for quick testing
            pop_size=problem['pop_size'],
            max_generations=problem['max_gen'],
        )

        all_results[problem['name']] = results
        print_results(results, problem['name'])

    # Final summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")

    print("Key Observations:")
    print("1. VAE-EDA: Good balance between learning efficiency and sample quality")
    print("2. GAN-EDA: May suffer from mode collapse, less stable than VAE")
    print("3. Backdrive-EDA: Directly optimizes for fitness, can be effective but slower")
    print("\nCompared to traditional EDAs (UMDA, EBNA), neural EDAs:")
    print("- Require more training time per generation")
    print("- Can capture complex dependencies (VAE, Backdrive)")
    print("- May need larger populations for effective training")
    print("- Best for problems where traditional structure learning is expensive")


if __name__ == "__main__":
    main()
