"""
Comprehensive Comparison of Diffusion-Based EDAs for Continuous Optimization

This script compares five diffusion-based EDAs on five continuous benchmark functions:
- DbD-CS, DbD-CD, DbD-UC, DbD-US (Denoising-by-Deblending variants)
- DenDiff (Denoising Diffusion)

All tests are performed with dimension n=20.
"""

import numpy as np
import time
import sys
import os
import json
from pathlib import Path

# Add pateda to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pateda.learning.dbd import learn_dbd, find_closest_neighbors, sample_univariate_gaussian
from pateda.sampling.dbd import sample_dbd
from pateda.learning.dendiff import learn_dendiff
from pateda.sampling.dendiff import sample_dendiff_fast


# ==================== Benchmark Functions ====================

def sphere_function(x):
    """
    Sphere function: f(x) = sum(x_i^2)
    Global minimum: f(0,...,0) = 0
    Domain: [-5.12, 5.12]^n
    """
    if x.ndim == 1:
        return np.sum(x**2)
    return np.sum(x**2, axis=1)


def rosenbrock_function(x):
    """
    Rosenbrock function: f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
    Global minimum: f(1,...,1) = 0
    Domain: [-5, 5]^n
    """
    if x.ndim == 1:
        return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    return np.sum(100 * (x[:, 1:] - x[:, :-1]**2)**2 + (1 - x[:, :-1])**2, axis=1)


def rastrigin_function(x):
    """
    Rastrigin function: f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))
    Global minimum: f(0,...,0) = 0
    Domain: [-5.12, 5.12]^n
    """
    if x.ndim == 1:
        n = len(x)
        return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
    n = x.shape[1]
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=1)


def ackley_function(x):
    """
    Ackley function:
    f(x) = -20*exp(-0.2*sqrt(sum(x_i^2)/n)) - exp(sum(cos(2*pi*x_i))/n) + 20 + e
    Global minimum: f(0,...,0) = 0
    Domain: [-5, 5]^n
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
    Griewank function: f(x) = 1 + sum(x_i^2)/4000 - prod(cos(x_i/sqrt(i+1)))
    Global minimum: f(0,...,0) = 0
    Domain: [-600, 600]^n (typically, but we'll use [-5, 5])
    """
    if x.ndim == 1:
        n = len(x)
        sum_term = np.sum(x**2) / 4000
        prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, n + 1))))
        return sum_term - prod_term + 1
    n = x.shape[1]
    sum_term = np.sum(x**2, axis=1) / 4000
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, n + 1))), axis=1)
    return sum_term - prod_term + 1


# Define benchmark suite
BENCHMARKS = {
    'Sphere': {
        'function': sphere_function,
        'bounds': np.array([[-5.12] * 20, [5.12] * 20]),
        'optimum': 0.0
    },
    'Rosenbrock': {
        'function': rosenbrock_function,
        'bounds': np.array([[-5.0] * 20, [5.0] * 20]),
        'optimum': 0.0
    },
    'Rastrigin': {
        'function': rastrigin_function,
        'bounds': np.array([[-5.12] * 20, [5.12] * 20]),
        'optimum': 0.0
    },
    'Ackley': {
        'function': ackley_function,
        'bounds': np.array([[-5.0] * 20, [5.0] * 20]),
        'optimum': 0.0
    },
    'Griewank': {
        'function': griewank_function,
        'bounds': np.array([[-5.0] * 20, [5.0] * 20]),
        'optimum': 0.0
    }
}


# ==================== DbD-EDA Implementation ====================

class DbDEDA:
    """Diffusion-by-Deblending EDA with restart mechanism."""

    def __init__(self, variant='CS', pop_size=200, selection_ratio=0.3,
                 dbd_params=None, restart_params=None):
        self.variant = variant.upper()
        assert self.variant in ['CS', 'CD', 'UC', 'US']

        self.pop_size = pop_size
        self.selection_size = int(pop_size * selection_ratio)

        # DbD parameters
        if dbd_params is None:
            dbd_params = {}
        self.dbd_params = {
            'num_alpha_samples': dbd_params.get('num_alpha_samples', 10),
            'hidden_dims': dbd_params.get('hidden_dims', [64, 64]),
            'epochs': dbd_params.get('epochs', 30),
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

        # Tracking
        self.generations_without_improvement = 0
        self.best_fitness = np.inf
        self.best_solution = None

    def _prepare_distributions(self, population, selected_population):
        """Prepare p0 and p1 distributions based on variant."""
        p_size = len(population)
        sel_p_size = len(selected_population)
        to_take = p_size * 2

        if self.variant == 'CS':
            p0_indices = np.random.randint(0, p_size, size=to_take)
            p0 = population[p0_indices]
            p1_indices = np.random.randint(0, sel_p_size, size=to_take)
            p1 = selected_population[p1_indices]
        elif self.variant == 'CD':
            p0_indices = np.random.randint(0, p_size, size=to_take)
            p0 = population[p0_indices]
            p1 = find_closest_neighbors(p0, selected_population)
        elif self.variant == 'UC':
            p0 = sample_univariate_gaussian(population, to_take)
            p1_indices = np.random.randint(0, p_size, size=to_take)
            p1 = population[p1_indices]
        elif self.variant == 'US':
            p0 = sample_univariate_gaussian(population, to_take)
            p1_indices = np.random.randint(0, sel_p_size, size=to_take)
            p1 = selected_population[p1_indices]

        return p0, p1

    def _get_sampling_p0(self, population, selected_population):
        """Get p0 for sampling based on variant."""
        if self.variant in ['CS', 'CD']:
            return selected_population
        else:
            return sample_univariate_gaussian(selected_population, self.pop_size)

    def _check_restart_condition(self, selected_fitness, current_best):
        """Check if restart should be triggered."""
        sel_diversity = np.std(selected_fitness[self.keep_best:])
        return (sel_diversity < self.diversity_threshold or
                self.generations_without_improvement >= self.trigger_no_improvement)

    def optimize(self, fitness_function, n_vars, bounds, n_generations, verbose=False):
        """Run DbD-EDA optimization."""
        # Initialize population
        population = np.random.uniform(bounds[0], bounds[1], (self.pop_size, n_vars))

        # Tracking
        fitness_history = []
        learning_times = []
        sampling_times = []
        function_evaluations = 0

        for gen in range(n_generations):
            # Evaluate
            fitness = np.array([fitness_function(ind) for ind in population])
            function_evaluations += len(population)

            # Track best
            best_idx = np.argmin(fitness)
            gen_best_fitness = fitness[best_idx]
            fitness_history.append(gen_best_fitness)

            if verbose and gen % 5 == 0:
                print(f"  Gen {gen+1:3d}: Best = {gen_best_fitness:.6e}")

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
                continue

            # Prepare distributions
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
                model, p0_sample, self.pop_size, bounds=bounds,
                params={'num_iterations': self.num_iterations}
            )
            sampling_time = time.time() - start_time
            sampling_times.append(sampling_time)

            # Replace population
            population = new_population

        # Final evaluation
        final_fitness = np.array([fitness_function(ind) for ind in population])
        function_evaluations += len(population)
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
            'function_evaluations': function_evaluations
        }


# ==================== DenDiff-EDA Implementation ====================

class DenDiffEDA:
    """Denoising Diffusion EDA."""

    def __init__(self, pop_size=200, selection_ratio=0.3, dendiff_params=None):
        self.pop_size = pop_size
        self.selection_size = int(pop_size * selection_ratio)

        if dendiff_params is None:
            dendiff_params = {}
        self.dendiff_params = {
            'n_timesteps': dendiff_params.get('n_timesteps', 1000),
            'beta_schedule': dendiff_params.get('beta_schedule', 'linear'),
            'hidden_dims': dendiff_params.get('hidden_dims', [128, 64]),
            'time_emb_dim': dendiff_params.get('time_emb_dim', 32),
            'epochs': dendiff_params.get('epochs', 30),
            'batch_size': dendiff_params.get('batch_size', 32),
            'learning_rate': dendiff_params.get('learning_rate', 1e-3)
        }
        self.fast_sampling_params = {
            'ddim_steps': dendiff_params.get('ddim_steps', 50),
            'ddim_eta': dendiff_params.get('ddim_eta', 0.0)
        }

        self.best_fitness = np.inf
        self.best_solution = None

    def optimize(self, fitness_function, n_vars, bounds, n_generations, verbose=False):
        """Run DenDiff-EDA optimization."""
        # Initialize population
        population = np.random.uniform(bounds[0], bounds[1], (self.pop_size, n_vars))

        # Tracking
        fitness_history = []
        learning_times = []
        sampling_times = []
        function_evaluations = 0

        for gen in range(n_generations):
            # Evaluate
            fitness = np.array([fitness_function(ind) for ind in population])
            function_evaluations += len(population)

            # Track best
            best_idx = np.argmin(fitness)
            gen_best_fitness = fitness[best_idx]
            fitness_history.append(gen_best_fitness)

            if verbose and gen % 5 == 0:
                print(f"  Gen {gen+1:3d}: Best = {gen_best_fitness:.6e}")

            # Update global best
            if gen_best_fitness < self.best_fitness:
                self.best_fitness = gen_best_fitness
                self.best_solution = population[best_idx].copy()

            # Select best individuals
            idx = np.argsort(fitness)[:self.selection_size]
            selected_pop = population[idx]
            selected_fit = fitness[idx]

            # Learn model
            start_time = time.time()
            model = learn_dendiff(selected_pop, selected_fit, params=self.dendiff_params)
            learning_time = time.time() - start_time
            learning_times.append(learning_time)

            # Sample new population (using fast sampling)
            start_time = time.time()
            new_population = sample_dendiff_fast(
                model, n_samples=self.pop_size, bounds=bounds,
                params=self.fast_sampling_params
            )
            sampling_time = time.time() - start_time
            sampling_times.append(sampling_time)

            # Replace population
            population = new_population

        # Final evaluation
        final_fitness = np.array([fitness_function(ind) for ind in population])
        function_evaluations += len(population)
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
            'function_evaluations': function_evaluations
        }


# ==================== Experimental Framework ====================

def run_single_experiment(eda_name, eda_class, eda_params, function_name,
                         benchmark_info, n_generations=50, seed=None):
    """Run a single experiment."""
    if seed is not None:
        np.random.seed(seed)

    # Create EDA instance
    eda = eda_class(**eda_params)

    # Run optimization
    fitness_function = benchmark_info['function']
    bounds = benchmark_info['bounds']
    n_vars = len(bounds[0])

    start_time = time.time()
    result = eda.optimize(fitness_function, n_vars, bounds, n_generations, verbose=False)
    total_time = time.time() - start_time

    # Compile results
    return {
        'eda_name': eda_name,
        'function_name': function_name,
        'best_fitness': result['best_fitness'],
        'best_solution': result['best_solution'].tolist(),
        'fitness_history': result['fitness_history'],
        'learning_times': result['learning_times'],
        'sampling_times': result['sampling_times'],
        'total_time': total_time,
        'avg_learning_time': np.mean(result['learning_times']),
        'avg_sampling_time': np.mean(result['sampling_times']),
        'avg_generation_time': total_time / n_generations,
        'function_evaluations': result['function_evaluations'],
        'final_error': result['best_fitness'] - benchmark_info['optimum']
    }


def run_comparison_experiments(n_runs=5, n_generations=50, save_results=True):
    """Run comprehensive comparison experiments."""
    print("="*80)
    print("DIFFUSION-BASED EDAs COMPARISON EXPERIMENT")
    print("="*80)
    print(f"Dimension: n = 20")
    print(f"Generations: {n_generations}")
    print(f"Independent runs: {n_runs}")
    print(f"Benchmark functions: {len(BENCHMARKS)}")
    print(f"EDAs: 5 (DbD-CS, DbD-CD, DbD-UC, DbD-US, DenDiff)")
    print("="*80)
    print()

    # Define EDAs to compare
    edas = {
        'DbD-CS': {
            'class': DbDEDA,
            'params': {
                'variant': 'CS',
                'pop_size': 200,
                'selection_ratio': 0.3,
                'dbd_params': {
                    'num_alpha_samples': 10,
                    'hidden_dims': [64, 64],
                    'epochs': 30,
                    'batch_size': 32,
                    'learning_rate': 1e-3,
                    'num_iterations': 10
                },
                'restart_params': {
                    'trigger_no_improvement': 5,
                    'diversity_threshold': 1e-6,
                    'keep_best': 2
                }
            }
        },
        'DbD-CD': {
            'class': DbDEDA,
            'params': {
                'variant': 'CD',
                'pop_size': 200,
                'selection_ratio': 0.3,
                'dbd_params': {
                    'num_alpha_samples': 10,
                    'hidden_dims': [64, 64],
                    'epochs': 30,
                    'batch_size': 32,
                    'learning_rate': 1e-3,
                    'num_iterations': 10
                },
                'restart_params': {
                    'trigger_no_improvement': 5,
                    'diversity_threshold': 1e-6,
                    'keep_best': 2
                }
            }
        },
        'DbD-UC': {
            'class': DbDEDA,
            'params': {
                'variant': 'UC',
                'pop_size': 200,
                'selection_ratio': 0.3,
                'dbd_params': {
                    'num_alpha_samples': 10,
                    'hidden_dims': [64, 64],
                    'epochs': 30,
                    'batch_size': 32,
                    'learning_rate': 1e-3,
                    'num_iterations': 10
                },
                'restart_params': {
                    'trigger_no_improvement': 5,
                    'diversity_threshold': 1e-6,
                    'keep_best': 2
                }
            }
        },
        'DbD-US': {
            'class': DbDEDA,
            'params': {
                'variant': 'US',
                'pop_size': 200,
                'selection_ratio': 0.3,
                'dbd_params': {
                    'num_alpha_samples': 10,
                    'hidden_dims': [64, 64],
                    'epochs': 30,
                    'batch_size': 32,
                    'learning_rate': 1e-3,
                    'num_iterations': 10
                },
                'restart_params': {
                    'trigger_no_improvement': 5,
                    'diversity_threshold': 1e-6,
                    'keep_best': 2
                }
            }
        },
        'DenDiff': {
            'class': DenDiffEDA,
            'params': {
                'pop_size': 200,
                'selection_ratio': 0.3,
                'dendiff_params': {
                    'n_timesteps': 1000,
                    'beta_schedule': 'linear',
                    'hidden_dims': [128, 64],
                    'time_emb_dim': 32,
                    'epochs': 30,
                    'batch_size': 32,
                    'learning_rate': 1e-3,
                    'ddim_steps': 50,
                    'ddim_eta': 0.0
                }
            }
        }
    }

    all_results = []
    total_experiments = len(edas) * len(BENCHMARKS) * n_runs
    experiment_count = 0

    # Run experiments
    for eda_name, eda_config in edas.items():
        print(f"\nTesting {eda_name}:")
        print("-" * 80)

        for func_name, benchmark_info in BENCHMARKS.items():
            print(f"  {func_name}:", end=" ", flush=True)

            for run in range(n_runs):
                experiment_count += 1
                seed = 42 + run  # Different seed for each run

                result = run_single_experiment(
                    eda_name=eda_name,
                    eda_class=eda_config['class'],
                    eda_params=eda_config['params'],
                    function_name=func_name,
                    benchmark_info=benchmark_info,
                    n_generations=n_generations,
                    seed=seed
                )

                result['run'] = run + 1
                all_results.append(result)

                print(f".", end="", flush=True)

            # Print summary for this function
            func_results = [r for r in all_results
                          if r['eda_name'] == eda_name and r['function_name'] == func_name]
            avg_best = np.mean([r['best_fitness'] for r in func_results])
            print(f" Avg: {avg_best:.6e} ({experiment_count}/{total_experiments})")

    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*80)

    # Save results
    if save_results:
        output_dir = Path("experiments/results")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"diffusion_eda_comparison_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\nResults saved to: {results_file}")

    return all_results


def analyze_results(results):
    """Analyze and display comparison results."""
    print("\n" + "="*80)
    print("ANALYSIS: STATISTICAL SUMMARY")
    print("="*80)

    # Group by EDA and function
    edas = sorted(set(r['eda_name'] for r in results))
    functions = sorted(set(r['function_name'] for r in results))

    # Table 1: Best Fitness (mean ± std)
    print("\nTable 1: Best Fitness Achieved (mean ± std)")
    print("-" * 80)
    print(f"{'EDA':<12} | " + " | ".join([f"{f:>18}" for f in functions]))
    print("-" * 80)

    for eda in edas:
        row = f"{eda:<12} | "
        for func in functions:
            func_results = [r['best_fitness'] for r in results
                          if r['eda_name'] == eda and r['function_name'] == func]
            mean_fit = np.mean(func_results)
            std_fit = np.std(func_results)
            row += f"{mean_fit:10.3e}±{std_fit:.2e} | "
        print(row)

    # Table 2: Computation Time
    print("\n" + "="*80)
    print("Table 2: Average Time per Generation (seconds)")
    print("-" * 80)
    print(f"{'EDA':<12} | {'Learning':<12} | {'Sampling':<12} | {'Total':<12}")
    print("-" * 80)

    for eda in edas:
        eda_results = [r for r in results if r['eda_name'] == eda]
        avg_learning = np.mean([r['avg_learning_time'] for r in eda_results])
        avg_sampling = np.mean([r['avg_sampling_time'] for r in eda_results])
        avg_total = np.mean([r['avg_generation_time'] for r in eda_results])
        print(f"{eda:<12} | {avg_learning:>12.4f} | {avg_sampling:>12.4f} | {avg_total:>12.4f}")

    # Table 3: Best EDA per function
    print("\n" + "="*80)
    print("Table 3: Best EDA per Function (lowest mean fitness)")
    print("-" * 80)
    print(f"{'Function':<15} | {'Best EDA':<12} | {'Mean Fitness':<15} | {'Std':<12}")
    print("-" * 80)

    for func in functions:
        func_by_eda = {}
        for eda in edas:
            func_results = [r['best_fitness'] for r in results
                          if r['eda_name'] == eda and r['function_name'] == func]
            func_by_eda[eda] = {
                'mean': np.mean(func_results),
                'std': np.std(func_results)
            }

        best_eda = min(func_by_eda.items(), key=lambda x: x[1]['mean'])
        print(f"{func:<15} | {best_eda[0]:<12} | {best_eda[1]['mean']:>15.6e} | {best_eda[1]['std']:>12.3e}")

    # Overall ranking
    print("\n" + "="*80)
    print("Table 4: Overall Ranking (based on average rank across functions)")
    print("-" * 80)

    eda_ranks = {eda: [] for eda in edas}

    for func in functions:
        func_means = {}
        for eda in edas:
            func_results = [r['best_fitness'] for r in results
                          if r['eda_name'] == eda and r['function_name'] == func]
            func_means[eda] = np.mean(func_results)

        # Rank EDAs for this function (1 = best)
        sorted_edas = sorted(func_means.items(), key=lambda x: x[1])
        for rank, (eda, _) in enumerate(sorted_edas, 1):
            eda_ranks[eda].append(rank)

    avg_ranks = {eda: np.mean(ranks) for eda, ranks in eda_ranks.items()}
    sorted_ranks = sorted(avg_ranks.items(), key=lambda x: x[1])

    print(f"{'Rank':<6} | {'EDA':<12} | {'Avg Rank':<12} | {'Rank Std':<12}")
    print("-" * 80)
    for i, (eda, avg_rank) in enumerate(sorted_ranks, 1):
        rank_std = np.std(eda_ranks[eda])
        print(f"{i:<6} | {eda:<12} | {avg_rank:>12.2f} | {rank_std:>12.2f}")

    print("="*80)


def main():
    """Main execution function."""
    print("\n" + "#"*80)
    print("# COMPREHENSIVE COMPARISON OF DIFFUSION-BASED EDAs")
    print("# Dimension: n=20")
    print("# Functions: Sphere, Rosenbrock, Rastrigin, Ackley, Griewank")
    print("# EDAs: DbD-CS, DbD-CD, DbD-UC, DbD-US, DenDiff")
    print("#"*80 + "\n")

    # Run experiments
    results = run_comparison_experiments(
        n_runs=5,
        n_generations=50,
        save_results=True
    )

    # Analyze results
    analyze_results(results)

    print("\n" + "#"*80)
    print("# EXPERIMENT COMPLETED SUCCESSFULLY")
    print("#"*80 + "\n")


if __name__ == '__main__':
    main()
