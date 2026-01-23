"""
Discrete GAN EDA - Command-Line Interface for GAN-EDA Variants
===============================================================

This program provides a unified interface to run various GAN-EDA algorithm
variants on benchmark problems with different seeds for cluster execution.

Supports seven GAN-EDA variants as described in Deeper_GAN_Critical_Analysis.md:

1. V1-WGAN-GP: Wasserstein Loss + Gradient Penalty
2. V2-Cond-Fit-GAN: Condition input on target fitness percentiles
3. V3-Aux-GAN: Auxiliary head for fitness prediction
4. V4-Repulsion-GAN: Batch-wide diversity penalty in Generator
5. V5-Weighted-D-GAN: Fitness-weighted Real/Fake classification
6. V6-Statistic-Match: MSE loss on mean/std of generated batch
7. V7-Hybrid-GAN-VAE: GAN with an Encoder (BiGAN)

Configurable Parameters:
- Activation Function for Generator: relu, tanh, sigmoid, leakyrelu, etc.
- Activation Function for Discriminator: relu, tanh, sigmoid, leakyrelu, etc.
- Activation Function for Encoder (V7 only): relu, tanh, sigmoid, leakyrelu, etc.
- Dropout rate: Discriminator dropout rate (default: 0.5 for stability)
- Temperature: Gumbel-Softmax temperature
- Learning rates: Separate for Generator and Discriminator
- Hidden dimensions: Automatically computed from n_vars and pop_size
- Batch size: Automatically computed from selected population size
- Truncation Percent: Selection ratio for truncation selection

Usage:
    python discrete_GAN_EDA.py <seed> <obj_func> <n> <pop_size> <n_gen> <trunc> <variant> \\
        <activation_g> <activation_d> <activation_e> <dropout> <temperature> <use_surrogate>

Example:
    python discrete_GAN_EDA.py 0 OneMax 20 80 20 0.5 WGAN-GP relu leaky_relu relu 0.5 1.0 0
    python discrete_GAN_EDA.py 1 Deceptive3 30 100 30 0.5 Aux-GAN relu leaky_relu relu 0.5 1.0 0

==============================================================================
"""

import sys
import os
import argparse
import random

# Add parent directory to path for running examples without installation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import time
import math
from typing import Dict, Any, Optional
import warnings

# GAN learning modules
from pateda.learning.discrete_gan import (
    learn_binary_gan,
    learn_binary_gan_wgan_gp,
    learn_binary_gan_cond_fit,
    learn_binary_gan_aux,
    learn_binary_gan_repulsion,
    learn_binary_gan_weighted_d,
    learn_binary_gan_statistic_match,
    learn_binary_gan_hybrid_vae,
)

# GAN sampling modules
from pateda.sampling.discrete_neural import (
    sample_binary_gan,
    sample_binary_gan_cond_fit,
    sample_binary_gan_aux,
    sample_binary_gan_hybrid_vae,
)

# Benchmark functions
from pateda.functions.discrete.additive_decomposable import (
    k_deceptive, decep3, decep_marta3, decep_marta3_new, decep3_mh,
    two_peaks_decep3, decep_venturini, hard_decep5,
    hiff, fhtrap1,
    first_polytree3_ochoa, first_polytree5_ochoa,
    fc2, fc3, fc4, fc5
)

# Mutation operators
from pateda.operators.mutation import frequency_balance_mutation


# ==============================================================================
# Constants
# ==============================================================================

# Success threshold as a fraction of optimal fitness
SUCCESS_THRESHOLD = 0.01


# ==============================================================================
# Seeding Utilities
# ==============================================================================

def set_seed(seed: int):
    """
    Set all random seeds for reproducibility.
    
    This function sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)
    - PyTorch deterministic operations
    
    Parameters
    ----------
    seed : int
        Random seed value
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        
        # Set deterministic behavior for reproducibility
        # Note: This may impact performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # For some operations on CUDA >= 10.2
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        # PyTorch not available, skip torch seeding
        pass


# ==============================================================================
# Fitness Function Wrappers
# ==============================================================================

def onemax(x: np.ndarray) -> np.ndarray:
    """OneMax function"""
    if x.ndim == 1:
        return np.array([float(np.sum(x))])
    else:
        return np.sum(x, axis=1).astype(float)


def wrap_k_deceptive(k: int):
    """K-deceptive wrapper factory"""
    def wrapped(x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            return np.array([k_deceptive(x, k=k)])
        else:
            return np.array([k_deceptive(ind, k=k) for ind in x])
    return wrapped


def wrap_decep3(overlap: bool = False):
    """Deceptive-3 wrapper factory"""
    def wrapped(x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            return np.array([decep3(x, overlap=overlap)])
        else:
            return np.array([decep3(ind, overlap=overlap) for ind in x])
    return wrapped


def wrap_function(func):
    """Generic function wrapper"""
    def wrapped(x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            return np.array([func(x)])
        else:
            return np.array([func(ind) for ind in x])
    return wrapped


def wrap_polytree3(overlap: bool):
    """Polytree-3 wrapper factory"""
    def wrapped(x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            return np.array([first_polytree3_ochoa(x, overlap=overlap)])
        else:
            return np.array([first_polytree3_ochoa(ind, overlap=overlap) for ind in x])
    return wrapped


# ==============================================================================
# Problem Configuration
# ==============================================================================

def parse_problem(obj_func: str, n: int):
    """
    Parse problem name and return fitness function, n_vars, and optimal fitness

    Parameters
    ----------
    obj_func : str
        Problem name (e.g., 'OneMax', 'Deceptive3', 'KDeceptive3', 'HIFF')
    n : int
        Number of variables

    Returns
    -------
    func : callable
        Fitness function
    n_vars : int
        Number of variables
    optimal : float
        Optimal fitness value (or approximate optimal if not known)
    """
    n_vars = n

    # OneMax family
    if obj_func == 'OneMax':
        return onemax, n_vars, float(n_vars)

    # K-Deceptive family
    elif obj_func.startswith('KDeceptive'):
        # Parse k value (e.g., KDeceptive3, KDeceptive5)
        try:
            k = int(obj_func[len('KDeceptive'):])  # Extract number after 'KDeceptive'
        except (ValueError, IndexError):
            raise ValueError(f"Invalid KDeceptive format: {obj_func}. Expected format: KDeceptive<k> (e.g., KDeceptive3)")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if n_vars % k != 0:
            raise ValueError(f"For KDeceptive{k}, n must be a multiple of {k}")
        return wrap_k_deceptive(k), n_vars, float(n_vars)

    # Deceptive-3 variants
    elif obj_func == 'Deceptive3':
        if n_vars % 3 != 0:
            raise ValueError(f"For Deceptive3, n must be a multiple of 3")
        n_blocks = n_vars // 3
        return wrap_decep3(overlap=False), n_vars, float(n_blocks)

    elif obj_func == 'Deceptive3Overlap':
        # With overlap=True, uses overlapping partitions with step=2
        # Number of partitions: (n - 2) // 2
        return wrap_decep3(overlap=True), n_vars, float((n_vars - 2) // 2)

    elif obj_func == 'DecepMarta3':
        if n_vars % 3 != 0:
            raise ValueError(f"For DecepMarta3, n must be a multiple of 3")
        return wrap_function(decep_marta3), n_vars, float(n_vars)

    elif obj_func == 'DecepMarta3New':
        if n_vars % 3 != 0:
            raise ValueError(f"For DecepMarta3New, n must be a multiple of 3")
        return wrap_function(decep_marta3_new), n_vars, float(n_vars)

    elif obj_func == 'Decep3MH':
        if n_vars % 3 != 0:
            raise ValueError(f"For Decep3MH, n must be a multiple of 3")
        return wrap_function(decep3_mh), n_vars, float(n_vars)

    elif obj_func == 'TwoPeaksDecep3':
        return wrap_function(two_peaks_decep3), n_vars, float(n_vars)

    elif obj_func == 'DecepVenturini':
        if n_vars % 3 != 0:
            raise ValueError(f"For DecepVenturini, n must be a multiple of 3")
        return wrap_function(decep_venturini), n_vars, float(n_vars)

    # Hard Deceptive-5
    elif obj_func == 'HardDecep5':
        if n_vars % 5 != 0:
            raise ValueError(f"For HardDecep5, n must be a multiple of 5")
        return wrap_function(hard_decep5), n_vars, float(n_vars)

    # HIFF (Hierarchical If and only If)
    elif obj_func == 'HIFF':
        # Check if n is a power of 2 using bit manipulation
        # Power of 2 numbers have only one bit set: n & (n-1) == 0
        if n_vars <= 0 or (n_vars & (n_vars - 1) != 0):
            raise ValueError(f"For HIFF, n must be a power of 2 (e.g., 1, 2, 4, 8, 16, 32, 64, 128)")
        # HIFF optimal is n * (log2(n) + 1)
        # Each level contributes n, and there are log2(n) + 1 levels (including base level)
        return wrap_function(hiff), n_vars, float(n_vars * (1 + int(math.log2(n_vars))))

    # FHTrap1 (Hierarchical Trap)
    elif obj_func == 'FHTrap1':
        # Check if n is a power of 3 using iterative division
        if n_vars <= 0:
            raise ValueError(f"For FHTrap1, n must be positive")
        # Check by repeatedly dividing by 3
        temp = n_vars
        while temp > 1 and temp % 3 == 0:
            temp //= 3
        if temp != 1:
            raise ValueError(f"For FHTrap1, n must be a power of 3 (e.g., 9, 27, 81, 243, 729)")
        return wrap_function(fhtrap1), n_vars, float(n_vars)

    # Polytree functions
    elif obj_func == 'Polytree3':
        return wrap_polytree3(overlap=False), n_vars, float(n_vars)

    elif obj_func == 'Polytree3Overlap':
        return wrap_polytree3(overlap=True), n_vars, float(n_vars)

    elif obj_func == 'Polytree5':
        if n_vars % 5 != 0:
            raise ValueError(f"For Polytree5, n must be a multiple of 5")
        return wrap_function(first_polytree5_ochoa), n_vars, float(n_vars)

    # Cuban functions
    elif obj_func == 'FC2':
        if n_vars % 5 != 0:
            raise ValueError(f"For FC2, n must be a multiple of 5")
        return wrap_function(fc2), n_vars, float(n_vars)

    elif obj_func == 'FC3':
        if n_vars % 5 != 0:
            raise ValueError(f"For FC3, n must be a multiple of 5")
        return wrap_function(fc3), n_vars, float(n_vars)

    elif obj_func == 'FC4':
        return wrap_function(fc4), n_vars, float(n_vars)

    elif obj_func == 'FC5':
        return wrap_function(fc5), n_vars, float(n_vars)

    else:
        raise ValueError(f"Unknown problem: {obj_func}")


# ==============================================================================
# GAN EDA Implementation
# ==============================================================================

class GANEDA:
    """
    Configurable GAN-EDA framework with seven variant options
    """

    def __init__(
        self,
        variant: str,
        n_vars: int,
        cardinality: np.ndarray,
        pop_size: int = 100,
        selection_ratio: float = 0.5,
        max_generations: int = 50,
        activation_g: str = 'relu',
        activation_d: str = 'leaky_relu',
        activation_e: str = 'relu',
        dropout: float = 0.5,
        temperature: float = 1.0,
        learning_params: Optional[Dict[str, Any]] = None,
        sampling_params: Optional[Dict[str, Any]] = None,
        random_seed: Optional[int] = None,
        alpha: float = 0.0,
    ):
        """
        Initialize GAN EDA

        Parameters
        ----------
        variant : str
            GAN variant: 'GAN', 'WGAN-GP', 'Cond-Fit-GAN', 'Aux-GAN',
                        'Repulsion-GAN', 'Weighted-D-GAN', 'Statistic-Match', 'Hybrid-GAN-VAE'
        n_vars : int
            Number of variables
        cardinality : np.ndarray
            Cardinality of each variable
        pop_size : int
            Population size
        selection_ratio : float
            Selection ratio (truncation percent)
        max_generations : int
            Maximum generations
        activation_g : str
            Activation function for Generator (relu, tanh, sigmoid, leaky_relu, etc.)
        activation_d : str
            Activation function for Discriminator (relu, tanh, sigmoid, leaky_relu, etc.)
        activation_e : str
            Activation function for Encoder (for Hybrid-GAN-VAE variant)
        dropout : float
            Dropout rate for discriminator (default: 0.5 for stability)
        temperature : float
            Gumbel-Softmax temperature
        learning_params : dict, optional
            Additional learning parameters
        sampling_params : dict, optional
            Additional sampling parameters
        random_seed : int, optional
            Random seed for reproducibility
        alpha : float
            Max frequency threshold for mutation (default: 0.0, no mutation)
        """
        self.variant = variant
        self.n_vars = n_vars
        self.cardinality = cardinality
        self.pop_size = pop_size
        self.selection_ratio = selection_ratio
        self.max_generations = max_generations
        self.activation_g = activation_g
        self.activation_d = activation_d
        self.activation_e = activation_e
        self.dropout = dropout
        self.temperature = temperature
        self.learning_params = learning_params or {}
        self.sampling_params = sampling_params or {}
        self.random_seed = random_seed
        self.alpha = alpha

        # Set random seed if provided
        if random_seed is not None:
            set_seed(random_seed)

        # Validate parameters
        valid_variants = ['GAN', 'WGAN-GP', 'Cond-Fit-GAN', 'Aux-GAN',
                         'Repulsion-GAN', 'Weighted-D-GAN', 'Statistic-Match', 'Hybrid-GAN-VAE']
        if variant not in valid_variants:
            raise ValueError(f"Invalid variant: {variant}. Must be one of {valid_variants}")

        # Map variant to learning and sampling functions
        self.learning_function_map = {
            'GAN': learn_binary_gan,
            'WGAN-GP': learn_binary_gan_wgan_gp,
            'Cond-Fit-GAN': learn_binary_gan_cond_fit,
            'Aux-GAN': learn_binary_gan_aux,
            'Repulsion-GAN': learn_binary_gan_repulsion,
            'Weighted-D-GAN': learn_binary_gan_weighted_d,
            'Statistic-Match': learn_binary_gan_statistic_match,
            'Hybrid-GAN-VAE': learn_binary_gan_hybrid_vae,
        }

        self.sampling_function_map = {
            'GAN': sample_binary_gan,
            'WGAN-GP': sample_binary_gan,
            'Cond-Fit-GAN': sample_binary_gan_cond_fit,
            'Aux-GAN': sample_binary_gan_aux,
            'Repulsion-GAN': sample_binary_gan,
            'Weighted-D-GAN': sample_binary_gan,
            'Statistic-Match': sample_binary_gan,
            'Hybrid-GAN-VAE': sample_binary_gan_hybrid_vae,
        }

    def run(self, fitness_func, verbose=True):
        """
        Run the GAN EDA

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
        # Get learning and sampling functions
        learn_fn = self.learning_function_map[self.variant]
        sample_fn = self.sampling_function_map[self.variant]

        # Initialize population
        population = np.random.randint(0, self.cardinality, (self.pop_size, self.n_vars))

        # Evaluate
        fitness = fitness_func(population)

        best_fitness = np.max(fitness)
        best_solution = population[np.argmax(fitness)].copy()
        generation_found = 0  # Track generation where best was found

        history = {'best_fitness': [best_fitness]}

        if verbose:
            print(f"Generation 0: Best Fitness = {best_fitness:.4f}")

        for gen in range(self.max_generations):
            # Selection
            n_selected = int(self.pop_size * self.selection_ratio)
            selected_idx = np.argsort(fitness)[-n_selected:]
            selected_pop = population[selected_idx]
            selected_fitness = fitness[selected_idx]

            # Prepare learning parameters
            learning_params = self.learning_params.copy()

            # Add activation functions
            if 'hidden_dims_g' in learning_params:
                n_hidden_g = len(learning_params['hidden_dims_g'])
                learning_params['list_act_functs_g'] = [self.activation_g] * n_hidden_g
            if 'hidden_dims_d' in learning_params:
                n_hidden_d = len(learning_params['hidden_dims_d'])
                learning_params['list_act_functs_d'] = [self.activation_d] * n_hidden_d

            # For Hybrid-GAN-VAE, add encoder activations
            if self.variant == 'Hybrid-GAN-VAE' and 'hidden_dims_e' in learning_params:
                n_hidden_e = len(learning_params['hidden_dims_e'])
                learning_params['list_act_functs_e'] = [self.activation_e] * n_hidden_e

            # Add dropout and temperature
            learning_params['dropout'] = self.dropout
            learning_params['temperature'] = self.temperature

            # Learn model
            try:
                model = learn_fn(selected_pop, selected_fitness, learning_params)

                # Prepare sampling parameters
                sampling_params = self.sampling_params.copy()

                # For Cond-Fit-GAN, add fitness percentile information
                if self.variant == 'Cond-Fit-GAN':
                    sampling_params['selected_fitness'] = selected_fitness

                # For Aux-GAN, add surrogate filtering option
                if self.variant == 'Aux-GAN':
                    sampling_params['use_surrogate'] = sampling_params.get('use_surrogate', False)

                # Sample new population
                population = sample_fn(model, self.pop_size, sampling_params)

            except Exception as e:
                if verbose:
                    print(f"  Warning: Learning/Sampling failed ({e}), using random population")
                population = np.random.randint(0, self.cardinality,
                                             (self.pop_size, self.n_vars))

            # Apply frequency balance mutation if alpha > 0
            best_solution_pre_mutation = best_solution.copy()
            if self.alpha > 0:
                mutation_params = {'alpha': self.alpha}
                population = frequency_balance_mutation(
                    self.n_vars,
                    self.cardinality,
                    population,
                    mutation_params
                )
                # Elitism: replace first individual with best from previous generation
                population[0] = best_solution_pre_mutation

            # Evaluate
            fitness = fitness_func(population)

            # Update best
            gen_best = np.max(fitness)
            if gen_best > best_fitness:
                best_fitness = gen_best
                best_solution = population[np.argmax(fitness)].copy()
                generation_found = gen + 1  # Update generation where best was found

            history['best_fitness'].append(best_fitness)

            if verbose and (gen + 1) % 1 == 0:
                print(f"Generation {gen+1}: Best Fitness = {best_fitness:.4f}")

        # Print completion summary
        if verbose:
            print(f"\nGAN-EDA completed after {self.max_generations} generations")
            print(f"Best fitness found: {best_fitness:.6f}")
            print(f"  at generation {generation_found}")

        return best_fitness, best_solution, history


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    """Main entry point for command-line execution"""

    # Create argument parser
    parser = argparse.ArgumentParser(
        description='Discrete GAN EDA - Seven GAN Algorithm Variants',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with WGAN-GP variant
  python discrete_GAN_EDA.py 0 OneMax 20 80 20 0.5 WGAN-GP relu leaky_relu relu 0.5 1.0 0

  # Cond-Fit-GAN with custom activations
  python discrete_GAN_EDA.py 0 Deceptive3 30 100 30 0.5 Cond-Fit-GAN tanh leaky_relu relu 0.5 1.0 0

  # Aux-GAN with surrogate filtering
  python discrete_GAN_EDA.py 0 HIFF 64 200 50 0.5 Aux-GAN relu leaky_relu relu 0.5 1.0 1
        """
    )

    # All positional arguments
    parser.add_argument('seed', type=int, help='Random seed')
    parser.add_argument('obj_func', type=str, help='Objective function name')
    parser.add_argument('n', type=int, help='Number of variables')
    parser.add_argument('pop_size', type=int, help='Population size')
    parser.add_argument('n_gen', type=int, help='Number of generations')
    parser.add_argument('trunc', type=float, help='Truncation percent (selection ratio, e.g., 0.5 for 50%)')
    parser.add_argument('variant', type=str,
                       choices=['GAN', 'WGAN-GP', 'Cond-Fit-GAN', 'Aux-GAN',
                               'Repulsion-GAN', 'Weighted-D-GAN', 'Statistic-Match', 'Hybrid-GAN-VAE'],
                       help='GAN variant to use')
    parser.add_argument('activation_g', type=str,
                       help='Activation function for Generator. Options: relu, tanh, sigmoid, leaky_relu, elu, selu, gelu, etc.')
    parser.add_argument('activation_d', type=str,
                       help='Activation function for Discriminator')
    parser.add_argument('activation_e', type=str,
                       help='Activation function for Encoder (Hybrid-GAN-VAE only)')
    parser.add_argument('dropout', type=float,
                       help='Dropout rate for discriminator')
    parser.add_argument('temperature', type=float,
                       help='Gumbel-Softmax temperature')
    parser.add_argument('use_surrogate', type=int, choices=[0, 1],
                       help='Use surrogate model for pre-filtering solutions (1=yes, 0=no, Aux-GAN only)')
    parser.add_argument('alpha', type=float, nargs='?', default=0.0,
                       help='Max frequency threshold for mutation (default: 0.0, no mutation)')

    # Parse arguments
    args = parser.parse_args()

    # Convert integer flags to boolean
    args.use_surrogate = bool(args.use_surrogate)

    # Validate truncation percent
    if args.trunc <= 0 or args.trunc > 1:
        print(f"Error: Truncation percent must be between 0 and 1, got {args.trunc}")
        sys.exit(1)

    # Suppress warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    # Parse problem
    try:
        fitness_func, n_vars, optimal_fitness = parse_problem(args.obj_func, args.n)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Print configuration
    print("=" * 80)
    print("DISCRETE GAN EDA - Configuration")
    print("=" * 80)
    print(f"Seed:               {args.seed}")
    print(f"Problem:            {args.obj_func}")
    print(f"Variables:          {n_vars}")
    print(f"Optimal Fitness:    {optimal_fitness}")
    print(f"Population Size:    {args.pop_size}")
    print(f"Generations:        {args.n_gen}")
    print(f"Truncation Percent: {args.trunc}")
    print(f"Variant:            {args.variant}")
    print(f"Activation (Gen):   {args.activation_g}")
    print(f"Activation (Disc):  {args.activation_d}")
    if args.variant == 'Hybrid-GAN-VAE':
        print(f"Activation (Enc):   {args.activation_e}")
    print(f"Dropout:            {args.dropout}")
    print(f"Temperature:        {args.temperature}")
    if args.variant == 'Aux-GAN':
        print(f"Use Surrogate:      {args.use_surrogate}")
    print("=" * 80)
    print()

    start_time = time.time()

    # Compute common parameters based on pop_size and n_vars
    selected_pop_size = int(args.pop_size * args.trunc)

    # CRITICAL: Dynamic hidden layer sizing based on population and problem size
    # As described in Deeper_GAN_Critical_Analysis.md:
    # "hidden layer width is a function of the number of selected individuals"
    adaptive_hidden_dims_g = [max(10, n_vars // 2), max(10, n_vars // 4)]
    adaptive_hidden_dims_d = list(reversed(adaptive_hidden_dims_g))

    # CRITICAL: Batch size depends on selected population size
    # "batch size depends on the size of the selected population, e.g., max(10,selected_pop_size/20)"
    batch_s = max(10, selected_pop_size // 20)

    # Configure learning parameters
    learning_params = {
        'epochs': 60,
        'latent_dim': max(10, n_vars // 2),
        'hidden_dims_g': adaptive_hidden_dims_g,
        'hidden_dims_d': adaptive_hidden_dims_d,
        'batch_size': batch_s,
        'learning_rate_g': 0.0002,
        'learning_rate_d': 0.0002,
        'dropout': args.dropout,
        'temperature': args.temperature,
    }

    # For Hybrid-GAN-VAE, add encoder dimensions
    if args.variant == 'Hybrid-GAN-VAE':
        learning_params['hidden_dims_e'] = adaptive_hidden_dims_g

    # Configure sampling parameters
    sampling_params = {
        'temperature': args.temperature,
    }

    if args.variant == 'Aux-GAN':
        sampling_params['use_surrogate'] = args.use_surrogate

    cardinality = np.full(n_vars, 2)

    # Create and run GAN EDA
    eda = GANEDA(
        variant=args.variant,
        n_vars=n_vars,
        cardinality=cardinality,
        pop_size=args.pop_size,
        selection_ratio=args.trunc,
        max_generations=args.n_gen,
        activation_g=args.activation_g,
        activation_d=args.activation_d,
        activation_e=args.activation_e,
        dropout=args.dropout,
        temperature=args.temperature,
        learning_params=learning_params,
        sampling_params=sampling_params,
        random_seed=args.seed,
        alpha=args.alpha,
    )

    best_fitness, best_solution, history = eda.run(fitness_func, verbose=True)

    elapsed_time = time.time() - start_time

    # Print results
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Best Fitness:     {best_fitness:.4f}")
    print(f"Optimal Fitness:  {optimal_fitness:.4f}")
    print(f"Gap:              {abs(best_fitness - optimal_fitness):.4f}")
    print(f"Success:          {'Yes' if abs(best_fitness - optimal_fitness) < SUCCESS_THRESHOLD * optimal_fitness else 'No'}")
    print(f"Elapsed Time:     {elapsed_time:.2f} seconds")
    print(f"Best Solution:    {best_solution[:20]}{'...' if len(best_solution) > 20 else ''}")
    print("=" * 80)


if __name__ == "__main__":
    main()
