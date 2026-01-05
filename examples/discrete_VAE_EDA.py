"""
Discrete VAE EDA - Command-Line Interface for VAE-EDA Variants
===============================================================

This program provides a unified interface to run various Variational Autoencoder (VAE)
based EDA algorithms on benchmark problems with different seeds for cluster execution.

Supports six VAE-EDA variants:
- VAE: Standard Variational Autoencoder with β-annealing
- E-VAE: Enhanced VAE with fitness predictor
- C-VAE: Conditional VAE conditioned on fitness and statistics
- Desc-VAE: Descriptor-augmented VAE with landscape information
- Reg-VAE: Regression-focused VAE with fitness-weighted reconstruction
- Mom-VAE: Moment-matching VAE with statistical alignment

Configurable Parameters:
- Activation Functions: For encoder and decoder layers
- Beta Annealing: KL divergence weight scheduling
- Latent Dimensions: Size of latent space
- Hidden Layer Dimensions: Defined in terms of number of variables
- Batch Size: Adaptive based on selected population size

Usage:
    python discrete_VAE_EDA.py <seed> <obj_func> <n> <pop_size> <n_gen> <trunc> <vae_variant> \\
        [--activation-enc ACT] [--activation-dec ACT] [--beta-start BETA] [--beta-end BETA]

Example:
    python discrete_VAE_EDA.py 0 OneMax 20 80 20 0.5 VAE --activation-enc relu --activation-dec tanh
    python discrete_VAE_EDA.py 1 Deceptive3 30 100 30 0.5 C-VAE --beta-start 0.0 --beta-end 1.0

==============================================================================
"""

import sys
import os
import argparse

# Add parent directory to path for running examples without installation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import time
import math
from typing import Dict, Any, Optional
import warnings

# VAE learning modules
from pateda.learning.discrete_vae import (
    learn_binary_vae,
    learn_binary_cvae,
    learn_binary_descvae,
    learn_binary_regvae,
    learn_binary_momvae
)

# VAE sampling modules
from pateda.sampling.discrete_neural import (
    sample_binary_vae,
    sample_binary_cvae,
    sample_binary_descvae,
    sample_binary_regvae,
    sample_binary_momvae
)

# Benchmark functions
from pateda.functions.discrete.additive_decomposable import (
    k_deceptive, decep3, decep_marta3, decep_marta3_new, decep3_mh,
    two_peaks_decep3, decep_venturini, hard_decep5,
    hiff, fhtrap1,
    first_polytree3_ochoa, first_polytree5_ochoa,
    fc2, fc3, fc4, fc5
)


# ==============================================================================
# Constants
# ==============================================================================

# Success threshold as a fraction of optimal fitness
SUCCESS_THRESHOLD = 0.01


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
        try:
            k = int(obj_func[len('KDeceptive'):])
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
        if n_vars <= 0 or (n_vars & (n_vars - 1) != 0):
            raise ValueError(f"For HIFF, n must be a power of 2 (e.g., 1, 2, 4, 8, 16, 32, 64, 128)")
        return wrap_function(hiff), n_vars, float(n_vars * (1 + int(math.log2(n_vars))))

    # FHTrap1 (Hierarchical Trap)
    elif obj_func == 'FHTrap1':
        if n_vars <= 0:
            raise ValueError(f"For FHTrap1, n must be positive")
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
# VAE EDA Implementation
# ==============================================================================

class VAEEDA:
    """
    Unified framework for VAE-based EDAs with configurable parameters
    """

    def __init__(
        self,
        variant: str,
        n_vars: int,
        cardinality: np.ndarray,
        pop_size: int = 100,
        selection_ratio: float = 0.5,
        max_generations: int = 50,
        activation_enc: str = 'relu',
        activation_dec: str = 'relu',
        beta_start: float = 0.0,
        beta_end: float = 1.0,
        learning_params: Optional[Dict[str, Any]] = None,
        sampling_params: Optional[Dict[str, Any]] = None,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize VAE EDA

        Parameters
        ----------
        variant : str
            VAE variant: 'vae', 'evae', 'cvae', 'descvae', 'regvae', 'momvae'
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
        activation_enc : str
            Activation function for encoder (relu, tanh, sigmoid, leakyrelu, etc.)
        activation_dec : str
            Activation function for decoder
        beta_start : float
            Initial KL weight for annealing
        beta_end : float
            Final KL weight for annealing
        learning_params : dict, optional
            Additional learning parameters
        sampling_params : dict, optional
            Additional sampling parameters
        random_seed : int, optional
            Random seed for reproducibility
        """
        self.variant = variant
        self.n_vars = n_vars
        self.cardinality = cardinality
        self.pop_size = pop_size
        self.selection_ratio = selection_ratio
        self.max_generations = max_generations
        self.activation_enc = activation_enc
        self.activation_dec = activation_dec
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.learning_params = learning_params or {}
        self.sampling_params = sampling_params or {}
        self.random_seed = random_seed

        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)

        # Map variant to learning and sampling functions
        self.variant_map = {
            'vae': (learn_binary_vae, sample_binary_vae),
            'evae': (learn_binary_vae, sample_binary_vae),  # E-VAE uses learn_binary_vae with use_extended=True
            'cvae': (learn_binary_cvae, sample_binary_cvae),
            'descvae': (learn_binary_descvae, sample_binary_descvae),
            'regvae': (learn_binary_regvae, sample_binary_regvae),
            'momvae': (learn_binary_momvae, sample_binary_momvae),
        }

        if variant not in self.variant_map:
            raise ValueError(f"Invalid variant: {variant}. Must be one of {list(self.variant_map.keys())}")

    def run(self, fitness_func, verbose=True):
        """
        Run the VAE EDA

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
        learn_fn, sample_fn = self.variant_map[self.variant]

        # Initialize population
        population = np.random.randint(0, self.cardinality, (self.pop_size, self.n_vars))

        # Evaluate
        fitness = fitness_func(population)

        best_fitness = np.max(fitness)
        best_solution = population[np.argmax(fitness)].copy()
        generation_found = 0

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
            # Dynamic hidden layer computation based on n_vars and population size
            h1 = min(self.n_vars, n_selected)
            latent_dim = self.learning_params.get('latent_dim', max(2, self.n_vars // 4))
            target_params = 4.5 * n_selected
            h2 = max(4, int((target_params - self.n_vars * h1) / (h1 + latent_dim)))
            adaptive_hidden_dims_enc = [h1, h2]
            adaptive_hidden_dims_dec = list(reversed(adaptive_hidden_dims_enc))

            # Adaptive batch size: max(10, selected_pop_size/20)
            batch_s = max(10, int(n_selected / 20))

            learning_params = self.learning_params.copy()
            learning_params['latent_dim'] = latent_dim
            learning_params['hidden_dims_enc'] = learning_params.get('hidden_dims_enc', adaptive_hidden_dims_enc)
            learning_params['hidden_dims_dec'] = learning_params.get('hidden_dims_dec', adaptive_hidden_dims_dec)
            learning_params['batch_size'] = learning_params.get('batch_size', batch_s)
            learning_params['epochs'] = learning_params.get('epochs', 30)
            learning_params['beta_start'] = self.beta_start
            learning_params['beta_end'] = self.beta_end
            learning_params['beta_annealing_epochs'] = learning_params.get('beta_annealing_epochs', 15)

            # Pass activation functions
            n_hidden_enc = len(learning_params['hidden_dims_enc'])
            n_hidden_dec = len(learning_params['hidden_dims_dec'])
            learning_params['list_act_functs_enc'] = [self.activation_enc] * n_hidden_enc
            learning_params['list_act_functs_dec'] = [self.activation_dec] * n_hidden_dec

            # Variant-specific parameters
            if self.variant == 'evae':
                learning_params['use_extended'] = True
                learning_params['fitness_weight'] = learning_params.get('fitness_weight', 0.1)
            elif self.variant == 'regvae':
                learning_params['fitness_weight'] = learning_params.get('fitness_weight', 0.1)
                learning_params['use_fitness_weighting'] = learning_params.get('use_fitness_weighting', True)
            elif self.variant == 'momvae':
                learning_params['moment_weight'] = learning_params.get('moment_weight', 0.1)

            # Learn model
            try:
                model = learn_fn(selected_pop, selected_fitness, learning_params)

                # Prepare sampling parameters
                sampling_params = self.sampling_params.copy()

                # Variant-specific sampling
                if self.variant == 'evae':
                    sampling_params['use_fitness_guidance'] = sampling_params.get('use_fitness_guidance', False)
                elif self.variant == 'cvae':
                    # C-VAE: can specify target condition for sampling
                    pass
                elif self.variant == 'regvae':
                    sampling_params['use_fitness_guidance'] = sampling_params.get('use_fitness_guidance', True)

                # Sample new population
                population = sample_fn(model, self.pop_size, sampling_params)

            except Exception as e:
                if verbose:
                    print(f"  Warning: Learning/Sampling failed ({e}), using random population")
                population = np.random.randint(0, self.cardinality,
                                             (self.pop_size, self.n_vars))

            # Evaluate
            fitness = fitness_func(population)

            # Update best
            gen_best = np.max(fitness)
            if gen_best > best_fitness:
                best_fitness = gen_best
                best_solution = population[np.argmax(fitness)].copy()
                generation_found = gen + 1

            history['best_fitness'].append(best_fitness)

            if verbose and (gen + 1) % 1 == 0:
                print(f"Generation {gen+1}: Best Fitness = {best_fitness:.4f}")

        # Print completion summary
        if verbose:
            print(f"\nVAE-EDA completed after {self.max_generations} generations")
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
        description='Discrete VAE EDA - Configurable VAE Algorithm Variants',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python discrete_VAE_EDA.py 0 OneMax 20 80 20 0.5 VAE

  # With custom activation functions
  python discrete_VAE_EDA.py 0 OneMax 20 80 20 0.5 E-VAE --activation-enc tanh --activation-dec relu

  # Conditional VAE with beta annealing
  python discrete_VAE_EDA.py 0 Deceptive3 30 100 30 0.5 C-VAE --beta-start 0.0 --beta-end 1.0

  # Regression VAE
  python discrete_VAE_EDA.py 0 HIFF 64 200 50 0.5 Reg-VAE --activation-enc leakyrelu
        """
    )

    # Required positional arguments
    parser.add_argument('seed', type=int, help='Random seed')
    parser.add_argument('obj_func', type=str, help='Objective function name')
    parser.add_argument('n', type=int, help='Number of variables')
    parser.add_argument('pop_size', type=int, help='Population size')
    parser.add_argument('n_gen', type=int, help='Number of generations')
    parser.add_argument('trunc', type=float, help='Truncation percent (selection ratio, e.g., 0.5 for 50%%)')
    parser.add_argument('vae_variant', type=str,
                        choices=['VAE', 'E-VAE', 'C-VAE', 'Desc-VAE', 'Reg-VAE', 'Mom-VAE'],
                        help='VAE variant to use')

    # Optional configuration arguments
    parser.add_argument('--activation-enc', type=str, default='relu',
                        help='Activation function for encoder (default: relu). Options: relu, tanh, sigmoid, leakyrelu, elu, selu, gelu, etc.')
    parser.add_argument('--activation-dec', type=str, default='relu',
                        help='Activation function for decoder (default: relu)')
    parser.add_argument('--beta-start', type=float, default=0.0,
                        help='Initial KL weight for beta annealing (default: 0.0)')
    parser.add_argument('--beta-end', type=float, default=1.0,
                        help='Final KL weight for beta annealing (default: 1.0)')
    parser.add_argument('--latent-dim', type=int, default=None,
                        help='Latent dimension (default: max(2, n_vars/4))')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Training epochs per generation (default: 30)')

    # Parse arguments
    args = parser.parse_args()

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
    print("DISCRETE VAE EDA - Configuration")
    print("=" * 80)
    print(f"Seed:               {args.seed}")
    print(f"Problem:            {args.obj_func}")
    print(f"Variables:          {n_vars}")
    print(f"Optimal Fitness:    {optimal_fitness}")
    print(f"Population Size:    {args.pop_size}")
    print(f"Generations:        {args.n_gen}")
    print(f"Truncation Percent: {args.trunc}")
    print(f"VAE Variant:        {args.vae_variant}")
    print(f"Activation Enc:     {args.activation_enc}")
    print(f"Activation Dec:     {args.activation_dec}")
    print(f"Beta Start:         {args.beta_start}")
    print(f"Beta End:           {args.beta_end}")
    if args.latent_dim:
        print(f"Latent Dim:         {args.latent_dim}")
    print(f"Epochs:             {args.epochs}")
    print("=" * 80)
    print()

    start_time = time.time()

    # Map variant names to internal names
    variant_map = {
        'VAE': 'vae',
        'E-VAE': 'evae',
        'C-VAE': 'cvae',
        'Desc-VAE': 'descvae',
        'Reg-VAE': 'regvae',
        'Mom-VAE': 'momvae',
    }

    variant_id = variant_map[args.vae_variant]

    # Configure learning parameters
    learning_params = {
        'epochs': args.epochs,
    }
    if args.latent_dim:
        learning_params['latent_dim'] = args.latent_dim

    # Configure sampling parameters
    sampling_params = {}

    cardinality = np.full(n_vars, 2)

    # Create and run VAE EDA
    eda = VAEEDA(
        variant=variant_id,
        n_vars=n_vars,
        cardinality=cardinality,
        pop_size=args.pop_size,
        selection_ratio=args.trunc,
        max_generations=args.n_gen,
        activation_enc=args.activation_enc,
        activation_dec=args.activation_dec,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        learning_params=learning_params,
        sampling_params=sampling_params,
        random_seed=args.seed,
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
