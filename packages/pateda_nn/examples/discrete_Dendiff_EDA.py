"""
Discrete Dendiff EDA - Command-Line Interface for Dendiff-EDA Variants
=======================================================================

This program provides a unified interface to run various Denoising Diffusion (Dendiff)
based EDA algorithms on benchmark problems with different seeds for cluster execution.

Supports comprehensive Dendiff-EDA variants:

**Standard Dendiff Variants:**
- Dendiff-Gumbel: Gumbel-Softmax based discrete diffusion
- Dendiff-Corruption: Corruption/denoising based discrete diffusion (BERT-style)

**New Alternative Sampling Strategies:**
- Dendiff-STE: Straight-Through Estimator - hard values in forward, gradient flow in backward
- Dendiff-HardConcrete: Hard Concrete distribution with stretching/folding for exact 0s and 1s
- Dendiff-Deterministic: Deterministic softmax without Gumbel noise for cleaner gradients

**Enhanced Dendiff Variants (with alternative loss functions):**
- Dendiff-Gumbel-WeightedMSE: Gumbel variant with fitness-weighted loss
- Dendiff-Gumbel-Ranking: Gumbel variant with ranking loss
- Dendiff-Gumbel-Huber: Gumbel variant with Huber loss (robust to outliers)
- Dendiff-Corruption-WeightedMSE: Corruption variant with fitness-weighted loss
- Dendiff-Corruption-Ranking: Corruption variant with ranking loss
- Dendiff-Corruption-Huber: Corruption variant with Huber loss

**Fitness-Guided Dendiff Variants:**
- Dendiff-Gumbel-FitnessGuided: Conditional on fitness (inspired by C-VAE)
- Dendiff-Corruption-FitnessGuided: Corruption variant with fitness conditioning

Configurable Parameters (all positional):
- seed: Random seed
- obj_func: Objective function name
- n: Number of variables
- pop_size: Population size
- n_gen: Number of generations
- trunc: Truncation percent (selection ratio, e.g., 0.5 for 50%)
- variant: Dendiff variant (dendiff_gumbel, dendiff_corruption, dendiff_ste, dendiff_hard_concrete, dendiff_deterministic)
- sampling_strategy: Differentiable sampling strategy (gumbel, corruption, ste, hard_concrete, deterministic)
- activation: Activation function (relu, tanh, sigmoid, leakyrelu, elu, selu, gelu, etc.)
- loss: Loss function (mse, weighted_mse, ranking, huber)
- n_timesteps: Number of diffusion timesteps (e.g., 50 or 100)
- n_sampling_steps: Number of denoising steps during sampling (e.g., 20)
- fitness_guided: Use fitness guidance/conditioning (1=yes, 0=no)
- temperature: Gumbel-Softmax temperature or sampling temperature (e.g., 1.0)
- beta_start: Starting noise/corruption level (e.g., 0.0001)
- beta_end: Ending noise/corruption level (e.g., 0.3 for Gumbel, 0.5 for Corruption)
- alpha: (Optional) Max frequency threshold for mutation (default: 0.0, no mutation)

Usage:
    python discrete_Dendiff_EDA.py <seed> <obj_func> <n> <pop_size> <n_gen> <trunc> \\
        <variant> <sampling_strategy> <activation> <loss> <n_timesteps> <n_sampling_steps> \\
        <fitness_guided> <temperature> <beta_start> <beta_end> [alpha]

Examples:
    # Standard Dendiff-Gumbel with default settings
    python discrete_Dendiff_EDA.py 0 OneMax 20 80 20 0.5 dendiff_gumbel gumbel relu mse 100 20 0 1.0 0.0001 0.3

    # Dendiff-Corruption with tanh activation
    python discrete_Dendiff_EDA.py 1 Deceptive3 30 100 30 0.5 dendiff_corruption corruption tanh mse 50 20 0 0.5 0.01 0.5

    # Dendiff-STE (Straight-Through Estimator)
    python discrete_Dendiff_EDA.py 0 OneMax 20 80 20 0.5 dendiff_ste ste relu mse 50 20 0 0.5 0.01 0.5

    # Dendiff-HardConcrete with exact 0s and 1s
    python discrete_Dendiff_EDA.py 0 OneMax 20 80 20 0.5 dendiff_hard_concrete hard_concrete relu mse 100 20 0 0.1 0.0001 0.3

    # Dendiff-Deterministic for optimization tasks
    python discrete_Dendiff_EDA.py 0 OneMax 20 80 20 0.5 dendiff_deterministic deterministic relu mse 100 20 0 1.0 0.0001 0.3

    # Dendiff-Gumbel with weighted MSE and fitness guidance
    python discrete_Dendiff_EDA.py 2 HIFF 64 200 50 0.5 dendiff_gumbel gumbel elu weighted_mse 100 20 1 1.0 0.0001 0.3

    # Dendiff-Corruption with fitness guidance and ranking loss
    python discrete_Dendiff_EDA.py 3 FC3 30 150 40 0.5 dendiff_corruption corruption relu ranking 50 20 1 0.5 0.01 0.5

    # Dendiff-Gumbel with frequency balance mutation (alpha=0.1)
    python discrete_Dendiff_EDA.py 0 OneMax 20 80 20 0.5 dendiff_gumbel gumbel relu mse 100 20 0 1.0 0.0001 0.3 0.1

==============================================================================
"""

import sys
import os
import argparse
import random

# Add parent directory to path for running examples without installation

import numpy as np
import time
import math
from typing import Dict, Any, Optional
import warnings

# Dendiff learning modules - base and enhanced versions
from pateda_nn.learning.discrete_dendiff_gumbel import learn_discrete_dendiff_gumbel
from pateda_nn.learning.discrete_dendiff_corruption import learn_discrete_dendiff_corruption
from pateda_nn.learning.discrete_dendiff_ste import learn_discrete_dendiff_ste
from pateda_nn.learning.discrete_dendiff_hard_concrete import learn_discrete_dendiff_hard_concrete
from pateda_nn.learning.discrete_dendiff_deterministic import learn_discrete_dendiff_deterministic

# Try to import enhanced versions, fall back to base versions if not available
try:
    from pateda_nn.learning.discrete_dendiff_gumbel_enhanced import learn_discrete_dendiff_gumbel_enhanced
    from pateda_nn.learning.discrete_dendiff_corruption_enhanced import learn_discrete_dendiff_corruption_enhanced
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False
    print("Warning: Enhanced Dendiff learning functions not available. Using base versions.")

# Dendiff sampling modules
from pateda_nn.sampling.discrete_dendiff import (
    sample_discrete_dendiff_gumbel,
    sample_discrete_dendiff_corruption,
    sample_discrete_dendiff_ste,
    sample_discrete_dendiff_hard_concrete,
    sample_discrete_dendiff_deterministic
)

# Benchmark functions
from pateda.functions.discrete.additive_decomposable import (
    k_deceptive, decep3, decep_marta3, decep_marta3_new, decep3_mh,
    two_peaks_decep3, decep_venturini, hard_decep5,
    hiff, fhtrap1,
    first_polytree3_ochoa, first_polytree5_ochoa,
    fc2, fc3, fc4, fc5
)

# Mutation modules
from pateda.mutation import frequency_balance_mutation


# ==============================================================================
# Constants
# ==============================================================================

# Success threshold as a fraction of optimal fitness
SUCCESS_THRESHOLD = 0.01

# Tolerance for checking if optimum is reached (absolute difference)
# A value of 1e-6 is used to account for floating-point arithmetic precision
# while being strict enough to ensure the optimum is truly reached
OPTIMUM_TOLERANCE = 1e-6

# Loss functions that require fitness values for computation
LOSS_FUNCTIONS_REQUIRING_FITNESS = ['weighted_mse', 'ranking']


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
# Dendiff EDA Implementation
# ==============================================================================

class DendiffEDA:
    """
    Unified framework for Dendiff-based EDAs with configurable parameters
    """

    def __init__(
        self,
        variant: str,
        n_vars: int,
        cardinality: np.ndarray,
        pop_size: int = 100,
        selection_ratio: float = 0.5,
        max_generations: int = 50,
        sampling_strategy: str = 'gumbel',
        activation: str = 'relu',
        loss_function: str = 'mse',
        n_timesteps: int = 100,
        n_sampling_steps: int = 20,
        fitness_guided: bool = False,
        temperature: float = 1.0,
        beta_start: float = 0.0001,
        beta_end: float = 0.3,
        learning_params: Optional[Dict[str, Any]] = None,
        sampling_params: Optional[Dict[str, Any]] = None,
        random_seed: Optional[int] = None,
        alpha: float = 0.0,
    ):
        """
        Initialize Dendiff EDA

        Parameters
        ----------
        variant : str
            Dendiff variant: 'dendiff_gumbel', 'dendiff_corruption'
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
        sampling_strategy : str
            Sampling strategy: 'gumbel' or 'corruption'
        activation : str
            Activation function (relu, tanh, sigmoid, leakyrelu, elu, selu, gelu, etc.)
        loss_function : str
            Loss function (mse, weighted_mse, ranking, huber)
        n_timesteps : int
            Number of diffusion timesteps during training
        n_sampling_steps : int
            Number of denoising steps during sampling
        fitness_guided : bool
            Use fitness guidance/conditioning
        temperature : float
            Gumbel-Softmax or sampling temperature
        beta_start : float
            Starting noise/corruption level
        beta_end : float
            Ending noise/corruption level
        learning_params : dict, optional
            Additional learning parameters
        sampling_params : dict, optional
            Additional sampling parameters
        random_seed : int, optional
            Random seed for reproducibility
        alpha : float
            Maximum allowed frequency for ones or zeros (default 0.0, no mutation)
            If alpha > 0, applies frequency balance mutation
        """
        self.variant = variant
        self.n_vars = n_vars
        self.cardinality = cardinality
        self.pop_size = pop_size
        self.selection_ratio = selection_ratio
        self.max_generations = max_generations
        self.sampling_strategy = sampling_strategy
        self.activation = activation
        self.loss_function = loss_function
        self.n_timesteps = n_timesteps
        self.n_sampling_steps = n_sampling_steps
        self.fitness_guided = fitness_guided
        self.temperature = temperature
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.learning_params = learning_params or {}
        self.sampling_params = sampling_params or {}
        self.random_seed = random_seed
        self.alpha = alpha

        # Set random seed if provided
        if random_seed is not None:
            set_seed(random_seed)

        # Map variant to learning and sampling functions
        self.variant_map = {
            'dendiff_gumbel': (learn_discrete_dendiff_gumbel, sample_discrete_dendiff_gumbel),
            'dendiff_corruption': (learn_discrete_dendiff_corruption, sample_discrete_dendiff_corruption),
            'dendiff_ste': (learn_discrete_dendiff_ste, sample_discrete_dendiff_ste),
            'dendiff_hard_concrete': (learn_discrete_dendiff_hard_concrete, sample_discrete_dendiff_hard_concrete),
            'dendiff_deterministic': (learn_discrete_dendiff_deterministic, sample_discrete_dendiff_deterministic),
        }

        if variant not in self.variant_map:
            raise ValueError(f"Invalid variant: {variant}. Must be one of {list(self.variant_map.keys())}")

    def run(self, fitness_func, verbose=True, optimal_fitness=None):
        """
        Run the Dendiff EDA

        Parameters
        ----------
        fitness_func : callable
            Fitness function
        verbose : bool
            Print progress
        optimal_fitness : float, optional
            Known optimal fitness value for early termination

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
        generation_found = 0  # Track generation where best was found

        history = {'best_fitness': [best_fitness]}

        if verbose:
            print(f"Generation 0: Best Fitness = {best_fitness:.4f}")

        # Check if optimum reached in initial population
        if optimal_fitness is not None and abs(best_fitness - optimal_fitness) < OPTIMUM_TOLERANCE:
            if verbose:
                print(f"\nOptimum reached!")
                print(f"\nDendiff-EDA completed after 0 generations")
                print(f"Best fitness found: {best_fitness:.6f}")
                print(f"  at generation {generation_found}")
            return best_fitness, best_solution, history

        for gen in range(self.max_generations):
            # Selection
            n_selected = int(self.pop_size * self.selection_ratio)
            selected_idx = np.argsort(fitness)[-n_selected:]
            selected_pop = population[selected_idx]
            selected_fitness = fitness[selected_idx]

            # Store the best solution from selected population before mutation (for elitism with mutation)
            if self.alpha > 0:
                best_idx_in_selected = np.argmax(selected_fitness)
                best_solution_pre_mutation = selected_pop[best_idx_in_selected].copy()

            # Prepare learning parameters
            # Dynamic hidden layer computation based on n_vars and population size
            # Following DISCRETE_DENDIFF_ANALYSIS.md recommendations
            adaptive_hidden_dims = [
                max(10, self.n_vars // 2),
                max(10, self.n_vars // 4)
            ]

            # Adaptive batch size: max(10, selected_pop_size/20)
            batch_s = max(10, int(n_selected / 20))

            learning_params = self.learning_params.copy()
            learning_params['hidden_dims'] = learning_params.get('hidden_dims', adaptive_hidden_dims)
            learning_params['batch_size'] = learning_params.get('batch_size', batch_s)
            learning_params['epochs'] = learning_params.get('epochs', 50)

            # Pass activation function as a list for all hidden layers
            n_hidden = len(learning_params['hidden_dims'])
            learning_params['list_act_functs'] = [self.activation] * n_hidden

            # Diffusion parameters
            if self.variant == 'dendiff_gumbel':
                learning_params['n_timesteps'] = self.n_timesteps
                learning_params['beta_schedule'] = 'linear'
                learning_params['beta_start'] = self.beta_start
                learning_params['beta_end'] = self.beta_end
                learning_params['temperature'] = self.temperature
                learning_params['temperature_decay'] = 0.99
                learning_params['min_temperature'] = 0.5
            elif self.variant == 'dendiff_corruption':
                learning_params['n_timesteps'] = self.n_timesteps
                learning_params['schedule'] = 'linear'
                learning_params['corruption_start'] = self.beta_start
                learning_params['corruption_end'] = self.beta_end
            elif self.variant == 'dendiff_ste':
                learning_params['n_timesteps'] = self.n_timesteps
                learning_params['schedule'] = 'linear'
                learning_params['noise_start'] = self.beta_start
                learning_params['noise_end'] = self.beta_end
            elif self.variant == 'dendiff_hard_concrete':
                learning_params['n_timesteps'] = self.n_timesteps
                learning_params['schedule'] = 'linear'
                learning_params['beta_start'] = self.beta_start
                learning_params['beta_end'] = self.beta_end
                learning_params['temperature'] = self.temperature
                learning_params['stretch_limits'] = (-0.1, 1.1)
            elif self.variant == 'dendiff_deterministic':
                learning_params['n_timesteps'] = self.n_timesteps
                learning_params['schedule'] = 'linear'
                learning_params['beta_start'] = self.beta_start
                learning_params['beta_end'] = self.beta_end

            # Time embedding dimension (smaller for smaller problems)
            learning_params['time_emb_dim'] = learning_params.get('time_emb_dim', min(32, max(4, self.n_vars // 8)))

            # Fitness guidance (inspired by C-VAE and fitness-guided DbD)
            if self.fitness_guided:
                learning_params['use_fitness_guidance'] = True
                learning_params['fitness_weight'] = learning_params.get('fitness_weight', 0.1)
                # Note: This requires modifications to the learning functions
                # For now, we'll pass it as a parameter and handle it in enhanced variants

            # Loss function parameter
            # Note: Current implementations use fixed loss functions
            # This would require creating enhanced learning functions with loss parameter
            learning_params['loss_function'] = self.loss_function

            # Learn model
            try:
                # Determine which learning function to use
                use_enhanced = (
                    ENHANCED_AVAILABLE and
                    (self.loss_function in LOSS_FUNCTIONS_REQUIRING_FITNESS or
                     self.fitness_guided or
                     self.loss_function != 'mse')
                )

                # Only dendiff_gumbel and dendiff_corruption have enhanced versions
                # For other variants (ste, hard_concrete, deterministic), use standard functions
                if use_enhanced and self.variant in ['dendiff_gumbel', 'dendiff_corruption']:
                    # Use enhanced learning functions with loss/fitness support
                    if self.variant == 'dendiff_gumbel':
                        model = learn_discrete_dendiff_gumbel_enhanced(
                            selected_pop, selected_fitness, learning_params
                        )
                    elif self.variant == 'dendiff_corruption':
                        model = learn_discrete_dendiff_corruption_enhanced(
                            selected_pop, selected_fitness, learning_params
                        )
                else:
                    # Use standard learning functions
                    model = learn_fn(selected_pop, selected_fitness, learning_params)

                # Prepare sampling parameters
                sampling_params = self.sampling_params.copy()
                sampling_params['n_steps'] = self.n_sampling_steps
                sampling_params['temperature'] = self.temperature
                sampling_params['deterministic'] = sampling_params.get('deterministic', False)

                # Sample new population
                population = sample_fn(model, self.pop_size, sampling_params)

            except Exception as e:
                if verbose:
                    print(f"  Warning: Learning/Sampling failed ({e}), using random population")
                    import traceback
                    traceback.print_exc()
                population = np.random.randint(0, self.cardinality,
                                             (self.pop_size, self.n_vars))

            # Apply frequency balance mutation if alpha > 0
            if self.alpha > 0:
                # Apply mutation
                mutation_params = {'alpha': self.alpha}
                population = frequency_balance_mutation(
                    self.n_vars,
                    self.cardinality,
                    population,
                    mutation_params
                )

                # Enforce elitism: replace one solution with the best from previous selected population
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

            # Check if optimum reached
            if optimal_fitness is not None and abs(best_fitness - optimal_fitness) < OPTIMUM_TOLERANCE:
                if verbose:
                    print(f"\nOptimum reached!")
                    print(f"\nDendiff-EDA completed after {gen+1} generations")
                    print(f"Best fitness found: {best_fitness:.6f}")
                    print(f"  at generation {generation_found}")
                return best_fitness, best_solution, history

        # Print completion summary
        if verbose:
            print(f"\nDendiff-EDA completed after {self.max_generations} generations")
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
        description='Discrete Dendiff EDA - Configurable Dendiff Algorithm Variants',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard Dendiff-Gumbel with default settings (no mutation)
  python discrete_Dendiff_EDA.py 0 OneMax 20 80 20 0.5 dendiff_gumbel gumbel relu mse 100 20 0 1.0 0.0001 0.3

  # Dendiff-Corruption with tanh activation
  python discrete_Dendiff_EDA.py 0 Deceptive3 30 100 30 0.5 dendiff_corruption corruption tanh mse 50 20 0 0.5 0.01 0.5

  # Dendiff-Gumbel with fitness guidance
  python discrete_Dendiff_EDA.py 0 HIFF 64 200 50 0.5 dendiff_gumbel gumbel elu mse 100 20 1 1.0 0.0001 0.3

  # Dendiff-Corruption with weighted MSE
  python discrete_Dendiff_EDA.py 0 FC3 30 150 40 0.5 dendiff_corruption corruption relu weighted_mse 50 20 0 0.5 0.01 0.5

  # Dendiff-Gumbel with frequency balance mutation (alpha=0.1)
  python discrete_Dendiff_EDA.py 0 OneMax 20 80 20 0.5 dendiff_gumbel gumbel relu mse 100 20 0 1.0 0.0001 0.3 0.1
        """
    )

    # All positional arguments
    parser.add_argument('seed', type=int, help='Random seed')
    parser.add_argument('obj_func', type=str, help='Objective function name')
    parser.add_argument('n', type=int, help='Number of variables')
    parser.add_argument('pop_size', type=int, help='Population size')
    parser.add_argument('n_gen', type=int, help='Number of generations')
    parser.add_argument('trunc', type=float, help='Truncation percent (selection ratio, e.g., 0.5 for 50%%)')
    parser.add_argument('variant', type=str,
                        choices=['dendiff_gumbel', 'dendiff_corruption', 'dendiff_ste', 'dendiff_hard_concrete', 'dendiff_deterministic'],
                        help='Dendiff variant')
    parser.add_argument('sampling_strategy', type=str,
                        choices=['gumbel', 'corruption', 'ste', 'hard_concrete', 'deterministic'],
                        help='Differentiable sampling strategy')
    parser.add_argument('activation', type=str,
                        help='Activation function (relu, tanh, sigmoid, leakyrelu, elu, selu, gelu, etc.)')
    parser.add_argument('loss', type=str,
                        choices=['mse', 'weighted_mse', 'ranking', 'huber'],
                        help='Loss function')
    parser.add_argument('n_timesteps', type=int,
                        help='Number of diffusion timesteps during training (e.g., 50 or 100)')
    parser.add_argument('n_sampling_steps', type=int,
                        help='Number of denoising steps during sampling (e.g., 20)')
    parser.add_argument('fitness_guided', type=int, choices=[0, 1],
                        help='Use fitness guidance/conditioning (1=yes, 0=no)')
    parser.add_argument('temperature', type=float,
                        help='Gumbel-Softmax or sampling temperature (e.g., 1.0)')
    parser.add_argument('beta_start', type=float,
                        help='Starting noise/corruption level (e.g., 0.0001 for Gumbel, 0.01 for Corruption)')
    parser.add_argument('beta_end', type=float,
                        help='Ending noise/corruption level (e.g., 0.3 for Gumbel, 0.5 for Corruption)')
    parser.add_argument('alpha', type=float, nargs='?', default=0.0,
                        help='Max frequency threshold for mutation (default: 0.0, no mutation)')

    # Parse arguments
    args = parser.parse_args()

    # Convert integer flags to boolean
    args.fitness_guided = bool(args.fitness_guided)

    # Validate truncation percent
    if args.trunc <= 0 or args.trunc > 1:
        print(f"Error: Truncation percent must be between 0 and 1, got {args.trunc}")
        sys.exit(1)

    # Validate variant and sampling_strategy compatibility
    if args.variant == 'dendiff_gumbel' and args.sampling_strategy != 'gumbel':
        print(f"Warning: dendiff_gumbel variant should use 'gumbel' sampling strategy")
    if args.variant == 'dendiff_corruption' and args.sampling_strategy != 'corruption':
        print(f"Warning: dendiff_corruption variant should use 'corruption' sampling strategy")

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
    print("DISCRETE DENDIFF EDA - Configuration")
    print("=" * 80)
    print(f"Seed:               {args.seed}")
    print(f"Problem:            {args.obj_func}")
    print(f"Variables:          {n_vars}")
    print(f"Optimal Fitness:    {optimal_fitness}")
    print(f"Population Size:    {args.pop_size}")
    print(f"Generations:        {args.n_gen}")
    print(f"Truncation Percent: {args.trunc}")
    print(f"Variant:            {args.variant}")
    print(f"Sampling Strategy:  {args.sampling_strategy}")
    print(f"Activation:         {args.activation}")
    print(f"Loss Function:      {args.loss}")
    print(f"Timesteps:          {args.n_timesteps}")
    print(f"Sampling Steps:     {args.n_sampling_steps}")
    print(f"Fitness Guided:     {args.fitness_guided}")
    print(f"Temperature:        {args.temperature}")
    print(f"Beta Start:         {args.beta_start}")
    print(f"Beta End:           {args.beta_end}")
    print(f"Alpha (mutation):   {args.alpha}")
    print("=" * 80)
    print()

    start_time = time.time()

    # Configure learning parameters
    learning_params = {
        'epochs': 50,
        # hidden_dims and batch_size will be computed adaptively in run()
    }

    # Configure sampling parameters
    sampling_params = {
        'deterministic': False,
    }

    cardinality = np.full(n_vars, 2)

    # Create and run Dendiff EDA
    eda = DendiffEDA(
        variant=args.variant,
        n_vars=n_vars,
        cardinality=cardinality,
        pop_size=args.pop_size,
        selection_ratio=args.trunc,
        max_generations=args.n_gen,
        sampling_strategy=args.sampling_strategy,
        activation=args.activation,
        loss_function=args.loss,
        n_timesteps=args.n_timesteps,
        n_sampling_steps=args.n_sampling_steps,
        fitness_guided=args.fitness_guided,
        temperature=args.temperature,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        learning_params=learning_params,
        sampling_params=sampling_params,
        random_seed=args.seed,
        alpha=args.alpha,
    )

    best_fitness, best_solution, history = eda.run(fitness_func, verbose=True, optimal_fitness=optimal_fitness)

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
