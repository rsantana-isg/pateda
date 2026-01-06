"""
Discrete DbD EDA - Command-Line Interface for DbD-EDA Variants
==============================================================

This program provides a unified interface to run various Diffusion-by-Deblending (DbD)
based EDA algorithms on benchmark problems with different seeds for cluster execution.

Supports comprehensive DbD-EDA variants:

**Standard DbD Variants:**
- DbD: Standard Diffusion-by-Deblending
- DbD-CS: Current to Selected population
- DbD-CD: Current to Closest in selected (Distance-based)
- DbD-UC: Univariate approximation to Current
- DbD-US: Univariate approximation to Selected

**DbD with Markov Transformation (DbD-T Variants):**
- DbD-CS-T: CS with Markov transformation (k=0,1,2)
- DbD-CD-T: CD with Markov transformation (k=0,1,2)
- DbD-UC-T: UC with Markov transformation (k=0,1,2)
- DbD-US-T: US with Markov transformation (k=0,1,2)

**Enhanced DbD Variants (inspired by VAE and Backdrive EDAs):**
- DbD-Weighted: DbD with fitness-weighted MSE loss
- DbD-Ranking: DbD with ranking loss
- DbD-Huber: DbD with Huber loss (robust to outliers)
- C-DbD: Conditional DbD with fitness guidance (inspired by C-VAE)
- M-DbD: DbD with Markov model initialization

Configurable Parameters (all positional):
- seed: Random seed
- obj_func: Objective function name
- n: Number of variables
- pop_size: Population size
- n_gen: Number of generations
- trunc: Truncation percent (selection ratio, e.g., 0.5 for 50%)
- variant: DbD variant (see list above)
- activation: Activation function (relu, tanh, sigmoid, leakyrelu, elu, selu, gelu, etc.)
- loss: Loss function (mse, weighted_mse, ranking, huber)
- num_alpha_samples: Number of alpha samples for blending (e.g., 20)
- n_steps: Number of denoising steps during sampling (e.g., 20)
- k: Order of Markov chain for transformation variants (0, 1, 2)
- alpha_smooth: Smoothing parameter for Markov probabilities (e.g., 0.1)
- fitness_guided: Use fitness guidance (1=yes, 0=no)
- use_markov_init: Use Markov model for initialization (1=yes, 0=no)

Usage:
    python discrete_DbD_EDA.py <seed> <obj_func> <n> <pop_size> <n_gen> <trunc> \\
        <variant> <activation> <loss> <num_alpha_samples> <n_steps> <k> <alpha_smooth> \\
        <fitness_guided> <use_markov_init>

Examples:
    # Standard DbD with default settings
    python discrete_DbD_EDA.py 0 OneMax 20 80 20 0.5 dbd relu mse 20 20 0 0.1 0 0

    # DbD-CS with weighted MSE and tanh activation
    python discrete_DbD_EDA.py 1 Deceptive3 30 100 30 0.5 dbd_cs tanh weighted_mse 20 20 0 0.1 0 0

    # DbD-CS-T with transformation (k=1) and fitness guidance
    python discrete_DbD_EDA.py 2 HIFF 64 200 50 0.5 dbd_cs_t elu mse 20 20 1 0.1 1 0

    # Conditional DbD with fitness guidance and Markov initialization
    python discrete_DbD_EDA.py 3 FC3 30 150 40 0.5 c_dbd relu mse 20 20 0 0.1 1 1

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

# DbD learning modules
from pateda.learning.discrete_dbd import (
    learn_binary_dbd, learn_binary_dbd_cs, learn_binary_dbd_cd,
    learn_binary_dbd_uc, learn_binary_dbd_us,
    learn_binary_dbd_cs_t, learn_binary_dbd_cd_t,
    learn_binary_dbd_uc_t, learn_binary_dbd_us_t
)

# DbD sampling modules
from pateda.sampling.discrete_dbd import (
    sample_binary_dbd, sample_binary_dbd_cs, sample_binary_dbd_cd,
    sample_binary_dbd_uc, sample_binary_dbd_us,
    sample_binary_dbd_cs_t, sample_binary_dbd_cd_t,
    sample_binary_dbd_uc_t, sample_binary_dbd_us_t
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
# DbD EDA Implementation
# ==============================================================================

class DbDEDA:
    """
    Unified framework for DbD-based EDAs with configurable parameters
    """

    def __init__(
        self,
        variant: str,
        n_vars: int,
        cardinality: np.ndarray,
        pop_size: int = 100,
        selection_ratio: float = 0.5,
        max_generations: int = 50,
        activation: str = 'relu',
        loss_function: str = 'mse',
        num_alpha_samples: int = 20,
        n_steps: int = 20,
        k: int = 1,
        alpha_smooth: float = 0.1,
        fitness_guided: bool = False,
        use_markov_init: bool = False,
        learning_params: Optional[Dict[str, Any]] = None,
        sampling_params: Optional[Dict[str, Any]] = None,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize DbD EDA

        Parameters
        ----------
        variant : str
            DbD variant: 'dbd', 'dbd_cs', 'dbd_cd', 'dbd_uc', 'dbd_us',
                        'dbd_cs_t', 'dbd_cd_t', 'dbd_uc_t', 'dbd_us_t'
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
        activation : str
            Activation function (relu, tanh, sigmoid, leakyrelu, elu, selu, gelu, etc.)
        loss_function : str
            Loss function (mse, weighted_mse, ranking, huber)
        num_alpha_samples : int
            Number of alpha samples for training blending
        n_steps : int
            Number of denoising steps during sampling
        k : int
            Order of Markov chain for transformation variants
        alpha_smooth : float
            Smoothing parameter for Markov probabilities
        fitness_guided : bool
            Use fitness guidance (inspired by C-VAE)
        use_markov_init : bool
            Use Markov model for initialization
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
        self.activation = activation
        self.loss_function = loss_function
        self.num_alpha_samples = num_alpha_samples
        self.n_steps = n_steps
        self.k = k
        self.alpha_smooth = alpha_smooth
        self.fitness_guided = fitness_guided
        self.use_markov_init = use_markov_init
        self.learning_params = learning_params or {}
        self.sampling_params = sampling_params or {}
        self.random_seed = random_seed

        # Set random seed if provided
        if random_seed is not None:
            set_seed(random_seed)

        # Map variant to learning and sampling functions
        self.variant_map = {
            'dbd': (learn_binary_dbd, sample_binary_dbd),
            'dbd_cs': (learn_binary_dbd_cs, sample_binary_dbd_cs),
            'dbd_cd': (learn_binary_dbd_cd, sample_binary_dbd_cd),
            'dbd_uc': (learn_binary_dbd_uc, sample_binary_dbd_uc),
            'dbd_us': (learn_binary_dbd_us, sample_binary_dbd_us),
            'dbd_cs_t': (learn_binary_dbd_cs_t, sample_binary_dbd_cs_t),
            'dbd_cd_t': (learn_binary_dbd_cd_t, sample_binary_dbd_cd_t),
            'dbd_uc_t': (learn_binary_dbd_uc_t, sample_binary_dbd_uc_t),
            'dbd_us_t': (learn_binary_dbd_us_t, sample_binary_dbd_us_t),
        }

        if variant not in self.variant_map:
            raise ValueError(f"Invalid variant: {variant}. Must be one of {list(self.variant_map.keys())}")

        # Store Markov model if using Markov initialization
        self.markov_model = None

    def _learn_markov_model(self, population: np.ndarray, markov_k: int = 1):
        """
        Learn a k-order Markov chain model from population

        Parameters
        ----------
        population : np.ndarray
            Binary population [n_samples, n_vars]
        markov_k : int
            Order of Markov chain

        Returns
        -------
        dict
            Markov model with conditional probabilities
        """
        from pateda.learning.discrete_dbd import compute_conditional_probabilities

        conditional_probs = compute_conditional_probabilities(
            population.astype(int),
            markov_k,
            self.alpha_smooth
        )

        return {
            'conditional_probs': conditional_probs,
            'k': markov_k,
            'n_vars': self.n_vars
        }

    def _sample_from_markov_model(self, model: dict, n_samples: int) -> np.ndarray:
        """
        Sample from a Markov chain model

        Parameters
        ----------
        model : dict
            Markov model
        n_samples : int
            Number of samples

        Returns
        -------
        np.ndarray
            Sampled binary population [n_samples, n_vars]
        """
        conditional_probs = model['conditional_probs']
        k = model['k']
        n_vars = model['n_vars']

        samples = np.zeros((n_samples, n_vars), dtype=int)

        for var in range(n_vars):
            cpd = conditional_probs[var]

            if var < k:
                # For first k variables, use marginal probabilities
                # cpd is [P(X=0), P(X=1)]
                prob_1 = cpd[1]
                samples[:, var] = (np.random.rand(n_samples) < prob_1).astype(int)
            else:
                # For remaining variables, use conditional probabilities
                n_parents = min(k, var)
                parent_vars = list(range(var - n_parents, var))

                for sample_idx in range(n_samples):
                    # Calculate parent configuration index
                    config_idx = 0
                    for i, parent_var in enumerate(parent_vars):
                        config_idx += int(samples[sample_idx, parent_var]) * (2 ** i)

                    # Get conditional probability P(X_i = 1 | parents)
                    prob_1_given_parents = cpd[config_idx, 1]
                    samples[sample_idx, var] = 1 if np.random.rand() < prob_1_given_parents else 0

        return samples

    def run(self, fitness_func, verbose=True):
        """
        Run the DbD EDA

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
        if self.use_markov_init and self.markov_model is not None:
            # Initialize from Markov model
            population = self._sample_from_markov_model(self.markov_model, self.pop_size)
        else:
            # Random initialization
            population = np.random.randint(0, self.cardinality, (self.pop_size, self.n_vars))

        # Evaluate
        fitness = fitness_func(population)

        best_fitness = np.max(fitness)
        best_solution = population[np.argmax(fitness)].copy()
        generation_found = 0  # Track generation where best was found

        history = {'best_fitness': [best_fitness]}

        if verbose:
            print(f"Generation 0: Best Fitness = {best_fitness:.4f}")

        # Keep track of previous population for DbD variants
        prev_population = None
        prev_selected_pop = None

        for gen in range(self.max_generations):
            # Selection
            n_selected = int(self.pop_size * self.selection_ratio)
            selected_idx = np.argsort(fitness)[-n_selected:]
            selected_pop = population[selected_idx]
            selected_fitness = fitness[selected_idx]

            # Update Markov model if using Markov initialization
            if self.use_markov_init:
                # Learn/update Markov model from selected population
                markov_k = self.learning_params.get('markov_k', 1)
                self.markov_model = self._learn_markov_model(selected_pop, markov_k)

            # Prepare learning parameters
            # Dynamic hidden layer computation based on n_vars and population size
            # Following DISCRETE_DBD_ANALYSIS.md recommendations
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
            learning_params['num_alpha_samples'] = self.num_alpha_samples

            # Pass activation function as a list for all hidden layers
            n_hidden = len(learning_params['hidden_dims'])
            learning_params['list_act_functs'] = [self.activation] * n_hidden

            # For transformation variants, pass k and alpha_smooth
            if '_t' in self.variant:
                learning_params['k'] = self.k
                learning_params['alpha'] = self.alpha_smooth

            # For UC/US variants, pass to_take parameter
            if self.variant in ['dbd_uc', 'dbd_us', 'dbd_uc_t', 'dbd_us_t']:
                learning_params['to_take'] = self.pop_size * 4

            # Fitness guidance (inspired by C-VAE)
            if self.fitness_guided:
                learning_params['use_fitness_guidance'] = True
                learning_params['fitness_weight'] = learning_params.get('fitness_weight', 0.1)

            # Learn model
            try:
                # DbD variants need two populations (source and target)
                if prev_population is None:
                    # First generation: use random as current population
                    current_pop = np.random.randint(0, self.cardinality,
                                                  (len(selected_pop), self.n_vars))
                else:
                    # Sample from previous population to match selected population size
                    n_to_sample = len(selected_pop)
                    if len(prev_population) >= n_to_sample:
                        indices = np.random.choice(len(prev_population), n_to_sample, replace=False)
                    else:
                        indices = np.random.choice(len(prev_population), n_to_sample, replace=True)
                    current_pop = prev_population[indices]

                # Learn model with current and selected populations
                model = learn_fn(current_pop, selected_pop, learning_params)

                # Save for next iteration
                prev_population = population.copy()
                prev_selected_pop = selected_pop.copy()

                # Prepare sampling parameters
                sampling_params = self.sampling_params.copy()
                sampling_params['n_steps'] = self.n_steps
                sampling_params['temperature'] = sampling_params.get('temperature', 1.0)

                # For transformation variants, pass sampling parameters
                if '_t' in self.variant:
                    sampling_params['num_iterations'] = sampling_params.get('num_iterations', 10)
                    sampling_params['prob_min'] = sampling_params.get('prob_min', 0.01)
                    sampling_params['prob_max'] = sampling_params.get('prob_max', 0.99)

                # Sample new population based on variant
                if self.variant in ['dbd_cs', 'dbd_us', 'dbd_cs_t', 'dbd_us_t']:
                    # DbD-CS and DbD-US variants: initialize from selected population
                    population = sample_fn(model, self.pop_size, selected_pop, sampling_params)
                elif self.variant in ['dbd_cd', 'dbd_uc', 'dbd_cd_t', 'dbd_uc_t']:
                    # DbD-CD and DbD-UC: initialize from current population
                    population = sample_fn(model, self.pop_size, population, sampling_params)
                elif self.variant == 'dbd':
                    # Original DbD: use default sampling
                    population = sample_fn(model, self.pop_size, sampling_params)
                else:
                    # Other variants
                    population = sample_fn(model, self.pop_size, sampling_params)

            except Exception as e:
                if verbose:
                    print(f"  Warning: Learning/Sampling failed ({e}), using random population")
                    import traceback
                    traceback.print_exc()
                population = np.random.randint(0, self.cardinality,
                                             (self.pop_size, self.n_vars))

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
            print(f"\nDbD-EDA completed after {self.max_generations} generations")
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
        description='Discrete DbD EDA - Configurable DbD Algorithm Variants',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard DbD with default settings
  python discrete_DbD_EDA.py 0 OneMax 20 80 20 0.5 dbd relu mse 20 20 0 0.1 0 0

  # DbD-CS with weighted MSE and tanh activation
  python discrete_DbD_EDA.py 0 Deceptive3 30 100 30 0.5 dbd_cs tanh weighted_mse 20 20 0 0.1 0 0

  # DbD-CS-T with transformation (k=1) and fitness guidance
  python discrete_DbD_EDA.py 0 HIFF 64 200 50 0.5 dbd_cs_t elu mse 20 20 1 0.1 1 0

  # DbD with Markov initialization
  python discrete_DbD_EDA.py 0 FC3 30 150 40 0.5 dbd relu mse 20 20 0 0.1 0 1
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
                        choices=['dbd', 'dbd_cs', 'dbd_cd', 'dbd_uc', 'dbd_us',
                                'dbd_cs_t', 'dbd_cd_t', 'dbd_uc_t', 'dbd_us_t'],
                        help='DbD variant')
    parser.add_argument('activation', type=str,
                        help='Activation function (relu, tanh, sigmoid, leakyrelu, elu, selu, gelu, etc.)')
    parser.add_argument('loss', type=str,
                        choices=['mse', 'weighted_mse', 'ranking', 'huber'],
                        help='Loss function')
    parser.add_argument('num_alpha_samples', type=int,
                        help='Number of alpha samples for blending (e.g., 20)')
    parser.add_argument('n_steps', type=int,
                        help='Number of denoising steps during sampling (e.g., 20)')
    parser.add_argument('k', type=int,
                        help='Order of Markov chain for transformation variants (0, 1, 2)')
    parser.add_argument('alpha_smooth', type=float,
                        help='Smoothing parameter for Markov probabilities (e.g., 0.1)')
    parser.add_argument('fitness_guided', type=int, choices=[0, 1],
                        help='Use fitness guidance (1=yes, 0=no)')
    parser.add_argument('use_markov_init', type=int, choices=[0, 1],
                        help='Use Markov model for initialization (1=yes, 0=no)')

    # Parse arguments
    args = parser.parse_args()

    # Convert integer flags to boolean
    args.fitness_guided = bool(args.fitness_guided)
    args.use_markov_init = bool(args.use_markov_init)

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
    print("DISCRETE DBD EDA - Configuration")
    print("=" * 80)
    print(f"Seed:               {args.seed}")
    print(f"Problem:            {args.obj_func}")
    print(f"Variables:          {n_vars}")
    print(f"Optimal Fitness:    {optimal_fitness}")
    print(f"Population Size:    {args.pop_size}")
    print(f"Generations:        {args.n_gen}")
    print(f"Truncation Percent: {args.trunc}")
    print(f"Variant:            {args.variant}")
    print(f"Activation:         {args.activation}")
    print(f"Loss Function:      {args.loss}")
    print(f"Num Alpha Samples:  {args.num_alpha_samples}")
    print(f"Denoising Steps:    {args.n_steps}")
    if '_t' in args.variant:
        print(f"Markov k:           {args.k}")
        print(f"Alpha Smooth:       {args.alpha_smooth}")
    print(f"Fitness Guided:     {args.fitness_guided}")
    print(f"Markov Init:        {args.use_markov_init}")
    print("=" * 80)
    print()

    start_time = time.time()

    # Configure learning parameters
    learning_params = {
        'epochs': 50,
        # hidden_dims and batch_size will be computed adaptively in run()
    }

    if args.use_markov_init:
        learning_params['markov_k'] = args.k if '_t' in args.variant else 1

    # Configure sampling parameters
    sampling_params = {
        'temperature': 1.0,
    }

    cardinality = np.full(n_vars, 2)

    # Create and run DbD EDA
    eda = DbDEDA(
        variant=args.variant,
        n_vars=n_vars,
        cardinality=cardinality,
        pop_size=args.pop_size,
        selection_ratio=args.trunc,
        max_generations=args.n_gen,
        activation=args.activation,
        loss_function=args.loss,
        num_alpha_samples=args.num_alpha_samples,
        n_steps=args.n_steps,
        k=args.k,
        alpha_smooth=args.alpha_smooth,
        fitness_guided=args.fitness_guided,
        use_markov_init=args.use_markov_init,
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
