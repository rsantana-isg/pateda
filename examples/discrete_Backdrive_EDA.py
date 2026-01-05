"""
Discrete Backdrive EDA - Command-Line Interface for Backdrive Variants
======================================================================

This program provides a unified interface to run various Backdrive-EDA algorithm
variants on benchmark problems with different seeds for cluster execution.

Supports configurable Backdrive-EDA variants:
- Backdrive: Standard network inversion approach
- Backdrive-Adaptive: Adaptive sampling with multiple target fitness levels
- Backdrive-Descriptors: Multi-descriptor variant predicting (fitness, mean, std)

Configurable Parameters (all positional):
- seed: Random seed
- obj_func: Objective function name
- n: Number of variables
- pop_size: Population size
- n_gen: Number of generations
- trunc: Truncation percent (selection ratio, e.g., 0.5 for 50%)
- variant: Backdrive variant (backdrive, backdrive_adaptive, backdrive_descriptors)
- init: Initialization method (random, perturb_best, perturb_selected)
- loss: Loss function (mse, weighted_mse, ranking, huber)
  * For backdrive_descriptors: All loss functions supported (adapted for descriptors)
  * weighted_mse: Weights solution reconstruction by fitness
  * ranking: Falls back to MSE (not applicable for descriptor prediction)
  * huber: Robust MSE for solution reconstruction
- activation: Activation function (relu, tanh, sigmoid, leaky_relu, elu, selu, gelu, etc.)
  * All variants now support any activation function
- weight_transfer: 1 to enable, 0 to disable
- early_stopping: 1 to enable, 0 to disable
- surrogate_filtering: 1 to enable, 0 to disable
  * For backdrive_descriptors: Uses forward model (solution → descriptors) to predict fitness

Usage:
    python discrete_Backdrive_EDA.py <seed> <obj_func> <n> <pop_size> <n_gen> <trunc> \\
        <variant> <init> <loss> <activation> <weight_transfer> <early_stopping> <surrogate_filtering>

Examples:
    # Basic backdrive with MSE and ReLU
    python discrete_Backdrive_EDA.py 0 OneMax 20 80 20 0.5 backdrive random mse relu 0 0 0

    # Backdrive with weighted MSE and tanh activation
    python discrete_Backdrive_EDA.py 1 Deceptive3 30 100 30 0.5 backdrive random weighted_mse tanh 1 1 0

    # Backdrive descriptors with Huber loss and ELU activation
    python discrete_Backdrive_EDA.py 2 HIFF 64 200 50 0.5 backdrive_descriptors random huber elu 0 1 0

    # Backdrive descriptors with surrogate filtering enabled
    python discrete_Backdrive_EDA.py 2 HIFF 64 200 50 0.5 backdrive_descriptors random mse relu 0 1 1

    # Adaptive backdrive with GELU activation
    python discrete_Backdrive_EDA.py 3 FC3 30 150 40 0.5 backdrive_adaptive perturb_best mse gelu 1 1 0

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

# Backdrive learning modules
from pateda.learning.discrete_backdrive import learn_binary_backdrive
from pateda.learning.discrete_backdrive_weighted_mse import learn_binary_backdrive_weighted_mse
from pateda.learning.discrete_backdrive_ranking import learn_binary_backdrive_ranking
from pateda.learning.discrete_backdrive_huber import learn_binary_backdrive_huber
from pateda.learning.discrete_backdrive_descriptors import learn_binary_backdrive_descriptors

# Backdrive sampling modules
from pateda.sampling.discrete_neural import (
    sample_binary_backdrive,
    sample_binary_backdrive_adaptive,
    sample_binary_backdrive_descriptors
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
# Backdrive EDA Implementation
# ==============================================================================

class BackdriveEDA:
    """
    Configurable Backdrive-EDA framework with multiple parameter options
    """

    def __init__(
        self,
        variant: str,
        n_vars: int,
        cardinality: np.ndarray,
        pop_size: int = 100,
        selection_ratio: float = 0.5,
        max_generations: int = 50,
        weight_transfer: bool = False,
        early_stopping: bool = True,
        init_method: str = 'random',
        loss_function: str = 'mse',
        activation: str = 'relu',
        surrogate_filtering: bool = False,
        learning_params: Optional[Dict[str, Any]] = None,
        sampling_params: Optional[Dict[str, Any]] = None,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize Backdrive EDA

        Parameters
        ----------
        variant : str
            Backdrive variant: 'backdrive', 'backdrive_adaptive', 'backdrive_descriptors'
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
        weight_transfer : bool
            Transfer neural network weights between generations
        early_stopping : bool
            Use early stopping during training
        init_method : str
            Initialization method: 'random', 'perturb_best', 'perturb_selected'
        loss_function : str
            Loss function: 'mse', 'weighted_mse', 'ranking', 'huber'
        activation : str
            Activation function (relu, tanh, sigmoid, leakyrelu, etc.)
        surrogate_filtering : bool
            Use surrogate model for pre-filtering solutions
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
        self.weight_transfer = weight_transfer
        self.early_stopping = early_stopping
        self.init_method = init_method
        self.loss_function = loss_function
        self.activation = activation
        self.surrogate_filtering = surrogate_filtering
        self.learning_params = learning_params or {}
        self.sampling_params = sampling_params or {}
        self.random_seed = random_seed

        # Set random seed if provided
        if random_seed is not None:
            set_seed(random_seed)

        # Validate parameters
        valid_variants = ['backdrive', 'backdrive_adaptive', 'backdrive_descriptors']
        if variant not in valid_variants:
            raise ValueError(f"Invalid variant: {variant}. Must be one of {valid_variants}")
        
        valid_init_methods = ['random', 'perturb_best', 'perturb_selected']
        if init_method not in valid_init_methods:
            raise ValueError(f"Invalid init_method: {init_method}. Must be one of {valid_init_methods}")
        
        valid_loss_functions = ['mse', 'weighted_mse', 'ranking', 'huber']
        if loss_function not in valid_loss_functions:
            raise ValueError(f"Invalid loss_function: {loss_function}. Must be one of {valid_loss_functions}")

        # Map loss function to learning function
        self.loss_function_map = {
            'mse': learn_binary_backdrive,
            'weighted_mse': learn_binary_backdrive_weighted_mse,
            'ranking': learn_binary_backdrive_ranking,
            'huber': learn_binary_backdrive_huber,
            'descriptors': learn_binary_backdrive_descriptors,
        }

        # Map variant to sampling function
        self.sampling_function_map = {
            'backdrive': sample_binary_backdrive,
            'backdrive_adaptive': sample_binary_backdrive_adaptive,
            'backdrive_descriptors': sample_binary_backdrive_descriptors,
        }

        # Store model for weight transfer
        self.previous_model = None

    def run(self, fitness_func, verbose=True):
        """
        Run the Backdrive EDA

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
        # For backdrive_descriptors variant, always use the descriptors learning function
        # but pass the loss_function parameter to it
        if self.variant == 'backdrive_descriptors':
            learn_fn = self.loss_function_map['descriptors']
        else:
            learn_fn = self.loss_function_map[self.loss_function]
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
            learning_params['early_stopping'] = self.early_stopping

            # Pass activation as a list for all hidden layers
            if 'hidden_dims' in learning_params:
                n_hidden = len(learning_params['hidden_dims'])
                learning_params['list_act_functs'] = [self.activation] * n_hidden
            elif 'hidden_layers' in learning_params:
                n_hidden = len(learning_params['hidden_layers'])
                learning_params['list_act_functs'] = [self.activation] * n_hidden

            # For backdrive_descriptors variant, pass loss function
            if self.variant == 'backdrive_descriptors':
                learning_params['loss_function'] = self.loss_function

            # For weight transfer, initialize from previous model
            if self.weight_transfer and self.previous_model is not None:
                learning_params['pretrained_model'] = self.previous_model

            # Learn model
            try:
                model = learn_fn(selected_pop, selected_fitness, learning_params)
                
                # Store model for weight transfer
                if self.weight_transfer:
                    self.previous_model = model

                # Prepare sampling parameters
                sampling_params = self.sampling_params.copy()
                sampling_params['init_method'] = self.init_method
                
                # For backdrive_descriptors variant, pass selected population
                if self.variant == 'backdrive_descriptors':
                    sampling_params['selected_population'] = selected_pop
                    sampling_params['selected_fitness'] = selected_fitness
                
                # For perturb methods, add current population and fitness
                if self.init_method in ['perturb_best', 'perturb_selected']:
                    sampling_params['current_population'] = selected_pop
                    sampling_params['current_fitness'] = selected_fitness

                # Sample new population
                new_population = sample_fn(model, self.pop_size, sampling_params)
                
                # Surrogate filtering (optional)
                if self.surrogate_filtering:
                    import torch
                    
                    if self.variant == 'backdrive_descriptors':
                        # Use forward model to predict descriptors (including fitness) from solutions
                        if 'forward_network_state' in model:
                            from pateda.learning.discrete_backdrive_descriptors import ForwardDescriptorNet
                            
                            # Reconstruct forward network for predictions
                            forward_network = ForwardDescriptorNet(
                                model['n_vars'],
                                model['n_descriptors'],
                                model.get('forward_hidden_layers', model['hidden_layers']),
                                dropout=0.0,  # No dropout during evaluation
                                list_act_functs=model.get('list_act_functs', None)
                            )
                            forward_network.load_state_dict(model['forward_network_state'])
                            forward_network.eval()
                            
                            # Generate more samples than needed
                            candidate_pop = sample_fn(model, self.pop_size * 3, sampling_params)
                            
                            # Get descriptor statistics for denormalization
                            descriptor_means, descriptor_stds = model['descriptor_stats']
                            
                            with torch.no_grad():
                                # For binary variables, solutions are already normalized to [0, 1]
                                # Same as during training (population.astype(float))
                                X = torch.FloatTensor(candidate_pop.astype(float))
                                # Predict normalized descriptors
                                pred_descriptors_norm = forward_network(X).numpy()
                                # Denormalize to get actual descriptor values
                                pred_descriptors = pred_descriptors_norm * descriptor_stds + descriptor_means
                                # Extract fitness (first component of descriptors)
                                pred_fitness = pred_descriptors[:, 0]
                            
                            # Select top predicted solutions based on fitness
                            top_indices = np.argsort(pred_fitness)[-self.pop_size:]
                            population = candidate_pop[top_indices]
                        else:
                            # Forward model not available, use sampled population directly
                            if verbose and gen == 0:
                                print("  Note: Forward model not available, surrogate filtering disabled")
                            population = new_population
                    else:
                        # Original backdrive variants
                        from pateda.learning.discrete_backdrive import DiscreteBackdriveNet
                        
                        # Reconstruct network for predictions with same configuration as training
                        network = DiscreteBackdriveNet(
                            model['n_vars'],
                            model['cardinality'],
                            model['hidden_layers'],
                            model['use_embeddings'],
                            model.get('embedding_dim', 8),
                            dropout=0.0,  # No dropout during evaluation
                            list_act_functs=model.get('list_act_functs', None)
                        )
                        network.load_state_dict(model['network_state'])
                        network.eval()

                        # Generate more samples than needed
                        candidate_pop = sample_fn(model, self.pop_size * 3, sampling_params)

                        with torch.no_grad():
                            X = torch.LongTensor(candidate_pop.astype(int))
                            pred_fitness = network(X).numpy().flatten()

                        # Select top predicted solutions
                        top_indices = np.argsort(pred_fitness)[-self.pop_size:]
                        population = candidate_pop[top_indices]
                else:
                    population = new_population

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
                generation_found = gen + 1  # Update generation where best was found

            history['best_fitness'].append(best_fitness)

            if verbose and (gen + 1) % 1 == 0:
                print(f"Generation {gen+1}: Best Fitness = {best_fitness:.4f}")

        # Print completion summary
        if verbose:
            print(f"\nBackdrive-EDA completed after {self.max_generations} generations")
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
        description='Discrete Backdrive EDA - Configurable Backdrive Algorithm Variants',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python discrete_Backdrive_EDA.py 0 OneMax 20 80 20 0.5 backdrive random mse relu 0 0 0

  # With weight transfer and early stopping
  python discrete_Backdrive_EDA.py 0 OneMax 20 80 20 0.5 backdrive random mse relu 1 1 0

  # Custom initialization and loss function
  python discrete_Backdrive_EDA.py 0 Deceptive3 30 100 30 0.5 backdrive perturb_best weighted_mse relu 0 0 0

  # Adaptive variant with custom activation
  python discrete_Backdrive_EDA.py 0 HIFF 64 200 50 0.5 backdrive_adaptive random mse tanh 0 1 0

  # With surrogate filtering
  python discrete_Backdrive_EDA.py 0 OneMax 20 80 20 0.5 backdrive random mse relu 0 0 1
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
                        choices=['backdrive', 'backdrive_adaptive', 'backdrive_descriptors'],
                        help='Backdrive variant')
    parser.add_argument('init', type=str,
                        choices=['random', 'perturb_best', 'perturb_selected'],
                        help='Initialization method')
    parser.add_argument('loss', type=str,
                        choices=['mse', 'weighted_mse', 'ranking', 'huber'],
                        help='Loss function')
    parser.add_argument('activation', type=str,
                        help='Activation function (relu, tanh, sigmoid, leaky_relu, elu, selu, gelu, etc.)')
    parser.add_argument('weight_transfer', type=int, choices=[0, 1],
                        help='Transfer neural network weights between generations (1=yes, 0=no)')
    parser.add_argument('early_stopping', type=int, choices=[0, 1],
                        help='Use early stopping during training (1=yes, 0=no)')
    parser.add_argument('surrogate_filtering', type=int, choices=[0, 1],
                        help='Use surrogate model for pre-filtering solutions (1=yes, 0=no)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Convert integer flags to boolean
    args.weight_transfer = bool(args.weight_transfer)
    args.early_stopping = bool(args.early_stopping)
    args.surrogate_filtering = bool(args.surrogate_filtering)
    
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
    print("DISCRETE BACKDRIVE EDA - Configuration")
    print("=" * 80)
    print(f"Seed:               {args.seed}")
    print(f"Problem:            {args.obj_func}")
    print(f"Variables:          {n_vars}")
    print(f"Optimal Fitness:    {optimal_fitness}")
    print(f"Population Size:    {args.pop_size}")
    print(f"Generations:        {args.n_gen}")
    print(f"Truncation Percent: {args.trunc}")
    print(f"Variant:            {args.variant}")
    print(f"Weight Transfer:    {args.weight_transfer}")
    print(f"Early Stopping:     {args.early_stopping}")
    print(f"Initialization:     {args.init}")
    print(f"Loss Function:      {args.loss}")
    print(f"Activation:         {args.activation}")
    print(f"Surrogate Filter:   {args.surrogate_filtering}")
    print("=" * 80)
    print()
    
    start_time = time.time()
    
    # Compute common parameters based on pop_size and n_vars
    selected_pop_size = int(args.pop_size * args.trunc)
    adaptive_hidden_dims = [max(10, n_vars // 2), max(10, n_vars // 4)]
    batch_s = min(32, int(selected_pop_size/10))
    
    # Configure learning parameters
    # Note: descriptors variant uses 'hidden_layers', others use 'hidden_dims'
    if args.variant == 'backdrive_descriptors':
        learning_params = {
            'epochs': 30,
            'hidden_layers': adaptive_hidden_dims,
            'batch_size': batch_s,
            'early_stopping': args.early_stopping,
            # Activation and loss_function will be passed in the run method
        }
    else:
        learning_params = {
            'epochs': 30,
            'hidden_dims': adaptive_hidden_dims,
            'batch_size': batch_s,
            'early_stopping': args.early_stopping,
            # Activation will be passed as list_act_functs in the run method
        }
    
    # Configure sampling parameters based on variant and initialization
    sampling_params = {
        'init_method': args.init,
        'n_iterations': 100,
    }
    
    if args.init in ['perturb_best', 'perturb_selected']:
        sampling_params['init_noise'] = 0.1
    
    if args.variant == 'backdrive_adaptive':
        sampling_params['target_levels'] = [100, 90, 80]
        sampling_params['level_fractions'] = [0.5, 0.3, 0.2]
    
    if args.variant == 'backdrive_descriptors':
        # For descriptor variant, specify how to sample descriptors
        sampling_params['descriptor_sampling'] = 'from_population'  # or 'gaussian', 'uniform'
    
    cardinality = np.full(n_vars, 2)
    
    # Create and run Backdrive EDA
    eda = BackdriveEDA(
        variant=args.variant,
        n_vars=n_vars,
        cardinality=cardinality,
        pop_size=args.pop_size,
        selection_ratio=args.trunc,
        max_generations=args.n_gen,
        weight_transfer=args.weight_transfer,
        early_stopping=args.early_stopping,
        init_method=args.init,
        loss_function=args.loss,
        activation=args.activation,
        surrogate_filtering=args.surrogate_filtering,
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


if __name__ == "__main__":
    main()
