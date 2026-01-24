"""
Discrete Backdrive EDA - Real-World Combinatorial Problems
==========================================================

This program combines discrete_Backdrive_EDA.py and discrete_EDA_RW.py to provide
a unified interface to run various Backdrive-EDA algorithm variants on real-world
combinatorial problems (SAT, Ising, UBQP) with different seeds for cluster execution.

Supports configurable Backdrive-EDA variants:
- Backdrive: Standard network inversion approach
- Backdrive-Adaptive: Adaptive sampling with multiple target fitness levels  
- Backdrive-Descriptors: Multi-descriptor variant predicting (fitness, mean, std)

Supports real-world combinatorial problems:
- SAT: Boolean satisfiability problem
- Ising: Ising spin glass model
- UBQP: Unconstrained Binary Quadratic Programming

Configurable Parameters (all positional):
- seed: Random seed
- problem_type: Problem type (SAT, Ising, UBQP)
- instance_name: Instance file name
- pop_size: Population size
- n_gen: Number of generations
- trunc: Truncation percent (selection ratio, e.g., 0.5 for 50%)
- variant: Backdrive variant (backdrive, backdrive_adaptive, backdrive_descriptors)
- init: Initialization method (random, perturb_best, perturb_selected)
- loss: Loss function (mse, weighted_mse, ranking, huber)
- activation: Activation function (relu, tanh, sigmoid, leaky_relu, elu, selu, gelu, etc.)
- weight_transfer: 1 to enable, 0 to disable
- early_stopping: 1 to enable, 0 to disable
- surrogate_filtering: 1 to enable, 0 to disable
- alpha: Maximum allowed frequency for ones or zeros (mutation control)

Usage:
    python discrete_Backdrive_EDA_RW.py <seed> <problem_type> <instance_name> <pop_size> <n_gen> <trunc> \\
        <variant> <init> <loss> <activation> <weight_transfer> <early_stopping> <surrogate_filtering> <alpha>

Examples:
    # Basic backdrive with MSE on SAT
    python discrete_Backdrive_EDA_RW.py 0 SAT uf20-01 80 20 0.5 backdrive random mse relu 0 0 0 0.0

    # Backdrive with weighted MSE on Ising with mutation
    python discrete_Backdrive_EDA_RW.py 1 Ising SG_16_1 100 30 0.5 backdrive random weighted_mse tanh 1 1 0 0.95

    # Backdrive descriptors on UBQP with surrogate filtering
    python discrete_Backdrive_EDA_RW.py 2 UBQP bqp50 200 50 0.5 backdrive_descriptors random mse relu 0 1 1 0.0

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

# Real-world problem functions
from pateda.functions.discrete.ising import load_ising, eval_ising
from pateda.functions.discrete.ubqp import UBQPInstance

# Mutation operators
from pateda.mutation import frequency_balance_mutation


# ==============================================================================
# Constants
# ==============================================================================

# Success threshold as a fraction of optimal fitness
SUCCESS_THRESHOLD = 0.01

# Constant for unknown optimal values
UNKNOWN_OPTIMAL = "Unknown"

# Tolerance for checking if optimal value is reached
OPTIMUM_TOLERANCE = 1e-6

# Surrogate filtering: factor for generating candidate solutions
SURROGATE_CANDIDATE_FACTOR = 3


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
# Real-World Problem Instance Loading and Evaluation
# ==============================================================================

def load_sat_instance(instance_name: str):
    """
    Load a SAT instance from CNF file
    
    Parameters
    ----------
    instance_name : str
        Name of the instance file (e.g., 'uf20-01')
    
    Returns
    -------
    tuple
        (n_vars, clauses, optimal_value)
        clauses is a list of tuples (var1, var2, var3) with negative values for negated literals
        optimal_value is "Unknown" for these instances
    """
    # Get the path to the SAT instances directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    instances_dir = os.path.join(os.path.dirname(script_dir), 'functions', 'SAT_instances')
    
    # Add .cnf extension if not present
    if not instance_name.endswith('.cnf'):
        instance_name = instance_name + '.cnf'
    
    instance_file = os.path.join(instances_dir, instance_name)
    
    if not os.path.exists(instance_file):
        raise FileNotFoundError(f"SAT instance file not found: {instance_file}")
    
    n_vars = 0
    n_clauses = 0
    clauses = []
    
    with open(instance_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments, empty lines, and special markers
            if not line or line.startswith('c') or line.startswith('%'):
                continue
            
            # Parse problem line
            if line.startswith('p'):
                parts = line.split()
                n_vars = int(parts[2])
                n_clauses = int(parts[3])
                continue
            
            # Parse clause - only if we have valid numbers
            try:
                literals = [int(x) for x in line.split() if x != '0']
                if len(literals) == 3:  # 3-SAT
                    clauses.append(tuple(literals))
            except ValueError:
                # Skip lines that can't be parsed as integers
                continue
    
    # Known optimal values for specific instances
    optimal_fitness = UNKNOWN_OPTIMAL
    instance_base = instance_name.replace('.cnf', '')
    if instance_base in ['uf100-01', 'uf100-02', 'uf100-03', 'uf100-04', 'uf100-05']:
        optimal_fitness = 430
    
    return n_vars, clauses, optimal_fitness


def evaluate_sat(solution: np.ndarray, clauses) -> float:
    """
    Evaluate a SAT solution
    
    Parameters
    ----------
    solution : np.ndarray
        Binary vector (0/1)
    clauses : list
        List of clauses, each clause is a tuple of 3 literals (negative for negated)
    
    Returns
    -------
    float
        Number of satisfied clauses
    """
    if solution.ndim == 1:
        satisfied = 0
        for clause in clauses:
            # Check if any literal in the clause is satisfied
            clause_sat = False
            for lit in clause:
                if lit > 0:
                    if solution[abs(lit) - 1] == 1:
                        clause_sat = True
                        break
                else:
                    if solution[abs(lit) - 1] == 0:
                        clause_sat = True
                        break
            if clause_sat:
                satisfied += 1
        return float(satisfied)
    else:
        return np.array([evaluate_sat(sol, clauses) for sol in solution])


def load_ising_instance(instance_name: str):
    """
    Load an Ising instance
    
    Parameters
    ----------
    instance_name : str
        Name of the instance (e.g., 'SG_16_1')
    
    Returns
    -------
    tuple
        (n_vars, lattice, inter, optimal_value)
        optimal_value is "Unknown" for these instances
    """
    # Parse n_vars and instance number from name (e.g., SG_16_1 -> n=16, inst=1)
    parts = instance_name.replace('.txt', '').split('_')
    if len(parts) != 3 or parts[0] != 'SG':
        raise ValueError(f"Invalid Ising instance name format: {instance_name}. Expected format: SG_<n>_<inst>")
    
    n_vars = int(parts[1])
    inst = int(parts[2])
    
    # Get the path to the Ising instances directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    instances_dir = os.path.join(os.path.dirname(script_dir), 'functions', 'Ising_Instances')
    
    # Add .txt extension if not present
    if not instance_name.endswith('.txt'):
        instance_name = instance_name + '.txt'
    
    instance_file = os.path.join(instances_dir, instance_name)
    
    if not os.path.exists(instance_file):
        raise FileNotFoundError(f"Ising instance file not found: {instance_file}")
    
    with open(instance_file, 'r') as fp:
        # Read header
        num_vars = int(fp.readline().strip())
        dim = int(fp.readline().strip())
        neigh = int(fp.readline().strip())
        width = int(fp.readline().strip())
        
        # Verify consistency
        if num_vars != n_vars:
            raise ValueError(f"Instance name suggests {n_vars} variables but file has {num_vars}")
        
        # Initialize lattice and inter
        # Each line has format: num_neighbors node1 node2 ... interaction1 interaction2 ...
        neighbor = int(2**neigh * dim)
        lattice = np.zeros((num_vars, neighbor + 1), dtype=int)
        inter = np.zeros((num_vars, neighbor), dtype=float)
        
        # Read the structures from file
        for i in range(num_vars):
            line = fp.readline().strip().split()
            n_neighbors = int(line[0])
            lattice[i, 0] = n_neighbors
            
            if n_neighbors > 0:
                # Read neighbor indices
                for j in range(n_neighbors):
                    lattice[i, j + 1] = int(line[1 + j]) + 1  # Convert to 1-indexed
                
                # Read interaction values
                for j in range(n_neighbors):
                    inter[i, j] = float(line[1 + n_neighbors + j])
    
    # Known optimal values for specific instances
    optimal_fitness = UNKNOWN_OPTIMAL
    if instance_name.replace('.txt', '') == 'SG_100_1':
        optimal_fitness = 130
    elif instance_name.replace('.txt', '') == 'SG_100_2':
        optimal_fitness = 136
    elif instance_name.replace('.txt', '') == 'SG_100_3':
        optimal_fitness = 136
    elif instance_name.replace('.txt', '') == 'SG_100_4':
        optimal_fitness = 130
    
    return n_vars, lattice, inter, optimal_fitness


def evaluate_ising(solution: np.ndarray, lattice, inter) -> float:
    """
    Evaluate an Ising solution
    
    Parameters
    ----------
    solution : np.ndarray
        Binary vector (0/1)
    lattice : np.ndarray
        Lattice structure
    inter : np.ndarray
        Interaction values
    
    Returns
    -------
    float
        Energy value (we maximize -energy, so this returns -eval_ising)
    """
    if solution.ndim == 1:
        # eval_ising returns negative energy, so we negate to maximize
        return -eval_ising(solution, lattice, inter)
    else:
        return np.array([-eval_ising(sol, lattice, inter) for sol in solution])


def load_ubqp_instance(instance_name: str):
    """
    Load a UBQP instance
    
    Parameters
    ----------
    instance_name : str
        Name of the instance (e.g., 'bqp50')
    
    Returns
    -------
    tuple
        (n_vars, ubqp_instance, optimal_value)
        optimal_value is "Unknown" for these instances
    """
    # Get the path to the UBQP instances directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    instances_dir = os.path.join(os.path.dirname(script_dir), 'functions', 'UBQP_Instances')
    
    # Add .txt extension if not present
    if not instance_name.endswith('.txt'):
        instance_name = instance_name + '.txt'
    
    instance_file = os.path.join(instances_dir, instance_name)
    
    if not os.path.exists(instance_file):
        raise FileNotFoundError(f"UBQP instance file not found: {instance_file}")
    
    with open(instance_file, 'r') as f:
        # First line: seed (not used)
        seed_line = f.readline().strip()
        
        # Second line: n_vars n_edges
        header = f.readline().strip().split()
        n_vars = int(header[0])
        n_edges = int(header[1])
        
        # Create UBQP instance
        ubqp_instance = UBQPInstance(n_vars, n_objectives=1)
        
        # Read edges
        for _ in range(n_edges):
            parts = f.readline().strip().split()
            i = int(parts[0])
            j = int(parts[1])
            weight = float(parts[2])
            ubqp_instance.add_interaction(0, i, j, weight)
    
    # Known optimal values for specific instances
    optimal_fitness = UNKNOWN_OPTIMAL
    instance_base = instance_name.replace('.txt', '')
    if instance_base == 'bqp100':
        optimal_fitness = 3955
    
    return n_vars, ubqp_instance, optimal_fitness


def evaluate_ubqp(solution: np.ndarray, ubqp_instance: UBQPInstance) -> float:
    """
    Evaluate a UBQP solution
    
    Parameters
    ----------
    solution : np.ndarray
        Binary vector (0/1)
    ubqp_instance : UBQPInstance
        UBQP instance
    
    Returns
    -------
    float
        Objective value
    """
    result = ubqp_instance.evaluate(solution)
    # Result is already in the right shape
    if result.ndim == 2:
        # Single objective, extract the value
        return result[:, 0] if len(result.shape) > 1 else result.flatten()
    return result.flatten() if hasattr(result, 'flatten') else result


# ==============================================================================
# Problem Configuration
# ==============================================================================

def parse_rw_problem(problem_type: str, instance_name: str):
    """
    Parse real-world problem and return fitness function, n_vars, and optimal fitness
    
    Parameters
    ----------
    problem_type : str
        Problem type: 'SAT', 'Ising', or 'UBQP'
    instance_name : str
        Name of the instance file
    
    Returns
    -------
    func : callable
        Fitness function
    n_vars : int
        Number of variables
    optimal : str or float
        Optimal fitness value (UNKNOWN_OPTIMAL for these instances)
    """
    problem_type = problem_type.upper()
    
    if problem_type == 'SAT':
        n_vars, clauses, optimal = load_sat_instance(instance_name)
        
        def fitness_func(solution):
            return evaluate_sat(solution, clauses)
        
        return fitness_func, n_vars, optimal
    
    elif problem_type == 'ISING':
        n_vars, lattice, inter, optimal = load_ising_instance(instance_name)
        
        def fitness_func(solution):
            return evaluate_ising(solution, lattice, inter)
        
        return fitness_func, n_vars, optimal
    
    elif problem_type == 'UBQP':
        n_vars, ubqp_instance, optimal = load_ubqp_instance(instance_name)
        
        def fitness_func(solution):
            return evaluate_ubqp(solution, ubqp_instance)
        
        return fitness_func, n_vars, optimal
    
    else:
        raise ValueError(f"Unknown problem type: {problem_type}. Supported types: SAT, Ising, UBQP")


# ==============================================================================
# GAN EDA Implementation
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
        alpha: float = 0.0,
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
        self.weight_transfer = weight_transfer
        self.early_stopping = early_stopping
        self.init_method = init_method
        self.loss_function = loss_function
        self.activation = activation
        self.surrogate_filtering = surrogate_filtering
        self.learning_params = learning_params or {}
        self.sampling_params = sampling_params or {}
        self.random_seed = random_seed
        self.alpha = alpha

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

    def run(self, fitness_func, verbose=True, optimal_fitness=None):
        """
        Run the Backdrive EDA

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

        # Check if optimum reached in initial population
        if optimal_fitness is not None and optimal_fitness != UNKNOWN_OPTIMAL and abs(best_fitness - optimal_fitness) < OPTIMUM_TOLERANCE:
            if verbose:
                print(f"\nOptimum reached!")
                print(f"\nBackdrive-EDA completed after 0 generations")
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
                            
                            # Generate more samples than needed for filtering
                            candidate_pop = sample_fn(model, self.pop_size * SURROGATE_CANDIDATE_FACTOR, sampling_params)
                            
                            # Get descriptor statistics for denormalization
                            if 'descriptor_stats' in model:
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
                                # descriptor_stats not found, cannot denormalize predictions
                                if verbose and gen == 0:
                                    print("  Warning: descriptor_stats not found in model, surrogate filtering disabled")
                                population = new_population
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

                        # Generate more samples than needed for filtering
                        candidate_pop = sample_fn(model, self.pop_size * SURROGATE_CANDIDATE_FACTOR, sampling_params)

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

            # Evaluate (only once)
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
            if optimal_fitness is not None and optimal_fitness != UNKNOWN_OPTIMAL and abs(best_fitness - optimal_fitness) < OPTIMUM_TOLERANCE:
                if verbose:
                    print(f"\nOptimum reached!")
                    print(f"\nBackdrive-EDA completed after {gen+1} generations")
                    print(f"Best fitness found: {best_fitness:.6f}")
                    print(f"  at generation {generation_found}")
                return best_fitness, best_solution, history

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
        description='Discrete Backdrive EDA - Real-World Combinatorial Problems',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic backdrive with MSE on SAT
  python discrete_Backdrive_EDA_RW.py 0 SAT uf20-01 80 20 0.5 backdrive random mse relu 0 0 0 0.0

  # Backdrive with weighted MSE on Ising with mutation
  python discrete_Backdrive_EDA_RW.py 1 Ising SG_16_1 100 30 0.5 backdrive random weighted_mse tanh 1 1 0 0.95

  # Backdrive descriptors on UBQP with surrogate filtering
  python discrete_Backdrive_EDA_RW.py 2 UBQP bqp50 200 50 0.5 backdrive_descriptors random mse relu 0 1 1 0.0
        """
    )

    # All positional arguments
    parser.add_argument('seed', type=int, help='Random seed')
    parser.add_argument('problem_type', type=str, help='Problem type (SAT, Ising, UBQP)')
    parser.add_argument('instance_name', type=str, help='Instance file name')
    parser.add_argument('pop_size', type=int, help='Population size')
    parser.add_argument('n_gen', type=int, help='Number of generations')
    parser.add_argument('trunc', type=float, help='Truncation percent (selection ratio, e.g., 0.5 for 50%)')
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
    parser.add_argument('alpha', type=float,
                        help='Max frequency threshold for mutation (default: 0.0, no mutation)')

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
        fitness_func, n_vars, optimal_fitness = parse_rw_problem(args.problem_type, args.instance_name)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Print configuration
    print("=" * 80)
    print("DISCRETE BACKDRIVE EDA - Real-World Problem Configuration")
    print("=" * 80)
    print(f"Seed:               {args.seed}")
    print(f"Problem Type:       {args.problem_type}")
    print(f"Instance:           {args.instance_name}")
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
    print(f"Alpha (mutation):   {args.alpha}")
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
        }
    else:
        learning_params = {
            'epochs': 30,
            'hidden_dims': adaptive_hidden_dims,
            'batch_size': batch_s,
            'early_stopping': args.early_stopping,
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
        sampling_params['descriptor_sampling'] = 'from_population'

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
        alpha=args.alpha,
    )

    # Pass optimal_fitness only if it's known (not UNKNOWN_OPTIMAL)
    optimal_fitness_param = None if optimal_fitness == UNKNOWN_OPTIMAL else optimal_fitness
    best_fitness, best_solution, history = eda.run(fitness_func, verbose=True, optimal_fitness=optimal_fitness_param)

    elapsed_time = time.time() - start_time

    # Print results
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Best Fitness:     {best_fitness:.4f}")
    print(f"Optimal Fitness:  {optimal_fitness}")
    
    # Only compute gap and success if optimal is known
    if optimal_fitness != UNKNOWN_OPTIMAL:
        gap = abs(best_fitness - float(optimal_fitness))
        success = gap < SUCCESS_THRESHOLD * float(optimal_fitness)
        print(f"Gap:              {gap:.4f}")
        print(f"Success:          {'Yes' if success else 'No'}")
    else:
        print(f"Gap:              Unknown")
        print(f"Success:          Unknown")
    
    print(f"Elapsed Time:     {elapsed_time:.2f} seconds")
    print(f"Best Solution:    {best_solution[:20]}{'...' if len(best_solution) > 20 else ''}")
    print("=" * 80)


if __name__ == "__main__":
    main()
