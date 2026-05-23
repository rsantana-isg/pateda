"""
Discrete DbD EDA - Real-World Combinatorial Problems
====================================================

This program combines discrete_DbD_EDA.py and discrete_EDA_RW.py to provide
a unified interface to run various DbD-EDA algorithm variants on real-world
combinatorial problems (SAT, Ising, UBQP) with different seeds for cluster execution.

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

**Enhanced DbD Variants:**
- DbD-Weighted: DbD with fitness-weighted MSE loss
- DbD-Ranking: DbD with ranking loss
- DbD-Huber: DbD with Huber loss (robust to outliers)
- C-DbD: Conditional DbD with fitness guidance
- M-DbD: DbD with Markov model initialization

Supports real-world combinatorial problems:
- SAT: Boolean satisfiability problem
- Ising: Ising spin glass model
- UBQP: Unconstrained Binary Quadratic Programming

Configurable Parameters:
- Activation Function: relu, tanh, sigmoid, leakyrelu, elu, selu, gelu, etc.
- Loss Function: mse, weighted_mse, ranking, huber
- Number of Alpha Samples: For blending
- Denoising Steps: Number of steps during sampling
- Markov Order: k=0,1,2 for transformation variants
- Alpha: Maximum allowed frequency for ones or zeros (mutation control)

Usage:
    python discrete_DbD_EDA_RW.py <seed> <problem_type> <instance_name> <pop_size> <n_gen> <trunc> \\
        <variant> <activation> <loss> <num_alpha_samples> <n_steps> <k> <alpha_smooth> \\
        <fitness_guided> <use_markov_init> <alpha>

Example:
    python discrete_DbD_EDA_RW.py 0 SAT uf20-01 80 20 0.5 dbd relu mse 20 20 0 0.1 0 0 0.0
    python discrete_DbD_EDA_RW.py 1 Ising SG_16_1 100 30 0.5 dbd_cs tanh weighted_mse 20 20 0 0.1 0 0 0.95
    python discrete_DbD_EDA_RW.py 0 UBQP bqp50 200 50 0.5 dbd_cs_t elu mse 20 20 1 0.1 1 0 0.0

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

# DbD learning modules
from pateda_nn.learning.discrete_dbd import (
    learn_binary_dbd, learn_binary_dbd_cs, learn_binary_dbd_cd,
    learn_binary_dbd_uc, learn_binary_dbd_us,
    learn_binary_dbd_cs_t, learn_binary_dbd_cd_t,
    learn_binary_dbd_uc_t, learn_binary_dbd_us_t
)

# DbD sampling modules
from pateda_nn.sampling.discrete_dbd import (
    sample_binary_dbd, sample_binary_dbd_cs, sample_binary_dbd_cd,
    sample_binary_dbd_uc, sample_binary_dbd_us,
    sample_binary_dbd_cs_t, sample_binary_dbd_cd_t,
    sample_binary_dbd_uc_t, sample_binary_dbd_us_t
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

# Tolerance for checking if optimum has been reached
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
        optimal_fitness = 132
    elif instance_name.replace('.txt', '') == 'SG_100_2':
        optimal_fitness = 142
    elif instance_name.replace('.txt', '') == 'SG_100_3':
        optimal_fitness = 142
    elif instance_name.replace('.txt', '') == 'SG_100_4':
        optimal_fitness = 138
    
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
        alpha: float = 0.0,
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
        self.alpha = alpha

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
        from pateda_nn.learning.discrete_dbd import compute_conditional_probabilities

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

    def run(self, fitness_func, verbose=True, optimal_fitness=None):
        """
        Run the DbD EDA

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
        if self.use_markov_init and self.markov_model is not None:
            # Initialize from Markov model
            population = self._sample_from_markov_model(self.markov_model, self.pop_size)
        else:
            # Random initialization (use scalar 2 for binary variables)
            population = np.random.randint(0, 2, (self.pop_size, self.n_vars))

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
                print(f"\nDbD-EDA completed after 0 generations")
                print(f"Best fitness found: {best_fitness:.6f}")
                print(f"  at generation {generation_found}")
            return best_fitness, best_solution, history

        # Keep track of previous population for DbD variants
        prev_population = None
        prev_selected_pop = None

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

            # Update Markov model if using Markov initialization
            if self.use_markov_init:
                # Learn/update Markov model from selected population
                markov_k = self.learning_params.get('markov_k', 1)
                self.markov_model = self._learn_markov_model(selected_pop, markov_k)

            # Prepare learning parameters
            # Dynamic hidden layer computation based on n_vars and population size
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
            
            # Pass loss function parameter
            learning_params['loss_function'] = self.loss_function

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
                    # First generation: use random as current population (use scalar 2 for binary)
                    current_pop = np.random.randint(0, 2, (len(selected_pop), self.n_vars))
                    # Evaluate fitness for random population if needed for loss function
                    if self.loss_function in LOSS_FUNCTIONS_REQUIRING_FITNESS or self.fitness_guided:
                        fitness_current = fitness_func(current_pop)
                    else:
                        fitness_current = None
                else:
                    # Sample from previous population to match selected population size
                    n_to_sample = len(selected_pop)
                    if len(prev_population) >= n_to_sample:
                        indices = np.random.choice(len(prev_population), n_to_sample, replace=False)
                    else:
                        indices = np.random.choice(len(prev_population), n_to_sample, replace=True)
                    current_pop = prev_population[indices]
                    
                    # Sample corresponding fitness values if needed
                    if self.loss_function in LOSS_FUNCTIONS_REQUIRING_FITNESS or self.fitness_guided:
                        # Get fitness for sampled current population
                        fitness_current = fitness_func(current_pop)
                    else:
                        fitness_current = None

                # Learn model with current and selected populations, including fitness
                # Pass fitness values to learning function
                model = learn_fn(current_pop, selected_pop, learning_params, 
                                fitness_current, selected_fitness)

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
                # Use scalar 2 for binary variables
                population = np.random.randint(0, 2, (self.pop_size, self.n_vars))

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
                    print(f"\nDbD-EDA completed after {gen+1} generations")
                    print(f"Best fitness found: {best_fitness:.6f}")
                    print(f"  at generation {generation_found}")
                return best_fitness, best_solution, history

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
        description='Discrete DbD EDA - Real-World Combinatorial Problems',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # SAT problem with standard DbD
  python discrete_DbD_EDA_RW.py 0 SAT uf20-01 80 20 0.5 dbd relu mse 20 20 0 0.1 0 0 0.0

  # Ising problem with DbD-CS and mutation
  python discrete_DbD_EDA_RW.py 1 Ising SG_16_1 100 30 0.5 dbd_cs tanh weighted_mse 20 20 0 0.1 0 0 0.95

  # UBQP problem with DbD-CS-T
  python discrete_DbD_EDA_RW.py 0 UBQP bqp50 200 50 0.5 dbd_cs_t elu mse 20 20 1 0.1 1 0 0.0
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
                       choices=['dbd', 'dbd_cs', 'dbd_cd', 'dbd_uc', 'dbd_us',
                               'dbd_cs_t', 'dbd_cd_t', 'dbd_uc_t', 'dbd_us_t'],
                       help='DbD variant to use')
    parser.add_argument('activation', type=str,
                       help='Activation function. Options: relu, tanh, sigmoid, leaky_relu, elu, selu, gelu, etc.')
    parser.add_argument('loss', type=str,
                       choices=['mse', 'weighted_mse', 'ranking', 'huber'],
                       help='Loss function')
    parser.add_argument('num_alpha_samples', type=int,
                       help='Number of alpha samples for blending')
    parser.add_argument('n_steps', type=int,
                       help='Number of denoising steps during sampling')
    parser.add_argument('k', type=int,
                       help='Order of Markov chain for transformation variants (0, 1, 2)')
    parser.add_argument('alpha_smooth', type=float,
                       help='Smoothing parameter for Markov probabilities')
    parser.add_argument('fitness_guided', type=int, choices=[0, 1],
                       help='Use fitness guidance (1=yes, 0=no)')
    parser.add_argument('use_markov_init', type=int, choices=[0, 1],
                       help='Use Markov model for initialization (1=yes, 0=no)')
    parser.add_argument('alpha', type=float,
                       help='Max frequency threshold for mutation (default: 0.0, no mutation)')

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
        fitness_func, n_vars, optimal_fitness = parse_rw_problem(args.problem_type, args.instance_name)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Print configuration
    print("=" * 80)
    print("DISCRETE DBD EDA - Real-World Problem Configuration")
    print("=" * 80)
    print(f"Seed:               {args.seed}")
    print(f"Problem Type:       {args.problem_type}")
    print(f"Instance:           {args.instance_name}")
    print(f"Variables:          {n_vars}")
    print(f"Optimal Fitness:    {optimal_fitness}")
    print(f"Population Size:    {args.pop_size}")
    print(f"Generations:        {args.n_gen}")
    print(f"Truncation Percent: {args.trunc}")
    print(f"DbD Variant:        {args.variant}")
    print(f"Activation:         {args.activation}")
    print(f"Loss Function:      {args.loss}")
    print(f"Num Alpha Samples:  {args.num_alpha_samples}")
    print(f"Denoising Steps:    {args.n_steps}")
    print(f"Markov Order (k):   {args.k}")
    print(f"Alpha Smooth:       {args.alpha_smooth}")
    print(f"Fitness Guided:     {args.fitness_guided}")
    print(f"Use Markov Init:    {args.use_markov_init}")
    print(f"Alpha (mutation):   {args.alpha}")
    print("=" * 80)
    print()

    start_time = time.time()

    # Configure learning and sampling parameters
    learning_params = {}
    sampling_params = {}

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
