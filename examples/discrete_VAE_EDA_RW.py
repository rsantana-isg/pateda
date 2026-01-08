"""
Discrete VAE EDA - Real-World Combinatorial Problems
====================================================

This program combines discrete_VAE_EDA.py and discrete_EDA_RW.py to provide
a unified interface to run various VAE-EDA algorithm variants on real-world
combinatorial problems (SAT, Ising, UBQP) with different seeds for cluster execution.

Supports twelve VAE-EDA variants:

**Original Variants:**
- VAE: Standard Variational Autoencoder with β-annealing
- E-VAE: Enhanced VAE with fitness predictor
- C-VAE: Conditional VAE conditioned on fitness and statistics
- Desc-VAE: Descriptor-augmented VAE with landscape information
- Reg-VAE: Regression-focused VAE with fitness-weighted reconstruction
- Mom-VAE: Moment-matching VAE with statistical alignment

**Enhanced Variants:**
- BA-VAE: Beta-Annealed VAE - Addresses posterior collapse
- AA-VAE: Adaptive-Architecture VAE - Addresses overfitting
- FW-VAE: Fitness-Weighted VAE - Better fitness guidance
- GS-VAE: Greedy-Sampling VAE - Deterministic sampling
- HS-VAE: Hybrid-Sampling VAE - Combined sampling
- TC-VAE: Temperature-Controlled VAE - Adaptive temperature

Supports real-world combinatorial problems:
- SAT: Boolean satisfiability problem
- Ising: Ising spin glass model
- UBQP: Unconstrained Binary Quadratic Programming

Configurable Parameters:
- Activation Functions: For encoder and decoder layers
- Beta Annealing: KL divergence weight scheduling
- Latent Dimensions: Size of latent space
- Hidden Layer Dimensions: Defined in terms of number of variables
- Batch Size: Adaptive based on selected population size
- Alpha: Maximum allowed frequency for ones or zeros (mutation control)

Usage:
    python discrete_VAE_EDA_RW.py <seed> <problem_type> <instance_name> <pop_size> <n_gen> <trunc> <vae_variant> \\
        <activation_enc> <activation_dec> <beta_start> <beta_end> <latent_dim> <epochs> <mi_layer> <alpha>

Example:
    python discrete_VAE_EDA_RW.py 0 SAT uf20-01 80 20 0.5 vae relu relu 0.0 1.0 0 30 0 0.0
    python discrete_VAE_EDA_RW.py 1 Ising SG_16_1 100 30 0.5 bavae relu relu 0.0 1.0 8 30 1 0.95
    python discrete_VAE_EDA_RW.py 0 UBQP bqp50 200 50 0.5 cvae relu relu 0.0 1.0 0 30 0 0.0

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

# VAE learning modules
from pateda.learning.discrete_vae import (
    learn_binary_vae,
    learn_binary_cvae,
    learn_binary_descvae,
    learn_binary_regvae,
    learn_binary_momvae,
    learn_binary_bavae,
    learn_binary_aavae,
    learn_binary_fwvae
)

# VAE sampling modules
from pateda.sampling.discrete_neural import (
    sample_binary_vae,
    sample_binary_cvae,
    sample_binary_descvae,
    sample_binary_regvae,
    sample_binary_momvae,
    sample_binary_bavae,
    sample_binary_aavae,
    sample_binary_fwvae,
    sample_binary_gsvae,
    sample_binary_hsvae,
    sample_binary_tcvae
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
    
    return n_vars, clauses, UNKNOWN_OPTIMAL


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
    
    return n_vars, lattice, inter, UNKNOWN_OPTIMAL


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
    
    return n_vars, ubqp_instance, UNKNOWN_OPTIMAL


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
        alpha: float = 0.0,
    ):
        """
        Initialize VAE EDA

        Parameters
        ----------
        variant : str
            VAE variant: 'vae', 'evae', 'cvae', 'descvae', 'regvae', 'momvae',
                        'bavae', 'aavae', 'fwvae', 'gsvae', 'hsvae', 'tcvae'
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
        self.activation_enc = activation_enc
        self.activation_dec = activation_dec
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
            # Original variants
            'vae': (learn_binary_vae, sample_binary_vae),
            'evae': (learn_binary_vae, sample_binary_vae),  # E-VAE uses learn_binary_vae with use_extended=True
            'cvae': (learn_binary_cvae, sample_binary_cvae),
            'descvae': (learn_binary_descvae, sample_binary_descvae),
            'regvae': (learn_binary_regvae, sample_binary_regvae),
            'momvae': (learn_binary_momvae, sample_binary_momvae),
            # Enhanced variants
            'bavae': (learn_binary_bavae, sample_binary_bavae),
            'aavae': (learn_binary_aavae, sample_binary_aavae),
            'fwvae': (learn_binary_fwvae, sample_binary_fwvae),
            'gsvae': (learn_binary_vae, sample_binary_gsvae),  # Uses standard learning, greedy sampling
            'hsvae': (learn_binary_vae, sample_binary_hsvae),  # Uses standard learning, hybrid sampling
            'tcvae': (learn_binary_vae, sample_binary_tcvae),  # Uses standard learning, temp-controlled sampling
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
                elif self.variant == 'tcvae':
                    # TC-VAE: pass generation info for temperature control
                    sampling_params['generation'] = gen + 1
                    sampling_params['max_generations'] = self.max_generations

                # Sample new population
                population = sample_fn(model, self.pop_size, sampling_params)

            except Exception as e:
                if verbose:
                    print(f"  Warning: Learning/Sampling failed ({e}), using random population")
                population = np.random.randint(0, self.cardinality,
                                             (self.pop_size, self.n_vars))

            # Evaluate
            fitness = fitness_func(population)
            
            # Apply frequency balance mutation if alpha > 0
            if self.alpha > 0:
                # Store the best solution before mutation to enforce elitism
                best_idx = np.argmax(fitness)
                best_solution_pre_mutation = population[best_idx].copy()
                
                mutation_params = {'alpha': self.alpha}
                population = frequency_balance_mutation(
                    self.n_vars,
                    self.cardinality,
                    population,
                    mutation_params
                )
                
                # Enforce elitism: ensure the best solution is not mutated
                # Replace the individual at best_idx with the original best solution
                population[best_idx] = best_solution_pre_mutation
                
                # Re-evaluate only if mutation was applied
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
        description='Discrete VAE EDA - Real-World Combinatorial Problems',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # SAT problem with standard VAE
  python discrete_VAE_EDA_RW.py 0 SAT uf20-01 80 20 0.5 vae relu relu 0.0 1.0 0 30 0 0.0

  # Ising problem with BA-VAE and mutation
  python discrete_VAE_EDA_RW.py 1 Ising SG_16_1 100 30 0.5 bavae relu relu 0.0 1.0 8 30 1 0.95

  # UBQP problem with C-VAE
  python discrete_VAE_EDA_RW.py 0 UBQP bqp50 200 50 0.5 cvae relu relu 0.0 1.0 0 30 0 0.0
        """
    )

    # All positional arguments
    parser.add_argument('seed', type=int, help='Random seed')
    parser.add_argument('problem_type', type=str, help='Problem type (SAT, Ising, UBQP)')
    parser.add_argument('instance_name', type=str, help='Instance file name')
    parser.add_argument('pop_size', type=int, help='Population size')
    parser.add_argument('n_gen', type=int, help='Number of generations')
    parser.add_argument('trunc', type=float, help='Truncation percent (selection ratio, e.g., 0.5 for 50%)')
    parser.add_argument('vae_variant', type=str,
                       choices=['vae', 'evae', 'cvae', 'descvae', 'regvae', 'momvae',
                               'bavae', 'aavae', 'fwvae', 'gsvae', 'hsvae', 'tcvae'],
                       help='VAE variant to use')
    parser.add_argument('activation_enc', type=str,
                       help='Activation function for Encoder. Options: relu, tanh, sigmoid, leaky_relu, elu, selu, gelu, etc.')
    parser.add_argument('activation_dec', type=str,
                       help='Activation function for Decoder')
    parser.add_argument('beta_start', type=float,
                       help='Initial KL weight for beta annealing')
    parser.add_argument('beta_end', type=float,
                       help='Final KL weight for beta annealing')
    parser.add_argument('latent_dim', type=int,
                       help='Latent dimension size (0 for automatic sizing)')
    parser.add_argument('epochs', type=int,
                       help='Number of training epochs')
    parser.add_argument('mi_layer', type=int, choices=[0, 1],
                       help='Use mutual information layer (1=yes, 0=no)')
    parser.add_argument('alpha', type=float,
                       help='Max frequency threshold for mutation (default: 0.0, no mutation)')

    # Parse arguments
    args = parser.parse_args()

    # Convert integer flags to boolean
    args.mi_layer = bool(args.mi_layer)

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
    print("DISCRETE VAE EDA - Real-World Problem Configuration")
    print("=" * 80)
    print(f"Seed:               {args.seed}")
    print(f"Problem Type:       {args.problem_type}")
    print(f"Instance:           {args.instance_name}")
    print(f"Variables:          {n_vars}")
    print(f"Optimal Fitness:    {optimal_fitness}")
    print(f"Population Size:    {args.pop_size}")
    print(f"Generations:        {args.n_gen}")
    print(f"Truncation Percent: {args.trunc}")
    print(f"VAE Variant:        {args.vae_variant}")
    print(f"Activation (Enc):   {args.activation_enc}")
    print(f"Activation (Dec):   {args.activation_dec}")
    print(f"Beta Start:         {args.beta_start}")
    print(f"Beta End:           {args.beta_end}")
    print(f"Latent Dim:         {args.latent_dim if args.latent_dim > 0 else 'Auto'}")
    print(f"Epochs:             {args.epochs}")
    print(f"MI Layer:           {args.mi_layer}")
    print(f"Alpha (mutation):   {args.alpha}")
    print("=" * 80)
    print()

    start_time = time.time()

    # Configure learning parameters
    learning_params = {
        'epochs': args.epochs,
        'use_mi_layer': args.mi_layer,
    }

    # Set latent_dim only if specified (otherwise use automatic)
    if args.latent_dim > 0:
        learning_params['latent_dim'] = args.latent_dim

    # Configure sampling parameters
    sampling_params = {}

    cardinality = np.full(n_vars, 2)

    # Create and run VAE EDA
    eda = VAEEDA(
        variant=args.vae_variant,
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
