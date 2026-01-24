"""
Discrete GAN EDA - Real-World Combinatorial Problems
====================================================

This program combines discrete_GAN_EDA.py and discrete_EDA_RW.py to provide
a unified interface to run various GAN-EDA algorithm variants on real-world
combinatorial problems (SAT, Ising, UBQP) with different seeds for cluster execution.

Supports seven GAN-EDA variants as described in Deeper_GAN_Critical_Analysis.md:

1. V1-WGAN-GP: Wasserstein Loss + Gradient Penalty
2. V2-Cond-Fit-GAN: Condition input on target fitness percentiles
3. V3-Aux-GAN: Auxiliary head for fitness prediction
4. V4-Repulsion-GAN: Batch-wide diversity penalty in Generator
5. V5-Weighted-D-GAN: Fitness-weighted Real/Fake classification
6. V6-Statistic-Match: MSE loss on mean/std of generated batch
7. V7-Hybrid-GAN-VAE: GAN with an Encoder (BiGAN)

Supports real-world combinatorial problems:
- SAT: Boolean satisfiability problem
- Ising: Ising spin glass model
- UBQP: Unconstrained Binary Quadratic Programming

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
- Alpha: Maximum allowed frequency for ones or zeros (mutation control)

Usage:
    python discrete_GAN_EDA_RW.py <seed> <problem_type> <instance_name> <pop_size> <n_gen> <trunc> <variant> \\
        <activation_g> <activation_d> <activation_e> <dropout> <temperature> <use_surrogate> <alpha>

Example:
    python discrete_GAN_EDA_RW.py 0 SAT uf20-01 80 20 0.5 WGAN-GP relu leaky_relu relu 0.5 1.0 0 0.0
    python discrete_GAN_EDA_RW.py 1 Ising SG_16_1 100 30 0.5 Cond-Fit-GAN tanh leaky_relu relu 0.5 1.0 0 0.95
    python discrete_GAN_EDA_RW.py 0 UBQP bqp50 200 50 0.5 Aux-GAN relu leaky_relu relu 0.5 1.0 1 0.0

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
            Maximum allowed frequency for ones or zeros (default 0.0, no mutation)
            If alpha > 0, applies frequency balance mutation
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

    def run(self, fitness_func, verbose=True, optimal_fitness=None):
        """
        Run the GAN EDA

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

        # Check if optimum reached in initial population
        if optimal_fitness is not None and optimal_fitness != UNKNOWN_OPTIMAL and abs(best_fitness - optimal_fitness) < OPTIMUM_TOLERANCE:
            if verbose:
                print(f"\nOptimum reached!")
                print(f"\nGAN-EDA completed after 0 generations")
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
                    print(f"\nGAN-EDA completed after {gen+1} generations")
                    print(f"Best fitness found: {best_fitness:.6f}")
                    print(f"  at generation {generation_found}")
                return best_fitness, best_solution, history

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
        description='Discrete GAN EDA - Real-World Combinatorial Problems',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # SAT problem with WGAN-GP variant
  python discrete_GAN_EDA_RW.py 0 SAT uf20-01 80 20 0.5 WGAN-GP relu leaky_relu relu 0.5 1.0 0 0.0

  # Ising problem with Cond-Fit-GAN and mutation
  python discrete_GAN_EDA_RW.py 1 Ising SG_16_1 100 30 0.5 Cond-Fit-GAN tanh leaky_relu relu 0.5 1.0 0 0.95

  # UBQP problem with Aux-GAN and surrogate filtering
  python discrete_GAN_EDA_RW.py 0 UBQP bqp50 200 50 0.5 Aux-GAN relu leaky_relu relu 0.5 1.0 1 0.0
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
    parser.add_argument('alpha', type=float,
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
        fitness_func, n_vars, optimal_fitness = parse_rw_problem(args.problem_type, args.instance_name)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Print configuration
    print("=" * 80)
    print("DISCRETE GAN EDA - Real-World Problem Configuration")
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
    print(f"Activation (Gen):   {args.activation_g}")
    print(f"Activation (Disc):  {args.activation_d}")
    if args.variant == 'Hybrid-GAN-VAE':
        print(f"Activation (Enc):   {args.activation_e}")
    print(f"Dropout:            {args.dropout}")
    print(f"Temperature:        {args.temperature}")
    if args.variant == 'Aux-GAN':
        print(f"Use Surrogate:      {args.use_surrogate}")
    print(f"Alpha (mutation):   {args.alpha}")
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
