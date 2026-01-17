"""
Discrete EDA - Real-World Combinatorial Problems
===============================================================

This program provides a unified interface to run various discrete EDA algorithms
on real-world combinatorial problems (SAT, Ising, UBQP).

Supports both neural network-based EDAs and traditional EDAs:

Neural EDAs:
- VAE: Variational Autoencoder with Gumbel-Softmax
- GAN: Generative Adversarial Network
- Backdrive: Network inversion approach
- DAE: Denoising Autoencoder
- RBM: Restricted Boltzmann Machine
- DbD: Diffusion-by-Deblending

Traditional EDAs:
- UMDA: Univariate Marginal Distribution Algorithm
- TreeEDA: Tree-based Factorized Distribution Algorithm
- EBNA: Estimation of Bayesian Network Algorithm
- MOA: Markovianity Based Optimization Algorithm
- MN-FDA: Markov Network Factorized Distribution Algorithm
- MN-FDAG: MN-FDA with G-test
- MK-EDA: k-order Markov Chain EDA (k=1,2,3)
- MT-EDA: Mixture of Trees EDA (k=2,3)

Usage:
    python discrete_EDA_RW.py <seed> <problem_type> <instance_name> <pop_size> <n_gen> <alg>

Example:
    python discrete_EDA_RW.py 0 SAT uf20-01 80 20 VAE
    python discrete_EDA_RW.py 1 Ising SG_16_1 100 30 UMDA
    python discrete_EDA_RW.py 0 UBQP bqp50 200 50 TreeEDA

==============================================================================
"""

import sys
import os

# Add parent directory to path for running examples without installation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import time
import math
from typing import Dict, Any
import warnings

# Neural learning modules
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
from pateda.learning.discrete_gan import learn_binary_gan
from pateda.learning.discrete_backdrive import learn_binary_backdrive
from pateda.learning.discrete_backdrive_weighted_mse import learn_binary_backdrive_weighted_mse
from pateda.learning.discrete_backdrive_ranking import learn_binary_backdrive_ranking
from pateda.learning.discrete_backdrive_huber import learn_binary_backdrive_huber
from pateda.learning.dae import learn_dae
from pateda.learning.rbm import learn_softmax_rbm
from pateda.learning.discrete_dbd import (
    learn_binary_dbd, learn_binary_dbd_cs, learn_binary_dbd_cd,
    learn_binary_dbd_uc, learn_binary_dbd_us,
    learn_binary_dbd_cs_t, learn_binary_dbd_cd_t,
    learn_binary_dbd_uc_t, learn_binary_dbd_us_t
)
from pateda.learning.discrete_dendiff_gumbel import learn_discrete_dendiff_gumbel
from pateda.learning.discrete_dendiff_corruption import learn_discrete_dendiff_corruption

# Neural sampling modules
from pateda.sampling.discrete_neural import (
    sample_binary_vae, sample_binary_cvae, sample_binary_descvae,
    sample_binary_regvae, sample_binary_momvae, sample_binary_bavae,
    sample_binary_aavae, sample_binary_fwvae, sample_binary_gsvae,
    sample_binary_hsvae, sample_binary_tcvae,
    sample_binary_gan, sample_binary_backdrive,
    sample_binary_backdrive_adaptive
)
from pateda.sampling.dae import sample_dae
from pateda.sampling.rbm import sample_softmax_rbm
from pateda.sampling.discrete_dbd import (
    sample_binary_dbd, sample_binary_dbd_cs, sample_binary_dbd_cd,
    sample_binary_dbd_uc, sample_binary_dbd_us,
    sample_binary_dbd_cs_t, sample_binary_dbd_cd_t,
    sample_binary_dbd_uc_t, sample_binary_dbd_us_t
)
from pateda.sampling.discrete_dendiff import (
    sample_discrete_dendiff_gumbel, sample_discrete_dendiff_corruption
)

# Traditional EDA modules
from pateda.core.eda import EDA, EDAComponents
from pateda.stop_conditions import MaxGenerations
from pateda.seeding import RandomInit
from pateda.selection import TruncationSelection
from pateda.replacement import ElitistReplacement

# Traditional learning methods
from pateda.learning.umda import LearnUMDA
from pateda.learning.ebna import LearnEBNA
from pateda.learning.tree import LearnTreeModel
from pateda.learning.moa import LearnMOA
from pateda.learning.mnfda import LearnMNFDA
from pateda.learning.mnfdag import LearnMNFDAG
from pateda.learning.markov import LearnMarkovChain
from pateda.learning.mixture_trees import LearnMixtureTrees

# Traditional sampling methods
from pateda.sampling.bayesian_network import SampleBayesianNetwork
from pateda.sampling.fda import SampleFDA
from pateda.sampling.gibbs import SampleGibbs
from pateda.sampling.markov import SampleMarkovChain
from pateda.sampling.mixture_trees import SampleMixtureTrees

# Benchmark functions
from pateda.functions.discrete.additive_decomposable import (
    k_deceptive, decep3, decep_marta3, decep_marta3_new, decep3_mh,
    two_peaks_decep3, decep_venturini, hard_decep5,
    hiff, fhtrap1,
    first_polytree3_ochoa, first_polytree5_ochoa,
    fc2, fc3, fc4, fc5
)

# Real-world problem functions
from pateda.functions.discrete.ising import load_ising, eval_ising
from pateda.functions.discrete.ubqp import UBQPInstance

# Mutation operators
from pateda.mutation import frequency_balance_mutation, FrequencyBalanceMutation


# ==============================================================================
# Constants
# ==============================================================================

# Success threshold as a fraction of optimal fitness
SUCCESS_THRESHOLD = 0.01

# Constant for unknown optimal values
UNKNOWN_OPTIMAL = "Unknown"


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
        
        # Validate n_vars against instance name if possible
        # Format: uf<n>-<inst>
        if instance_name.startswith('uf'):
            try:
                expected_n = int(instance_name.split('-')[0][2:])
                if expected_n != n_vars:
                    raise ValueError(f"Instance name suggests {expected_n} variables but file has {n_vars}")
            except (ValueError, IndexError):
                pass  # Can't parse, skip validation
        
        def fitness_func(solution):
            return evaluate_sat(solution, clauses)
        
        return fitness_func, n_vars, optimal
    
    elif problem_type == 'ISING':
        n_vars, lattice, inter, optimal = load_ising_instance(instance_name)
        
        # Validate n_vars against instance name
        # Format: SG_<n>_<inst>
        parts = instance_name.replace('.txt', '').split('_')
        if len(parts) >= 2:
            try:
                expected_n = int(parts[1])
                if expected_n != n_vars:
                    raise ValueError(f"Instance name suggests {expected_n} variables but file has {n_vars}")
            except (ValueError, IndexError):
                pass  # Can't parse, skip validation
        
        def fitness_func(solution):
            return evaluate_ising(solution, lattice, inter)
        
        return fitness_func, n_vars, optimal
    
    elif problem_type == 'UBQP':
        n_vars, ubqp_instance, optimal = load_ubqp_instance(instance_name)
        
        # Validate n_vars against instance name if possible
        # Format: bqp<n>
        if instance_name.startswith('bqp'):
            try:
                expected_n = int(instance_name.replace('.txt', '')[3:])
                if expected_n != n_vars:
                    raise ValueError(f"Instance name suggests {expected_n} variables but file has {n_vars}")
            except (ValueError, IndexError):
                pass  # Can't parse, skip validation
        
        def fitness_func(solution):
            return evaluate_ubqp(solution, ubqp_instance)
        
        return fitness_func, n_vars, optimal
    
    else:
        raise ValueError(f"Unknown problem type: {problem_type}. Supported types: SAT, Ising, UBQP")




# ==============================================================================
# Neural EDA Implementation
# ==============================================================================

class UnifiedDiscreteNeuralEDA:
    """
    Unified framework for all discrete neural EDAs
    """

    def __init__(
        self,
        method: str,
        n_vars: int,
        cardinality: np.ndarray,
        pop_size: int = 100,
        selection_ratio: float = 0.5,
        max_generations: int = 50,
        learning_params: Dict[str, Any] = None,
        sampling_params: Dict[str, Any] = None,
        random_seed: int = None,
        alpha: float = 0.0,
    ):
        """
        Initialize Unified Neural EDA

        Parameters
        ----------
        method : str
            Method: 'vae', 'gan', 'backdrive', 'backdrive_random', 'backdrive_perturb_best', 
                    'backdrive_perturb_selected', 'backdrive_adaptive', 'dae', 'rbm', 'dbd'
        n_vars : int
            Number of variables
        cardinality : np.ndarray
            Cardinality of each variable
        pop_size : int
            Population size
        selection_ratio : float
            Selection ratio
        max_generations : int
            Maximum generations
        learning_params : dict
            Learning parameters
        sampling_params : dict
            Sampling parameters
        random_seed : int
            Random seed for reproducibility
        alpha : float
            Maximum allowed frequency for ones or zeros (default 0.0, no mutation)
            If alpha > 0, applies frequency balance mutation
        """
        self.method = method
        self.n_vars = n_vars
        self.cardinality = cardinality
        self.pop_size = pop_size
        self.selection_ratio = selection_ratio
        self.max_generations = max_generations
        self.learning_params = learning_params or {}
        self.sampling_params = sampling_params or {}
        self.random_seed = random_seed
        self.alpha = alpha

        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)

        # Map methods to functions
        # Format: (learn_function, sample_function, needs_two_pops, variant)
        self.method_map = {
            'vae': (learn_binary_vae, sample_binary_vae, False, None),
            'vae_extended': (learn_binary_vae, sample_binary_vae, False, 'extended'),  # E-VAE with fitness guidance
            'cvae': (learn_binary_cvae, sample_binary_cvae, False, None),
            'descvae': (learn_binary_descvae, sample_binary_descvae, False, None),
            'regvae': (learn_binary_regvae, sample_binary_regvae, False, None),
            'momvae': (learn_binary_momvae, sample_binary_momvae, False, None),
            'bavae': (learn_binary_bavae, sample_binary_bavae, False, None),
            'aavae': (learn_binary_aavae, sample_binary_aavae, False, None),
            'fwvae': (learn_binary_fwvae, sample_binary_fwvae, False, None),
            'gsvae': (learn_binary_vae, sample_binary_gsvae, False, None),  # Uses standard learning, greedy sampling
            'hsvae': (learn_binary_vae, sample_binary_hsvae, False, None),  # Uses standard learning, hybrid sampling
            'tcvae': (learn_binary_vae, sample_binary_tcvae, False, None),  # Uses standard learning, temp-controlled sampling
            'gan': (learn_binary_gan, sample_binary_gan, False, None),
            'backdrive': (learn_binary_backdrive, sample_binary_backdrive, False, None),
            'backdrive_random': (learn_binary_backdrive, sample_binary_backdrive, False, None),
            'backdrive_perturb_best': (learn_binary_backdrive, sample_binary_backdrive, False, None),
            'backdrive_perturb_selected': (learn_binary_backdrive, sample_binary_backdrive, False, None),
            'backdrive_adaptive': (learn_binary_backdrive, sample_binary_backdrive_adaptive, False, None),
            'backdrive_weighted_mse': (learn_binary_backdrive_weighted_mse, sample_binary_backdrive, False, None),
            'backdrive_ranking': (learn_binary_backdrive_ranking, sample_binary_backdrive, False, None),
            'backdrive_huber': (learn_binary_backdrive_huber, sample_binary_backdrive, False, None),
            'dae': (learn_dae, sample_dae, False, None),
            'rbm': (learn_softmax_rbm, sample_softmax_rbm, True, None),  # Needs cardinality
            'dbd': (learn_binary_dbd, sample_binary_dbd, True, None),  # Needs two populations
            'dbd_cs': (learn_binary_dbd_cs, sample_binary_dbd_cs, True, 'cs'),  # Current to Selected
            'dbd_cd': (learn_binary_dbd_cd, sample_binary_dbd_cd, True, 'cd'),  # Current to Distance
            'dbd_uc': (learn_binary_dbd_uc, sample_binary_dbd_uc, True, 'uc'),  # Univariate to Current
            'dbd_us': (learn_binary_dbd_us, sample_binary_dbd_us, True, 'us'),  # Univariate to Selected
            'dbd_cs_t_k0': (learn_binary_dbd_cs_t, sample_binary_dbd_cs_t, True, 'cs_t'),  # CS with Transformation K=0
            'dbd_cs_t_k1': (learn_binary_dbd_cs_t, sample_binary_dbd_cs_t, True, 'cs_t'),  # CS with Transformation K=1
            'dbd_cs_t_k2': (learn_binary_dbd_cs_t, sample_binary_dbd_cs_t, True, 'cs_t'),  # CS with Transformation K=2
            'dendiff_gumbel': (learn_discrete_dendiff_gumbel, sample_discrete_dendiff_gumbel, False, None),
            'dendiff_corruption': (learn_discrete_dendiff_corruption, sample_discrete_dendiff_corruption, False, None),
        }
        
        # Methods that need current population for initialization
        # This list is maintained separately to avoid hardcoding in multiple places
        self.methods_needing_current_pop = {
            'backdrive_perturb_best', 'backdrive_perturb_selected', 'backdrive_adaptive',
            'backdrive_weighted_mse', 'backdrive_ranking', 'backdrive_huber'
        }

    def run(self, fitness_func, verbose=True):
        """
        Run the EDA

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
        learn_fn, sample_fn, needs_two_pops, variant = self.method_map[self.method]

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

        # Keep track of previous population for DbD variants
        prev_population = None
        prev_selected_pop = None

        for gen in range(self.max_generations):
            # Selection
            n_selected = int(self.pop_size * self.selection_ratio)
            selected_idx = np.argsort(fitness)[-n_selected:]
            selected_pop = population[selected_idx]
            selected_fitness = fitness[selected_idx]

            # Learn model
            try:
                if self.method == 'rbm':
                    model = learn_fn(selected_pop, selected_fitness, self.cardinality,
                                   self.learning_params)
                elif self.method in ['dbd', 'dbd_cs', 'dbd_cd', 'dbd_uc', 'dbd_us',
                                     'dbd_cs_t_k0', 'dbd_cs_t_k1', 'dbd_cs_t_k2']:
                    # DbD variants need two populations (source and target)
                    # Determine p0 (source) and p1 (target) based on variant
                    if prev_population is None:
                        # First generation: use random as current population
                        current_pop = np.random.randint(0, self.cardinality,
                                                      (len(selected_pop), self.n_vars))
                    else:
                        # Sample from previous population to match selected population size
                        # This ensures p0 and p1 have the same size for blending
                        n_to_sample = len(selected_pop)
                        if len(prev_population) >= n_to_sample:
                            indices = np.random.choice(len(prev_population), n_to_sample, replace=False)
                        else:
                            indices = np.random.choice(len(prev_population), n_to_sample, replace=True)
                        current_pop = prev_population[indices]

                    # Learn model with current and selected populations
                    model = learn_fn(current_pop, selected_pop, self.learning_params)

                    # Save for next iteration
                    prev_population = population.copy()
                    prev_selected_pop = selected_pop.copy()
                else:
                    model = learn_fn(selected_pop, selected_fitness, self.learning_params)

                # Prepare sampling parameters
                sampling_params = self.sampling_params.copy()

                # For backdrive perturb methods, add selected population and fitness
                # (not current population, as these methods initialize from best/selected solutions)
                if self.method in self.methods_needing_current_pop:
                    sampling_params['current_population'] = selected_pop
                    sampling_params['current_fitness'] = selected_fitness

                # VAE variant-specific sampling parameters
                if self.method == 'vae_extended':
                    sampling_params['use_fitness_guidance'] = sampling_params.get('use_fitness_guidance', False)
                elif self.method == 'cvae':
                    # C-VAE: can specify target condition for sampling
                    # Default: use max fitness from selected population
                    if 'condition' not in sampling_params:
                        sampling_params['condition'] = np.max(selected_fitness)
                elif self.method == 'regvae':
                    sampling_params['use_fitness_guidance'] = sampling_params.get('use_fitness_guidance', True)
                elif self.method == 'tcvae':
                    # TC-VAE: pass generation info for temperature control
                    sampling_params['generation'] = gen + 1
                    sampling_params['max_generations'] = self.max_generations

                # Sample new population based on method
                if self.method in ['dbd_cs', 'dbd_us', 'dbd_cs_t_k0', 'dbd_cs_t_k1', 'dbd_cs_t_k2']:
                    # DbD-CS and DbD-US variants: initialize from selected population
                    population = sample_fn(model, self.pop_size, selected_pop, sampling_params)
                elif self.method in ['dbd_cd', 'dbd_uc']:
                    # DbD-CD and DbD-UC: initialize from current population
                    population = sample_fn(model, self.pop_size, population, sampling_params)
                elif self.method == 'dbd':
                    # Original DbD: use default sampling
                    population = sample_fn(model, self.pop_size, sampling_params)
                else:
                    # Other methods
                    population = sample_fn(model, self.pop_size, sampling_params)

            except Exception as e:
                if verbose:
                    print(f"  Warning: Sampling failed ({e}), using random population")
                population = np.random.randint(0, self.cardinality,
                                             (self.pop_size, self.n_vars))

            # Evaluate the newly sampled population
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
                # Note: We could optimize further by only evaluating changed individuals,
                # but for simplicity and since fitness is typically fast, we re-evaluate all
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

        # Print completion summary (matching traditional EDA format)
        if verbose:
            print(f"\nEDA completed after {self.max_generations} generations")
            print(f"Best fitness found: {best_fitness:.6f}")
            print(f"  at generation {generation_found}")

        return best_fitness, best_solution, history


# ==============================================================================
# Traditional EDA Implementation
# ==============================================================================

def run_traditional_eda(
    alg: str,
    fitness_func,
    n_vars: int,
    pop_size: int,
    max_generations: int,
    random_seed: int = None,
    verbose: bool = True,
    alpha: float = 0.0,
    truncation_ratio: float = 0.5,
):
    """
    Run a traditional EDA algorithm
    
    Parameters
    ----------
    alg : str
        Algorithm name: 'UMDA', 'TreeEDA', 'EBNA', 'MOA', 'MN-FDA', 'MN-FDAG',
                        'MK-EDA1', 'MK-EDA2', 'MK-EDA3', 'MT-EDA2', 'MT-EDA3'
    fitness_func : callable
        Fitness function
    n_vars : int
        Number of variables
    pop_size : int
        Population size
    max_generations : int
        Maximum generations
    random_seed : int
        Random seed
    verbose : bool
        Print progress
    alpha : float
        Maximum allowed frequency for ones or zeros (default 0.0, no mutation)
    truncation_ratio : float
        Truncation selection ratio (default 0.5)
        
    Returns
    -------
    best_fitness : float
        Best fitness found
    best_solution : np.ndarray
        Best solution found
    history : dict
        History dictionary
    """
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    cardinality = np.full(n_vars, 2)  # Binary
    
    # Configure algorithm
    if alg == 'UMDA':
        learning = LearnUMDA(alpha=1.0)
        sampling = SampleFDA(n_samples=pop_size)
        
    elif alg == 'TreeEDA':
        learning = LearnTreeModel(alpha=0.1)
        sampling = SampleFDA(n_samples=pop_size)
        
    elif alg == 'EBNA':
        learning = LearnEBNA(max_parents=3, score_metric='bic')
        sampling = SampleBayesianNetwork(n_samples=pop_size)
        
    elif alg == 'MOA':
        learning = LearnMOA(k_neighbors=5, threshold_factor=1.5)
        sampling = SampleGibbs(n_samples=pop_size, IT=4, temperature=1.0)
    
    elif alg == 'MN-FDA':
        learning = LearnMNFDA(max_clique_size=3, threshold=0.05, return_factorized=True)
        sampling = SampleFDA(n_samples=pop_size)
    
    elif alg == 'MN-FDAG':
        learning = LearnMNFDAG(max_clique_size=5, alpha=0.01, return_factorized=True)
        sampling = SampleFDA(n_samples=pop_size)
    
    elif alg == 'MK-EDA1':
        learning = LearnMarkovChain(k=1, alpha=0.1)
        sampling = SampleMarkovChain(n_samples=pop_size)
    
    elif alg == 'MK-EDA2':
        learning = LearnMarkovChain(k=2, alpha=0.1)
        sampling = SampleMarkovChain(n_samples=pop_size)
    
    elif alg == 'MK-EDA3':
        learning = LearnMarkovChain(k=3, alpha=0.1)
        sampling = SampleMarkovChain(n_samples=pop_size)
    
    elif alg == 'MT-EDA2':
        learning = LearnMixtureTrees(
            n_components=2, 
            component_learning="tree",
            alpha=0.1,
            weight_learning="uniform",
            random_seed=random_seed
        )
        sampling = SampleMixtureTrees(n_samples=pop_size)
    
    elif alg == 'MT-EDA3':
        learning = LearnMixtureTrees(
            n_components=3,
            component_learning="tree", 
            alpha=0.1,
            weight_learning="uniform",
            random_seed=random_seed
        )
        sampling = SampleMixtureTrees(n_samples=pop_size)
        
    else:
        raise ValueError(f"Unknown traditional EDA: {alg}")
    
    # Create EDA components
    components = EDAComponents(
        seeding=RandomInit(),
        selection=TruncationSelection(ratio=truncation_ratio),
        learning=learning,
        sampling=sampling,
        replacement=ElitistReplacement(),
        stop_condition=MaxGenerations(max_gen=max_generations),
        mutation=FrequencyBalanceMutation(alpha=alpha) if alpha > 0 else None,
    )
    
    # Create and run EDA
    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        fitness_func=fitness_func,
        cardinality=cardinality,
        components=components,
        random_seed=random_seed,
    )
    
    stats, _ = eda.run(verbose=verbose)
    
    # Extract results
    best_fitness = stats.best_fitness_overall
    best_solution = stats.best_individual
    history = {'best_fitness': stats.best_fitness}
    
    return best_fitness, best_solution, history


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    """Main entry point for command-line execution"""
    
    # Check arguments (sys.argv[0] is script name, so 7-15 total = 6-14 arguments + script name)
    if len(sys.argv) < 7 or len(sys.argv) > 15:
        print("Usage: python discrete_EDA_RW.py <seed> <problem_type> <instance_name> <pop_size> <n_gen> <alg> [alpha] [truncation] [activation_enc] [activation_dec] [beta_start] [beta_end] [epochs] [mi_layer]")
        print()
        print("Arguments:")
        print("  seed          : Random seed (integer)")
        print("  problem_type  : Problem type (SAT, Ising, UBQP)")
        print("  instance_name : Instance file name")
        print("  pop_size      : Population size (integer)")
        print("  n_gen         : Number of generations (integer)")
        print("  alg           : Algorithm name")
        print("  alpha         : (Optional) Max frequency threshold for mutation (default: 0.0, no mutation)")
        print("  truncation    : (Optional) Truncation selection ratio (default: 0.5)")
        print("  activation_enc: (Optional) Encoder activation function for VAE (default: relu)")
        print("  activation_dec: (Optional) Decoder activation function for VAE (default: relu)")
        print("  beta_start    : (Optional) Beta annealing start for VAE (default: 0.0)")
        print("  beta_end      : (Optional) Beta annealing end for VAE (default: 1.0)")
        print("  epochs        : (Optional) Training epochs for VAE (default: 50)")
        print("  mi_layer      : (Optional) Use MI layer for VAE: 0 or 1 (default: 0)")
        print()
        print("Supported problem types:")
        print("  SAT   - Boolean satisfiability problem")
        print("         Examples: uf20-01, uf100-01")
        print("  Ising - Ising spin glass model")
        print("         Examples: SG_16_1, SG_100_1, SG_256_1")
        print("  UBQP  - Unconstrained Binary Quadratic Programming")
        print("         Examples: bqp50, bqp100, bqp250")
        print()
        print("Supported algorithms:")
        print("  VAE variants: VAE, E-VAE, C-VAE, Desc-VAE, Reg-VAE, Mom-VAE,")
        print("                BA-VAE, AA-VAE, FW-VAE, GS-VAE, HS-VAE, TC-VAE")
        print("  Neural EDAs: GAN, Backdrive, DAE, RBM, DbD")
        print("  Backdrive variants: Backdrive-Random, Backdrive-PerturbBest,")
        print("                      Backdrive-PerturbSelected, Backdrive-Adaptive")
        print("  DbD variants: DbD-CS, DbD-CD, DbD-UC, DbD-US")
        print("  DbD-T variants: DbD-CS-T-K0, DbD-CS-T-K1, DbD-CS-T-K2")
        print("  Dendiff variants: Dendiff-Gumbel, Dendiff-Corruption")
        print("  Traditional EDAs: UMDA, TreeEDA, EBNA, MOA")
        print("  Markov EDAs: MN-FDA, MN-FDAG, MK-EDA1, MK-EDA2, MK-EDA3")
        print("  Mixture EDAs: MT-EDA2, MT-EDA3")
        print()
        print("Example:")
        print("  python discrete_EDA_RW.py 0 SAT uf20-01 80 20 VAE")
        print("  python discrete_EDA_RW.py 0 SAT uf20-01 80 20 C-VAE 0.0 0.5 relu tanh 0.0 1.0 50 1")
        print("  python discrete_EDA_RW.py 1 Ising SG_16_1 100 30 UMDA 0.95")
        print("  python discrete_EDA_RW.py 0 UBQP bqp50 200 50 TreeEDA 0.95 0.5")
        sys.exit(1)
    
    # Parse arguments
    myseed = int(sys.argv[1])
    problem_type = sys.argv[2]
    instance_name = sys.argv[3]
    pop_size = int(sys.argv[4])
    n_gen = int(sys.argv[5])
    alg = sys.argv[6]
    
    # Optional arguments
    alpha = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0
    truncation_ratio = float(sys.argv[8]) if len(sys.argv) > 8 else 0.5
    
    # VAE-specific optional arguments
    activation_enc = sys.argv[9] if len(sys.argv) > 9 else 'relu'
    activation_dec = sys.argv[10] if len(sys.argv) > 10 else 'relu'
    beta_start = float(sys.argv[11]) if len(sys.argv) > 11 else 0.0
    beta_end = float(sys.argv[12]) if len(sys.argv) > 12 else 1.0
    epochs_vae = int(sys.argv[13]) if len(sys.argv) > 13 else 50
    mi_layer = int(sys.argv[14]) if len(sys.argv) > 14 else 0
    
    # Suppress warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    # Parse problem
    try:
        fitness_func, n_vars, optimal_fitness = parse_rw_problem(problem_type, instance_name)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Print configuration
    print("=" * 80)
    print("DISCRETE EDA - Real-World Problem Configuration")
    print("=" * 80)
    print(f"Seed:             {myseed}")
    print(f"Problem Type:     {problem_type}")
    print(f"Instance:         {instance_name}")
    print(f"Variables:        {n_vars}")
    print(f"Optimal Fitness:  {optimal_fitness}")
    print(f"Population Size:  {pop_size}")
    print(f"Generations:      {n_gen}")
    print(f"Algorithm:        {alg}")
    print(f"Alpha (mutation): {alpha}")
    print(f"Truncation:       {truncation_ratio}")
    # Print VAE-specific parameters if applicable
    vae_variants = ['VAE', 'E-VAE', 'C-VAE', 'Desc-VAE', 'Reg-VAE', 'Mom-VAE',
                    'BA-VAE', 'AA-VAE', 'FW-VAE', 'GS-VAE', 'HS-VAE', 'TC-VAE']
    if alg in vae_variants:
        print(f"Activation Enc:   {activation_enc}")
        print(f"Activation Dec:   {activation_dec}")
        print(f"Beta Start:       {beta_start}")
        print(f"Beta End:         {beta_end}")
        print(f"Epochs:           {epochs_vae}")
        print(f"MI Layer:         {bool(mi_layer)}")
    print("=" * 80)
    print()
    
    # Determine if neural or traditional EDA
    neural_edas = ['VAE', 'E-VAE', 'C-VAE', 'Desc-VAE', 'Reg-VAE', 'Mom-VAE',
                   'BA-VAE', 'AA-VAE', 'FW-VAE', 'GS-VAE', 'HS-VAE', 'TC-VAE',
                   'GAN', 'Backdrive', 'Backdrive-Random', 'Backdrive-PerturbBest',
                   'Backdrive-PerturbSelected', 'Backdrive-Adaptive', 'Backdrive-WeightedMSE',
                   'Backdrive-Ranking', 'Backdrive-Huber', 'DAE', 'RBM', 'DbD',
                   'DbD-CS', 'DbD-CD', 'DbD-UC', 'DbD-US', 
                   'DbD-CS-T-K0', 'DbD-CS-T-K1', 'DbD-CS-T-K2',
                   'Dendiff-Gumbel', 'Dendiff-Corruption']
    traditional_edas = ['UMDA', 'TreeEDA', 'EBNA', 'MOA', 'MN-FDA', 'MN-FDAG',
                       'MK-EDA1', 'MK-EDA2', 'MK-EDA3', 'MT-EDA2', 'MT-EDA3']
    
    start_time = time.time()
    
    if alg in neural_edas:
        # Run neural EDA
        method_map = {
            'VAE': 'vae',
            'E-VAE': 'vae_extended',
            'C-VAE': 'cvae',
            'Desc-VAE': 'descvae',
            'Reg-VAE': 'regvae',
            'Mom-VAE': 'momvae',
            'BA-VAE': 'bavae',
            'AA-VAE': 'aavae',
            'FW-VAE': 'fwvae',
            'GS-VAE': 'gsvae',
            'HS-VAE': 'hsvae',
            'TC-VAE': 'tcvae',
            'GAN': 'gan',
            'Backdrive': 'backdrive',
            'Backdrive-Random': 'backdrive_random',
            'Backdrive-PerturbBest': 'backdrive_perturb_best',
            'Backdrive-PerturbSelected': 'backdrive_perturb_selected',
            'Backdrive-Adaptive': 'backdrive_adaptive',
            'Backdrive-WeightedMSE': 'backdrive_weighted_mse',
            'Backdrive-Ranking': 'backdrive_ranking',
            'Backdrive-Huber': 'backdrive_huber',
            'DAE': 'dae',
            'RBM': 'rbm',
            'DbD': 'dbd',
            'DbD-CS': 'dbd_cs',
            'DbD-CD': 'dbd_cd',
            'DbD-UC': 'dbd_uc',
            'DbD-US': 'dbd_us',
            'DbD-CS-T-K0': 'dbd_cs_t_k0',
            'DbD-CS-T-K1': 'dbd_cs_t_k1',
            'DbD-CS-T-K2': 'dbd_cs_t_k2',
            'Dendiff-Gumbel': 'dendiff_gumbel',
            'Dendiff-Corruption': 'dendiff_corruption',
        }
        
        method_id = method_map[alg]
        
        # Compute common parameters based on pop_size and n_vars
        selection_ratio = truncation_ratio
        selected_pop_size = int(pop_size * selection_ratio)
        adaptive_hidden_dims = [max(10, n_vars // 2), max(10, n_vars // 4)]
        batch_s = min(32, selected_pop_size // 10)
        
        # Configure learning parameters
        params_map = {
            'vae': {
                'epochs': epochs_vae,
                'latent_dim': max(2, n_vars // 4),
                'batch_size': batch_s,
                # Dynamic hidden layers will be computed in learn_binary_vae
                # Beta annealing from command-line arguments
                'beta_start': beta_start,
                'beta_end': beta_end,
                'beta_annealing_epochs': epochs_vae // 2,  # Half of epochs
                'mi_layer': bool(mi_layer),
                # Activation functions (2 layers is the default for learn_binary_vae)
                'list_act_functs_enc': [activation_enc, activation_enc],
                'list_act_functs_dec': [activation_dec, activation_dec],
            },
            'vae_extended': {
                'epochs': epochs_vae,
                'latent_dim': max(2, n_vars // 4),
                'batch_size':  batch_s,
                # Dynamic hidden layers will be computed in learn_binary_vae
                # Beta annealing from command-line arguments
                'beta_start': beta_start,
                'beta_end': beta_end,
                'beta_annealing_epochs': epochs_vae // 2,  # Half of epochs
                # Extended VAE specific parameters
                'use_extended': True,
                'fitness_weight': 0.1,
                'mi_layer': bool(mi_layer),
                # Activation functions (2 layers is the default for learn_binary_vae)
                'list_act_functs_enc': [activation_enc, activation_enc],
                'list_act_functs_dec': [activation_dec, activation_dec],
            },
            'cvae': {
                'epochs': epochs_vae,
                'latent_dim': max(2, n_vars // 4),
                'batch_size': batch_s,
                'beta_start': beta_start,
                'beta_end': beta_end,
                'beta_annealing_epochs': epochs_vae // 2,
                'mi_layer': bool(mi_layer),
                'list_act_functs_enc': [activation_enc, activation_enc],
                'list_act_functs_dec': [activation_dec, activation_dec],
            },
            'descvae': {
                'epochs': epochs_vae,
                'latent_dim': max(2, n_vars // 4),
                'batch_size': batch_s,
                'beta_start': beta_start,
                'beta_end': beta_end,
                'beta_annealing_epochs': epochs_vae // 2,
                'mi_layer': bool(mi_layer),
                'list_act_functs_enc': [activation_enc, activation_enc],
                'list_act_functs_dec': [activation_dec, activation_dec],
            },
            'regvae': {
                'epochs': epochs_vae,
                'latent_dim': max(2, n_vars // 4),
                'batch_size': batch_s,
                'beta_start': beta_start,
                'beta_end': beta_end,
                'beta_annealing_epochs': epochs_vae // 2,
                'fitness_weight': 0.1,
                'use_fitness_weighting': True,
                'mi_layer': bool(mi_layer),
                'list_act_functs_enc': [activation_enc, activation_enc],
                'list_act_functs_dec': [activation_dec, activation_dec],
            },
            'momvae': {
                'epochs': epochs_vae,
                'latent_dim': max(2, n_vars // 4),
                'batch_size': batch_s,
                'beta_start': beta_start,
                'beta_end': beta_end,
                'beta_annealing_epochs': epochs_vae // 2,
                'moment_weight': 0.1,
                'mi_layer': bool(mi_layer),
                'list_act_functs_enc': [activation_enc, activation_enc],
                'list_act_functs_dec': [activation_dec, activation_dec],
            },
            'bavae': {
                'epochs': epochs_vae,
                'latent_dim': max(2, n_vars // 4),
                'batch_size': batch_s,
                'beta_start': beta_start,
                'beta_end': beta_end,
                'beta_annealing_epochs': epochs_vae // 2,
                'mi_layer': bool(mi_layer),
                'list_act_functs_enc': [activation_enc, activation_enc],
                'list_act_functs_dec': [activation_dec, activation_dec],
            },
            'aavae': {
                'epochs': epochs_vae,
                'latent_dim': max(2, n_vars // 4),
                'batch_size': batch_s,
                'beta_start': beta_start,
                'beta_end': beta_end,
                'beta_annealing_epochs': epochs_vae // 2,
                'mi_layer': bool(mi_layer),
                'list_act_functs_enc': [activation_enc, activation_enc],
                'list_act_functs_dec': [activation_dec, activation_dec],
            },
            'fwvae': {
                'epochs': epochs_vae,
                'latent_dim': max(2, n_vars // 4),
                'batch_size': batch_s,
                'beta_start': beta_start,
                'beta_end': beta_end,
                'beta_annealing_epochs': epochs_vae // 2,
                'fitness_weight': 0.1,
                'mi_layer': bool(mi_layer),
                'list_act_functs_enc': [activation_enc, activation_enc],
                'list_act_functs_dec': [activation_dec, activation_dec],
            },
            'gsvae': {
                'epochs': epochs_vae,
                'latent_dim': max(2, n_vars // 4),
                'batch_size': batch_s,
                'beta_start': beta_start,
                'beta_end': beta_end,
                'beta_annealing_epochs': epochs_vae // 2,
                'mi_layer': bool(mi_layer),
                'list_act_functs_enc': [activation_enc, activation_enc],
                'list_act_functs_dec': [activation_dec, activation_dec],
            },
            'hsvae': {
                'epochs': epochs_vae,
                'latent_dim': max(2, n_vars // 4),
                'batch_size': batch_s,
                'beta_start': beta_start,
                'beta_end': beta_end,
                'beta_annealing_epochs': epochs_vae // 2,
                'mi_layer': bool(mi_layer),
                'list_act_functs_enc': [activation_enc, activation_enc],
                'list_act_functs_dec': [activation_dec, activation_dec],
            },
            'tcvae': {
                'epochs': epochs_vae,
                'latent_dim': max(2, n_vars // 4),
                'batch_size': batch_s,
                'beta_start': beta_start,
                'beta_end': beta_end,
                'beta_annealing_epochs': epochs_vae // 2,
                'mi_layer': bool(mi_layer),
                'list_act_functs_enc': [activation_enc, activation_enc],
                'list_act_functs_dec': [activation_dec, activation_dec],
            },
            'gan': {
                'epochs': 60,
                'latent_dim': max(10, n_vars // 2),
                'batch_size':  batch_s,
            },
            'backdrive': {
                'epochs': 30,
                'hidden_dims': adaptive_hidden_dims,
                'batch_size':  batch_s,
            },
            'backdrive_random': {
                'epochs': 30,
                'hidden_dims': adaptive_hidden_dims,
                'batch_size':  batch_s,
            },
            'backdrive_perturb_best': {
                'epochs': 30,
                'hidden_dims': adaptive_hidden_dims,
                'batch_size':  batch_s,
            },
            'backdrive_perturb_selected': {
                'epochs': 30,
                'hidden_dims': adaptive_hidden_dims,
                'batch_size':  batch_s,
            },
            'backdrive_adaptive': {
                'epochs': 30,
                'hidden_dims': adaptive_hidden_dims,
                'batch_size':  batch_s,
            },
            'backdrive_weighted_mse': {
                'epochs': 30,
                'hidden_dims': adaptive_hidden_dims,
                'batch_size':  batch_s,
            },
            'backdrive_ranking': {
                'epochs': 30,
                'hidden_dims': adaptive_hidden_dims,
                'batch_size':  batch_s,
            },
            'backdrive_huber': {
                'epochs': 30,
                'hidden_dims': adaptive_hidden_dims,
                'batch_size':  batch_s,
            },
            'dae': {
                'epochs': 30,
                'hidden_dim': max(n_vars // 2, 10),
                'corruption_level': 0.1,
            },
            'rbm': {
                'epochs': 15,
                'n_hidden': n_vars,
                'k_cd': 1,
            },
            'dbd': {
                'epochs': 50,
                'hidden_dims': adaptive_hidden_dims,
                # CRITICAL FIX: Increased training data
                # Old: 5 alpha samples = 375 total, New: 20 = 1500 total
                'num_alpha_samples': 20,
            },
            'dbd_cs': {
                'epochs': 50,
                'hidden_dims': adaptive_hidden_dims,
                'num_alpha_samples': 20,
            },
            'dbd_cd': {
                'epochs': 50,
                'hidden_dims': adaptive_hidden_dims,
                'num_alpha_samples': 20,
            },
            'dbd_uc': {
                'epochs': 50,
                'hidden_dims': adaptive_hidden_dims,
                'num_alpha_samples': 20,
                'to_take': pop_size * 4,  # Increased training pairs
            },
            'dbd_us': {
                'epochs': 50,
                'hidden_dims': adaptive_hidden_dims,
                'num_alpha_samples': 20,
                'to_take': pop_size * 4,  # Increased training pairs
            },
            'dbd_cs_t_k0': {
                'epochs': 50,
                'k': 0,  # No conditioning (marginal probabilities only)
                'alpha': 0.1,  # Smoothing parameter
                'num_alpha_samples': 10,  # For continuous DbD
                'hidden_dims': adaptive_hidden_dims,
                'normalize': True,  # Normalize continuous values
            },
            'dbd_cs_t_k1': {
                'epochs': 50,
                'k': 1,  # First-order Markov chain
                'alpha': 0.1,
                'num_alpha_samples': 10,
                'hidden_dims': adaptive_hidden_dims,
                'normalize': True,
            },
            'dbd_cs_t_k2': {
                'epochs': 50,
                'k': 2,  # Second-order Markov chain
                'alpha': 0.1,
                'num_alpha_samples': 10,
                'hidden_dims': adaptive_hidden_dims,
                'normalize': True,
            },
            'dendiff_gumbel': {
                'epochs': 50,
                'n_timesteps': 100,  # Number of diffusion steps
                'beta_schedule': 'linear',
                'beta_start': 0.0001,
                'beta_end': 0.3,  # Max bit-flip probability
                'hidden_dims': adaptive_hidden_dims,
                'time_emb_dim': 4,
                'batch_size':  batch_s,
                'learning_rate': 1e-3,
                'temperature': 1.0,
                'temperature_decay': 0.99,
                'min_temperature': 0.5,
            },
            'dendiff_corruption': {
                'epochs': 50,
                'n_timesteps': 50,  # Fewer steps for corruption approach
                'schedule': 'linear',
                'corruption_start': 0.01,
                'corruption_end': 0.5,
                'hidden_dims': adaptive_hidden_dims,
                'time_emb_dim': 4,
                'batch_size':  batch_s,
                'learning_rate': 1e-3,
            },
        }
        
        learning_params = params_map[method_id]
        
        # Configure sampling parameters specific to backdrive variants
        sampling_params = {}
        if method_id == 'backdrive_random':
            sampling_params = {
                'init_method': 'random',
                'n_iterations': 100,
                # Use new improved defaults (learning_rate, gradient_clip, temperature, etc.)
                # are now set in sampling function
            }
        elif method_id == 'backdrive_perturb_best':
            sampling_params = {
                'init_method': 'random', #'perturb_best',
                'init_noise': 0.1,
                'n_iterations': 100,
            }
        elif method_id == 'backdrive_perturb_selected':
            sampling_params = {
                'init_method': 'random', #'perturb_selected',
                'init_noise': 0.1,
                'n_iterations': 100,
            }
        elif method_id == 'backdrive_adaptive':
            sampling_params = {
                'init_method': 'random', #'perturb_best',
                'init_noise': 0.1,
                'n_iterations': 100,
                'target_levels': [100, 90, 80],
                'level_fractions': [0.5, 0.3, 0.2],
            }
        elif method_id in ['backdrive_weighted_mse', 'backdrive_ranking', 'backdrive_huber']:
            # New loss function variants use same sampling as standard backdrive
            sampling_params = {
                'init_method': 'random', #'perturb_best',
                'init_noise': 0.1,
                'n_iterations': 100,
            }
        
        cardinality = np.full(n_vars, 2)
        
        eda = UnifiedDiscreteNeuralEDA(
            method=method_id,
            n_vars=n_vars,
            cardinality=cardinality,
            pop_size=pop_size,
            selection_ratio=selection_ratio,
            max_generations=n_gen,
            learning_params=learning_params,
            sampling_params=sampling_params,
            random_seed=myseed,
            alpha=alpha,
        )
        
        best_fitness, best_solution, history = eda.run(fitness_func, verbose=True)
        
    elif alg in traditional_edas:
        # Run traditional EDA
        best_fitness, best_solution, history = run_traditional_eda(
            alg=alg,
            fitness_func=fitness_func,
            n_vars=n_vars,
            pop_size=pop_size,
            max_generations=n_gen,
            random_seed=myseed,
            verbose=True,
            alpha=alpha,
            truncation_ratio=truncation_ratio,
        )
        
    else:
        print(f"Error: Unknown algorithm '{alg}'")
        print(f"Supported algorithms: {', '.join(neural_edas + traditional_edas)}")
        sys.exit(1)
    
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
