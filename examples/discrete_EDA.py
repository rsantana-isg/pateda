"""
Discrete EDA - Unified Command-Line Interface for EDA Variants
===============================================================

This program provides a unified interface to run various discrete EDA algorithms
on benchmark problems with different seeds for cluster execution.

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
- TreeEDA: Tree-based Fctorized Distribution Algorithm
- EBNA: Estimation of Bayesian Network Algorithm
- MOA: Markovianity Based Optimization Algorithm
- MN-FDA: Markov Network Factorized Distribution Algorithm
- MN-FDAG: MN-FDA with G-test
- MK-EDA: k-order Markov Chain EDA (k=1,2,3)
- MT-EDA: Mixture of Trees EDA (k=2,3)

Usage:
    python discrete_EDA.py <seed> <obj_func> <n> <pop_size> <n_gen> <alg>

Example:
    python discrete_EDA.py 0 OneMax 20 80 20 VAE
    python discrete_EDA.py 1 Deceptive3 30 100 30 UMDA
    python discrete_EDA.py 0 HIFF 64 200 50 MN-FDA

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
from pateda.learning.discrete_vae import learn_binary_vae
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
    sample_binary_vae, sample_binary_gan, sample_binary_backdrive,
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

# Mutation modules
from pateda.mutation import frequency_balance_mutation, FrequencyBalanceMutation

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

            # Store the best solution from selected population before mutation (for elitism with mutation)
            if self.alpha > 0:
                best_idx_in_selected = np.argmax(selected_fitness)
                best_solution_pre_mutation = selected_pop[best_idx_in_selected].copy()

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
    alpha: float,
    random_seed: int = None,
    verbose: bool = True,
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
        selection=TruncationSelection(ratio=0.5),
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
    
    # Check arguments (sys.argv[0] is script name, 6 required args + optional alpha)
    if len(sys.argv) < 7 or len(sys.argv) > 8:
        print("Usage: python discrete_EDA.py <seed> <obj_func> <n> <pop_size> <n_gen> <alg> [alpha]")
        print()
        print("Arguments:")
        print("  seed      : Random seed (integer)")
        print("  obj_func  : Objective function name")
        print("  n         : Number of variables (integer)")
        print("  pop_size  : Population size (integer)")
        print("  n_gen     : Number of generations (integer)")
        print("  alg       : Algorithm name")
        print("  alpha     : (Optional) Max frequency threshold for mutation (default: 0.0, no mutation)")
        print()
        print("Supported objective functions:")
        print("  OneMax, KDeceptive3, KDeceptive5, Deceptive3, Deceptive3Overlap")
        print("  DecepMarta3, DecepMarta3New, Decep3MH, TwoPeaksDecep3, DecepVenturini")
        print("  HardDecep5, HIFF, FHTrap1")
        print("  Polytree3, Polytree3Overlap, Polytree5")
        print("  FC2, FC3, FC4, FC5")
        print()
        print("Supported algorithms:")
        print("  Neural EDAs: VAE, VAE-Extended, GAN, Backdrive, DAE, RBM, DbD")
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
        print("  python discrete_EDA.py 0 OneMax 20 80 20 VAE")
        print("  python discrete_EDA.py 0 HIFF 64 200 50 MN-FDA")
        sys.exit(1)
    
    # Parse arguments
    myseed = int(sys.argv[1])
    obj_func = sys.argv[2]
    n = int(sys.argv[3])
    pop_size = int(sys.argv[4])
    n_gen = int(sys.argv[5])
    alg = sys.argv[6]
    
    # Optional arguments
    alpha = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0
    
    # Suppress warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    # Parse problem
    try:
        fitness_func, n_vars, optimal_fitness = parse_problem(obj_func, n)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Print configuration
    print("=" * 80)
    print("DISCRETE EDA - Single Run Configuration")
    print("=" * 80)
    print(f"Seed:             {myseed}")
    print(f"Problem:          {obj_func}")
    print(f"Variables:        {n_vars}")
    print(f"Optimal Fitness:  {optimal_fitness}")
    print(f"Population Size:  {pop_size}")
    print(f"Generations:      {n_gen}")
    print(f"Algorithm:        {alg}")
    print(f"Alpha (mutation): {alpha}")
    print("=" * 80)
    print()
    
    # Determine if neural or traditional EDA
    neural_edas = ['VAE', 'VAE-Extended', 'GAN', 'Backdrive', 'Backdrive-Random', 'Backdrive-PerturbBest',
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
            'VAE-Extended': 'vae_extended',
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
        selection_ratio = 0.5
        selected_pop_size = int(pop_size * selection_ratio)
        adaptive_hidden_dims = [max(10, n_vars // 2), max(10, n_vars // 4)]
        batch_s =  min(32, int(selected_pop_size/10))
        
        # Configure learning parameters
        params_map = {
            'vae': {
                'epochs': 30,
                'latent_dim': max(2, n_vars // 4),
                'batch_size': batch_s,
                # Dynamic hidden layers will be computed in learn_binary_vae
                # Beta annealing enabled by default
                'beta_start': 0.0,
                'beta_end': 1.0,
                'beta_annealing_epochs': 15,  # Half of epochs
            },
            'vae_extended': {
                'epochs': 30,
                'latent_dim': max(2, n_vars // 4),
                'batch_size':  batch_s,
                # Dynamic hidden layers will be computed in learn_binary_vae
                # Beta annealing enabled by default
                'beta_start': 0.0,
                'beta_end': 1.0,
                'beta_annealing_epochs': 15,  # Half of epochs
                # Extended VAE specific parameters
                'use_extended': True,
                'fitness_weight': 0.1,
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
            selection_ratio=0.5,
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
            alpha=alpha,
            random_seed=myseed,
            verbose=True,
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
    print(f"Optimal Fitness:  {optimal_fitness:.4f}")
    print(f"Gap:              {abs(best_fitness - optimal_fitness):.4f}")
    print(f"Success:          {'Yes' if abs(best_fitness - optimal_fitness) < SUCCESS_THRESHOLD * optimal_fitness else 'No'}")
    print(f"Elapsed Time:     {elapsed_time:.2f} seconds")
    print(f"Best Solution:    {best_solution[:20]}{'...' if len(best_solution) > 20 else ''}")
    print("=" * 80)


if __name__ == "__main__":
    main()
