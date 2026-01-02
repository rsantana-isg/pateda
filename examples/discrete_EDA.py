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
- TreeEDA: Tree-based Factorized Distribution Algorithm
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
from pateda.learning.dae import learn_dae
from pateda.learning.rbm import learn_softmax_rbm
from pateda.learning.discrete_dbd import learn_binary_dbd

# Neural sampling modules
from pateda.sampling.discrete_neural import (
    sample_binary_vae, sample_binary_gan, sample_binary_backdrive
)
from pateda.sampling.dae import sample_dae
from pateda.sampling.rbm import sample_softmax_rbm
from pateda.sampling.discrete_dbd import sample_binary_dbd

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
            k = int(obj_func[10:])  # Extract number after 'KDeceptive'
        except (ValueError, IndexError):
            raise ValueError(f"Invalid KDeceptive format: {obj_func}. Expected format: KDeceptive<k> (e.g., KDeceptive3)")
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
        # Check if n is a power of 2
        if n_vars & (n_vars - 1) != 0 or n_vars == 0:
            raise ValueError(f"For HIFF, n must be a power of 2 (e.g., 16, 32, 64, 128)")
        # HIFF optimal is complex to calculate, approximate
        return wrap_function(hiff), n_vars, float(n_vars * np.log2(n_vars))
    
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
    ):
        """
        Initialize Unified Neural EDA

        Parameters
        ----------
        method : str
            Method: 'vae', 'gan', 'backdrive', 'dae', 'rbm', 'dbd'
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

        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)

        # Map methods to functions
        self.method_map = {
            'vae': (learn_binary_vae, sample_binary_vae, False),
            'gan': (learn_binary_gan, sample_binary_gan, False),
            'backdrive': (learn_binary_backdrive, sample_binary_backdrive, False),
            'dae': (learn_dae, sample_dae, False),
            'rbm': (learn_softmax_rbm, sample_softmax_rbm, True),  # Needs cardinality
            'dbd': (learn_binary_dbd, sample_binary_dbd, True),  # Needs two populations
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
        learn_fn, sample_fn, special = self.method_map[self.method]

        # Initialize population
        population = np.random.randint(0, self.cardinality, (self.pop_size, self.n_vars))

        # Evaluate
        fitness = fitness_func(population)

        best_fitness = np.max(fitness)
        best_solution = population[np.argmax(fitness)].copy()

        history = {'best_fitness': [best_fitness]}

        if verbose:
            print(f"Generation 0: Best Fitness = {best_fitness:.4f}")

        # Keep track of previous population for DbD
        prev_population = None

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
                elif self.method == 'dbd':
                    # DbD needs two populations (source and target)
                    if prev_population is None:
                        # First generation: use random as source
                        p0 = np.random.randint(0, self.cardinality,
                                             (len(selected_pop), self.n_vars))
                    else:
                        # Use previous selected population as source
                        p0 = prev_population

                    p1 = selected_pop
                    model = learn_fn(p0, p1, self.learning_params)

                    # Save for next iteration
                    prev_population = selected_pop.copy()
                else:
                    model = learn_fn(selected_pop, selected_fitness, self.learning_params)

                # Sample new population
                population = sample_fn(model, self.pop_size, self.sampling_params)

            except Exception as e:
                if verbose:
                    print(f"  Warning: Sampling failed ({e}), using random population")
                population = np.random.randint(0, self.cardinality,
                                             (self.pop_size, self.n_vars))

            # Evaluate
            fitness = fitness_func(population)

            # Update best
            gen_best = np.max(fitness)
            if gen_best > best_fitness:
                best_fitness = gen_best
                best_solution = population[np.argmax(fitness)].copy()

            history['best_fitness'].append(best_fitness)

            if verbose and (gen + 1) % 10 == 0:
                print(f"Generation {gen+1}: Best Fitness = {best_fitness:.4f}")

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
    
    # Check arguments
    if len(sys.argv) != 7:
        print("Usage: python discrete_EDA.py <seed> <obj_func> <n> <pop_size> <n_gen> <alg>")
        print()
        print("Arguments:")
        print("  seed      : Random seed (integer)")
        print("  obj_func  : Objective function name")
        print("  n         : Number of variables (integer)")
        print("  pop_size  : Population size (integer)")
        print("  n_gen     : Number of generations (integer)")
        print("  alg       : Algorithm name")
        print()
        print("Supported objective functions:")
        print("  OneMax, KDeceptive3, KDeceptive5, Deceptive3, Deceptive3Overlap")
        print("  DecepMarta3, DecepMarta3New, Decep3MH, TwoPeaksDecep3, DecepVenturini")
        print("  HardDecep5, HIFF, FHTrap1")
        print("  Polytree3, Polytree3Overlap, Polytree5")
        print("  FC2, FC3, FC4, FC5")
        print()
        print("Supported algorithms:")
        print("  Neural EDAs: VAE, GAN, Backdrive, DAE, RBM, DbD")
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
    print("=" * 80)
    print()
    
    # Determine if neural or traditional EDA
    neural_edas = ['VAE', 'GAN', 'Backdrive', 'DAE', 'RBM', 'DbD']
    traditional_edas = ['UMDA', 'TreeEDA', 'EBNA', 'MOA', 'MN-FDA', 'MN-FDAG',
                       'MK-EDA1', 'MK-EDA2', 'MK-EDA3', 'MT-EDA2', 'MT-EDA3']
    
    start_time = time.time()
    
    if alg in neural_edas:
        # Run neural EDA
        method_map = {
            'VAE': 'vae',
            'GAN': 'gan',
            'Backdrive': 'backdrive',
            'DAE': 'dae',
            'RBM': 'rbm',
            'DbD': 'dbd',
        }
        
        method_id = method_map[alg]
        
        # Configure learning parameters
        params_map = {
            'vae': {
                'epochs': 30,
                'latent_dim': max(2, n_vars // 4),
                'batch_size': min(32, pop_size // 2),
            },
            'gan': {
                'epochs': 60,
                'latent_dim': max(10, n_vars // 2),
                'batch_size': min(32, pop_size // 2),
            },
            'backdrive': {
                'epochs': 30,
                'hidden_layers': [64, 32],
                'batch_size': min(32, pop_size // 2),
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
                'hidden_dims': [64, 32],
                'num_alpha_samples': 5,
            },
        }
        
        learning_params = params_map[method_id]
        cardinality = np.full(n_vars, 2)
        
        eda = UnifiedDiscreteNeuralEDA(
            method=method_id,
            n_vars=n_vars,
            cardinality=cardinality,
            pop_size=pop_size,
            selection_ratio=0.5,
            max_generations=n_gen,
            learning_params=learning_params,
            sampling_params={},
            random_seed=myseed,
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
