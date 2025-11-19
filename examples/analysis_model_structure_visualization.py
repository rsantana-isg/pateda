"""
Model Structure Visualization and Analysis for EDAs

This script demonstrates how to extract, visualize, and analyze the probabilistic
models learned by EDAs during evolution. This is inspired by the MATLAB
AnalysisScripts but adapted for Python.

Features demonstrated:
1. Extract learned model structures across generations
2. Visualize Bayesian Network structures
3. Track edge frequency and stability
4. Analyze fitness evolution and convergence
5. Examine probability distributions

This corresponds to functionality in:
- ScriptsMateda/AnalysisScripts/BN_StructureVisualization.m
- ScriptsMateda/AnalysisScripts/FitnessMeasuresComp.m

Requirements:
- matplotlib for plotting
- networkx for graph visualization (optional)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from collections import defaultdict

from pateda.core.eda import EDA, EDAComponents
from pateda.stop_conditions import MaxGenerations
from pateda.seeding import RandomInit
from pateda.selection import TruncationSelection
from pateda.replacement import GenerationalReplacement
from pateda.learning import LearnEBNA
from pateda.sampling.bayesian_network import SampleBayesianNetwork
from pateda.functions.discrete.trap import trap_n


# Try to import networkx for graph visualization
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("Warning: networkx not available. Graph visualization will be limited.")


def extract_bn_structures(cache: Dict) -> List[np.ndarray]:
    """
    Extract Bayesian network structures from cache

    Args:
        cache: EDA cache containing models

    Returns:
        List of adjacency matrices, one per generation
    """
    structures = []

    if 'models' in cache:
        for gen, model in enumerate(cache['models']):
            if hasattr(model, 'structure'):
                # Convert to adjacency matrix
                adj_matrix = model_to_adjacency_matrix(model)
                structures.append(adj_matrix)

    return structures


def model_to_adjacency_matrix(model) -> np.ndarray:
    """
    Convert a Bayesian network model to adjacency matrix

    Args:
        model: Bayesian network model

    Returns:
        Adjacency matrix (n_vars x n_vars) where adj[i,j] = 1 if edge i->j exists
    """
    if hasattr(model, 'structure'):
        structure = model.structure
        n_vars = len(structure)
        adj_matrix = np.zeros((n_vars, n_vars))

        for child, parents in enumerate(structure):
            if parents is not None and len(parents) > 0:
                for parent in parents:
                    adj_matrix[parent, child] = 1

        return adj_matrix

    return None


def analyze_edge_frequency(structures: List[np.ndarray]) -> np.ndarray:
    """
    Compute edge frequency across all generations

    Args:
        structures: List of adjacency matrices

    Returns:
        Frequency matrix where freq[i,j] = fraction of generations with edge i->j
    """
    if not structures:
        return None

    n_vars = structures[0].shape[0]
    frequency_matrix = np.zeros((n_vars, n_vars))

    for adj_matrix in structures:
        frequency_matrix += adj_matrix

    frequency_matrix /= len(structures)

    return frequency_matrix


def visualize_bn_structure(adj_matrix: np.ndarray, title: str = "Bayesian Network Structure",
                           threshold: float = 0.0, ax=None):
    """
    Visualize Bayesian network structure

    Args:
        adj_matrix: Adjacency matrix
        title: Plot title
        threshold: Only show edges with frequency above this threshold
        ax: Matplotlib axis (optional)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    n_vars = adj_matrix.shape[0]

    if NETWORKX_AVAILABLE:
        # Use networkx for nice graph layout
        G = nx.DiGraph()

        # Add nodes
        for i in range(n_vars):
            G.add_node(i)

        # Add edges above threshold
        for i in range(n_vars):
            for j in range(n_vars):
                if adj_matrix[i, j] > threshold:
                    G.add_edge(i, j, weight=adj_matrix[i, j])

        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)

        # Draw
        nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                              node_size=500, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)

        # Draw edges with varying thickness based on weight
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=[w * 3 for w in weights],
                              alpha=0.6, arrows=True, arrowsize=20,
                              connectionstyle='arc3,rad=0.1', ax=ax)

    else:
        # Fallback: show adjacency matrix as heatmap
        im = ax.imshow(adj_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        ax.set_xlabel('Child Variable')
        ax.set_ylabel('Parent Variable')
        ax.set_xticks(range(n_vars))
        ax.set_yticks(range(n_vars))
        plt.colorbar(im, ax=ax, label='Edge Frequency')

    ax.set_title(title)
    ax.axis('off' if NETWORKX_AVAILABLE else 'on')


def plot_fitness_evolution(cache: Dict):
    """
    Plot fitness evolution over generations

    Shows:
    - Best fitness per generation
    - Average fitness per generation
    - Fitness diversity (std)
    """
    if 'statistics' not in cache:
        print("Warning: No statistics in cache")
        return

    stats = cache['statistics']
    generations = range(len(stats['best_fitness']))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Fitness evolution
    ax1.plot(generations, stats['best_fitness'], 'b-', linewidth=2, label='Best')
    ax1.plot(generations, stats['mean_fitness'], 'g--', linewidth=2, label='Mean')
    ax1.fill_between(generations,
                     np.array(stats['mean_fitness']) - np.array(stats['std_fitness']),
                     np.array(stats['mean_fitness']) + np.array(stats['std_fitness']),
                     alpha=0.3, color='green', label='±1 Std')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness')
    ax1.set_title('Fitness Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Fitness diversity
    ax2.plot(generations, stats['std_fitness'], 'r-', linewidth=2)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Fitness Std Dev')
    ax2.set_title('Population Diversity (Fitness Standard Deviation)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()


def plot_structure_evolution(structures: List[np.ndarray], generations_to_show: List[int] = None):
    """
    Plot evolution of BN structure across generations

    Args:
        structures: List of adjacency matrices
        generations_to_show: List of generation indices to visualize
    """
    if not structures:
        print("Warning: No structures to visualize")
        return

    if generations_to_show is None:
        # Show first, middle, and last
        n_gens = len(structures)
        generations_to_show = [0, n_gens // 2, n_gens - 1]

    n_plots = len(generations_to_show)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))

    if n_plots == 1:
        axes = [axes]

    for idx, gen in enumerate(generations_to_show):
        if gen < len(structures):
            visualize_bn_structure(
                structures[gen],
                title=f"Generation {gen}",
                ax=axes[idx]
            )

    plt.tight_layout()


def plot_edge_frequency_heatmap(frequency_matrix: np.ndarray):
    """
    Plot heatmap of edge frequencies across all generations
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(frequency_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax.set_xlabel('Child Variable', fontsize=12)
    ax.set_ylabel('Parent Variable', fontsize=12)
    ax.set_title('Edge Frequency Across All Generations', fontsize=14)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Frequency', rotation=270, labelpad=20, fontsize=12)

    # Add grid
    n_vars = frequency_matrix.shape[0]
    ax.set_xticks(np.arange(n_vars))
    ax.set_yticks(np.arange(n_vars))
    ax.grid(which='both', color='white', linestyle='-', linewidth=0.5)

    plt.tight_layout()


def run_analysis_example():
    """
    Run a complete example: train EDA and analyze results
    """
    print("=" * 80)
    print("Model Structure Visualization and Analysis")
    print("=" * 80)
    print()

    # Problem configuration
    n_vars = 15  # 3 traps of size 5
    pop_size = 200
    max_generations = 50

    print("Configuration:")
    print(f"  - Problem: Trap-5 function (3 blocks)")
    print(f"  - Variables: {n_vars}")
    print(f"  - Population size: {pop_size}")
    print(f"  - Generations: {max_generations}")
    print(f"  - Algorithm: EBNA (Bayesian Network)")
    print()

    # Create fitness function
    def fitness_func(x):
        return trap_n(x, n_trap=5)

    # Configure EDA with caching enabled
    components = EDAComponents(
        seeding=RandomInit(),
        selection=TruncationSelection(ratio=0.5),
        learning=LearnEBNA(
            max_parents=4,
            score_metric='bic',
        ),
        sampling=SampleBayesianNetwork(n_samples=pop_size),
        replacement=GenerationalReplacement(),
        stop_condition=MaxGenerations(max_generations),
    )

    eda = EDA(
        pop_size=pop_size,
        n_vars=n_vars,
        fitness_func=fitness_func,
        cardinality=np.full(n_vars, 2),
        components=components,
    )

    print("Running EBNA...")
    stats, cache = eda.run(verbose=True)

    print()
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print()

    # Extract structures
    structures = extract_bn_structures(cache)
    print(f"Extracted {len(structures)} model structures")

    # Analyze edge frequency
    if structures:
        frequency_matrix = analyze_edge_frequency(structures)

        print("\nMost frequent edges (>50% of generations):")
        n_vars = frequency_matrix.shape[0]
        for i in range(n_vars):
            for j in range(n_vars):
                if frequency_matrix[i, j] > 0.5:
                    print(f"  {i} -> {j}: {frequency_matrix[i, j]:.2%}")

        # Visualizations
        print("\nGenerating visualizations...")

        # Plot fitness evolution
        plot_fitness_evolution(cache)

        # Plot structure evolution
        plot_structure_evolution(structures, generations_to_show=[0, 25, 49])

        # Plot edge frequency heatmap
        plot_edge_frequency_heatmap(frequency_matrix)

        # Plot final structure with threshold
        fig, ax = plt.subplots(figsize=(10, 8))
        visualize_bn_structure(
            frequency_matrix,
            title="Aggregated Structure (All Generations)",
            threshold=0.3,  # Only show edges present in >30% of generations
            ax=ax
        )

        plt.show()

        print("\nVisualization complete. Plots displayed.")

    else:
        print("Warning: No structures found in cache for visualization")

    print()
    print("Analysis complete!")

    return stats, cache, structures


if __name__ == "__main__":
    stats, cache, structures = run_analysis_example()
