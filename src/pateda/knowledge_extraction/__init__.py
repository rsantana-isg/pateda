"""
Knowledge extraction module for PATEDA.

This module provides comprehensive tools for extracting knowledge from
Estimation of Distribution Algorithms (EDAs) during and after optimization.
It includes fitness-related measures, dependency analysis, and advanced
visualization techniques for understanding learned structures.

Modules
-------
fitness_measures
    Response to selection, amount of selection, realized heritability.
dependency_analysis
    A posteriori dependency analysis including correlation networks and
    probabilistic graphical model learning.
model_visualizations
    Advanced visualizations including dendrograms and glyph representations
    of learned structures.

Author: Roberto Santana (roberto.santana@ehu.es)
Ported to Python: 2025
"""

from pateda.knowledge_extraction.fitness_measures import (
    response_to_selection,
    amount_of_selection,
    realized_heritability,
    compute_objective_distribution
)

from pateda.knowledge_extraction.dependency_analysis import (
    compute_correlation_matrix,
    learn_bayesian_network,
    learn_gaussian_network,
    analyze_variable_dependencies
)

from pateda.knowledge_extraction.model_visualizations import (
    view_dendrogram_structure,
    view_glyph_structure
)

from pateda.knowledge_extraction.eda_strategies import (
    extract_bayesian_network_evolution,
    extract_gaussian_parameters_evolution,
    extract_probability_distribution_evolution,
    generate_comprehensive_report,
    compare_eda_runs
)

from pateda.knowledge_extraction.network_measures import (
    model_to_adjacency,
    cliques_to_adjacency,
    tree_to_adjacency,
    compute_network_measures,
    compute_measures_evolution,
    edge_frequency_matrix,
    aggregate_degree_distribution,
    triad_census,
    triad_census_series,
    motif_number,
    motif_spectrum,
    max_modularity,
    participation_coefficient,
    dagdif,
    SCALAR_MEASURE_KEYS,
)

from pateda.knowledge_extraction.network_visualizations import (
    plot_measures_evolution,
    compare_measure_evolution,
    compare_measures_grid,
    plot_edge_frequency_matrix,
    plot_degree_distribution,
    plot_motif_evolution,
    plot_network_snapshots,
    plot_betweenness_two_approaches,
)

from pateda.knowledge_extraction.gaussian_networks import (
    extract_gaussian_parameters,
    covariance_to_precision,
    partial_correlation_matrix,
    glasso_precision,
    gaussian_interaction_network,
    orient_edges_likelihood_score,
    compare_networks,
    combine_networks,
    gaussian_network_evolution,
)

from pateda.knowledge_extraction.vine_analysis import (
    get_vine_model,
    vine_structure,
    first_tree_network,
    family_composition,
    tau_by_tree,
    effective_truncation,
    tau_matrix,
    analyze_vine,
    vine_evolution,
)

from pateda.knowledge_extraction.continuous_visualizations import (
    plot_gaussian_parameter_evolution,
    plot_precision_heatmap,
    plot_partial_correlation_network,
    plot_network_comparison,
    plot_vine_first_tree,
    plot_family_composition,
    plot_tau_by_tree,
    plot_vine_evolution,
)

__all__ = [
    # Fitness measures
    'response_to_selection',
    'amount_of_selection',
    'realized_heritability',
    'compute_objective_distribution',

    # Dependency analysis
    'compute_correlation_matrix',
    'learn_bayesian_network',
    'learn_gaussian_network',
    'analyze_variable_dependencies',

    # Model visualizations
    'view_dendrogram_structure',
    'view_glyph_structure',

    # EDA-specific strategies
    'extract_bayesian_network_evolution',
    'extract_gaussian_parameters_evolution',
    'extract_probability_distribution_evolution',
    'generate_comprehensive_report',
    'compare_eda_runs',

    # Network measures (structural analysis of learned PGMs)
    'model_to_adjacency',
    'cliques_to_adjacency',
    'tree_to_adjacency',
    'compute_network_measures',
    'compute_measures_evolution',
    'edge_frequency_matrix',
    'aggregate_degree_distribution',
    'triad_census',
    'triad_census_series',
    'motif_number',
    'motif_spectrum',
    'max_modularity',
    'participation_coefficient',
    'dagdif',
    'SCALAR_MEASURE_KEYS',

    # Network visualizations
    'plot_measures_evolution',
    'compare_measure_evolution',
    'compare_measures_grid',
    'plot_edge_frequency_matrix',
    'plot_degree_distribution',
    'plot_motif_evolution',
    'plot_network_snapshots',
    'plot_betweenness_two_approaches',

    # Gaussian interaction networks (continuous EDAs)
    'extract_gaussian_parameters',
    'covariance_to_precision',
    'partial_correlation_matrix',
    'glasso_precision',
    'gaussian_interaction_network',
    'orient_edges_likelihood_score',
    'compare_networks',
    'combine_networks',
    'gaussian_network_evolution',

    # Vine copula analysis (continuous EDAs)
    'get_vine_model',
    'vine_structure',
    'first_tree_network',
    'family_composition',
    'tau_by_tree',
    'effective_truncation',
    'tau_matrix',
    'analyze_vine',
    'vine_evolution',

    # Continuous-EDA visualizations
    'plot_gaussian_parameter_evolution',
    'plot_precision_heatmap',
    'plot_partial_correlation_network',
    'plot_network_comparison',
    'plot_vine_first_tree',
    'plot_family_composition',
    'plot_tau_by_tree',
    'plot_vine_evolution',
]
