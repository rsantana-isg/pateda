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
    'compare_eda_runs'
]
