"""
Statistics module for PATEDA.

This module provides comprehensive statistical analysis tools for tracking,
analyzing, and comparing EDA performance. It includes population statistics,
convergence analysis, and multi-run comparisons.

Main Components:
- Population statistics: Compute fitness and diversity metrics
- Statistics tracker: Class-based interface for tracking evolution
- Analysis utilities: Statistical tests, success rates, convergence detection

Author: Roberto Santana (roberto.santana@ehu.es)
Ported to Python: 2025
"""

# Population statistics functions
from pateda.statistics.population_stats import (
    simple_population_statistics,
    compute_fitness_statistics,
    compute_population_diversity,
    compute_convergence_metrics,
)

# Statistics tracker
from pateda.statistics.tracker import StatisticsTracker

# Analysis utilities
from pateda.statistics.analysis import (
    analyze_multiple_runs,
    compute_statistical_tests,
    compute_success_rate,
    compute_auc,
    detect_convergence,
)

__all__ = [
    # Population statistics
    "simple_population_statistics",
    "compute_fitness_statistics",
    "compute_population_diversity",
    "compute_convergence_metrics",
    # Tracker
    "StatisticsTracker",
    # Analysis
    "analyze_multiple_runs",
    "compute_statistical_tests",
    "compute_success_rate",
    "compute_auc",
    "detect_convergence",
]
