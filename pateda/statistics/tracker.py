"""
Statistics tracker for monitoring EDA evolution.

This module provides a class-based interface for tracking and analyzing
statistics throughout the EDA optimization process.

Author: Roberto Santana (roberto.santana@ehu.es)
Ported to Python: 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable
import json
from pathlib import Path

from pateda.statistics.population_stats import (
    simple_population_statistics,
    compute_convergence_metrics,
    compute_population_diversity
)


class StatisticsTracker:
    """
    Track and analyze statistics throughout EDA evolution.

    This class provides a convenient interface for collecting, storing,
    and analyzing population statistics during optimization.

    Parameters
    ----------
    find_best_method : callable, optional
        Function to find best individual(s) from population.
        If None, uses default minimization on first objective.
    track_time : bool, default=True
        Whether to track time operations.
    track_diversity : bool, default=True
        Whether to track population diversity metrics.

    Attributes
    ----------
    statistics : dict
        Dictionary storing statistics for each generation.
    time_operations : list
        List of time operation arrays for each generation.
    evaluations : list
        List of evaluation counts for each generation.

    Examples
    --------
    >>> tracker = StatisticsTracker()
    >>> for gen in range(max_generations):
    ...     # ... EDA operations ...
    ...     tracker.update(gen, population, fitness, time_ops, n_evals)
    >>> tracker.print_summary()
    >>> df = tracker.to_dataframe()
    >>> tracker.save('results.json')
    """

    def __init__(
        self,
        find_best_method: Optional[Callable] = None,
        track_time: bool = True,
        track_diversity: bool = True
    ):
        self.statistics: Dict[int, Dict[str, Any]] = {}
        self.time_operations: List[np.ndarray] = []
        self.evaluations: List[int] = []
        self.find_best_method = find_best_method
        self.track_time = track_time
        self.track_diversity = track_diversity

    def update(
        self,
        generation: int,
        population: np.ndarray,
        fitness_values: np.ndarray,
        time_operations: Optional[np.ndarray] = None,
        n_evaluations: Optional[int] = None
    ) -> None:
        """
        Update statistics for the current generation.

        Parameters
        ----------
        generation : int
            Current generation number.
        population : np.ndarray
            Current population.
        fitness_values : np.ndarray
            Fitness values for population.
        time_operations : np.ndarray, optional
            Time spent in various operations.
        n_evaluations : int, optional
            Number of evaluations in this generation.
        """
        # Prepare time operations array
        if time_operations is None and self.track_time:
            time_operations = np.zeros(8)  # Default 8 operation types

        # Store time and evaluations
        if self.track_time and time_operations is not None:
            self.time_operations.append(time_operations)
        else:
            self.time_operations.append(np.zeros(8))

        if n_evaluations is not None:
            self.evaluations.append(n_evaluations)
        else:
            self.evaluations.append(len(population))

        # Convert lists to arrays for statistics function
        time_ops_array = np.array(self.time_operations)
        evals_array = np.array(self.evaluations)

        # Compute statistics
        self.statistics = simple_population_statistics(
            generation,
            population,
            fitness_values,
            time_ops_array,
            evals_array,
            self.statistics,
            self.find_best_method
        )

        # Add diversity metrics if requested
        if self.track_diversity:
            diversity = compute_population_diversity(population)
            self.statistics[generation]['diversity'] = diversity

    def get_best_individual(self, generation: Optional[int] = None) -> np.ndarray:
        """
        Get best individual from specified generation.

        Parameters
        ----------
        generation : int, optional
            Generation number. If None, returns best from last generation.

        Returns
        -------
        np.ndarray
            Best individual.
        """
        if generation is None:
            generation = max(self.statistics.keys())

        return self.statistics[generation]['best_individual']

    def get_best_fitness(self, generation: Optional[int] = None) -> float:
        """
        Get best fitness value from specified generation.

        Parameters
        ----------
        generation : int, optional
            Generation number. If None, returns best from last generation.

        Returns
        -------
        float
            Best fitness value (minimum).
        """
        if generation is None:
            generation = max(self.statistics.keys())

        fitness_stats = self.statistics[generation]['fitness_stats']
        return fitness_stats[3, 0] if fitness_stats.ndim > 1 else fitness_stats[3]

    def get_convergence_curve(self, objective_idx: int = 0) -> Dict[str, np.ndarray]:
        """
        Get convergence metrics across all generations.

        Parameters
        ----------
        objective_idx : int, default=0
            Objective index for multi-objective problems.

        Returns
        -------
        dict
            Convergence metrics including best and mean fitness per generation.
        """
        return compute_convergence_metrics(self.statistics, objective_idx)

    def to_dataframe(self, objective_idx: int = 0) -> pd.DataFrame:
        """
        Convert statistics to pandas DataFrame for analysis.

        Parameters
        ----------
        objective_idx : int, default=0
            Objective index for multi-objective problems.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns for generation, fitness statistics,
            diversity, evaluations, and time.
        """
        records = []

        for gen in sorted(self.statistics.keys()):
            stats = self.statistics[gen]
            fitness_stats = stats['fitness_stats']

            record = {
                'generation': gen,
                'best_fitness': fitness_stats[3, objective_idx] if fitness_stats.ndim > 1 else fitness_stats[3],
                'mean_fitness': fitness_stats[1, objective_idx] if fitness_stats.ndim > 1 else fitness_stats[1],
                'median_fitness': fitness_stats[2, objective_idx] if fitness_stats.ndim > 1 else fitness_stats[2],
                'worst_fitness': fitness_stats[0, objective_idx] if fitness_stats.ndim > 1 else fitness_stats[0],
                'std_fitness': fitness_stats[4, objective_idx] if fitness_stats.ndim > 1 else fitness_stats[4],
                'n_unique': stats['n_unique'],
                'n_evaluations': stats['n_evaluations'],
            }

            # Add diversity metrics if available
            if 'diversity' in stats:
                record['uniqueness_ratio'] = stats['diversity']['uniqueness_ratio']
                record['entropy'] = stats['diversity']['entropy']

            # Add time metrics if available
            if self.track_time and len(stats['time_operations']) > 0:
                time_ops = stats['time_operations']
                record['time_total'] = time_ops[-1] if len(time_ops) > 7 else time_ops.sum()

            records.append(record)

        return pd.DataFrame(records)

    def print_summary(self, last_n: Optional[int] = None) -> None:
        """
        Print a summary of statistics.

        Parameters
        ----------
        last_n : int, optional
            Only print last N generations. If None, prints all.
        """
        generations = sorted(self.statistics.keys())
        if last_n is not None:
            generations = generations[-last_n:]

        print("=" * 80)
        print("EDA Statistics Summary")
        print("=" * 80)
        print(f"{'Gen':<6} {'Best':<12} {'Mean':<12} {'Std':<12} {'Unique':<8} {'Evals':<8}")
        print("-" * 80)

        for gen in generations:
            stats = self.statistics[gen]
            fitness_stats = stats['fitness_stats']

            best = fitness_stats[3, 0] if fitness_stats.ndim > 1 else fitness_stats[3]
            mean = fitness_stats[1, 0] if fitness_stats.ndim > 1 else fitness_stats[1]
            std = fitness_stats[4, 0] if fitness_stats.ndim > 1 else fitness_stats[4]

            print(f"{gen:<6} {best:<12.6f} {mean:<12.6f} {std:<12.6f} "
                  f"{stats['n_unique']:<8} {stats['n_evaluations']:<8}")

        print("=" * 80)

    def save(self, filepath: str) -> None:
        """
        Save statistics to file.

        Parameters
        ----------
        filepath : str
            Path to save file. Supports .json, .csv, or .pkl formats.
        """
        path = Path(filepath)
        suffix = path.suffix.lower()

        if suffix == '.json':
            self._save_json(filepath)
        elif suffix == '.csv':
            self._save_csv(filepath)
        elif suffix == '.pkl':
            self._save_pickle(filepath)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def _save_json(self, filepath: str) -> None:
        """Save statistics as JSON."""
        # Convert numpy arrays to lists for JSON serialization
        json_data = {}
        for gen, stats in self.statistics.items():
            json_data[str(gen)] = {
                'fitness_stats': stats['fitness_stats'].tolist(),
                'best_individual': stats['best_individual'].tolist(),
                'n_unique': int(stats['n_unique']),
                'variable_stats': stats['variable_stats'].tolist(),
                'n_evaluations': int(stats['n_evaluations']),
                'time_operations': stats['time_operations'].tolist()
            }
            if 'diversity' in stats:
                json_data[str(gen)]['diversity'] = {
                    k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in stats['diversity'].items()
                }

        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=2)

    def _save_csv(self, filepath: str) -> None:
        """Save statistics as CSV."""
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)

    def _save_pickle(self, filepath: str) -> None:
        """Save statistics as pickle."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.statistics, f)

    def load(self, filepath: str) -> None:
        """
        Load statistics from file.

        Parameters
        ----------
        filepath : str
            Path to load file from.
        """
        path = Path(filepath)
        suffix = path.suffix.lower()

        if suffix == '.json':
            self._load_json(filepath)
        elif suffix == '.pkl':
            self._load_pickle(filepath)
        else:
            raise ValueError(f"Loading not supported for format: {suffix}")

    def _load_json(self, filepath: str) -> None:
        """Load statistics from JSON."""
        with open(filepath, 'r') as f:
            json_data = json.load(f)

        self.statistics = {}
        for gen_str, stats in json_data.items():
            gen = int(gen_str)
            self.statistics[gen] = {
                'fitness_stats': np.array(stats['fitness_stats']),
                'best_individual': np.array(stats['best_individual']),
                'n_unique': stats['n_unique'],
                'variable_stats': np.array(stats['variable_stats']),
                'n_evaluations': stats['n_evaluations'],
                'time_operations': np.array(stats['time_operations'])
            }
            if 'diversity' in stats:
                self.statistics[gen]['diversity'] = stats['diversity']

    def _load_pickle(self, filepath: str) -> None:
        """Load statistics from pickle."""
        import pickle
        with open(filepath, 'rb') as f:
            self.statistics = pickle.load(f)
