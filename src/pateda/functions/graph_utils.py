"""
Graph file utilities for parsing graph instances in PATEDA.

Supports:
- DIMACS clique (.clq) and coloring (.col) formats.
- Max-Cut (.0, .1, etc.) file formats.
"""

from pathlib import Path
from typing import Tuple, List
import numpy as np


def graph_instances_dir(subdir: str = None) -> Path:
    """
    Return the path to the packaged graph-instances directory.

    Instances live in ``pateda/functions/graph_instances/<subdir>``. This helper
    keeps instance lookup independent of the caller's location, mirroring the
    ``_default_*_instances_dir`` helpers used by the Ising, SAT and UBQP problems.

    Args:
        subdir: Optional problem subdirectory (e.g. ``"max_cut"``,
            ``"maximum_clique"``, ``"graph_coloring"``, ``"max_independent_set"``,
            ``"dominating_set"``, ``"clique_covering"``). If ``None``, the root
            ``graph_instances`` directory is returned.

    Returns:
        Absolute :class:`~pathlib.Path` to the requested directory.
    """
    base = Path(__file__).resolve().parent / "graph_instances"
    return base / subdir if subdir is not None else base


def read_dimacs_graph(filepath: str) -> Tuple[int, np.ndarray]:
    """
    Read a graph in DIMACS format (clique .clq or coloring .col).

    Args:
        filepath: Path to the DIMACS file.

    Returns:
        Tuple of (n_nodes, adj_matrix):
            - n_nodes: Number of vertices in the graph.
            - adj_matrix: 2D boolean adjacency matrix of shape (n_nodes, n_nodes).
    """
    n_nodes = 0
    edges = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('c'):
                continue
            if line.startswith('p'):
                parts = line.split()
                # p edge n_nodes n_edges
                # p col n_nodes n_edges
                n_nodes = int(parts[2])
            elif line.startswith('e'):
                parts = line.split()
                # e u v
                u = int(parts[1]) - 1  # Convert 1-indexed to 0-indexed
                v = int(parts[2]) - 1  # Convert 1-indexed to 0-indexed
                edges.append((u, v))

    # Initialize adjacency matrix
    adj_matrix = np.zeros((n_nodes, n_nodes), dtype=bool)
    for u, v in edges:
        if u < n_nodes and v < n_nodes:
            adj_matrix[u, v] = True
            adj_matrix[v, u] = True

    return n_nodes, adj_matrix


def read_max_cut_graph(filepath: str) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Read a Max-Cut graph instance file (e.g. g05_60.0).

    File format:
    - First line: n_nodes n_edges
    - Subsequent lines: u v weight

    Args:
        filepath: Path to the Max-Cut file.

    Returns:
        Tuple of (n_nodes, adj_matrix, weights):
            - n_nodes: Number of vertices in the graph.
            - adj_matrix: 2D boolean adjacency matrix of shape (n_nodes, n_nodes).
            - weights: 2D float weight matrix of shape (n_nodes, n_nodes).
    """
    n_nodes = 0
    edges = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    if len(lines) > 0:
        first_line = lines[0].strip().split()
        n_nodes = int(first_line[0])

        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            u = int(parts[0]) - 1  # Convert 1-indexed to 0-indexed
            v = int(parts[1]) - 1  # Convert 1-indexed to 0-indexed
            w = float(parts[2])
            edges.append((u, v, w))

    # Initialize matrices
    adj_matrix = np.zeros((n_nodes, n_nodes), dtype=bool)
    weights = np.zeros((n_nodes, n_nodes), dtype=float)

    for u, v, w in edges:
        if u < n_nodes and v < n_nodes:
            adj_matrix[u, v] = True
            adj_matrix[v, u] = True
            weights[u, v] = w
            weights[v, u] = w

    return n_nodes, adj_matrix, weights
