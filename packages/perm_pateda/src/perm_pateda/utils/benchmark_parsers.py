"""
benchmark_parsers.py
--------------------
Instance loaders for the four standard benchmarks used in Ceberio et al. (2015):

  - LOLIB  → Linear Ordering Problem
  - Taillard → Permutation Flowshop Scheduling (PFSP)
  - QAPLIB  → Quadratic Assignment Problem
  - TSPLIB  → Travelling Salesman Problem

Each parser returns a plain numpy array (or tuple of arrays) that can be fed
directly into the corresponding perm_pateda problem class.

Usage
-----
    from perm_pateda.utils.benchmark_parsers import (
        parse_lolib,
        parse_taillard_pfsp,
        parse_qaplib,
        parse_tsplib,
    )
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# LOLIB  ─ Linear Ordering Problem
# ---------------------------------------------------------------------------

def parse_lolib(filepath: str | Path) -> np.ndarray:
    """Parse a LOLIB / N-be instance file.

    Supports the two most common layouts:

    **Format 1 – plain matrix** (original LOLIB .sop / .lop)::

        n
        w_11 w_12 … w_1n
        w_21 …
        …

    **Format 2 – sparse "n_pairs" header** (N-be* files)::

        n  n_pairs
        i  j  w_ij
        …

    Parameters
    ----------
    filepath:
        Path to the instance file.

    Returns
    -------
    np.ndarray of shape (n, n)
        Weight matrix W where W[i, j] is the benefit of placing i before j.
        The diagonal is set to 0.
    """
    path = Path(filepath)
    lines = [ln.rstrip() for ln in path.read_text().splitlines() if ln.strip()]

    tokens_first = lines[0].split()

    # ── Format 2: sparse triplet list ─────────────────────────────────────
    if len(tokens_first) == 2:
        n, n_pairs = int(tokens_first[0]), int(tokens_first[1])
        matrix = np.zeros((n, n), dtype=np.float64)
        for ln in lines[1:]:
            parts = ln.split()
            if len(parts) < 3:
                continue
            i, j, w = int(parts[0]) - 1, int(parts[1]) - 1, float(parts[2])
            matrix[i, j] = w
        np.fill_diagonal(matrix, 0)
        return matrix

    # ── Format 1: dense matrix ─────────────────────────────────────────────
    n = int(tokens_first[0])
    all_numbers: list[float] = []
    for ln in lines[1:]:
        all_numbers.extend(float(x) for x in ln.split())

    matrix = np.array(all_numbers[: n * n], dtype=np.float64).reshape(n, n)
    np.fill_diagonal(matrix, 0)
    return matrix


# ---------------------------------------------------------------------------
# Taillard PFSP  ─ Permutation Flowshop Scheduling
# ---------------------------------------------------------------------------

def parse_taillard_pfsp(filepath: str | Path) -> np.ndarray:
    """Parse a Taillard PFSP instance file.

    Taillard's original format::

        n_jobs  n_machines  <optional seed / info …>
        <processing times: n_machines rows × n_jobs columns>

    The file may contain several instances separated by blank lines or by
    lines beginning with ``number of jobs``.  Only the **first** instance is
    returned.

    Parameters
    ----------
    filepath:
        Path to the instance file.

    Returns
    -------
    np.ndarray of shape (n_jobs, n_machines)
        ``P[j, m]`` = processing time of job *j* on machine *m*.
    """
    path = Path(filepath)
    text = path.read_text()

    # Strip comment lines that start with "number of jobs", "number of machines" …
    lines = [
        ln
        for ln in text.splitlines()
        if ln.strip() and not re.match(r"^\s*(number|seed|Nb)", ln, re.I)
    ]

    # First non-comment line: n_jobs n_machines [extra …]
    header = lines[0].split()
    n_jobs, n_machines = int(header[0]), int(header[1])

    all_numbers: list[int] = []
    for ln in lines[1:]:
        all_numbers.extend(int(x) for x in ln.split())
        if len(all_numbers) >= n_machines * n_jobs:
            break

    # Taillard stores machines × jobs, transpose to jobs × machines
    raw = np.array(all_numbers[: n_machines * n_jobs], dtype=np.int64)
    processing_times = raw.reshape(n_machines, n_jobs).T  # (n_jobs, n_machines)
    return processing_times


# ---------------------------------------------------------------------------
# QAPLIB  ─ Quadratic Assignment Problem
# ---------------------------------------------------------------------------

def parse_qaplib(filepath: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Parse a QAPLIB instance file.

    Standard QAPLIB format::

        n

        <flow matrix, n×n>

        <distance matrix, n×n>

    Parameters
    ----------
    filepath:
        Path to the ``.dat`` instance file.

    Returns
    -------
    (flow, distance) : tuple of two np.ndarray of shape (n, n)
        ``flow[i, j]``     = flow between facilities *i* and *j*.
        ``distance[a, b]`` = distance between locations *a* and *b*.

    Notes
    -----
    The QAP objective is minimise Σ_ij flow[i,j] · distance[π(i), π(j)].
    """
    path = Path(filepath)
    numbers = list(
        map(int, re.findall(r"-?\d+", path.read_text()))
    )

    n = numbers[0]
    flow = np.array(numbers[1: 1 + n * n], dtype=np.float64).reshape(n, n)
    distance = np.array(numbers[1 + n * n: 1 + 2 * n * n], dtype=np.float64).reshape(n, n)
    return flow, distance


# ---------------------------------------------------------------------------
# TSPLIB  ─ Travelling Salesman Problem
# ---------------------------------------------------------------------------

def parse_tsplib(filepath: str | Path) -> np.ndarray:
    """Parse a TSPLIB instance file and return a full distance matrix.

    Handles the following ``EDGE_WEIGHT_TYPE`` values:
      ``EUC_2D``, ``CEIL_2D``, ``GEO``, ``ATT``,
      ``EXPLICIT`` (with ``LOWER_DIAG_ROW``, ``UPPER_DIAG_ROW``,
      ``LOWER_ROW``, ``UPPER_ROW``, ``FULL_MATRIX``).

    Parameters
    ----------
    filepath:
        Path to the ``.tsp`` instance file.

    Returns
    -------
    np.ndarray of shape (n, n)
        Symmetric distance matrix D where D[i, j] is the travel cost between
        cities *i* and *j*.  Diagonal is 0.
    """
    path = Path(filepath)
    text = path.read_text()

    # ── Parse header ──────────────────────────────────────────────────────
    header: dict[str, str] = {}
    for ln in text.splitlines():
        if ":" in ln:
            key, _, val = ln.partition(":")
            header[key.strip().upper()] = val.strip()

    n = int(header.get("DIMENSION", 0))
    ew_type = header.get("EDGE_WEIGHT_TYPE", "EUC_2D").upper()
    ew_format = header.get("EDGE_WEIGHT_FORMAT", "").upper()

    dist = np.zeros((n, n), dtype=np.float64)

    # ── EXPLICIT weight matrix ─────────────────────────────────────────────
    if ew_type == "EXPLICIT":
        # Extract numbers after EDGE_WEIGHT_SECTION
        section = re.split(r"EDGE_WEIGHT_SECTION", text, flags=re.I)[-1]
        section = re.split(r"(DISPLAY_DATA_SECTION|NODE_COORD_SECTION|EOF)", section, flags=re.I)[0]
        nums = list(map(float, section.split()))
        idx = 0
        if ew_format in ("LOWER_DIAG_ROW",):
            for i in range(n):
                for j in range(i + 1):
                    dist[i, j] = nums[idx]; idx += 1
            dist = dist + dist.T - np.diag(np.diag(dist))
        elif ew_format in ("UPPER_DIAG_ROW",):
            for i in range(n):
                for j in range(i, n):
                    dist[i, j] = nums[idx]; idx += 1
            dist = dist + dist.T - np.diag(np.diag(dist))
        elif ew_format == "LOWER_ROW":
            for i in range(1, n):
                for j in range(i):
                    dist[i, j] = nums[idx]; idx += 1
            dist = dist + dist.T
        elif ew_format == "UPPER_ROW":
            for i in range(n - 1):
                for j in range(i + 1, n):
                    dist[i, j] = nums[idx]; idx += 1
            dist = dist + dist.T
        elif ew_format == "FULL_MATRIX":
            dist = np.array(nums[: n * n]).reshape(n, n)
        np.fill_diagonal(dist, 0)
        return dist

    # ── Node-coordinate formats ────────────────────────────────────────────
    coord_section = re.split(r"NODE_COORD_SECTION", text, flags=re.I)
    if len(coord_section) < 2:
        raise ValueError(f"Cannot find NODE_COORD_SECTION in {filepath}")
    coord_text = re.split(r"(EDGE_WEIGHT_SECTION|DISPLAY_DATA_SECTION|EOF)", coord_section[1], flags=re.I)[0]

    coords: dict[int, tuple[float, float]] = {}
    for ln in coord_text.splitlines():
        parts = ln.split()
        if len(parts) >= 3:
            try:
                node = int(parts[0])
                coords[node] = (float(parts[1]), float(parts[2]))
            except ValueError:
                pass

    nodes = sorted(coords.keys())
    xs = np.array([coords[k][0] for k in nodes])
    ys = np.array([coords[k][1] for k in nodes])

    if ew_type == "EUC_2D":
        for i in range(n):
            for j in range(n):
                dist[i, j] = round(math.sqrt((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2))

    elif ew_type == "CEIL_2D":
        for i in range(n):
            for j in range(n):
                dist[i, j] = math.ceil(math.sqrt((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2))

    elif ew_type == "GEO":
        def to_rad(deg_dec: float) -> float:
            deg = int(deg_dec)
            minutes = deg_dec - deg
            return math.pi * (deg + 5 * minutes / 3) / 180.0

        lats = np.array([to_rad(v) for v in xs])
        lons = np.array([to_rad(v) for v in ys])
        RRR = 6378.388
        for i in range(n):
            for j in range(n):
                q1 = math.cos(lons[i] - lons[j])
                q2 = math.cos(lats[i] - lats[j])
                q3 = math.cos(lats[i] + lats[j])
                dist[i, j] = int(RRR * math.acos(0.5 * ((1 + q1) * q2 - (1 - q1) * q3)) + 1)

    elif ew_type == "ATT":
        for i in range(n):
            for j in range(n):
                xd = xs[i] - xs[j]
                yd = ys[i] - ys[j]
                rij = math.sqrt((xd * xd + yd * yd) / 10.0)
                tij = round(rij)
                dist[i, j] = tij + 1 if tij < rij else tij

    np.fill_diagonal(dist, 0)
    return dist
