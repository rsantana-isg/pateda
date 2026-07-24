"""
bn_io.py — Parse bnRep R files and sample Bayesian network datasets.

BN definitions are sourced from the bnRep R package:
    Leonelli, M., Varando, G. (2023).
    bnRep: A repository of Bayesian networks from the academic literature.
    R package. https://github.com/manueleleonelli/bnRep

The original bnRep network specifications were collected from the academic
literature cited within each R source file; the CPT arrays are taken verbatim
from those R files and converted here to NumPy format.

Parsing strategy
----------------
1.  Strip comments, join logical lines (track open-parenthesis depth).
2.  Evaluate simple R expressions (c(), rep(), arithmetic, variable refs).
3.  Find model2network("...") and extract the DAG.
4.  Find all array(...) calls, extract values/dims/dimnames, build CPDs.
5.  Match each array to its variable via the first dimnames key.
"""

from __future__ import annotations

import re
import sys
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class BNDefinition:
    """A fully parsed Bayesian network definition."""
    name: str
    n_vars: int
    var_names: List[str]          # order from model2network
    var_states: Dict[str, List[str]]   # var_name -> state labels
    cardinality: np.ndarray       # shape (n_vars,)
    adjacency: np.ndarray         # shape (n_vars, n_vars), adj[i,j]=1 means i->j
    cpds: Dict[int, Dict]         # {var_idx: {"parents": [...], "cpd": np.ndarray}}


# ---------------------------------------------------------------------------
# model2network parser
# ---------------------------------------------------------------------------

def _parse_model2network(s: str) -> Tuple[List[str], np.ndarray]:
    """
    Parse a bnlearn model2network string and return (var_names, adjacency).

    Format: "[A][B|A][C|A:B]"
      - Each bracketed token is "[child|parent1:parent2:...]"
      - "[child]" means no parents.
    """
    tokens = re.findall(r'\[([^\]]+)\]', s)
    var_names: List[str] = []
    parents_map: Dict[str, List[str]] = {}

    for tok in tokens:
        if '|' in tok:
            child, par_str = tok.split('|', 1)
            parents = [p.strip() for p in par_str.split(':')]
        else:
            child = tok
            parents = []
        child = child.strip()
        if child not in var_names:
            var_names.append(child)
        parents_map[child] = parents

    n = len(var_names)
    idx = {v: i for i, v in enumerate(var_names)}
    adj = np.zeros((n, n), dtype=int)
    for child, parents in parents_map.items():
        ci = idx[child]
        for p in parents:
            if p:
                pi = idx[p]
                adj[pi, ci] = 1
    return var_names, adj


# ---------------------------------------------------------------------------
# R expression evaluator (lightweight)
# ---------------------------------------------------------------------------

def _eval_r_expr(expr: str, env: Dict[str, object]) -> object:
    """
    Evaluate a simple R expression to a Python object.

    Handles:
    - Numeric literals (int, float, scientific notation)
    - Arithmetic: +, -, *, /, unary minus
    - c(v1, v2, ...) — vector constructor
    - rep(expr, n) — repeat a vector n times
    - Variable references (looked up in env)
    - Quoted strings "..." or '...'
    """
    expr = expr.strip()
    if not expr:
        return []

    # Strip outer parentheses
    while expr.startswith('(') and expr.endswith(')'):
        inner = expr[1:-1]
        if _paren_depth(inner) >= 0:
            expr = inner.strip()
        else:
            break

    # Quoted string
    m = re.fullmatch(r'"([^"]*)"', expr)
    if m:
        return m.group(1)
    m = re.fullmatch(r"'([^']*)'", expr)
    if m:
        return m.group(1)

    # c(...) or c(...)
    if re.match(r'c\s*\(', expr):
        inner = expr[expr.index('(') + 1: _matching_close(expr, expr.index('('))]
        parts = _split_args(inner)
        result = []
        for p in parts:
            v = _eval_r_expr(p.strip(), env)
            if isinstance(v, list):
                result.extend(v)
            else:
                result.append(v)
        return result

    # rep(expr, n)
    m = re.match(r'rep\s*\(', expr)
    if m:
        inner = expr[expr.index('(') + 1: _matching_close(expr, expr.index('('))]
        parts = _split_args(inner)
        if len(parts) >= 2:
            val = _eval_r_expr(parts[0].strip(), env)
            n_rep = int(float(_eval_r_expr(parts[1].strip(), env)))
            if isinstance(val, list):
                return val * n_rep
            else:
                return [val] * n_rep
        return []

    # round(expr, n)
    m = re.match(r'round\s*\(', expr)
    if m:
        inner = expr[expr.index('(') + 1: _matching_close(expr, expr.index('('))]
        parts = _split_args(inner)
        val = _eval_r_expr(parts[0].strip(), env)
        return val

    # sum(expr)  — used in some normalisation tricks
    m = re.match(r'sum\s*\(', expr)
    if m:
        inner = expr[expr.index('(') + 1: _matching_close(expr, expr.index('('))]
        parts = _split_args(inner)
        vals = []
        for p in parts:
            v = _eval_r_expr(p.strip(), env)
            if isinstance(v, list):
                vals.extend(v)
            else:
                vals.append(v)
        return float(sum(float(x) for x in vals))

    # Variable reference (must not contain operators/parens at top level)
    if re.fullmatch(r'[A-Za-z_.][A-Za-z0-9_.]*', expr):
        if expr in env:
            v = env[expr]
            if isinstance(v, (int, float)):
                return v
            return v
        # Try as float (e.g. "Inf", "NA" → treat as NaN/1)
        if expr == 'TRUE':
            return 1
        if expr == 'FALSE':
            return 0
        if expr in ('Inf', 'NA', 'NaN'):
            return float('nan')
        return expr  # unknown symbol, return as-is

    # Arithmetic: try to split at top-level +/- (respecting parens)
    # Scan for top-level binary +/- from right (lowest precedence)
    result = _try_arithmetic(expr, env)
    if result is not None:
        return result

    # Numeric literal (possibly scientific)
    try:
        return float(expr)
    except ValueError:
        pass

    # Fallback: unknown
    return expr


def _paren_depth(s: str) -> int:
    """Return minimum paren depth (negative means unbalanced close)."""
    depth = 0
    min_depth = 0
    for c in s:
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
            min_depth = min(min_depth, depth)
    return min_depth


def _matching_close(s: str, open_pos: int) -> int:
    """Return index of the closing ')' matching s[open_pos]='('."""
    depth = 0
    for i in range(open_pos, len(s)):
        if s[i] == '(':
            depth += 1
        elif s[i] == ')':
            depth -= 1
            if depth == 0:
                return i
    raise ValueError(f"Unmatched parenthesis in: {s[:50]}")


def _split_args(s: str) -> List[str]:
    """Split s at top-level commas."""
    parts = []
    depth = 0
    start = 0
    for i, c in enumerate(s):
        if c in ('(', '['):
            depth += 1
        elif c in (')', ']'):
            depth -= 1
        elif c == ',' and depth == 0:
            parts.append(s[start:i])
            start = i + 1
    parts.append(s[start:])
    return parts


def _try_arithmetic(expr: str, env: Dict) -> Optional[float]:
    """
    Attempt to evaluate expr as a simple arithmetic expression.
    Handles +, -, *, / at top level; also handles division like 25/3.
    Falls back to Python eval with a restricted namespace.
    """
    # Build a safe eval namespace
    safe_ns: Dict[str, object] = {}
    for k, v in env.items():
        if isinstance(v, (int, float)):
            safe_ns[k] = v

    # Replace ^ with ** (R power operator)
    py_expr = expr.replace('^', '**')

    # Try to parse as arithmetic
    try:
        result = eval(py_expr, {"__builtins__": {}}, safe_ns)  # noqa: S eval
        if isinstance(result, (int, float)):
            return float(result)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# R file tokenizer / line joiner
# ---------------------------------------------------------------------------

def _preprocess_r_source(text: str) -> List[str]:
    """
    Strip comments and join multi-line statements.
    Returns a list of complete R statements (one logical line each).
    """
    lines = text.splitlines()
    statements = []
    current = ''
    depth = 0

    for raw in lines:
        # Remove inline comments (but not # inside strings)
        line = _strip_comment(raw)
        line = line.rstrip()
        if not line and depth == 0:
            if current.strip():
                statements.append(current.strip())
                current = ''
            continue

        if current:
            current += ' ' + line
        else:
            current = line

        depth += line.count('(') - line.count(')')
        if depth <= 0:
            depth = 0
            if current.strip():
                statements.append(current.strip())
            current = ''

    if current.strip():
        statements.append(current.strip())
    return statements


def _strip_comment(line: str) -> str:
    """Remove R-style # comments that are not inside a string."""
    result = []
    in_str = False
    str_char = ''
    i = 0
    while i < len(line):
        c = line[i]
        if in_str:
            result.append(c)
            if c == str_char:
                in_str = False
        else:
            if c in ('"', "'"):
                in_str = True
                str_char = c
                result.append(c)
            elif c == '#':
                break  # rest is comment
            else:
                result.append(c)
        i += 1
    return ''.join(result)


# ---------------------------------------------------------------------------
# Assignment parser
# ---------------------------------------------------------------------------

def _extract_assignments(statements: List[str]) -> Dict[str, str]:
    """
    Extract all `name <- expr` (or `name = expr`) assignments.
    Returns dict mapping name -> raw_expr string.
    """
    assigns: Dict[str, str] = {}
    pattern = re.compile(r'^([A-Za-z_.][A-Za-z0-9_.]*)\s*(?:<-|=)\s*(.+)$', re.DOTALL)
    for stmt in statements:
        m = pattern.match(stmt)
        if m:
            name = m.group(1)
            expr_raw = m.group(2).strip()
            assigns[name] = expr_raw
    return assigns


# ---------------------------------------------------------------------------
# array() parser
# ---------------------------------------------------------------------------

def _parse_array_call(expr: str, env: Dict[str, object]) -> Optional[Tuple[List[float], List[int], List[str], List[List[str]]]]:
    """
    Parse an R array(c(...), dim=c(...), dimnames=list(...)) call.

    Returns (values, dims, dim_var_names, dim_state_lists) or None.
      - values: flat list of floats (column-major fill order)
      - dims: list of ints
      - dim_var_names: variable name for each dimension [child, parent1, ...]
      - dim_state_lists: state labels for each dimension
    """
    expr = expr.strip()
    if not re.match(r'array\s*\(', expr):
        return None

    # Extract the args of array(...)
    try:
        open_p = expr.index('(')
        close_p = _matching_close(expr, open_p)
    except ValueError:
        return None
    inner = expr[open_p + 1: close_p]
    args = _split_args(inner)
    if len(args) < 1:
        return None

    # --- values ---
    values_expr = args[0].strip()
    raw_values = _eval_r_expr(values_expr, env)
    if not isinstance(raw_values, list):
        raw_values = [raw_values]
    values = []
    for v in raw_values:
        try:
            values.append(float(v))
        except (TypeError, ValueError):
            values.append(float('nan'))

    # --- dim ---
    dims: List[int] = []
    dim_expr = ''
    for a in args[1:]:
        a = a.strip()
        m = re.match(r'dim\s*=\s*(.+)', a, re.DOTALL)
        if m:
            dim_expr = m.group(1).strip()
            break
    if dim_expr:
        raw_dim = _eval_r_expr(dim_expr, env)
        if isinstance(raw_dim, list):
            dims = [int(x) for x in raw_dim]
        else:
            dims = [int(raw_dim)]
    else:
        # Positional second arg
        if len(args) >= 2:
            a = args[1].strip()
            # Check it's not dimnames=
            if not re.match(r'dimnames\s*=', a):
                raw_dim = _eval_r_expr(a, env)
                if isinstance(raw_dim, list):
                    dims = [int(x) for x in raw_dim]
                else:
                    dims = [int(raw_dim)]

    if not dims:
        # Infer from values
        dims = [len(values)]

    # --- dimnames ---
    dim_var_names: List[str] = []
    dim_state_lists: List[List[str]] = []
    dimnames_expr = ''
    for a in args:
        a = a.strip()
        m = re.match(r'dimnames\s*=\s*(.+)', a, re.DOTALL)
        if m:
            dimnames_expr = m.group(1).strip()
            break

    if not dimnames_expr:
        # Positional fallback: look for any arg that starts with "list("
        for a in args[1:]:
            a = a.strip()
            if re.match(r'list\s*\(', a):
                dimnames_expr = a
                break

    if dimnames_expr:
        # list(Key1=states1, Key2=states2, ...)
        dim_var_names, dim_state_lists = _parse_dimnames_list(dimnames_expr, env, dims)

    return values, dims, dim_var_names, dim_state_lists


def _parse_dimnames_list(expr: str, env: Dict[str, object], dims: List[int]) -> Tuple[List[str], List[List[str]]]:
    """
    Parse list(Key1=c(...), Key2=c(...), ...) into (keys, state_lists).
    """
    expr = expr.strip()
    if not re.match(r'list\s*\(', expr):
        return [], [[] for _ in dims]

    try:
        open_p = expr.index('(')
        close_p = _matching_close(expr, open_p)
    except ValueError:
        return [], [[] for _ in dims]
    inner = expr[open_p + 1: close_p]
    args = _split_args(inner)

    keys: List[str] = []
    state_lists: List[List[str]] = []

    for a in args:
        a = a.strip()
        m = re.match(r'([A-Za-z_.`][A-Za-z0-9_. `]*)\s*=\s*(.+)', a, re.DOTALL)
        if m:
            key = m.group(1).strip().strip('`')
            val_expr = m.group(2).strip()
            states_raw = _eval_r_expr(val_expr, env)
            if isinstance(states_raw, list):
                states = [str(s) for s in states_raw]
            elif states_raw is None:
                states = []
            else:
                states = [str(states_raw)]
            keys.append(key)
            state_lists.append(states)
        else:
            # Positional (NULL or expression) — skip
            keys.append('')
            state_lists.append([])

    return keys, state_lists


# ---------------------------------------------------------------------------
# CPD builder
# ---------------------------------------------------------------------------

def _build_cpd(values: List[float], dims: List[int],
               dim_var_names: List[str], dim_state_lists: List[List[str]],
               var_idx: Dict[str, int]) -> Optional[Tuple[int, List[int], np.ndarray]]:
    """
    Convert R array data to CPD in our format.

    Returns (child_idx, parent_indices, cpd_array) where
    cpd[parent_config, child_val] are normalised probabilities.
    """
    if not dim_var_names:
        return None
    child_name = dim_var_names[0]
    if child_name not in var_idx:
        return None
    child_idx = var_idx[child_name]
    parent_names = dim_var_names[1:]
    parent_indices = []
    for pn in parent_names:
        if pn in var_idx:
            parent_indices.append(var_idx[pn])
        # if parent not found, skip (treated as root var later)

    n_total = int(np.prod(dims))

    # Pad or trim values to match total size
    if len(values) < n_total:
        values = values + [float('nan')] * (n_total - len(values))
    else:
        values = values[:n_total]

    # Replace NaN with 0 for CPD building; normalise anyway
    vals_arr = np.array(values, dtype=float)
    vals_arr = np.where(np.isnan(vals_arr), 0.0, vals_arr)

    # Reshape in Fortran (column-major) order: shape = (d_child, d_p1, d_p2, ...)
    arr = vals_arr.reshape(dims, order='F')

    # Move child axis (0) to last: (d_p1, d_p2, ..., d_child)
    arr_t = np.moveaxis(arr, 0, -1)

    # Flatten parent dims: shape = (n_parent_configs, d_child)
    d_child = dims[0]
    cpd = arr_t.reshape(-1, d_child, order='F')

    # Normalise each row
    row_sums = cpd.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    cpd = cpd / row_sums

    return child_idx, parent_indices, cpd


# ---------------------------------------------------------------------------
# Main parse function
# ---------------------------------------------------------------------------

def parse_bnrep_r_file(filepath: str, name: str = None) -> BNDefinition:
    """
    Parse a bnRep R data file into a BNDefinition.

    Parameters
    ----------
    filepath : str
        Path to the .R file.
    name : str, optional
        Name for the BN. Defaults to the filename stem.

    Returns
    -------
    BNDefinition
    """
    path = Path(filepath)
    if name is None:
        name = path.stem

    text = path.read_text(encoding='utf-8', errors='replace')
    statements = _preprocess_r_source(text)

    # ------------------------------------------------------------------
    # 1. Build a basic R environment by evaluating simple assignments
    # ------------------------------------------------------------------
    assigns_raw = _extract_assignments(statements)
    env: Dict[str, object] = {}

    # First pass: evaluate scalars and simple vectors (no array calls)
    for vname, expr in assigns_raw.items():
        if re.match(r'array\s*\(', expr):
            continue  # handle later
        if re.match(r'(model2network|custom\.fit|usethis|library)\s*[\(\[]', expr):
            continue
        try:
            val = _eval_r_expr(expr, env)
            env[vname] = val
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 2. Find model2network string
    # ------------------------------------------------------------------
    model_str = None
    for stmt in statements:
        m = re.search(r'model2network\s*\(\s*"([^"]+)"\s*\)', stmt)
        if m:
            model_str = m.group(1)
            break
    if model_str is None:
        raise ValueError(f"No model2network(...) found in {filepath}")

    var_names, adjacency = _parse_model2network(model_str)
    n_vars = len(var_names)
    var_idx = {v: i for i, v in enumerate(var_names)}

    # ------------------------------------------------------------------
    # 3. Parse all array(...) assignments
    # ------------------------------------------------------------------
    cpd_raw: Dict[str, Tuple] = {}  # child_name -> (child_idx, parent_indices, cpd)
    var_states: Dict[str, List[str]] = {v: [] for v in var_names}

    for vname, expr in assigns_raw.items():
        if not re.match(r'array\s*\(', expr):
            continue
        try:
            result = _parse_array_call(expr, env)
        except Exception as exc:
            # Non-fatal: print warning and continue
            print(f"  [bn_io] Warning: failed to parse array for '{vname}': {exc}",
                  file=sys.stderr)
            continue
        if result is None:
            continue
        values, dims, dim_var_names, dim_state_lists = result

        if not dim_var_names:
            continue
        child_name = dim_var_names[0]
        if child_name not in var_idx:
            # Some arrays are named differently from the variable (e.g. land.prob vs Land_Use)
            # Try to match by checking if any var_name is a substring or prefix
            candidate = _fuzzy_match_var(child_name, var_names)
            if candidate:
                child_name = candidate
                if dim_var_names:
                    dim_var_names[0] = candidate
            else:
                continue

        # Store state labels
        if dim_state_lists and dim_state_lists[0]:
            var_states[child_name] = [str(s) for s in dim_state_lists[0]]

        try:
            res = _build_cpd(values, dims, dim_var_names, dim_state_lists, var_idx)
        except Exception as exc:
            print(f"  [bn_io] Warning: failed to build CPD for '{child_name}': {exc}",
                  file=sys.stderr)
            continue
        if res is None:
            continue
        child_idx, parent_indices, cpd = res
        cpd_raw[child_name] = (child_idx, parent_indices, cpd)

    # ------------------------------------------------------------------
    # 4. Build cardinality vector
    # ------------------------------------------------------------------
    cardinality = np.zeros(n_vars, dtype=int)
    for vname, states in var_states.items():
        if vname in var_idx and states:
            cardinality[var_idx[vname]] = len(states)

    # Fill missing cardinalities from CPD shapes
    for child_name, (ci, pi, cpd) in cpd_raw.items():
        if cardinality[ci] == 0:
            cardinality[ci] = cpd.shape[1]

    # Fill any remaining zeros with 2 (binary default)
    cardinality = np.where(cardinality == 0, 2, cardinality)

    # ------------------------------------------------------------------
    # 5. Build cpds dict (our format)
    # ------------------------------------------------------------------
    # First, verify adjacency consistency with parsed parents
    # (Some R files define CPDs in a different order than model2network)
    cpds: Dict[int, Dict] = {}
    for child_name, (ci, parent_indices, cpd_arr) in cpd_raw.items():
        # Get expected parents from adjacency
        expected_parents = list(np.where(adjacency[:, ci] > 0)[0])

        # Re-derive parents from dim_var_names to keep correct order
        # parent_indices are already in the order matching CPD columns
        cpds[ci] = {
            "parents": parent_indices,
            "cpd": cpd_arr,
        }

    # Fill in missing CPDs (uniform distribution)
    for i, vname in enumerate(var_names):
        if i not in cpds:
            k = int(cardinality[i])
            pa = list(np.where(adjacency[:, i] > 0)[0])
            n_pa_configs = int(np.prod([cardinality[p] for p in pa])) if pa else 1
            uniform = np.full((n_pa_configs, k), 1.0 / k)
            cpds[i] = {"parents": pa, "cpd": uniform}

    # Fill in default states for variables missing them
    for i, vname in enumerate(var_names):
        if not var_states.get(vname):
            k = int(cardinality[i])
            var_states[vname] = [str(j) for j in range(k)]

    return BNDefinition(
        name=name,
        n_vars=n_vars,
        var_names=var_names,
        var_states=var_states,
        cardinality=cardinality,
        adjacency=adjacency,
        cpds=cpds,
    )


def _fuzzy_match_var(candidate: str, var_names: List[str]) -> Optional[str]:
    """Try to match candidate to a var_name using case-insensitive or partial match."""
    cl = candidate.lower().replace(' ', '').replace('_', '')
    for v in var_names:
        vl = v.lower().replace(' ', '').replace('_', '')
        if cl == vl:
            return v
    return None


# ---------------------------------------------------------------------------
# Ancestral sampler
# ---------------------------------------------------------------------------

def sample_bn(bn_def: BNDefinition, n_samples: int, rng=None) -> np.ndarray:
    """
    Draw n_samples i.i.d. samples from the BN via ancestral (forward) sampling.

    Parameters
    ----------
    bn_def : BNDefinition
    n_samples : int
    rng : np.random.Generator, optional
        If None, uses np.random.default_rng().

    Returns
    -------
    np.ndarray of shape (n_samples, n_vars) with dtype int.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = bn_def.n_vars
    adj = bn_def.adjacency
    cardinality = bn_def.cardinality
    cpds = bn_def.cpds

    # Topological sort (Kahn's algorithm)
    in_degree = adj.sum(axis=0).copy()
    queue = list(np.where(in_degree == 0)[0])
    topo_order: List[int] = []
    while queue:
        v = queue.pop(0)
        topo_order.append(v)
        for c in range(n):
            if adj[v, c]:
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
    if len(topo_order) != n:
        raise ValueError("BN adjacency contains a cycle — cannot sample.")

    samples = np.zeros((n_samples, n), dtype=int)

    for vi in topo_order:
        k = int(cardinality[vi])
        if vi not in cpds:
            # Uniform fallback
            samples[:, vi] = rng.integers(0, k, size=n_samples)
            continue

        cpd_info = cpds[vi]
        parents = cpd_info["parents"]
        cpd = cpd_info["cpd"]  # shape (n_parent_configs, k)

        if not parents:
            # Root variable: same distribution for all samples
            probs = cpd[0]  # shape (k,)
            samples[:, vi] = rng.choice(k, size=n_samples, p=probs)
        else:
            # Compute parent configuration index for each sample
            # pc = p1 + d_p1 * p2 + d_p1*d_p2 * p3 + ...
            # (first parent varies fastest — matching CPD layout)
            pc = np.zeros(n_samples, dtype=int)
            stride = 1
            for pi in parents:
                pc += samples[:, pi] * stride
                stride *= int(cardinality[pi])

            # Sample child for each unique parent config
            unique_pcs = np.unique(pc)
            for upc in unique_pcs:
                mask = pc == upc
                n_mask = int(mask.sum())
                if upc < cpd.shape[0]:
                    probs = cpd[upc]
                else:
                    probs = np.ones(k, dtype=float) / k
                # Ensure valid probability vector
                probs = np.clip(probs, 0, None)
                s = probs.sum()
                if s <= 0:
                    probs = np.ones(k, dtype=float) / k
                else:
                    probs = probs / s
                samples[np.where(mask)[0], vi] = rng.choice(k, size=n_mask, p=probs)

    return samples
