"""
Vine Copula Model Learning for Continuous EDAs

This module provides learning algorithms for vine copula-based
probabilistic models used in continuous optimization.

Vine copulas are flexible dependency models that decompose multivariate
distributions into bivariate building blocks (pair-copulas). This allows
modeling complex dependencies between variables while maintaining
computational tractability.

References
----------
- Soto et al. (2011): "Vine Estimation of Distribution Algorithms with
  Application to Molecular Docking"
- pyvinecopulib documentation: https://vinecopulib.github.io/pyvinecopulib/
"""

import numpy as np
from typing import Dict, Any, List, Optional

try:
    import pyvinecopulib as pv
    PYVINECOPULIB_AVAILABLE = True
except ImportError:
    PYVINECOPULIB_AVAILABLE = False
    print("Warning: pyvinecopulib not available. Install with: pip install pyvinecopulib")


# Map copula family indices to pyvinecopulib families
COPULA_FAMILIES = {
    0: 'gaussian',
    1: 'gumbel',
    2: 'frank',
    3: 'joe',
    4: 'indep',
    5: 'clayton',
    6: 'bb1',
    7: 'bb6',
    8: 'bb7',
    9: 'bb8',
    10: 'tll',
}


def _get_copula_family(copula_index: int):
    """Get pyvinecopulib copula family from index."""
    if not PYVINECOPULIB_AVAILABLE:
        raise RuntimeError("pyvinecopulib is not installed")

    family_name = COPULA_FAMILIES.get(copula_index, 'gaussian')
    return getattr(pv.BicopFamily, family_name)


def learn_vine_copula_cvine(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn a C-vine copula model from the population.

    C-vines (canonical vines) have a star structure where each tree has
    one node connected to all others. They are useful when one variable
    acts as a central driver of dependencies.

    Parameters
    ----------
    population : np.ndarray
        Population of shape (pop_size, n_vars) from which to learn
    fitness : np.ndarray
        Fitness values (not used in learning, but kept for API consistency)
    params : dict, optional
        Parameters containing:
        - 'copula_family': int or str, copula family to use (default: 0/gaussian)
        - 'truncation_level': int, vine truncation level (default: None, use all trees)
        - 'family_set': list, list of copula families to consider (default: [gaussian])
        - 'select_families': bool, whether to select best family for each pair (default: False)

    Returns
    -------
    model : dict
        Dictionary containing:
        - 'vine_model': the learned pyvinecopulib Vinecop object
        - 'bounds': array of shape (2, n_vars) with [min, max] for each variable
        - 'type': 'vine_copula_cvine'
        - 'copula_family': the copula family used
        - 'structure_type': 'cvine'

    Examples
    --------
    >>> import numpy as np
    >>> from pateda.learning.vine_copula import learn_vine_copula_cvine
    >>> # Simulate correlated data
    >>> mean = np.array([0, 0, 0])
    >>> cov = np.array([[1, 0.5, 0.3], [0.5, 1, 0.4], [0.3, 0.4, 1]])
    >>> pop = np.random.multivariate_normal(mean, cov, size=100)
    >>> fitness = np.sum(pop**2, axis=1)
    >>> model = learn_vine_copula_cvine(pop, fitness, {'copula_family': 0})
    >>> print(model['type'])
    vine_copula_cvine
    """
    if not PYVINECOPULIB_AVAILABLE:
        raise RuntimeError("pyvinecopulib is required. Install with: pip install pyvinecopulib")

    if params is None:
        params = {}

    # Extract parameters
    copula_family_idx = params.get('copula_family', 0)
    truncation_level = params.get('truncation_level', None)
    select_families = params.get('select_families', False)

    # Get bounds for rescaling
    bounds = np.vstack([np.min(population, axis=0), np.max(population, axis=0)])

    # Transform to pseudo-observations (uniform [0,1])
    u = pv.to_pseudo_obs(population)
    n_vars = population.shape[1]

    # Create C-vine structure
    cvine_structure = pv.CVineStructure([i + 1 for i in range(n_vars)])

    # Set up fit controls
    if select_families:
        # Auto-select best family for each pair
        family_set = [
            pv.BicopFamily.gaussian,
            pv.BicopFamily.clayton,
            pv.BicopFamily.gumbel,
            pv.BicopFamily.frank,
        ]
    else:
        # Use specified family
        copula_family = _get_copula_family(copula_family_idx)
        family_set = [copula_family]

    controls = pv.FitControlsVinecop(family_set=family_set)

    # Fit the vine copula model
    vine_model = pv.Vinecop.from_data(data=u, structure=cvine_structure, controls=controls)

    # Apply truncation if specified
    if truncation_level is not None:
        vine_model.truncate(truncation_level)

    return {
        'vine_model': vine_model,
        'bounds': bounds,
        'type': 'vine_copula_cvine',
        'copula_family': copula_family_idx,
        'structure_type': 'cvine',
        'n_vars': n_vars
    }


def learn_vine_copula_dvine(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn a D-vine copula model from the population.

    D-vines (drawable vines) have a path structure where no node is connected
    to more than two edges. They are more flexible than C-vines for capturing
    diverse dependency patterns.

    Parameters
    ----------
    population : np.ndarray
        Population of shape (pop_size, n_vars) from which to learn
    fitness : np.ndarray
        Fitness values (not used in learning, but kept for API consistency)
    params : dict, optional
        Parameters containing:
        - 'copula_family': int or str, copula family to use (default: 0/gaussian)
        - 'truncation_level': int, vine truncation level (default: None, use all trees)
        - 'family_set': list, list of copula families to consider (default: [gaussian])
        - 'select_families': bool, whether to select best family for each pair (default: True)

    Returns
    -------
    model : dict
        Dictionary containing:
        - 'vine_model': the learned pyvinecopulib Vinecop object
        - 'bounds': array of shape (2, n_vars) with [min, max] for each variable
        - 'type': 'vine_copula_dvine'
        - 'copula_family': the copula family used
        - 'structure_type': 'dvine' or 'rvine'

    Examples
    --------
    >>> import numpy as np
    >>> from pateda.learning.vine_copula import learn_vine_copula_dvine
    >>> # Simulate correlated data
    >>> mean = np.array([0, 0, 0])
    >>> cov = np.array([[1, 0.5, 0.3], [0.5, 1, 0.4], [0.3, 0.4, 1]])
    >>> pop = np.random.multivariate_normal(mean, cov, size=100)
    >>> fitness = np.sum(pop**2, axis=1)
    >>> model = learn_vine_copula_dvine(pop, fitness, {'copula_family': 0})
    >>> print(model['type'])
    vine_copula_dvine
    """
    if not PYVINECOPULIB_AVAILABLE:
        raise RuntimeError("pyvinecopulib is required. Install with: pip install pyvinecopulib")

    if params is None:
        params = {}

    # Extract parameters
    copula_family_idx = params.get('copula_family', 0)
    truncation_level = params.get('truncation_level', None)
    select_families = params.get('select_families', True)

    # Get bounds for rescaling
    bounds = np.vstack([np.min(population, axis=0), np.max(population, axis=0)])

    # Transform to pseudo-observations (uniform [0,1])
    u = pv.to_pseudo_obs(population)
    n_vars = population.shape[1]

    # Set up fit controls
    if select_families:
        # Auto-select best family for each pair
        family_set = [
            pv.BicopFamily.gaussian,
            pv.BicopFamily.clayton,
            pv.BicopFamily.gumbel,
            pv.BicopFamily.frank,
            pv.BicopFamily.joe,
            pv.BicopFamily.bb1,
        ]
    else:
        # Use specified family
        copula_family = _get_copula_family(copula_family_idx)
        family_set = [copula_family]

    controls = pv.FitControlsVinecop(family_set=family_set)

    # Fit the vine copula model (structure auto-selected)
    vine_model = pv.Vinecop.from_data(data=u, controls=controls)

    # Apply truncation if specified
    if truncation_level is not None:
        vine_model.truncate(truncation_level)

    return {
        'vine_model': vine_model,
        'bounds': bounds,
        'type': 'vine_copula_dvine',
        'copula_family': copula_family_idx,
        'structure_type': 'rvine',  # R-vine (general vine)
        'n_vars': n_vars
    }


def learn_vine_copula_auto(
    population: np.ndarray,
    fitness: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Learn a vine copula model with automatic structure and family selection.

    This function automatically selects the best vine structure and copula
    families for each pair based on the data. It provides the most flexibility
    but may require more computation.

    Parameters
    ----------
    population : np.ndarray
        Population of shape (pop_size, n_vars) from which to learn
    fitness : np.ndarray
        Fitness values (not used in learning, but kept for API consistency)
    params : dict, optional
        Parameters containing:
        - 'truncation_level': int, vine truncation level (default: None)
        - 'family_set': list of copula families to consider (default: standard set)
        - 'tree_criterion': criterion for tree selection (default: 'tau')
        - 'selection_criterion': criterion for model selection (default: 'bic')

    Returns
    -------
    model : dict
        Dictionary containing:
        - 'vine_model': the learned pyvinecopulib Vinecop object
        - 'bounds': array of shape (2, n_vars) with [min, max] for each variable
        - 'type': 'vine_copula_auto'
        - 'structure_type': 'auto'

    Examples
    --------
    >>> import numpy as np
    >>> from pateda.learning.vine_copula import learn_vine_copula_auto
    >>> # Simulate mixed dependencies
    >>> n = 100
    >>> x1 = np.random.normal(0, 1, n)
    >>> x2 = x1 + np.random.normal(0, 0.5, n)
    >>> x3 = x2**2 + np.random.normal(0, 0.5, n)
    >>> pop = np.column_stack([x1, x2, x3])
    >>> fitness = np.sum(pop**2, axis=1)
    >>> model = learn_vine_copula_auto(pop, fitness)
    >>> print(model['type'])
    vine_copula_auto
    """
    if not PYVINECOPULIB_AVAILABLE:
        raise RuntimeError("pyvinecopulib is required. Install with: pip install pyvinecopulib")

    if params is None:
        params = {}

    # Extract parameters
    truncation_level = params.get('truncation_level', None)
    family_set = params.get('family_set', None)
    tree_criterion = params.get('tree_criterion', 'tau')
    selection_criterion = params.get('selection_criterion', 'bic')

    # Get bounds for rescaling
    bounds = np.vstack([np.min(population, axis=0), np.max(population, axis=0)])

    # Transform to pseudo-observations (uniform [0,1])
    u = pv.to_pseudo_obs(population)
    n_vars = population.shape[1]

    # Set up fit controls with comprehensive family set
    if family_set is None:
        family_set = [
            pv.BicopFamily.gaussian,
            pv.BicopFamily.clayton,
            pv.BicopFamily.gumbel,
            pv.BicopFamily.frank,
            pv.BicopFamily.joe,
            pv.BicopFamily.bb1,
            pv.BicopFamily.bb6,
            pv.BicopFamily.bb7,
            pv.BicopFamily.bb8,
            pv.BicopFamily.tll,
            pv.BicopFamily.indep,
        ]

    controls = pv.FitControlsVinecop(
        family_set=family_set,
        tree_criterion=tree_criterion,
        selection_criterion=selection_criterion,
        select_families=True,
        select_trunc_lvl=(truncation_level is None)
    )

    # Fit the vine copula model (both structure and families auto-selected)
    vine_model = pv.Vinecop.from_data(data=u, controls=controls)

    # Apply truncation if specified and not auto-selected
    if truncation_level is not None:
        vine_model.truncate(truncation_level)

    return {
        'vine_model': vine_model,
        'bounds': bounds,
        'type': 'vine_copula_auto',
        'structure_type': 'auto',
        'n_vars': n_vars
    }
