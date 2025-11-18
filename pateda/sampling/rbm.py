"""
RBM Model Sampling for Discrete EDAs

This module provides sampling algorithms for Restricted Boltzmann Machine (RBM) based
probabilistic models used in combinatorial optimization.
"""

import numpy as np
from typing import Dict, Any, Optional
import torch

# Import RBM class from learning module
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from learning.rbm import SoftmaxRBM


def sample_softmax_rbm(
    model: Dict[str, Any],
    n_samples: int,
    cardinality: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from a Softmax RBM model using Gibbs sampling.

    Parameters
    ----------
    model : dict
        Model containing RBM state and parameters
    n_samples : int
        Number of samples to generate
    cardinality : np.ndarray
        Array of shape (n_vars,) with cardinality for each variable
    params : dict, optional
        Sampling parameters:
        - 'n_gibbs_steps': number of Gibbs sampling steps (default: 10)
        - 'burn_in': number of burn-in steps (default: 100)

    Returns
    -------
    population : np.ndarray
        Sampled population of shape (n_samples, n_vars)
    """
    n_vars = model['n_vars']
    n_hidden = model['n_hidden']
    stored_cardinality = model['cardinality']

    if params is None:
        params = {}

    n_gibbs_steps = params.get('n_gibbs_steps', 10)
    burn_in = params.get('burn_in', 100)

    # Recreate RBM
    rbm = SoftmaxRBM(n_vars, stored_cardinality, n_hidden)
    rbm.load_state_dict(model['rbm_state'])
    rbm.eval()

    with torch.no_grad():
        # Initialize with random values
        init_pop = np.random.randint(0, stored_cardinality[:, np.newaxis],
                                     size=(n_vars, n_samples)).T
        visible = rbm._encode_population(init_pop)

        # Burn-in phase
        for _ in range(burn_in):
            hidden = rbm.sample_hidden(visible)
            visible = rbm.sample_visible(hidden)

        # Sampling phase
        samples = []
        for _ in range(n_samples):
            for _ in range(n_gibbs_steps):
                hidden = rbm.sample_hidden(visible)
                visible = rbm.sample_visible(hidden)

            # Decode one sample
            sample = rbm._decode_visible(visible[[0]])
            samples.append(sample[0])

            # Replace first sample with new random initialization for next sample
            if len(samples) < n_samples:
                new_init = np.random.randint(0, stored_cardinality, size=n_vars)
                visible[0] = rbm._encode_population(new_init.reshape(1, -1))[0]

        population = np.array(samples)

    return population


def sample_softmax_rbm_with_surrogate(
    model: Dict[str, Any],
    n_samples: int,
    cardinality: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Sample from a Softmax RBM model with energy-based surrogate filtering.

    This function generates more samples than needed and filters them based on
    their free energy (which approximates fitness).

    Parameters
    ----------
    model : dict
        Model containing RBM state and parameters
    n_samples : int
        Number of samples to generate
    cardinality : np.ndarray
        Array of shape (n_vars,) with cardinality for each variable
    params : dict, optional
        Sampling parameters:
        - 'n_gibbs_steps': number of Gibbs sampling steps (default: 10)
        - 'burn_in': number of burn-in steps (default: 100)
        - 'oversample_factor': how many more samples to generate (default: 2)
        - 'energy_percentile': percentile of energy to keep (default: 50)

    Returns
    -------
    population : np.ndarray
        Sampled population of shape (n_samples, n_vars) filtered by energy
    """
    if params is None:
        params = {}

    oversample_factor = params.get('oversample_factor', 2)
    energy_percentile = params.get('energy_percentile', 50)

    # Generate more samples
    n_total_samples = n_samples * oversample_factor
    all_samples = sample_softmax_rbm(model, n_total_samples, cardinality, params)

    # Recreate RBM to compute free energy
    n_vars = model['n_vars']
    n_hidden = model['n_hidden']
    stored_cardinality = model['cardinality']

    rbm = SoftmaxRBM(n_vars, stored_cardinality, n_hidden)
    rbm.load_state_dict(model['rbm_state'])
    rbm.eval()

    with torch.no_grad():
        # Encode samples
        visible = rbm._encode_population(all_samples)

        # Compute free energy (lower is better for the RBM)
        energy = rbm.free_energy(visible).numpy()

    # Select samples with lowest energy (best according to RBM)
    threshold = np.percentile(energy, energy_percentile)
    selected_idx = np.where(energy <= threshold)[0][:n_samples]

    # If not enough samples, just take the first n_samples
    if len(selected_idx) < n_samples:
        selected_idx = np.arange(min(n_samples, len(all_samples)))

    population = all_samples[selected_idx]

    return population
