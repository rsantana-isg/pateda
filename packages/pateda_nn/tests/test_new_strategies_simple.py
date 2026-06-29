"""
Simple smoke tests for the alternative discrete Dendiff learning strategies.

These mirror :mod:`test_new_sampling_strategies` but only exercise the
*learning* step and the basic structure of the returned model.  They use
ordinary package imports (the previous version loaded modules from a
hard-coded CI path via ``importlib`` -- see ROADMAP "Replace sys.path hacks").
"""
import numpy as np
import pytest
import torch

from pateda_nn.learning.discrete_dendiff_ste import learn_discrete_dendiff_ste
from pateda_nn.learning.discrete_dendiff_hard_concrete import (
    learn_discrete_dendiff_hard_concrete,
)
from pateda_nn.learning.discrete_dendiff_deterministic import (
    learn_discrete_dendiff_deterministic,
)


STRATEGIES = [
    ("STE", learn_discrete_dendiff_ste),
    ("HardConcrete", learn_discrete_dendiff_hard_concrete),
    ("Deterministic", learn_discrete_dendiff_deterministic),
]


@pytest.mark.parametrize("strategy_name,learn_fn", STRATEGIES)
def test_discrete_dendiff_learning_smoke(strategy_name, learn_fn):
    """Each learning strategy returns a well-formed model dictionary."""
    np.random.seed(42)
    torch.manual_seed(42)

    n_vars = 10
    pop_size = 20
    population = np.random.randint(0, 2, (pop_size, n_vars)).astype(np.float32)
    fitness = np.sum(population, axis=1)

    params = {
        "n_timesteps": 5,
        "epochs": 3,
        "batch_size": 5,
        "hidden_dims": [16, 8],
        "learning_rate": 1e-3,
    }

    model = learn_fn(population, fitness, params)

    for key in ("model_state", "input_dim", "type", "hidden_dims"):
        assert key in model, f"{strategy_name}: model missing key '{key}'"
    assert model["input_dim"] == n_vars
