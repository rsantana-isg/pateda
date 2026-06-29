"""
Tests for the unified discrete sampling dispatcher (sample_discrete_nn).
"""
import numpy as np
import pytest

from pateda_nn.sampling.dispatch import (
    sample_discrete_nn,
    supported_discrete_types,
)
from pateda_nn.learning.discrete_vae import learn_binary_vae, learn_binary_regvae
from pateda_nn.learning.discrete_gan import learn_binary_gan
from pateda_nn.learning.discrete_backdrive import learn_discrete_backdrive
from pateda_nn.learning.discrete_dbd import learn_binary_dbd_cs, learn_binary_dbd_cd
from pateda_nn.learning.discrete_dendiff_ste import learn_discrete_dendiff_ste


@pytest.fixture
def binary_population():
    rng = np.random.default_rng(0)
    n_vars, pop_size = 12, 40
    pop = rng.integers(0, 2, size=(pop_size, n_vars)).astype(np.float32)
    fitness = pop.sum(axis=1).astype(float)
    return pop, fitness, n_vars


def _assert_valid_binary(samples, n_samples, n_vars):
    assert samples.shape == (n_samples, n_vars)
    uniq = np.unique(samples)
    assert set(uniq.tolist()).issubset({0, 1, 0.0, 1.0})


def test_supported_types_nonempty_and_sorted():
    types = supported_discrete_types()
    assert "binary_vae" in types
    assert "binary_dbd" in types
    assert types == sorted(types)


def test_dispatch_vae(binary_population):
    pop, fitness, n_vars = binary_population
    model = learn_binary_vae(pop, fitness, params={"epochs": 5, "latent_dim": 4})
    samples = sample_discrete_nn(model, n_samples=15)
    _assert_valid_binary(samples, 15, n_vars)


def test_dispatch_regvae_extended(binary_population):
    pop, fitness, n_vars = binary_population
    model = learn_binary_regvae(pop, fitness, params={"epochs": 5, "latent_dim": 4})
    assert model["type"] == "binary_regvae"
    samples = sample_discrete_nn(model, n_samples=10)
    _assert_valid_binary(samples, 10, n_vars)


def test_dispatch_gan(binary_population):
    pop, fitness, n_vars = binary_population
    model = learn_binary_gan(pop, fitness, params={"epochs": 5})
    samples = sample_discrete_nn(model, n_samples=10)
    _assert_valid_binary(samples, 10, n_vars)


def test_dispatch_backdrive(binary_population):
    pop, fitness, n_vars = binary_population
    model = learn_discrete_backdrive(pop, fitness, params={"epochs": 5})
    samples = sample_discrete_nn(model, n_samples=10)
    _assert_valid_binary(samples, 10, n_vars)


def test_dispatch_dendiff(binary_population):
    pop, fitness, n_vars = binary_population
    model = learn_discrete_dendiff_ste(
        pop, fitness, {"n_timesteps": 5, "epochs": 3, "batch_size": 5,
                       "hidden_dims": [16, 8]}
    )
    samples = sample_discrete_nn(model, n_samples=10)
    _assert_valid_binary(samples, 10, n_vars)


def test_dispatch_dbd_cs_with_seed_pop(binary_population):
    pop, fitness, n_vars = binary_population
    model = learn_binary_dbd_cs(pop, pop, params={"epochs": 5})
    assert model.get("variant") == "cs"
    samples = sample_discrete_nn(model, n_samples=10, seed_pop=pop)
    _assert_valid_binary(samples, 10, n_vars)


def test_dispatch_dbd_cd_fallback_without_seed_pop(binary_population):
    pop, fitness, n_vars = binary_population
    model = learn_binary_dbd_cd(pop, pop, params={"epochs": 5})
    assert model.get("variant") == "cd"
    # No seed_pop -> falls back to unconditional sampling, still valid output.
    samples = sample_discrete_nn(model, n_samples=10)
    _assert_valid_binary(samples, 10, n_vars)


def test_dispatch_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown discrete model type"):
        sample_discrete_nn({"type": "does_not_exist"}, n_samples=5)


def test_dispatch_missing_type_raises():
    with pytest.raises(KeyError):
        sample_discrete_nn({}, n_samples=5)
