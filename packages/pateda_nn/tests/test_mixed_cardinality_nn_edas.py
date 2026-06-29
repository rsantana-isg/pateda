"""
Mixed-cardinality round-trip tests for the discrete NN-EDAs.

Every learner/sampler pair exercised by
``scripts/compare_mixed_cardinality_nn_edas_rw.py`` must accept a *per-variable*
cardinality vector (different K_i per variable) and return a population whose
values respect those cardinalities.  This covers:
  * categorical VAE / GAN / DBD,
  * discrete Backdrive (per-variable embeddings; the sampler must handle the
    different one-hot widths -- regression test for the torch.stack fix),
  * the new categorical denoising-diffusion model.
"""
import numpy as np
import pytest

from pateda_nn.learning.discrete_vae import learn_categorical_vae
from pateda_nn.learning.discrete_gan import learn_categorical_gan
from pateda_nn.learning.discrete_dbd import learn_categorical_dbd
from pateda_nn.learning.discrete_backdrive import learn_discrete_backdrive
from pateda_nn.learning.categorical_dendiff import learn_categorical_dendiff
from pateda_nn.sampling.discrete_neural import (
    sample_categorical_vae,
    sample_categorical_gan,
    sample_discrete_backdrive,
)
from pateda_nn.sampling.discrete_dbd import sample_categorical_dbd
from pateda_nn.sampling.categorical_dendiff import sample_categorical_dendiff


# Deliberately mixed cardinalities (binary + several integer cardinalities).
CARDINALITY = np.array([2, 4, 8, 2, 16, 3, 5, 2])


@pytest.fixture
def mixed_population():
    rng = np.random.default_rng(0)
    n = 50
    pop = np.column_stack([rng.integers(0, c, size=n) for c in CARDINALITY])
    fitness = pop.sum(axis=1).astype(float)
    return pop, fitness


def _assert_respects_cardinality(samples):
    samples = np.asarray(samples)
    assert samples.shape[1] == len(CARDINALITY)
    assert samples.min() >= 0
    for i, c in enumerate(CARDINALITY):
        assert samples[:, i].max() < c, f"variable {i} exceeds cardinality {c}"


def test_categorical_vae_mixed(mixed_population):
    pop, fit = mixed_population
    model = learn_categorical_vae(pop, fit, CARDINALITY, {"epochs": 8, "latent_dim": 6})
    samples = sample_categorical_vae(model, 30, {"temperature": 0.5})
    _assert_respects_cardinality(samples)


def test_categorical_gan_mixed(mixed_population):
    pop, fit = mixed_population
    model = learn_categorical_gan(pop, fit, CARDINALITY, {"epochs": 8, "latent_dim": 12})
    samples = sample_categorical_gan(model, 30, {"temperature": 0.5})
    _assert_respects_cardinality(samples)


def test_categorical_dbd_mixed(mixed_population):
    pop, fit = mixed_population
    # DBD pairs equal-sized source/target populations.
    model = learn_categorical_dbd(pop, pop, CARDINALITY, {"epochs": 8})
    samples = sample_categorical_dbd(model, 30, {"n_steps": 5})
    _assert_respects_cardinality(samples)


def test_discrete_backdrive_mixed(mixed_population):
    """Regression: the backdrive sampler must stack different one-hot widths."""
    pop, fit = mixed_population
    model = learn_discrete_backdrive(pop, fit, CARDINALITY, {"epochs": 8})
    samples = sample_discrete_backdrive(model, 30, {"n_iterations": 10,
                                                    "init_method": "random"})
    _assert_respects_cardinality(samples)


def test_categorical_dendiff_mixed(mixed_population):
    pop, fit = mixed_population
    model = learn_categorical_dendiff(pop, fit, CARDINALITY,
                                      {"epochs": 8, "n_timesteps": 20, "seed": 1})
    assert model["type"] == "categorical_dendiff"
    samples = sample_categorical_dendiff(model, 30, {"seed": 2})
    _assert_respects_cardinality(samples)


def test_categorical_dendiff_via_dispatcher(mixed_population):
    from pateda_nn import sample_discrete_nn, supported_discrete_types
    assert "categorical_dendiff" in supported_discrete_types()
    pop, fit = mixed_population
    model = learn_categorical_dendiff(pop, fit, CARDINALITY,
                                      {"epochs": 5, "n_timesteps": 10, "seed": 1})
    samples = sample_discrete_nn(model, 20)
    _assert_respects_cardinality(samples)
