"""
Unified dispatcher for discrete neural-network EDA sampling.

All discrete learning methods in :mod:`pateda_nn.learning` return a model
dictionary that carries a ``'type'`` key identifying the generative model.
:func:`sample_discrete_nn` reads that key and routes the call to the matching
sampler, so callers do not need to remember which ``sample_*`` function goes
with which ``learn_*`` function.

Example
-------
>>> from pateda_nn.learning.discrete_vae import learn_binary_vae
>>> from pateda_nn.sampling.dispatch import sample_discrete_nn
>>> model = learn_binary_vae(pop, fitness, params={'epochs': 10})
>>> new_pop = sample_discrete_nn(model, n_samples=100)
"""

from typing import Any, Callable, Dict, Optional

import numpy as np

from pateda_nn.sampling.discrete_neural import (
    sample_binary_vae,
    sample_categorical_vae,
    sample_binary_cvae,
    sample_binary_descvae,
    sample_binary_regvae,
    sample_binary_momvae,
    sample_binary_bavae,
    sample_binary_aavae,
    sample_binary_fwvae,
    sample_binary_gan,
    sample_categorical_gan,
    sample_binary_gan_cond_fit,
    sample_binary_gan_aux,
    sample_binary_gan_hybrid_vae,
    sample_discrete_backdrive,
    sample_discrete_backdrive_descriptors,
)
from pateda_nn.sampling.discrete_dbd import (
    sample_binary_dbd,
    sample_categorical_dbd,
    sample_binary_dbd_cs,
    sample_binary_dbd_cd,
)
from pateda_nn.sampling.discrete_dendiff import (
    sample_discrete_dendiff_gumbel,
    sample_discrete_dendiff_corruption,
    sample_discrete_dendiff_ste,
    sample_discrete_dendiff_deterministic,
    sample_discrete_dendiff_hard_concrete,
)
from pateda_nn.sampling.categorical_dendiff import sample_categorical_dendiff


# ---------------------------------------------------------------------------
# Registry: model 'type' -> sampler taking (model, n_samples, params).
# DBD CS/CD are handled specially below because they additionally require a
# seed population.  GAN variants that share the plain GAN generator architecture
# reuse :func:`sample_binary_gan`.
# ---------------------------------------------------------------------------
_SIMPLE_SAMPLERS: Dict[str, Callable[..., np.ndarray]] = {
    # VAE family
    "binary_vae": sample_binary_vae,
    "binary_evae": sample_binary_vae,
    "categorical_vae": sample_categorical_vae,
    "binary_cvae": sample_binary_cvae,
    "binary_descvae": sample_binary_descvae,
    "binary_regvae": sample_binary_regvae,
    "binary_momvae": sample_binary_momvae,
    "binary_bavae": sample_binary_bavae,
    "binary_aavae": sample_binary_aavae,
    "binary_fwvae": sample_binary_fwvae,
    # GAN family
    "binary_gan": sample_binary_gan,
    "binary_gan_wgan_gp": sample_binary_gan,
    "binary_gan_weighted_d": sample_binary_gan,
    "binary_gan_statistic_match": sample_binary_gan,
    "binary_gan_repulsion": sample_binary_gan,
    "binary_gan_cond_fit": sample_binary_gan_cond_fit,
    "binary_gan_aux": sample_binary_gan_aux,
    "binary_gan_hybrid_vae": sample_binary_gan_hybrid_vae,
    "categorical_gan": sample_categorical_gan,
    # Backdrive family
    "discrete_backdrive": sample_discrete_backdrive,
    "discrete_backdrive_weighted_mse": sample_discrete_backdrive,
    "discrete_backdrive_ranking": sample_discrete_backdrive,
    "discrete_backdrive_huber": sample_discrete_backdrive,
    "discrete_backdrive_descriptors": sample_discrete_backdrive_descriptors,
    # DBD (categorical and plain binary; CS/CD handled separately)
    "categorical_dbd": sample_categorical_dbd,
    # Categorical (mixed-cardinality) denoising diffusion
    "categorical_dendiff": sample_categorical_dendiff,
}

# Dendiff variants: both the base and the "_enhanced" model share one sampler.
_DENDIFF_SAMPLERS: Dict[str, Callable[..., np.ndarray]] = {
    "gumbel": sample_discrete_dendiff_gumbel,
    "corruption": sample_discrete_dendiff_corruption,
    "ste": sample_discrete_dendiff_ste,
    "deterministic": sample_discrete_dendiff_deterministic,
    "hard_concrete": sample_discrete_dendiff_hard_concrete,
}
for _name, _fn in list(_DENDIFF_SAMPLERS.items()):
    _SIMPLE_SAMPLERS[f"discrete_dendiff_{_name}"] = _fn
    _SIMPLE_SAMPLERS[f"discrete_dendiff_{_name}_enhanced"] = _fn


def supported_discrete_types() -> list:
    """Return the sorted list of model ``'type'`` values understood by the dispatcher."""
    return sorted(set(_SIMPLE_SAMPLERS) | {"binary_dbd"})


def sample_discrete_nn(
    model: Dict[str, Any],
    n_samples: int,
    cardinality: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None,
    seed_pop: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Sample a new discrete population from any pateda_nn discrete model.

    Parameters
    ----------
    model : dict
        Model produced by one of the ``learn_*`` functions; must contain a
        ``'type'`` key.
    n_samples : int
        Number of solutions to generate.
    cardinality : np.ndarray, optional
        Variable cardinalities.  Accepted for interface compatibility with the
        pateda sampling protocol; most discrete neural samplers infer
        cardinality from the model and ignore this argument.
    params : dict, optional
        Sampler-specific parameters forwarded unchanged.
    seed_pop : np.ndarray, optional
        Seed / reference population required by the DBD CS and CD variants
        (the current or selected population the diffusion blends from).  If a
        DBD-CS/CD model is supplied without ``seed_pop``, the dispatcher falls
        back to the unconditional :func:`sample_binary_dbd`.

    Returns
    -------
    np.ndarray
        Sampled population of shape ``(n_samples, n_vars)``.

    Raises
    ------
    KeyError
        If ``model`` has no ``'type'`` key.
    ValueError
        If the model type is not recognised by the dispatcher.
    """
    if "type" not in model:
        raise KeyError(
            "model dictionary has no 'type' key; cannot dispatch discrete sampling"
        )

    model_type = model["type"]

    # DBD with an explicit CS/CD variant needs the seed population.
    if model_type == "binary_dbd":
        variant = model.get("variant")
        if variant == "cs":
            if seed_pop is None:
                return sample_binary_dbd(model, n_samples, params)
            return sample_binary_dbd_cs(model, n_samples, seed_pop, params)
        if variant == "cd":
            if seed_pop is None:
                return sample_binary_dbd(model, n_samples, params)
            return sample_binary_dbd_cd(model, n_samples, seed_pop, params)
        return sample_binary_dbd(model, n_samples, params)

    sampler = _SIMPLE_SAMPLERS.get(model_type)
    if sampler is None:
        raise ValueError(
            f"Unknown discrete model type {model_type!r}. "
            f"Supported types: {supported_discrete_types()}"
        )
    return sampler(model, n_samples, params)
