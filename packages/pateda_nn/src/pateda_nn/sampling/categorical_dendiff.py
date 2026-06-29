"""
Sampler for the mixed-cardinality categorical denoising diffusion EDA.

Companion to :func:`pateda_nn.learning.categorical_dendiff.learn_categorical_dendiff`.
"""

from typing import Any, Dict, Optional

import numpy as np
import torch

from pateda_nn.learning.categorical_dendiff import (
    CategoricalDenoisingMLP,
    one_hot_encode,
    q_sample_categorical,
)


def _categorical_sample_from_logits(
    logits: torch.Tensor,
    cum_card: np.ndarray,
    n_vars: int,
    temperature: float,
    deterministic: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """Turn flat per-variable logits into integer category indices."""
    out = np.empty((logits.shape[0], n_vars), dtype=int)
    for i in range(n_vars):
        var_logits = logits[:, cum_card[i]:cum_card[i + 1]]
        if deterministic:
            out[:, i] = torch.argmax(var_logits, dim=-1).cpu().numpy()
        else:
            probs = torch.softmax(var_logits / max(temperature, 1e-6), dim=-1)
            probs = probs.cpu().numpy()
            probs = probs / probs.sum(axis=1, keepdims=True)
            # Vectorised categorical draw per row.
            cdf = np.cumsum(probs, axis=1)
            u = rng.random((probs.shape[0], 1))
            out[:, i] = (u < cdf).argmax(axis=1)
    return out


def sample_categorical_dendiff(
    model: Dict[str, Any],
    n_samples: int,
    params: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Reverse-diffusion sampling for the categorical denoising diffusion model.

    Starts from a uniform random categorical population and iteratively denoises
    from ``t = T-1`` down to ``0``, progressively mixing the predicted clean
    values into the running sample.

    Parameters
    ----------
    model : dict
        Trained model from ``learn_categorical_dendiff``.
    n_samples : int
        Number of solutions to generate.
    params : dict, optional
        ``temperature`` (0.5), ``n_steps`` (all timesteps), ``deterministic``
        (False), ``seed`` (optional).

    Returns
    -------
    np.ndarray
        Integer population ``(n_samples, n_vars)`` with values in ``[0, K_i)``.
    """
    if params is None:
        params = {}

    cardinality = np.asarray(model["cardinality"], dtype=int)
    n_vars = int(model["n_vars"])
    n_timesteps = int(model["n_timesteps"])
    hidden_dims = model["hidden_dims"]
    time_emb_dim = model["time_emb_dim"]
    cum_card = np.concatenate([[0], np.cumsum(cardinality)]).astype(int)

    temperature = params.get("temperature", 0.5)
    n_steps = params.get("n_steps", n_timesteps)
    deterministic = params.get("deterministic", False)
    seed = params.get("seed", None)
    rng = np.random.default_rng(seed)

    alphas_cumprod = np.asarray(model["diffusion_params"]["alphas_cumprod"], dtype=np.float32)

    net = CategoricalDenoisingMLP(cardinality, time_emb_dim, hidden_dims)
    net.load_state_dict(model["model_state"])
    net.eval()

    # Start from a uniform random categorical population.
    x_idx = np.empty((n_samples, n_vars), dtype=int)
    for i in range(n_vars):
        x_idx[:, i] = rng.integers(0, cardinality[i], size=n_samples)

    if n_steps < n_timesteps:
        timestep_schedule = np.linspace(n_timesteps - 1, 0, n_steps, dtype=int)
    else:
        timestep_schedule = list(reversed(range(n_timesteps)))

    with torch.no_grad():
        for t_idx in timestep_schedule:
            x_onehot = torch.from_numpy(one_hot_encode(x_idx, cardinality))
            t_tensor = torch.full((n_samples,), int(t_idx), dtype=torch.long)
            logits = net(x_onehot, t_tensor)

            x_pred = _categorical_sample_from_logits(
                logits, cum_card, n_vars, temperature, deterministic, rng
            )

            if t_idx > 0:
                alpha_bar_t = float(alphas_cumprod[t_idx])
                alpha_bar_prev = float(alphas_cumprod[t_idx - 1])
                mixing_prob = min(max(alpha_bar_prev / (alpha_bar_t + 1e-8), 0.0), 1.0)
                keep_pred = rng.random((n_samples, n_vars)) < mixing_prob
                x_idx = np.where(keep_pred, x_pred, x_idx)
            else:
                x_idx = x_pred

    return x_idx.astype(int)
