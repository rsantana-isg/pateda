"""
Categorical (mixed-cardinality) Denoising Diffusion EDA.

This generalises the binary discrete denoising diffusion model
(:mod:`pateda_nn.learning.discrete_dendiff_gumbel`) to variables with
arbitrary, possibly different, cardinalities ``K_i``.

Forward (corruption) process
----------------------------
A *uniform* categorical corruption is used.  At timestep ``t`` each variable
independently keeps its value with probability ``alpha_bar_t`` or is resampled
uniformly from ``{0, ..., K_i - 1}`` with probability ``1 - alpha_bar_t``.  As
``t`` grows the data converges to a uniform categorical distribution.

Reverse (denoising) network
---------------------------
An MLP receives the one-hot encoding of the corrupted sample (``sum(K_i)``
inputs) plus a sinusoidal time embedding, and predicts, for every variable, the
logits of its ``K_i`` categories.  Training minimises the per-variable
cross-entropy against the clean category indices.

The companion sampler is
:func:`pateda_nn.sampling.categorical_dendiff.sample_categorical_dendiff`.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pateda_nn.learning.nn_utils import (
    compute_default_batch_size,
    compute_default_hidden_dims,
)
from pateda_nn.learning.discrete_dendiff_gumbel import (
    TimeEmbedding,
    make_beta_schedule_discrete,
    compute_diffusion_params_discrete,
)


def one_hot_encode(x_idx: np.ndarray, cardinality: np.ndarray) -> np.ndarray:
    """One-hot encode an integer population ``(n, n_vars)`` to ``(n, sum(K_i))``."""
    cardinality = np.asarray(cardinality, dtype=int)
    cum_card = np.concatenate([[0], np.cumsum(cardinality)]).astype(int)
    n, n_vars = x_idx.shape
    out = np.zeros((n, int(cum_card[-1])), dtype=np.float32)
    for i in range(n_vars):
        vals = x_idx[:, i].astype(int)
        out[np.arange(n), cum_card[i] + vals] = 1.0
    return out


class CategoricalDenoisingMLP(nn.Module):
    """Denoising network predicting per-variable categorical logits."""

    def __init__(
        self,
        cardinality: np.ndarray,
        time_emb_dim: int = 32,
        hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        self.cardinality = np.asarray(cardinality, dtype=int)
        self.n_vars = int(len(self.cardinality))
        self.total_categories = int(np.sum(self.cardinality))
        self.cum_card = np.concatenate([[0], np.cumsum(self.cardinality)]).astype(int)
        self.time_emb_dim = time_emb_dim

        if hidden_dims is None:
            hidden_dims = [64, 32]
        self.hidden_dims = list(hidden_dims)

        self.time_embed = TimeEmbedding(time_emb_dim)

        layers: List[nn.Module] = []
        prev_dim = self.total_categories + time_emb_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, self.total_categories))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_onehot: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return flat logits of shape ``(batch, total_categories)``."""
        t_emb = self.time_embed(t)
        h = torch.cat([x_onehot, t_emb], dim=1)
        return self.mlp(h)


def q_sample_categorical(
    x_0_idx: np.ndarray,
    t: np.ndarray,
    alphas_cumprod: np.ndarray,
    cardinality: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Forward diffusion for categorical variables (uniform corruption).

    Each entry keeps its value with probability ``alpha_bar_t`` and is otherwise
    resampled uniformly from its category set.
    """
    cardinality = np.asarray(cardinality, dtype=int)
    n, n_vars = x_0_idx.shape
    alpha_bar = alphas_cumprod[t][:, None]                 # (n, 1)
    keep = rng.random((n, n_vars)) < alpha_bar             # keep original where True
    # Uniform categorical resample per variable column (respects per-var K_i).
    resampled = np.empty_like(x_0_idx)
    for i in range(n_vars):
        resampled[:, i] = rng.integers(0, cardinality[i], size=n)
    return np.where(keep, x_0_idx, resampled).astype(int)


def learn_categorical_dendiff(
    population: np.ndarray,
    fitness: np.ndarray,
    cardinality: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Learn a mixed-cardinality categorical denoising diffusion model.

    Parameters
    ----------
    population : np.ndarray
        Integer population ``(pop_size, n_vars)`` with values in ``[0, K_i)``.
    fitness : np.ndarray
        Fitness values (unused by the basic model; accepted for a uniform
        ``learn_*(population, fitness, cardinality, params)`` signature).
    cardinality : np.ndarray
        Per-variable cardinalities ``K_i``.
    params : dict, optional
        ``n_timesteps`` (100), ``beta_schedule`` ('linear'), ``beta_start``
        (1e-4), ``beta_end`` (0.5), ``hidden_dims`` (auto), ``time_emb_dim``
        (32), ``epochs`` (50), ``batch_size`` (auto), ``learning_rate`` (1e-3),
        ``seed`` (optional).

    Returns
    -------
    dict
        Model dictionary (``type == 'categorical_dendiff'``).
    """
    if params is None:
        params = {}

    cardinality = np.asarray(cardinality, dtype=int)
    population = np.asarray(population)
    pop_size, n_vars = population.shape

    n_timesteps = params.get("n_timesteps", 100)
    beta_schedule = params.get("beta_schedule", "linear")
    beta_start = params.get("beta_start", 1e-4)
    beta_end = params.get("beta_end", 0.5)
    hidden_dims = params.get("hidden_dims", compute_default_hidden_dims(n_vars, pop_size))
    time_emb_dim = params.get("time_emb_dim", 32)
    epochs = params.get("epochs", 50)
    batch_size = params.get("batch_size", compute_default_batch_size(n_vars, pop_size))
    learning_rate = params.get("learning_rate", 1e-3)
    seed = params.get("seed", None)

    if seed is not None:
        torch.manual_seed(int(seed))
    rng = np.random.default_rng(seed)

    betas = make_beta_schedule_discrete(beta_schedule, n_timesteps, beta_start, beta_end)
    diffusion_params = compute_diffusion_params_discrete(betas)
    alphas_cumprod = diffusion_params["alphas_cumprod"].astype(np.float32)

    cum_card = np.concatenate([[0], np.cumsum(cardinality)]).astype(int)

    net = CategoricalDenoisingMLP(cardinality, time_emb_dim, hidden_dims)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    data_idx = population.astype(int)
    net.train()

    for _ in range(epochs):
        perm = rng.permutation(pop_size)
        for start in range(0, pop_size, batch_size):
            idx = perm[start:start + batch_size]
            batch_idx = data_idx[idx]
            bsz = batch_idx.shape[0]

            t = rng.integers(0, n_timesteps, size=bsz)
            x_t_idx = q_sample_categorical(batch_idx, t, alphas_cumprod, cardinality, rng)
            x_t_onehot = torch.from_numpy(one_hot_encode(x_t_idx, cardinality))
            t_tensor = torch.from_numpy(t.astype(np.int64))

            logits = net(x_t_onehot, t_tensor)            # (bsz, total_categories)
            target = torch.from_numpy(batch_idx.astype(np.int64))

            # Per-variable cross-entropy (cardinalities differ, so segment it).
            loss = 0.0
            for i in range(n_vars):
                var_logits = logits[:, cum_card[i]:cum_card[i + 1]]
                loss = loss + F.cross_entropy(var_logits, target[:, i])
            loss = loss / n_vars

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

    return {
        "model_state": net.state_dict(),
        "cardinality": cardinality,
        "n_vars": n_vars,
        "input_dim": n_vars,
        "n_timesteps": n_timesteps,
        "hidden_dims": list(hidden_dims),
        "time_emb_dim": time_emb_dim,
        "diffusion_params": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in diffusion_params.items()
        },
        "type": "categorical_dendiff",
    }
