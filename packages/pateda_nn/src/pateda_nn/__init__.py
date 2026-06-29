"""
pateda_nn — Neural Network EDA implementations (PyTorch)

Builds on the pateda package to provide EDA algorithms powered by deep
generative models: VAE, GAN, Diffusion, Backdrive, DBD, RBM, and DAE.

Requires:
    pateda >= 0.1.0
    torch >= 2.0.0
"""

__version__ = "0.1.0"
__author__ = "Roberto Santana"

from pateda_nn.learning import (
    learn_vae,
    learn_extended_vae,
    learn_conditional_extended_vae,
    learn_gan,
    learn_dbd,
    learn_dae,
    learn_dendiff,
    learn_backdrive,
    learn_rbm,
    LearnNNEDA,
    learn_categorical_dendiff,
)
from pateda_nn.sampling import (
    sample_vae,
    sample_extended_vae,
    sample_conditional_extended_vae,
    sample_gan,
    sample_dbd,
    sample_dae,
    sample_dendiff,
    sample_backdrive,
    sample_backdrive_adaptive,
    sample_rbm,
    SampleNNEDA,
    sample_categorical_dendiff,
    sample_discrete_nn,
    supported_discrete_types,
)

__all__ = [
    "learn_vae", "learn_extended_vae", "learn_conditional_extended_vae",
    "learn_gan", "learn_dbd", "learn_dae", "learn_dendiff",
    "learn_backdrive", "learn_rbm", "LearnNNEDA",
    "learn_categorical_dendiff",
    "sample_vae", "sample_extended_vae", "sample_conditional_extended_vae",
    "sample_gan", "sample_dbd", "sample_dae", "sample_dendiff",
    "sample_backdrive", "sample_backdrive_adaptive",
    "sample_rbm", "SampleNNEDA", "sample_categorical_dendiff",
    "sample_discrete_nn", "supported_discrete_types",
]
