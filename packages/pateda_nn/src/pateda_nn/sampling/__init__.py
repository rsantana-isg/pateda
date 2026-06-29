"""Neural network-based sampling methods (PyTorch)"""

from pateda_nn.sampling.vae import (
    sample_vae,
    sample_extended_vae,
    sample_conditional_extended_vae,
)
from pateda_nn.sampling.gan import sample_gan
from pateda_nn.sampling.dbd import sample_dbd
from pateda_nn.sampling.dae import sample_dae
from pateda_nn.sampling.dendiff import sample_dendiff
from pateda_nn.sampling.backdrive import sample_backdrive, sample_backdrive_adaptive
from pateda_nn.sampling.rbm import sample_rbm
from pateda_nn.sampling.nn_eda import SampleNNEDA
from pateda_nn.sampling.discrete_neural import (
    sample_discrete_vae,
    sample_discrete_gan,
    sample_discrete_backdrive,
)
from pateda_nn.sampling.discrete_dbd import (
    sample_discrete_dbd_cs,
    sample_discrete_dbd_cd,
)
from pateda_nn.sampling.categorical_dendiff import sample_categorical_dendiff
from pateda_nn.sampling.dispatch import (
    sample_discrete_nn,
    supported_discrete_types,
)

__all__ = [
    "sample_vae", "sample_extended_vae", "sample_conditional_extended_vae",
    "sample_gan", "sample_dbd", "sample_dae", "sample_dendiff",
    "sample_backdrive", "sample_backdrive_adaptive",
    "sample_rbm", "SampleNNEDA",
    "sample_discrete_vae", "sample_discrete_gan", "sample_discrete_backdrive",
    "sample_discrete_dbd_cs", "sample_discrete_dbd_cd",
    "sample_categorical_dendiff",
    "sample_discrete_nn", "supported_discrete_types",
]
