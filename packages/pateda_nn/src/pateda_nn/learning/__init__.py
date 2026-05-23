"""Neural network-based probabilistic model learning methods (PyTorch)"""

from pateda_nn.learning.nn_utils import (
    get_activation,
    apply_weight_init,
    compute_default_hidden_dims,
    compute_default_batch_size,
)
from pateda_nn.learning.vae import (
    learn_vae,
    learn_extended_vae,
    learn_conditional_extended_vae,
)
from pateda_nn.learning.gan import learn_gan
from pateda_nn.learning.dbd import learn_dbd
from pateda_nn.learning.dae import learn_dae
from pateda_nn.learning.dendiff import learn_dendiff
from pateda_nn.learning.backdrive import learn_backdrive
from pateda_nn.learning.rbm import learn_rbm
from pateda_nn.learning.nn_eda import LearnNNEDA
from pateda_nn.learning.discrete_vae import (
    learn_discrete_vae,
    learn_discrete_extended_vae,
)
from pateda_nn.learning.discrete_gan import learn_discrete_gan
from pateda_nn.learning.discrete_dbd import (
    learn_discrete_dbd_cs,
    learn_discrete_dbd_cd,
)
from pateda_nn.learning.discrete_backdrive import learn_discrete_backdrive

__all__ = [
    "get_activation", "apply_weight_init",
    "compute_default_hidden_dims", "compute_default_batch_size",
    "learn_vae", "learn_extended_vae", "learn_conditional_extended_vae",
    "learn_gan", "learn_dbd", "learn_dae", "learn_dendiff",
    "learn_backdrive", "learn_rbm", "LearnNNEDA",
    "learn_discrete_vae", "learn_discrete_extended_vae",
    "learn_discrete_gan",
    "learn_discrete_dbd_cs", "learn_discrete_dbd_cd",
    "learn_discrete_backdrive",
]
