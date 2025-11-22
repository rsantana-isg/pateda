"""
Tests for Neural Network Utilities (nn_utils.py)

Tests the new activation functions, initialization functions, and helper utilities
for neural network-based EDAs.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn

from pateda.learning.nn_utils import (
    get_activation,
    apply_weight_init,
    compute_default_hidden_dims,
    compute_default_batch_size,
    compute_default_latent_dim,
    build_hidden_layers,
    validate_list_params,
    SUPPORTED_ACTIVATIONS,
    SUPPORTED_INITIALIZATIONS,
)


class TestActivationFunctions:
    """Test activation function utilities"""

    def test_all_supported_activations(self):
        """Test that all supported activation functions work"""
        assert len(SUPPORTED_ACTIVATIONS) == 15

        for act_name in SUPPORTED_ACTIVATIONS:
            activation = get_activation(act_name, in_features=10)
            assert isinstance(activation, nn.Module)

            # Test forward pass
            x = torch.randn(5, 10)
            y = activation(x)
            assert y.shape == (5, 10)

    def test_activation_case_insensitivity(self):
        """Test activation names are case insensitive"""
        act1 = get_activation('relu')
        act2 = get_activation('ReLU')
        act3 = get_activation('RELU')

        x = torch.randn(5, 10)
        # All should produce valid output
        for act in [act1, act2, act3]:
            y = act(x)
            assert y.shape == (5, 10)

    def test_unknown_activation_raises_error(self):
        """Test that unknown activation raises ValueError"""
        with pytest.raises(ValueError):
            get_activation('unknown_activation')

    def test_prelu_with_features(self):
        """Test PReLU activation with in_features parameter"""
        activation = get_activation('prelu', in_features=10)
        assert isinstance(activation, nn.PReLU)

        x = torch.randn(5, 10)
        y = activation(x)
        assert y.shape == (5, 10)


class TestInitializationFunctions:
    """Test weight initialization utilities"""

    def test_all_supported_initializations(self):
        """Test that all supported initialization functions work"""
        assert len(SUPPORTED_INITIALIZATIONS) == 15

        for init_name in SUPPORTED_INITIALIZATIONS:
            linear = nn.Linear(10, 20)
            # Should not raise an error
            apply_weight_init(linear, init_name)
            # Weight should still have the same shape
            assert linear.weight.shape == (20, 10)

    def test_initialization_case_insensitivity(self):
        """Test initialization names are case insensitive"""
        linear1 = nn.Linear(10, 20)
        linear2 = nn.Linear(10, 20)
        linear3 = nn.Linear(10, 20)

        apply_weight_init(linear1, 'xavier_uniform')
        apply_weight_init(linear2, 'Xavier_Uniform')
        apply_weight_init(linear3, 'XAVIER_UNIFORM')

        # All should work without error
        assert linear1.weight.shape == (20, 10)
        assert linear2.weight.shape == (20, 10)
        assert linear3.weight.shape == (20, 10)

    def test_unknown_initialization_raises_error(self):
        """Test that unknown initialization raises ValueError"""
        linear = nn.Linear(10, 20)
        with pytest.raises(ValueError):
            apply_weight_init(linear, 'unknown_init')

    def test_initialization_with_kwargs(self):
        """Test initialization with custom kwargs"""
        linear = nn.Linear(10, 20)

        # Normal initialization with custom std
        apply_weight_init(linear, 'normal', std=0.1)

        # Uniform initialization with custom range
        apply_weight_init(linear, 'uniform', a=-0.5, b=0.5)

        # Xavier with custom gain
        apply_weight_init(linear, 'xavier_uniform', gain=2.0)


class TestDefaultParameters:
    """Test default parameter computation functions"""

    def test_default_hidden_dims(self):
        """Test compute_default_hidden_dims function"""
        # Test with various input sizes
        dims = compute_default_hidden_dims(n_inputs=100, pop_size=200)
        assert len(dims) == 2
        assert dims[0] == max(5, 100 // 10)  # 10
        assert dims[1] == max(10, 200 // 10)  # 20

        # Test with small inputs
        dims_small = compute_default_hidden_dims(n_inputs=10, pop_size=20)
        assert dims_small[0] >= 5
        assert dims_small[1] >= 10

        # Test with single layer
        dims_single = compute_default_hidden_dims(n_inputs=50, pop_size=100, n_layers=1)
        assert len(dims_single) == 1

        # Test with three layers
        dims_three = compute_default_hidden_dims(n_inputs=50, pop_size=100, n_layers=3)
        assert len(dims_three) == 3

    def test_default_batch_size(self):
        """Test compute_default_batch_size function"""
        # Basic test
        batch_size = compute_default_batch_size(n_inputs=100, pop_size=200)
        assert batch_size == max(8, 100 // 50)  # max(8, 2) = 8

        # Test with larger inputs
        batch_size_large = compute_default_batch_size(n_inputs=500, pop_size=200)
        assert batch_size_large == 10  # 500 // 50 = 10

        # Test that batch_size doesn't exceed half of pop_size
        batch_size_small_pop = compute_default_batch_size(n_inputs=1000, pop_size=10)
        assert batch_size_small_pop <= 5  # pop_size // 2 = 5

    def test_default_latent_dim(self):
        """Test compute_default_latent_dim function"""
        # Basic tests
        latent = compute_default_latent_dim(n_inputs=100)
        assert latent == max(2, 100 // 50)  # 2

        latent_large = compute_default_latent_dim(n_inputs=500)
        assert latent_large == 10  # 500 // 50 = 10

        # Test minimum
        latent_small = compute_default_latent_dim(n_inputs=10)
        assert latent_small >= 2


class TestValidateListParams:
    """Test list parameter validation"""

    def test_validate_list_params_defaults(self):
        """Test that validate_list_params sets defaults correctly"""
        hidden_dims = [64, 32, 16]

        act_functs, init_functs = validate_list_params(hidden_dims, None, None)

        assert len(act_functs) == 3
        assert len(init_functs) == 3
        assert all(act == 'relu' for act in act_functs)
        assert all(init == 'default' for init in init_functs)

    def test_validate_list_params_custom(self):
        """Test validate_list_params with custom values"""
        hidden_dims = [64, 32]
        list_act = ['tanh', 'relu']
        list_init = ['xavier_uniform', 'kaiming_normal']

        act_functs, init_functs = validate_list_params(hidden_dims, list_act, list_init)

        assert act_functs == ['tanh', 'relu']
        assert init_functs == ['xavier_uniform', 'kaiming_normal']

    def test_validate_list_params_length_mismatch(self):
        """Test that length mismatch raises ValueError"""
        hidden_dims = [64, 32, 16]
        list_act = ['relu', 'tanh']  # Only 2, need 3

        with pytest.raises(ValueError):
            validate_list_params(hidden_dims, list_act, None)

    def test_validate_list_params_invalid_activation(self):
        """Test that invalid activation raises ValueError"""
        hidden_dims = [64, 32]
        list_act = ['relu', 'invalid_act']

        with pytest.raises(ValueError):
            validate_list_params(hidden_dims, list_act, None)


class TestBuildHiddenLayers:
    """Test hidden layer building utility"""

    def test_build_hidden_layers_basic(self):
        """Test basic hidden layer construction"""
        network = build_hidden_layers(
            input_dim=10,
            hidden_dims=[32, 16],
            output_dim=5
        )

        assert isinstance(network, nn.Sequential)

        # Test forward pass
        x = torch.randn(4, 10)
        y = network(x)
        assert y.shape == (4, 5)

    def test_build_hidden_layers_with_activations(self):
        """Test hidden layers with custom activations"""
        network = build_hidden_layers(
            input_dim=10,
            hidden_dims=[32, 16],
            output_dim=5,
            list_act_functs=['tanh', 'relu'],
            output_activation='sigmoid'
        )

        x = torch.randn(4, 10)
        y = network(x)
        assert y.shape == (4, 5)
        # Output should be in [0, 1] due to sigmoid
        assert torch.all(y >= 0) and torch.all(y <= 1)

    def test_build_hidden_layers_with_dropout(self):
        """Test hidden layers with dropout"""
        network = build_hidden_layers(
            input_dim=10,
            hidden_dims=[32, 16],
            output_dim=5,
            dropout=0.2
        )

        network.train()
        x = torch.randn(4, 10)
        y = network(x)
        assert y.shape == (4, 5)

    def test_build_hidden_layers_with_batch_norm(self):
        """Test hidden layers with batch normalization"""
        network = build_hidden_layers(
            input_dim=10,
            hidden_dims=[32, 16],
            output_dim=5,
            batch_norm=True
        )

        x = torch.randn(8, 10)  # Batch size > 1 for batch norm
        y = network(x)
        assert y.shape == (8, 5)


class TestGANWithNewParams:
    """Test GAN with new activation and initialization parameters"""

    def test_gan_with_custom_activations(self):
        """Test GAN learning with custom activation functions"""
        from pateda.learning.gan import learn_gan
        from pateda.sampling.gan import sample_gan

        np.random.seed(42)
        torch.manual_seed(42)

        population = np.random.randn(100, 5)
        fitness = np.sum(population ** 2, axis=1)

        # Learn with custom activations
        model = learn_gan(
            population, fitness,
            params={
                'latent_dim': 3,
                'hidden_dims_g': [32, 64],
                'hidden_dims_d': [64, 32],
                'list_act_functs_g': ['leaky_relu', 'relu'],
                'list_act_functs_d': ['selu', 'elu'],
                'epochs': 30
            }
        )

        assert model['list_act_functs_g'] == ['leaky_relu', 'relu']
        assert model['list_act_functs_d'] == ['selu', 'elu']

        # Sampling should work
        new_pop = sample_gan(model, n_samples=50)
        assert new_pop.shape == (50, 5)

    def test_gan_with_custom_initializations(self):
        """Test GAN learning with custom initialization functions"""
        from pateda.learning.gan import learn_gan

        np.random.seed(42)
        torch.manual_seed(42)

        population = np.random.randn(100, 5)
        fitness = np.sum(population ** 2, axis=1)

        model = learn_gan(
            population, fitness,
            params={
                'latent_dim': 3,
                'hidden_dims_g': [32, 64],
                'list_init_functs_g': ['xavier_uniform', 'kaiming_normal'],
                'list_init_functs_d': ['orthogonal', 'normal'],
                'epochs': 30
            }
        )

        assert model['list_init_functs_g'] == ['xavier_uniform', 'kaiming_normal']


class TestVAEWithNewParams:
    """Test VAE with new parameters"""

    def test_vae_with_new_defaults(self):
        """Test VAE uses new default latent dimension"""
        from pateda.learning.vae import learn_vae

        np.random.seed(42)
        torch.manual_seed(42)

        # With 100 variables, default latent_dim should be max(2, 100/50) = 2
        population = np.random.randn(50, 100)
        fitness = np.sum(population ** 2, axis=1)

        model = learn_vae(population, fitness, params={'epochs': 10})

        # Default latent_dim should be 2 for n_vars=100
        assert model['latent_dim'] == 2

    def test_vae_with_custom_activations(self):
        """Test VAE with custom activation functions"""
        from pateda.learning.vae import learn_vae
        from pateda.sampling.vae import sample_vae

        np.random.seed(42)
        torch.manual_seed(42)

        population = np.random.randn(100, 10)
        fitness = np.sum(population ** 2, axis=1)

        model = learn_vae(
            population, fitness,
            params={
                'latent_dim': 3,
                'hidden_dims': [20, 10],
                'list_act_functs_enc': ['gelu', 'silu'],
                'list_act_functs_dec': ['elu', 'relu'],
                'epochs': 10
            }
        )

        assert model['list_act_functs_enc'] == ['gelu', 'silu']
        assert model['list_act_functs_dec'] == ['elu', 'relu']

        # Sampling should work
        new_pop = sample_vae(model, n_samples=50)
        assert new_pop.shape == (50, 10)


class TestDBDWithNewParams:
    """Test DBD with new parameters"""

    def test_dbd_with_new_defaults(self):
        """Test DBD uses new default parameters"""
        from pateda.learning.dbd import learn_dbd

        np.random.seed(42)
        torch.manual_seed(42)

        # Create source and target distributions
        n_vars = 50
        p0 = np.random.randn(100, n_vars)
        p1 = np.random.randn(100, n_vars) + 1.0

        model = learn_dbd(p0, p1, params={'epochs': 20})

        # Should have list_act_functs and list_init_functs in the model
        assert 'list_act_functs' in model
        assert 'list_init_functs' in model
        assert all(act == 'relu' for act in model['list_act_functs'])


class TestDendiffWithNewParams:
    """Test Denoising Diffusion with new parameters"""

    def test_dendiff_with_custom_activations(self):
        """Test Denoising Diffusion with custom activations"""
        from pateda.learning.dendiff import learn_dendiff

        np.random.seed(42)
        torch.manual_seed(42)

        population = np.random.randn(100, 10)
        fitness = np.sum(population ** 2, axis=1)

        model = learn_dendiff(
            population, fitness,
            params={
                'hidden_dims': [32, 16],
                'list_act_functs': ['gelu', 'mish'],
                'epochs': 10
            }
        )

        assert model['list_act_functs'] == ['gelu', 'mish']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
