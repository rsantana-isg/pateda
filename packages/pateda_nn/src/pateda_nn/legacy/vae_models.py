import tensorflow as tf
import copy
#from tensorflow import keras
#from tensorflow.keras import layers
import numpy as np


def build_encoder(input_dim, latent_dim):
    """Builds the encoder network and returns its variables."""
    dim1 = 5
    dim2 = 5
    # Encoder weights and biases
    W1 = tf.Variable(tf.initializers.GlorotUniform()([input_dim, dim1]), name="W1")
    b1 = tf.Variable(tf.zeros([dim1]), name="b1")
    W2 = tf.Variable(tf.initializers.GlorotUniform()([dim1, dim2]), name="W2")
    b2 = tf.Variable(tf.zeros([dim2]), name="b2")
    W_mean = tf.Variable(tf.initializers.GlorotUniform()([dim2, latent_dim]), name="W_mean")
    b_mean = tf.Variable(tf.zeros([latent_dim]), name="b_mean")
    W_logvar = tf.Variable(tf.initializers.GlorotUniform()([dim2, latent_dim]), name="W_logvar")
    b_logvar = tf.Variable(tf.zeros([latent_dim]), name="b_logvar")
    
    encoder_vars = [W1, b1, W2, b2, W_mean, b_mean, W_logvar, b_logvar]
    return encoder_vars

def build_decoder(latent_dim, input_dim):
    """Builds the decoder network and returns its variables."""
    # Decoder weights and biases
    dim1 = 5
    dim2 = 5
    W1 = tf.Variable(tf.initializers.GlorotUniform()([latent_dim, dim2]), name="W1")
    b1 = tf.Variable(tf.zeros([dim2]), name="b1")
    W2 = tf.Variable(tf.initializers.GlorotUniform()([dim2, dim1]), name="W2")
    b2 = tf.Variable(tf.zeros([dim1]), name="b2")
    W_out = tf.Variable(tf.initializers.GlorotUniform()([dim1, input_dim]), name="W_out")
    b_out = tf.Variable(tf.zeros([input_dim]), name="b_out")
    
    decoder_vars = [W1, b1, W2, b2, W_out, b_out]
    return decoder_vars

def encoder(x, encoder_vars):
    """Defines the encoder forward pass."""
    W1, b1, W2, b2, W_mean, b_mean, W_logvar, b_logvar = encoder_vars
    
    # Encoder forward pass
    x = tf.cast(x, dtype=tf.float32)  # Cast input to float32
    hidden1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
    hidden2 = tf.nn.tanh(tf.matmul(hidden1, W2) + b2)
    
    z_mean =  tf.matmul(hidden2, W_mean) + b_mean
    z_log_var = tf.matmul(hidden2, W_logvar) + b_logvar
    
    # Reparameterization trick
    epsilon = tf.random.normal(shape=tf.shape(z_mean))
    z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    return z_mean, z_log_var, z

def decoder(z, decoder_vars):
    """Defines the decoder forward pass."""
    W1, b1, W2, b2, W_out, b_out = decoder_vars
    
    # Decoder forward pass
    z = tf.cast(z, dtype=tf.float32)  # Cast input to float32
    hidden1 = tf.nn.relu(tf.matmul(z, W1) + b1)
    hidden2 = tf.nn.relu(tf.matmul(hidden1, W2) + b2)
    
    output = tf.nn.sigmoid(tf.matmul(hidden2, W_out) + b_out)  # Assuming normalized input
    
    return output

def train_vae(data, latent_dim=5, batch_size=32, epochs=50, learning_rate=0.01):
    """Trains the Variational Autoencoder."""
    input_dim = data.shape[1]
    data = data.astype(np.float32)  # Cast data to float32
    
    # Build encoder and decoder variables
    encoder_vars = build_encoder(input_dim, latent_dim)
    decoder_vars = build_decoder(latent_dim, input_dim)
    
    # Define optimizer
    optimizer = tf.optimizers.Adam(learning_rate)
    
    @tf.function
    def train_step(batch):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = encoder(batch, encoder_vars)
            x_reconstructed = decoder(z, decoder_vars)
            reconstruction_loss = tf.reduce_mean(tf.square(batch - x_reconstructed))
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            loss = reconstruction_loss + kl_loss
        trainable_vars = encoder_vars + decoder_vars
        grads = tape.gradient(loss, trainable_vars)
        optimizer.apply_gradients(zip(grads, trainable_vars))
        return loss
    
    for epoch in range(epochs):
        #np.random.shuffle(data)
        for i in range(0, data.shape[0], batch_size):
            batch = data[i:i+batch_size]
            loss_value = train_step(batch)
        #print(f"Epoch {epoch+1}/{epochs}, Loss: {loss_value:.4f}")
    
    # Return decoder parameters
    return {v.name: v.numpy() for v in decoder_vars}

def sample_vae(decoder_params, num_samples, latent_dim, input_dim):
    """Generates new samples from the trained VAE decoder."""
    # Convert decoder_params dictionary back to tf.Variable objects
    decoder_vars = [
        tf.Variable(decoder_params["W1:0"], name="W1"),
        tf.Variable(decoder_params["b1:0"], name="b1"),
        tf.Variable(decoder_params["W2:0"], name="W2"),
        tf.Variable(decoder_params["b2:0"], name="b2"),
        tf.Variable(decoder_params["W_out:0"], name="W_out"),
        tf.Variable(decoder_params["b_out:0"], name="b_out"),
    ]
    
    z_sampled = np.random.normal(size=(num_samples, latent_dim)).astype(np.float32)
    z = tf.Variable(z_sampled, dtype=tf.float32)
    x_generated = decoder(z, decoder_vars)
    
    return x_generated.numpy()

def learn_vae_model(selected_population, num_epochs, batch_size, latent_dim):
    """
    Learn a variational autoencoder model of the selected solutions.

    Parameters:
    - selected_population: The selected individuals.
    - num_epochs: int, number of training epochs.
    - batch_size: int, batch size for training.
    - latent_dim: int, latent dimension of the VAE.

    Returns:
    - decoder: the trained decoder model.
    """
    ranges=np.vstack((np.min(selected_population,0),np.max(selected_population,0)))
    norm_sel_pop =  (selected_population - selected_population.min(axis=0)) / (0.0000001+ (selected_population.max(axis=0)-selected_population.min(axis=0))) 
    #print(norm_sel_pop)
    
    decoder_params = train_vae(norm_sel_pop, latent_dim=latent_dim, batch_size=batch_size, epochs=num_epochs)
    return decoder_params, ranges

def sample_from_vae_model(decoder_params,ranges, population_size, latent_dim):
    """
    Sample new solutions from the learned model.

    Parameters:
    - decoder_params: the trained decoder parameters.
    - population_size: int, number of individuals to sample.
    - latent_dim: int, dimension of the latent space.

    Returns:
    - new_population: np.ndarray of shape (population_size, n_dim).
    """
    input_dim = len([v for v in decoder_params.keys() if "W_out" in v])  # Infer input_dim from decoder_params
    u_sim = sample_vae(decoder_params, num_samples=population_size, latent_dim=latent_dim, input_dim=input_dim)

    new_population = copy.copy(u_sim)
    
    for i in range(0,input_dim):
       new_population[:,i] = ranges[0,i] + (ranges[1,i]-ranges[0,i])*u_sim[:, i]
       
    return new_population
