import tensorflow as tf
import numpy as np


# Define the neural network D


def create_deblending_network(input_shape):
    """
    Creates a simple neural network for deblending.
    
    Args:
    input_shape: The shape of the input tensor (x_alpha concatenated with alpha).
    
    Returns:
    A TensorFlow Keras model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(input_shape[0] - 1, activation='linear')  # Output has the same dimension as x
    ])
    return model
    

"""
@tf.function
def train_step(x0, x1, alpha):
    #Performs a single training step.
    with tf.GradientTape() as tape:
          # Blend x0 and x1 to get x_alpha
          x_alpha = (1 - alpha) * x0 + alpha * x1
  
          # Predict the difference (x1 - x0) using the network
          predicted_diff = model(tf.concat([x_alpha, tf.expand_dims(alpha, axis=-1)], axis=-1))
  
          # Calculate the loss
          true_diff = x1 - x0
          loss = mse_loss(true_diff, predicted_diff)
  
          # Calculate gradients and update weights
          gradients = tape.gradient(loss, model.trainable_variables)
          optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss 
"""

@tf.function
def train_step(model, optimizer, mse_loss, x0, x1, alpha):
    """Performs a single training step."""
    with tf.GradientTape() as tape:
        # Expand alpha to match the shape of x0 and x1
        alpha_expanded = tf.expand_dims(alpha, axis=-1)  # Shape: (batch_size, 1)
        
        # Blend x0 and x1 to get x_alpha
        x_alpha = (1 - alpha_expanded) * x0 + alpha_expanded * x1  # Shapes now match
        
        # Predict the difference (x1 - x0) using the network
        predicted_diff = model(tf.concat([x_alpha, alpha_expanded], axis=-1))
        
        # Calculate the loss
        true_diff = x1 - x0
        loss = mse_loss(true_diff, predicted_diff)
        
        # Calculate gradients and update weights
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
    


# Algorithm 3: Training
def train_deblending_network(model, p0, p1, epochs=10, batch_size=32, learning_rate=1e-3):
    """
    Trains the deblending neural network.

    Args:
    model: The TensorFlow Keras model to train.
    p0: A function that samples from density p0.
    p1: A function that samples from density p1.
    epochs: The number of training epochs.
    batch_size: The batch size.
    learning_rate: The learning rate for the Adam optimizer.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    mse_loss = tf.keras.losses.MeanSquaredError()

    # Training loop
    for epoch in range(epochs):
        for batch in range(len(p0) // batch_size):
            # Sample x0, x1, and alpha
            x0 = p0[batch * batch_size:(batch + 1) * batch_size]
            x1 = p1[batch * batch_size:(batch + 1) * batch_size]
            alpha = tf.random.uniform(shape=(batch_size,), minval=0, maxval=1)

            # Perform a training step
            loss = train_step(model, optimizer, mse_loss, x0, x1, alpha)
        print(f"Epoch {epoch + 1}, Batch {batch + 1}, Loss: {loss.numpy()}")


def iterative_deblending_sampling(model, x0, num_iterations):
    """
    Implements the iterative alpha-deblending sampling algorithm.
    
    Args:
    model: The trained deblending neural network.
    x0: Initial sample from density p0.
    num_iterations: The number of iterations T.
    
    Returns:
    The final sample x_alphaT, which should be a sample from p1.
    """
    alpha_values = np.linspace(0, 1, num=num_iterations + 1)
    x_alpha = tf.identity(x0)  # Create a copy of x0 using tf.identity
    
    for t in range(num_iterations):
        alpha_t = alpha_values[t]
        alpha_t_plus_1 = alpha_values[t + 1]
        
        # Prepare the input for the network
        # Expand alpha_t to match the shape of x_alpha
        alpha_t_expanded = tf.expand_dims(tf.fill(tf.shape(x_alpha)[:-1], alpha_t), axis=-1)
        alpha_t_expanded = tf.cast(alpha_t_expanded, dtype=tf.float32)  # Ensure correct dtype
        
        # Concatenate x_alpha and alpha_t_expanded
        input_tensor = tf.concat([x_alpha, alpha_t_expanded], axis=-1)
        
        # Predict the difference (x1 - x0)
        predicted_diff = model(input_tensor)
        
        # Update x_alpha
        x_alpha = x_alpha + (alpha_t_plus_1 - alpha_t) * predicted_diff
    
    return x_alpha


# Define the densities p0 and p1 (example: Gaussian)
def sample_p0(shape):
    """Samples from a Gaussian distribution with mean 0 and std 1."""
    return tf.random.normal(shape=shape, mean=0, stddev=1)
  
def sample_p1(shape):
    """Samples from a Gaussian distribution with mean 3 and std 0.5."""
    return tf.random.normal(shape=shape, mean=3, stddev=0.5)


if __name__ == '__main__':
    # Set parameters
    input_dimension = 1000  # Dimension of the data
    num_epochs = 500
    batch_size = 32
    num_iterations = 10  # Number of iterations for sampling
    num_samples = 1000

    # Create samples from p0 and p1 for training
    p0_samples = sample_p0((num_samples, input_dimension))
    p1_samples = sample_p1((num_samples, input_dimension))

    # Create the deblending network
    # Input shape is (input_dimension + 1,) to account for concatenation with alpha
    deblending_model = create_deblending_network((input_dimension + 1,))

    # Train the deblending network
    train_deblending_network(deblending_model, p0_samples, p1_samples, epochs=num_epochs, batch_size=batch_size)

    for i in range(100):
        # Generate a sample from p0
        x0 = sample_p0((1, input_dimension))
        # Perform iterative deblending to generate a sample from p1
        x1 = iterative_deblending_sampling(deblending_model, x0, num_iterations)
        print("Generated sample ",i," of x1:", np.mean(x1.numpy()))

