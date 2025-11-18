import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
#tf.debugging.set_log_device_placement(True)
import numpy as np
from nn_architectures import *



# Define the neural network D
def create_network(input_shape,output_shape):
    """
    Creates a simple neural network for deblending.
    
    Args:
    input_shape: The shape of the input tensor (x_alpha concatenated with alpha).
    
    Returns:
    A TensorFlow Keras model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape, dtype=tf.float64),  # Set input dtype to float64
        tf.keras.layers.Dense(100, activation='tanh', dtype=tf.float64),  # Set layer dtype to float64
        
        tf.keras.layers.Dense(100, activation='tanh', dtype=tf.float64),  # Set layer dtype to float64
        #tf.keras.layers.Dense(20, activation='relu', dtype=tf.float64),  # Set layer dtype to float64
        #tf.keras.layers.Dense(10, activation='leaky_relu', dtype=tf.float64),  # Set layer dtype to float64
        #tf.keras.layers.Dense(20, activation='tanh', dtype=tf.float64),  # Set layer dtype to float64
        #tf.keras.layers.Dense(16, activation='tanh', dtype=tf.float64),  # Set layer dtype to float64        
        #tf.keras.layers.Dense(input_shape[0] - 1, activation='linear', dtype=tf.float64)  # Output has the same dimension as x
        #tf.keras.layers.Dropout(0.2), 
        tf.keras.layers.Dense(output_shape, activation='linear', dtype=tf.float64)  # Output has the same dimension as x
    ])
    return model

#@tf.function
def train_step(model, optimizer, mse_loss, x0, x1):
    """Performs a single training step."""
    with tf.GradientTape() as tape:
        # Cast inputs to float64 to ensure consistent data types
        x0 = tf.cast(x0, dtype=tf.float64)
        x1 = tf.cast(x1, dtype=tf.float64)
        
        predicted_x1 = model(x0)
        
        # Calculate the loss
        loss = mse_loss(predicted_x1, x1)
        
        # Calculategradients and update weights
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


# Algorithm 3: Training
def train_network(model, p0, p1, epochs=10, batch_size=32, learning_rate=1e-2, opt_method=0):
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
    print("Learning rate", learning_rate)
    optimizer = get_optimizer(opt_method, learning_rate)
        
    mse_loss = tf.keras.losses.MeanSquaredError()


    # Training loop
    for epoch in range(epochs):
        for batch in range(len(p0) // batch_size):
            # Sample x0, x1, and alpha
            x0 = p0[batch * batch_size:(batch + 1) * batch_size]
            x1 = p1[batch * batch_size:(batch + 1) * batch_size]
            
            # Perform a training step
            loss = train_step(model, optimizer, mse_loss, x0, x1)
        print(f"Epoch {epoch + 1}, Batch {batch + 1}, Loss: {loss.numpy()}")

def get_samples(model, inputs):
    """
    Samples from the network
    
    Args:
    model: The trained  neural network.
    inputs: Inputs to the network

    
    Returns:
    The new solutions predicted from the fitness and statistics
    """
    x0 = tf.cast(inputs, dtype=tf.float64)
    outputs = model(inputs)           
    return outputs




def learn_backdrive_model(selected_population, selected_fitness, num_epochs, batch_size, network, opt_method):
    """
    Learn a backdrive model to bridge the distribution between current population and selected solutions

    :param current_population: The selected individuals.    
    :param selected_population: The selected individuals.
    :return: Learned model
    """

    n = selected_population.shape[1]
    ranges=np.vstack((np.min(selected_population,0),np.max(selected_population,0)))
    #ranges=np.vstack((-5*np.ones((1,n)),5*np.ones((1,n)))) 
    selected_population = (selected_population - ranges[0,:]) / (10**(-16)+ (ranges[1,:]-ranges[0,:]))
    #print(selected_fitness)
    selected_fitness = (selected_fitness - np.min(selected_fitness))/(10**(-16) + np.max(selected_fitness)-np.min(selected_fitness))        
    
    sel_p_size, dim  = selected_population.shape
    
    descriptors = np.zeros((sel_p_size,5))  
    descriptors[:,0] = selected_fitness
    descriptors[:,1] = np.min(selected_population,1)
    descriptors[:,2] = np.mean(selected_population,1)
    descriptors[:,3] = np.max(selected_population,1)
    descriptors[:,4] = np.random.uniform(low=0.0, high=1.0, size=sel_p_size)
    
    mean = np.mean(descriptors,0)
    cov  = np.cov(descriptors,rowvar=False)
    
    
    # Train the deblending network
    #train_network(network, descriptors, selected_population, epochs=num_epochs, batch_size=batch_size, opt_method=opt_method)

    #print("Training Network Inputs")
    #print(descriptors[:10,:])

    #print("Selected Population")
    #print(selected_population[:10,:])
    
    network.fit(descriptors, selected_population, epochs=num_epochs, batch_size=batch_size, verbose=0)
    #train_outputs = network.predict(descriptors, verbose=0)

    #print("Training Network Predictions ")
    #print(train_outputs[:10,:])
    
    return network, mean, cov, ranges


def learn_elm_backdrive_model(selected_population, selected_fitness, network):
    """
    Learn a backdrive model to bridge the distribution between current population and selected solutions

    :param current_population: The selected individuals.    
    :param selected_population: The selected individuals.
    :return: Learned model
    """

    n = selected_population.shape[1]
    ranges=np.vstack((np.min(selected_population,0),np.max(selected_population,0)))

    selected_population = (selected_population - ranges[0,:]) / (10**(-16)+ (ranges[1,:]-ranges[0,:]))

    selected_fitness = (selected_fitness - np.min(selected_fitness))/(10**(-16) + np.max(selected_fitness)-np.min(selected_fitness))        
    
    sel_p_size, dim  = selected_population.shape
    
    descriptors = np.zeros((sel_p_size,5))  
    descriptors[:,0] = selected_fitness
    descriptors[:,1] = np.min(selected_population,1)
    descriptors[:,2] = np.mean(selected_population,1)
    descriptors[:,3] = np.max(selected_population,1)
    descriptors[:,4] = np.random.uniform(low=0.0, high=1.0, size=sel_p_size)
    
    mean = np.mean(descriptors,0)
    cov  = np.cov(descriptors,rowvar=False)

    network.fit(descriptors, selected_population)
    
    return network, mean, cov, ranges



def sample_from_backdrive_model(model, mean, cov, ranges, population_size):
    """
    Sample new solutions from the learned model.
    
    :model: Neural network learned to deblend solutions
    :param population_size: Number of individuals to sample.
    :return: A numpy array containing the new population.
    """
    mean[0] = 0
    
    network_inputs = np.clip(np.random.multivariate_normal(mean, cov, population_size),0,1)
    #network_inputs = np.random.multivariate_normal(mean, cov, population_size)
    #print("Test Network Inputs")
    #print(network_inputs[:10,:])
    

    #inputs = tf.convert_to_tensor(network_inputs, dtype=tf.float64)
    #outputs = get_samples(model, inputs)
    #new_population = model.predict(network_inputs, verbose=0)
    new_population = model.predict(network_inputs)

    

    n_dim = ranges.shape[1]
    
    for i in range(0,n_dim):
       new_population[:,i] = ranges[0,i] + (ranges[1,i]-ranges[0,i]+10**(-16))*new_population[:, i]

    #print("Test Network Predictions")
    #print(new_population[:10,:])
    return new_population


def learn_large_backdrive_model(selected_population, selected_fitness, num_epochs, batch_size, network, opt_method):
    """
    Learn a backdrive model to bridge the distribution between current population and selected solutions

    :param current_population: The selected individuals.    
    :param selected_population: The selected individuals.
    :return: Learned model
    """

    n = selected_population.shape[1]
    ranges=np.vstack((np.min(selected_population,0),np.max(selected_population,0)))
    #ranges=np.vstack((-5*np.ones((1,n)),5*np.ones((1,n)))) 
    selected_population = (selected_population - ranges[0,:]) / (10**(-16)+ (ranges[1,:]-ranges[0,:]))
    #print(selected_fitness)
    selected_fitness = (selected_fitness - np.min(selected_fitness))/(10**(-16) + np.max(selected_fitness)-np.min(selected_fitness))        
    
    sel_p_size, dim  = selected_population.shape
    
    descriptors = np.zeros((sel_p_size,5))  
    descriptors[:,0] = selected_fitness
    descriptors[:,1] = np.min(selected_population,1)
    descriptors[:,2] = np.mean(selected_population,1)
    descriptors[:,3] = np.max(selected_population,1)
    descriptors[:,4] = np.median(selected_population,1)
    
    mean = np.mean(descriptors,0)
    cov  = np.cov(descriptors,rowvar=False)
    
    
    # Train the deblending network
    #train_network(network, descriptors, selected_population, epochs=num_epochs, batch_size=batch_size, opt_method=opt_method)

    #print("Training Network Inputs")
    #print(descriptors[:10,:])

    #print("Selected Population")
    #print(selected_population[:10,:])
    
    network.fit(descriptors, selected_population, epochs=num_epochs, batch_size=batch_size, verbose=0)
    #train_outputs = network.predict(descriptors, verbose=0)

    #print("Training Network Predictions ")
    #print(train_outputs[:10,:])
    
    return network, mean, cov, ranges


def learn_referential_backdrive_model(selected_population, selected_fitness, num_epochs, batch_size, network, opt_method):
    """
    Learn a backdrive model to bridge the distribution between current population and selected solutions

    :param current_population: The selected individuals.    
    :param selected_population: The selected individuals.
    :return: Learned model
    """

    n = selected_population.shape[1]
    ranges=np.vstack((np.min(selected_population,0),np.max(selected_population,0)))
    #ranges=np.vstack((-5*np.ones((1,n)),5*np.ones((1,n)))) 
    selected_population = (selected_population - ranges[0,:]) / (10**(-16)+ (ranges[1,:]-ranges[0,:]))
    #print(selected_fitness)
    selected_fitness = (selected_fitness - np.min(selected_fitness))/(10**(-16) + np.max(selected_fitness)-np.min(selected_fitness))        
    
    sel_p_size, dim  = selected_population.shape

    n_descriptors = 7
    descriptors = np.zeros((sel_p_size,n_descriptors))  
    descriptors[:,0] = selected_fitness
    descriptors[:,1] = np.min(selected_population,1)
    descriptors[:,2] = np.mean(selected_population,1)
    descriptors[:,3] = np.max(selected_population,1)
    descriptors[:,4] = np.median(selected_population,1)
    d = np.linalg.norm(selected_population - selected_population[0,:], axis=1) # Distance to first individual in the selpop
    d = (d-np.min(d))/(np.max(d)-np.min(d))
    descriptors[:,5] = d
    d = np.linalg.norm(selected_population - selected_population[-1,:], axis=1)  # Distance to last individual in the selpop
    d = (d-np.min(d))/(np.max(d)-np.min(d))
    descriptors[:,6] = d    
    
    
    mean = np.mean(descriptors,0)
    cov  = np.cov(descriptors,rowvar=False)
    
    
    # Train the deblending network
    #train_network(network, descriptors, selected_population, epochs=num_epochs, batch_size=batch_size, opt_method=opt_method)

    #print("Training Network Inputs")
    #print(descriptors[:10,:])

    #print("Selected Population")
    #print(selected_population[:10,:])
    
    network.fit(descriptors, selected_population, epochs=num_epochs, batch_size=batch_size, verbose=0)
    train_outputs = network.predict(descriptors, verbose=0)

    #print("Training Network Predictions ")
    #print(train_outputs[:10,:])
    
    return network, mean, cov, ranges

def initialize_backdrive_model(n, nas_nn_index, expand_factor):

    learning_rate = 0.0001
    opt_method = 0 # Adam optimizer
    
    the_nas_function = nas_select_function(nas_nn_index)   
    backdrive_model = the_nas_function((5,), (n,), expand_factor)
    #print(deblending_model.summary())

    mse_loss = tf.keras.losses.MeanSquaredError()
    optimizer = get_optimizer(opt_method, learning_rate)

    
    backdrive_model.compile(optimizer=optimizer, loss=mse_loss)

    return backdrive_model
