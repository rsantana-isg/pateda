import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
from nn_architectures import *
from gaussian_models import *
from TfELM.Layers.ELMLayer import ELMLayer
from TfELM.Models.ELMModel import ELMModel
from TfELM.Models.DeepELMModel import DeepELMModel
from TfELM.Models.EHDrELMModel import  EHDrELMModel

def Create_Data_Set(p0, p1, num_iterations):
    n,m = p0.shape
    x0 = np.repeat(p0,num_iterations,axis=0)
    x1 = np.repeat(p1,num_iterations,axis=0)
    alpha = np.random.uniform(low=0, high=1, size=n*num_iterations).reshape(-1,1)
    x_alpha = (1 - alpha) * x0 + alpha * x1
    true_diff = x1-x0
    return alpha, x_alpha, true_diff
    

# Algorithm 3: Training
def train_deblending_network(model, p0, p1, num_iterations, num_epochs=10, batch_size=32, learning_rate=1e-3):
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

    alpha, x_alpha, true_diff = Create_Data_Set(p0, p1, num_iterations)
    
    enlarged_x_alpha = np.hstack((x_alpha,alpha))
    model.fit(enlarged_x_alpha, true_diff, epochs=num_epochs, batch_size=batch_size, verbose=0)

    return model


# Algorithm 3: Training with ELM model
def train_elm_deblending_network(model, p0, p1, num_iterations):
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

    alpha, x_alpha, true_diff = Create_Data_Set(p0, p1, num_iterations)
    
    enlarged_x_alpha = np.hstack((x_alpha,alpha))
    model.fit(enlarged_x_alpha, true_diff)

    return model



def iterative_elm_deblending_sampling(model, x0, num_iterations):
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
    x_alpha = x0 
    n = x0.shape[0]
    
    for t in range(num_iterations):
        alpha_t = alpha_values[t]
        alpha_t_plus_1 = alpha_values[t + 1]

        alpha_t_expanded = np.repeat(alpha_t,n).reshape(-1,1)        
        enlarged_x_alpha = np.hstack((x_alpha, alpha_t_expanded))
        
        # Predict the difference (x1 - x0)
        predicted_diff = model.predict(enlarged_x_alpha)
        
        # Update x_alpha
        x_alpha = x_alpha + (alpha_t_plus_1 - alpha_t) * predicted_diff
    
    return x_alpha



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
    x_alpha = x0 
    n = x0.shape[0]
    
    for t in range(num_iterations):
        alpha_t = alpha_values[t]
        alpha_t_plus_1 = alpha_values[t + 1]

        alpha_t_expanded = np.repeat(alpha_t,n).reshape(-1,1)        
        enlarged_x_alpha = np.hstack((x_alpha, alpha_t_expanded))
        
        # Predict the difference (x1 - x0)
        predicted_diff = model(enlarged_x_alpha)
        
        # Update x_alpha
        x_alpha = x_alpha + (alpha_t_plus_1 - alpha_t) * predicted_diff
    
    return x_alpha.numpy()

def learn_diffusion_model(current_population, selected_population, num_iterations, num_epochs, batch_size, deblending_model):
    """
    Learn a diffusion model to bridge the distribution between current population and selected solutions

    :param current_population: The selected individuals.    
    :param selected_population: The selected individuals.
    :return: Learned model
    """

    ranges=np.vstack((np.min(current_population,0),np.max(current_population,0)))
    current_population =  (current_population - ranges[0,:])  / (10**(-16) + (ranges[1,:]-ranges[0,:]))
    selected_population = (selected_population - ranges[0,:]) / (10**(-16) + (ranges[1,:]-ranges[0,:]))
        
    p_size, n_dim = current_population.shape
    sel_p_size = selected_population.shape[0]
    to_take = p_size*2
    
    p0_indices = np.random.randint(0, high=p_size, size=to_take)
    p0_samples = current_population[p0_indices, :]

    
    selected_indices = np.random.randint(0, high=sel_p_size, size=to_take)
    p1_samples = selected_population[selected_indices, :]

    # Input shape is (input_dimension + 1,) to account for concatenation with alpha
    # deblending_model = create_deblending_network((n_dim + 1,))

    # Train the deblending network
    train_deblending_network(deblending_model, p0_samples, p1_samples, num_iterations, num_epochs, batch_size)

    return deblending_model, ranges


def learn_elm_diffusion_model(current_population, selected_population, num_iterations, deblending_model):
    """
    Learn a diffusion model to bridge the distribution between current population and selected solutions

    :param current_population: The selected individuals.    
    :param selected_population: The selected individuals.
    :return: Learned model
    """

    ranges=np.vstack((np.min(current_population,0),np.max(current_population,0)))
    current_population =  (current_population - ranges[0,:])  / (10**(-16) + (ranges[1,:]-ranges[0,:]))
    selected_population = (selected_population - ranges[0,:]) / (10**(-16) + (ranges[1,:]-ranges[0,:]))
        
    p_size, n_dim = current_population.shape
    sel_p_size = selected_population.shape[0]
    to_take = p_size*2
    
    p0_indices = np.random.randint(0, high=p_size, size=to_take)
    p0_samples = current_population[p0_indices, :]

    
    selected_indices = np.random.randint(0, high=sel_p_size, size=to_take)
    p1_samples = selected_population[selected_indices, :]

    # Input shape is (input_dimension + 1,) to account for concatenation with alpha
    # deblending_model = create_deblending_network((n_dim + 1,))

    # Train the deblending network
    train_elm_deblending_network(deblending_model, p0_samples, p1_samples, num_iterations)

    return deblending_model, ranges


def sample_from_diffusion_model(deblending_model, ranges, population_size, selected_population, num_iterations):
    """
    Sample new solutions from the learned model.
    
    :deblending_model: Neural network learned to deblend solutions
    :param population_size: Number of individuals to sample.
    :return: A numpy array containing the new population.
    """
    selected_population = (selected_population - ranges[0,:]) / (10**(-16) + (ranges[1,:]-ranges[0,:]))
    sel_p_size, n_dim = selected_population.shape    
    selected_indices = np.random.randint(0, high=sel_p_size, size=population_size)
    p0_samples = selected_population[selected_indices, :]
    
    new_population = iterative_deblending_sampling(deblending_model, p0_samples, num_iterations)

            
    for i in range(0,n_dim):
       #new_population[:,i] = (new_population[:,i] - np.min(new_population[:,i]))/(np.max(new_population[:,i]) - np.min(new_population[:,i]))  
       new_population[:,i] = ranges[0,i] + (10**(-16) + ranges[1,i]-ranges[0,i])*new_population[:, i]
        
    return new_population

def sample_from_elm_diffusion_model(deblending_model, ranges, population_size, selected_population, num_iterations):
    """
    Sample new solutions from the learned model.
    
    :deblending_model: Neural network learned to deblend solutions
    :param population_size: Number of individuals to sample.
    :return: A numpy array containing the new population.
    """
    selected_population = (selected_population - ranges[0,:]) / (10**(-16) + (ranges[1,:]-ranges[0,:]))
    sel_p_size, n_dim = selected_population.shape    
    selected_indices = np.random.randint(0, high=sel_p_size, size=population_size)
    p0_samples = selected_population[selected_indices, :]
    
    new_population = iterative_elm_deblending_sampling(deblending_model, p0_samples, num_iterations)

            
    for i in range(0,n_dim):
       #new_population[:,i] = (new_population[:,i] - np.min(new_population[:,i]))/(np.max(new_population[:,i]) - np.min(new_population[:,i]))  
       new_population[:,i] = ranges[0,i] + (10**(-16) + ranges[1,i]-ranges[0,i])*new_population[:, i]
        
    return new_population



def find_closest_neighbors(source_matrix, reference_matrix):
    """
    Finds the closest row in reference_matrix for each row in source_matrix
    based on mean squared error distance.
    
    Parameters:
    source_matrix (numpy.ndarray): Source matrix of shape (N, m)
    reference_matrix (numpy.ndarray): Reference matrix of shape (M, m)
    
    Returns:
    numpy.ndarray: Closest neighbors matrix of shape (N, m)
    """
    N, m = source_matrix.shape
    M = reference_matrix.shape[0]
    
    # Compute pairwise mean squared error distances
    distances = np.sum((source_matrix[:, np.newaxis, :] - reference_matrix[np.newaxis, :, :]) ** 2, axis=2)
    
    # Find the index of the closest reference row for each source row
    closest_indices = np.argmin(distances, axis=1)
    
    # Retrieve the closest rows from reference_matrix
    closest_neighbor_matrix = reference_matrix[closest_indices]
    
    return closest_neighbor_matrix


def learn_focused_diffusion_model(current_population, selected_population, num_iterations, num_epochs, batch_size, deblending_model):
    """
    Learn a diffusion model to bridge the distribution between current population and selected solutions

    :param current_population: The selected individuals.    
    :param selected_population: The selected individuals.
    :return: Learned model
    """

    ranges=np.vstack((np.min(current_population,0),np.max(current_population,0)))
    current_population =  (current_population - ranges[0,:])  / (10**(-16) + (ranges[1,:]-ranges[0,:]))
    selected_population = (selected_population - ranges[0,:]) / (10**(-16) + (ranges[1,:]-ranges[0,:]))
        
    p_size, n_dim = current_population.shape
    sel_p_size = selected_population.shape[0]
    to_take = p_size*2
    
    p0_indices = np.random.randint(0, high=p_size, size=to_take)
    p0_samples = current_population[p0_indices, :]

    p1_samples = find_closest_neighbors(p0_samples, selected_population)
    
    # Input shape is (input_dimension + 1,) to account for concatenation with alpha
    # deblending_model = create_deblending_network((n_dim + 1,))

    # Train the deblending network

    train_deblending_network(deblending_model, p0_samples, p1_samples, num_iterations, num_epochs, batch_size)

    return deblending_model, ranges


def learn_p0_univariate_diffusion_model(current_population, selected_population, num_iterations,  num_epochs, batch_size, deblending_model):
    """
    Learn a diffusion model to bridge the distribution between current population and selected solutions

    :param current_population: The selected individuals.    
    :param selected_population: The selected individuals.
    :return: Learned model
    """

    ranges=np.vstack((np.min(current_population,0),np.max(current_population,0)))
    current_population =  (current_population - ranges[0,:])  / (10**(-16) + (ranges[1,:]-ranges[0,:]))
    selected_population = (selected_population - ranges[0,:]) / (10**(-16) + (ranges[1,:]-ranges[0,:]))
        
    p_size, n_dim = current_population.shape
    sel_p_size = selected_population.shape[0]
    to_take = p_size*2
    
    p0_indices = np.random.randint(0, high=p_size, size=to_take)
    p0_samples = current_population[p0_indices, :]

    mean, cov = learn_univariate_gaussian_model(current_population)
    p0_samples = sample_from_multivariate_gaussian_model(mean, cov, to_take, diversity=0)

    selected_indices = np.random.randint(0, high=sel_p_size, size=to_take)
    p1_samples = selected_population[selected_indices, :]    
    #p1_samples = find_closest_neighbors(p0_samples, selected_population)
    
    # Input shape is (input_dimension + 1,) to account for concatenation with alpha
    # deblending_model = create_deblending_network((n_dim + 1,))

    # Train the deblending network

    train_deblending_network(deblending_model, p0_samples, p1_samples, num_iterations, num_epochs, batch_size)

    return deblending_model, ranges

def sample_p1_univariate_diffusion_model(deblending_model, ranges, population_size, selected_population, num_iterations):
    """
    Sample new solutions from the learned model.
    
    :deblending_model: Neural network learned to deblend solutions
    :param population_size: Number of individuals to sample.
    :return: A numpy array containing the new population.
    """
    selected_population = (selected_population - ranges[0,:]) / (10**(-16) + (ranges[1,:]-ranges[0,:]))
    sel_p_size, n_dim = selected_population.shape    
    selected_indices = np.random.randint(0, high=sel_p_size, size=population_size)
    p1_samples = selected_population[selected_indices, :]

    mean, cov = learn_univariate_gaussian_model(p1_samples)
    p1_samples = sample_from_multivariate_gaussian_model(mean, cov, sel_p_size, diversity=0)   
    
    new_population = iterative_deblending_sampling(deblending_model, p1_samples, num_iterations)    
    for i in range(0,n_dim):
       new_population[:,i] = ranges[0,i] + (ranges[1,i]-ranges[0,i])*new_population[:, i]
        
    return new_population


def sample_elm_p1_univariate_diffusion_model(deblending_model, ranges, population_size, selected_population, num_iterations):
    """
    Sample new solutions from the learned model.
    
    :deblending_model: Neural network learned to deblend solutions
    :param population_size: Number of individuals to sample.
    :return: A numpy array containing the new population.
    """
    selected_population = (selected_population - ranges[0,:]) / (10**(-16) + (ranges[1,:]-ranges[0,:]))
    sel_p_size, n_dim = selected_population.shape    
    selected_indices = np.random.randint(0, high=sel_p_size, size=population_size)
    p1_samples = selected_population[selected_indices, :]

    mean, cov = learn_univariate_gaussian_model(p1_samples)
    p1_samples = sample_from_multivariate_gaussian_model(mean, cov, sel_p_size, diversity=0)   
    
    new_population = iterative_elm_deblending_sampling(deblending_model, p1_samples, num_iterations)    
    for i in range(0,n_dim):
       new_population[:,i] = ranges[0,i] + (ranges[1,i]-ranges[0,i])*new_population[:, i]
        
    return new_population


def learn_p1_univariate_diffusion_model(current_population, selected_population, num_iterations, num_epochs, batch_size, deblending_model):
    """
    Learn a diffusion model to bridge the distribution between current population and selected solutions

    :param current_population: The selected individuals.    
    :param selected_population: The selected individuals.
    :return: Learned model
    """

    ranges=np.vstack((np.min(current_population,0),np.max(current_population,0)))
    current_population =  (current_population - ranges[0,:])  / (10**(-16) + (ranges[1,:]-ranges[0,:]))
    selected_population = (selected_population - ranges[0,:]) / (10**(-16) + (ranges[1,:]-ranges[0,:]))
        
    p_size, n_dim = current_population.shape
    sel_p_size = selected_population.shape[0]
    to_take = p_size*2
    
    p0_indices = np.random.randint(0, high=p_size, size=to_take)
    p0_samples = current_population[p0_indices, :]

    mean, cov = learn_univariate_gaussian_model(selected_population)
    p1_samples = sample_from_multivariate_gaussian_model(mean, cov, to_take, diversity=0)

    
    #p1_samples = find_closest_neighbors(p0_samples, selected_population)
    
    # Input shape is (input_dimension + 1,) to account for concatenation with alpha
    # deblending_model = create_deblending_network((n_dim + 1,))

    # Train the deblending network

    train_deblending_network(deblending_model, p0_samples, p1_samples, num_iterations, num_epochs, batch_size)

    return deblending_model, ranges


def learn_elm_p1_univariate_diffusion_model(current_population, selected_population, num_iterations, num_epochs, batch_size, deblending_model):
    """
    Learn a diffusion model to bridge the distribution between current population and selected solutions

    :param current_population: The selected individuals.    
    :param selected_population: The selected individuals.
    :return: Learned model
    """

    ranges=np.vstack((np.min(current_population,0),np.max(current_population,0)))
    current_population =  (current_population - ranges[0,:])  / (10**(-16) + (ranges[1,:]-ranges[0,:]))
    selected_population = (selected_population - ranges[0,:]) / (10**(-16) + (ranges[1,:]-ranges[0,:]))
        
    p_size, n_dim = current_population.shape
    sel_p_size = selected_population.shape[0]
    to_take = p_size*2
    
    p0_indices = np.random.randint(0, high=p_size, size=to_take)
    p0_samples = current_population[p0_indices, :]

    mean, cov = learn_univariate_gaussian_model(selected_population)
    p1_samples = sample_from_multivariate_gaussian_model(mean, cov, to_take, diversity=0)

    train_elm_deblending_network(deblending_model, p0_samples, p1_samples, num_iterations)

    return deblending_model, ranges


def learn_p01_univariate_diffusion_model(current_population, selected_population, num_iterations, num_epochs, batch_size, deblending_model):
    """
    Learn a diffusion model to bridge the distribution between current population and selected solutions

    :param current_population: The selected individuals.    
    :param selected_population: The selected individuals.
    :return: Learned model
    """

    ranges=np.vstack((np.min(current_population,0),np.max(current_population,0)))
    current_population =  (current_population - ranges[0,:])  / (10**(-16) + (ranges[1,:]-ranges[0,:]))
    selected_population = (selected_population - ranges[0,:]) / (10**(-16) + (ranges[1,:]-ranges[0,:]))
        
    p_size, n_dim = current_population.shape
    sel_p_size = selected_population.shape[0]
    to_take = p_size*2
    
    mean_p0, cov_p0 = learn_univariate_gaussian_model(current_population)
    p0_samples = sample_from_multivariate_gaussian_model(mean_p0, cov_p0, to_take, diversity=0)
    
    mean_p1, cov_p1 = learn_univariate_gaussian_model(selected_population)
    p1_samples = sample_from_multivariate_gaussian_model(mean_p1, cov_p1, to_take, diversity=0)

    
    #p1_samples = find_closest_neighbors(p0_samples, selected_population)
    
    # Input shape is (input_dimension + 1,) to account for concatenation with alpha
    # deblending_model = create_deblending_network((n_dim + 1,))

    # Train the deblending network

    train_deblending_network(deblending_model, p0_samples, p1_samples, num_iterations, num_epochs, batch_size)

    return deblending_model, ranges


def learn_elm_p01_univariate_diffusion_model(current_population, selected_population, num_iterations, num_epochs, batch_size, deblending_model):
    """
    Learn a diffusion model to bridge the distribution between current population and selected solutions

    :param current_population: The selected individuals.    
    :param selected_population: The selected individuals.
    :return: Learned model
    """

    ranges=np.vstack((np.min(current_population,0),np.max(current_population,0)))
    current_population =  (current_population - ranges[0,:])  / (10**(-16) + (ranges[1,:]-ranges[0,:]))
    selected_population = (selected_population - ranges[0,:]) / (10**(-16) + (ranges[1,:]-ranges[0,:]))
        
    p_size, n_dim = current_population.shape
    sel_p_size = selected_population.shape[0]
    to_take = p_size*2
    
    mean_p0, cov_p0 = learn_univariate_gaussian_model(current_population)
    p0_samples = sample_from_multivariate_gaussian_model(mean_p0, cov_p0, to_take, diversity=0)
    
    mean_p1, cov_p1 = learn_univariate_gaussian_model(selected_population)
    p1_samples = sample_from_multivariate_gaussian_model(mean_p1, cov_p1, to_take, diversity=0)
    
  
    train_elm_deblending_network(deblending_model, p0_samples, p1_samples, num_iterations)

    return deblending_model, ranges

def learn_p11_univariate_diffusion_model(current_population, selected_population, num_iterations, num_epochs, batch_size, deblending_model):
    """
    Learn a diffusion model to bridge the distribution between current population and selected solutions

    :param current_population: The selected individuals.    
    :param selected_population: The selected individuals.
    :return: Learned model
    """

    ranges=np.vstack((np.min(current_population,0),np.max(current_population,0)))
    current_population =  (current_population - ranges[0,:])  / (10**(-16) + (ranges[1,:]-ranges[0,:]))
    selected_population = (selected_population - ranges[0,:]) / (10**(-16) + (ranges[1,:]-ranges[0,:]))
        
    p_size, n_dim = current_population.shape
    sel_p_size = selected_population.shape[0]
    to_take = p_size*2


    selected_indices = np.random.randint(0, high=sel_p_size, size=to_take)
    p1_samples = selected_population[selected_indices, :]
    
    mean_p0, cov_p0 = learn_univariate_gaussian_model(selected_population)
    p0_samples = sample_from_multivariate_gaussian_model(mean_p0, cov_p0, to_take, diversity=0)
    
    
    #p1_samples = find_closest_neighbors(p0_samples, selected_population)
    
    # Input shape is (input_dimension + 1,) to account for concatenation with alpha
    # deblending_model = create_deblending_network((n_dim + 1,))

    # Train the deblending network

    train_deblending_network(deblending_model, p0_samples, p1_samples, num_iterations, num_epochs, batch_size)
    return deblending_model, ranges


def learn_elm_p11_univariate_diffusion_model(current_population, selected_population, num_iterations, deblending_model):
    """
    Learn a diffusion model to bridge the distribution between current population and selected solutions

    :param current_population: The selected individuals.    
    :param selected_population: The selected individuals.
    :return: Learned model
    """

    ranges=np.vstack((np.min(current_population,0),np.max(current_population,0)))
    current_population =  (current_population - ranges[0,:])  / (10**(-16) + (ranges[1,:]-ranges[0,:]))
    selected_population = (selected_population - ranges[0,:]) / (10**(-16) + (ranges[1,:]-ranges[0,:]))
        
    p_size, n_dim = current_population.shape
    sel_p_size = selected_population.shape[0]
    to_take = p_size*2


    selected_indices = np.random.randint(0, high=sel_p_size, size=to_take)
    p1_samples = selected_population[selected_indices, :]
    
    mean_p0, cov_p0 = learn_univariate_gaussian_model(selected_population)
    p0_samples = sample_from_multivariate_gaussian_model(mean_p0, cov_p0, to_take, diversity=0)
    
    
    train_elm_deblending_network(deblending_model, p0_samples, p1_samples, num_iterations)
    return deblending_model, ranges

def learn_p00_univariate_diffusion_model(current_population, selected_population, num_iterations, num_epochs, batch_size, deblending_model):
    """
    Learn a diffusion model to bridge the distribution between current population and selected solutions

    :param current_population: The selected individuals.    
    :param selected_population: The selected individuals.
    :return: Learned model
    """

    ranges=np.vstack((np.min(current_population,0),np.max(current_population,0)))
    current_population =  (current_population - ranges[0,:])  / (10**(-16) + (ranges[1,:]-ranges[0,:]))
    selected_population = (selected_population - ranges[0,:]) / (10**(-16) + (ranges[1,:]-ranges[0,:]))
        
    p_size, n_dim = current_population.shape
    to_take = p_size*2


    selected_indices = np.random.randint(0, high=p_size, size=to_take)
    p1_samples = current_population[selected_indices, :]
    
    mean_p0, cov_p0 = learn_univariate_gaussian_model(current_population)
    p0_samples = sample_from_multivariate_gaussian_model(mean_p0, cov_p0, to_take, diversity=0)
    
    
    #p1_samples = find_closest_neighbors(p0_samples, selected_population)
    
    # Input shape is (input_dimension + 1,) to account for concatenation with alpha
    # deblending_model = create_deblending_network((n_dim + 1,))

    # Train the deblending network

    train_deblending_network(deblending_model, p0_samples, p1_samples, num_iterations, num_epochs, batch_size)
    return deblending_model, ranges



def learn_elm_p00_univariate_diffusion_model(current_population, selected_population, num_iterations, deblending_model):
    """
    Learn a diffusion model to bridge the distribution between current population and selected solutions

    :param current_population: The selected individuals.    
    :param selected_population: The selected individuals.
    :return: Learned model
    """

    ranges=np.vstack((np.min(current_population,0),np.max(current_population,0)))
    current_population =  (current_population - ranges[0,:])  / (10**(-16) + (ranges[1,:]-ranges[0,:]))
    selected_population = (selected_population - ranges[0,:]) / (10**(-16) + (ranges[1,:]-ranges[0,:]))
        
    p_size, n_dim = current_population.shape
    to_take = p_size*2


    selected_indices = np.random.randint(0, high=p_size, size=to_take)
    p1_samples = current_population[selected_indices, :]
    
    mean_p0, cov_p0 = learn_univariate_gaussian_model(current_population)
    p0_samples = sample_from_multivariate_gaussian_model(mean_p0, cov_p0, to_take, diversity=0)
    

    train_elm_deblending_network(deblending_model, p0_samples, p1_samples, num_iterations)
    return deblending_model, ranges


def initialize_diffusion_model(n, nas_nn_index, expand_factor):

    learning_rate = 0.0001
    opt_method = 0 # Adam optimizer
    
    the_nas_function = nas_select_function(nas_nn_index)
    deblending_model = the_nas_function((n + 1,), (n,), expand_factor)
    #print(deblending_model.summary())

    mse_loss = tf.keras.losses.MeanSquaredError()
    optimizer = get_optimizer(opt_method, learning_rate)

    
    deblending_model.compile(optimizer=optimizer, loss=mse_loss)

    return deblending_model

def initialize_elm_diffusion_model(number_neurons, nas_nn_index, expand_factor):
    # 'gompertz', 'triangular'
    elm = ELMLayer(number_neurons=number_neurons*expand_factor, activation='triangular')
    deblending_model = ELMModel(elm,classification=False)
    #deblending_model = EHDrELMModel(classification=False,layers=[elm,elm])

    
    return deblending_model
