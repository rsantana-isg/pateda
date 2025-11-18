import numpy as np
from sklearn.mixture import GaussianMixture


def weighted_mean_variance(data, weights):
    """
    Compute the weighted mean and weighted variance for each feature in a (N, d) data array.

    Parameters:
    - data: np.ndarray of shape (N, d), where N is the number of samples, d is the number of features.
    - weights: np.ndarray of shape (N,), containing weights for each sample.

    Returns:
    - weighted_mean: np.ndarray of shape (d,), the weighted mean for each feature.
    - weighted_variance: np.ndarray of shape (d,), the weighted variance for each feature.
    """
    weights = weights.reshape(-1, 1)  # Reshape to (N,1) for broadcasting
    sum_weights = np.sum(weights, axis=0)

    # Compute weighted mean
    weighted_mean = np.sum(weights * data, axis=0) / sum_weights

    # Compute weighted variance
    weighted_variance = np.sum(weights * (data - weighted_mean) ** 2, axis=0) / sum_weights

    return weighted_mean, weighted_variance


def learn_univariate_gaussian_model(selected_population):
    """
    Learn a univariate Gaussian model from the selected population.
    
    :param selected_population: The selected individuals.
    :return: Mean vector and covariance matrix of the Gaussian model.
    """
    mean = np.mean(selected_population, axis=0)
    variance = np.var(selected_population,0)
    cov = np.diag(variance)
    
    return mean, cov


def learn_weighted_univariate_gaussian_model(population,fitness):
    """
    Learn a weighted univariate Gaussian model from the selected population.
    
    :param selected_population: The selected individuals.
    :return: Mean vector and covariance matrix of the Gaussian model.
    """
    alpha = 0.0000000001
    beta = 0.1
    
    weights = (fitness-np.min(fitness)+alpha)/(np.max(fitness)-np.min(fitness)+alpha)    
    weights = np.exp(beta*weights)
    weighted_mean, weighted_variance = weighted_mean_variance(population, weights)
    #print(weighted_mean, weighted_variance)
    weighted_cov = np.diag(weighted_variance)
   
    return weighted_mean, weighted_cov


            


def learn_multivariate_gaussian_model(selected_population):
    """
    Learn a multivariate Gaussian model from the selected population.
    
    :param selected_population: The selected individuals.
    :return: Mean vector and covariance matrix of the Gaussian model.
    """
    mean = np.mean(selected_population, axis=0)
    cov = np.cov(selected_population, rowvar=False)
    return mean, cov


def learn_weighted_multivariate_gaussian_model(population,fitness):
    """
    Learn a weighted multivariate Gaussian model from the selected population.
    
    :param selected_population: The selected individuals.
    :return: Mean vector and covariance matrix of the Gaussian model.
    """
    alpha = 0.0000000001
    beta = 0.1
    
    weights = (fitness-np.min(fitness)+alpha)/(np.max(fitness)-np.min(fitness)+alpha)    
    weights = np.exp(beta*weights)
    
    mean = np.average(population, axis=0, weights=weights)
    cov = np.cov(population, rowvar=False, aweights=weights)
    return mean, cov





def sample_from_multivariate_gaussian_model(mean, cov, population_size, diversity):
    """
    Sample new solutions from the learned multivariate Gaussian model.
    
    :param mean: Mean vector of the Gaussian model.
    :param cov: Covariance matrix of the Gaussian model.
    :param population_size: Number of individuals to sample.
    :return: A numpy array containing the new population.
    """
    if diversity<=0:
        return np.random.multivariate_normal(mean, cov, population_size)
    else:
        mean_std = np.mean(np.diag(cov))
        if mean_std<diversity:            
            return np.random.multivariate_normal(mean, (1+mean_std)*cov, population_size) 
        else:
            return np.random.multivariate_normal(mean, cov, population_size)

def learn_gaussian_mixture_model_EM(selected_population, n_components):
    """
    Learn a mixture Gaussian model with n_components from the selected population.
    
    :param selected_population: The selected individuals.
    :n_components: Number of components 
    :return: Gaussian mixture model
    """
    gm = GaussianMixture(n_components=n_components).fit(selected_population)
    
    return gm

def sample_from_gaussian_mixture_model_EM(gm, population_size):
    """
    Sample new solutions from the learned multivariate Gaussian model.
    
    :param gm: Gaussian mixture model learned by EM as represented in sklearn
    :return: A numpy array containing the new population.
    """

    new_pop = gm.sample(population_size)

    return new_pop 
