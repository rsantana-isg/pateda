import numpy as np

def truncation_selection(population, fitness, percent_truncation):
    """
    Select the top individuals from the population based on their fitness.
   

    :param population: The current population.
    :param fitness: The fitness values of the population.
    :param selection_size: Number of individuals to select.
    :return: A numpy array containing the selected individuals.
    """
    selection_size =  int((percent_truncation/100.0)*len(population))
    # Sort the population by fitness and select the top individuals
    sorted_indices = np.argsort(fitness)
    selected_population = population[sorted_indices[:selection_size]]
    selected_fitness = fitness[sorted_indices[:selection_size]]
    return selected_population, selected_fitness



def merge_populations(population, fitness, selected_population, selected_fitness, psize):
    """
    Merges two populations and selects the top `psize` individuals based on fitness.

    Args:
        population (numpy.ndarray): The first population, shape (n, d), where n is the number of individuals and d is the number of dimensions.
        fitness (numpy.ndarray): The fitness values of the first population, shape (n,).
        selected_population (numpy.ndarray): The second population, shape (m, d), where m is the number of individuals and d is the number of dimensions.
        selected_fitness (numpy.ndarray): The fitness values of the second population, shape (m,).
        psize (int): The number of individuals to keep in the merged population.

    Returns:
        merged_population (numpy.ndarray): The merged population, shape (psize, d).
        merged_fitness (numpy.ndarray): The fitness values of the merged population, shape (psize,).
    """
    # Concatenate the two populations and their fitness values
    combined_population = np.vstack((population, selected_population))
    combined_fitness = np.concatenate((fitness, selected_fitness))

    # Sort the combined population based on fitness (ascending order for minimization problems)
    sorted_indices = np.argsort(combined_fitness)
    sorted_population = combined_population[sorted_indices]
    sorted_fitness = combined_fitness[sorted_indices]

    # Select the top `psize` individuals
    merged_population = sorted_population[:psize]
    merged_fitness = sorted_fitness[:psize]

    return merged_population, merged_fitness
