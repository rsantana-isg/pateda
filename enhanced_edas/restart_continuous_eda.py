import numpy as np
import sys
import copy
from selection_methods import *
from initialization_methods import *
from gaussian_models import *
from copula_models import *
from efficient_diffusion_models import *
from efficient_backdrive_models import *
from vae_models import *
from functions import *


def reinitialize(model):
    for l in model.layers:
        if isinstance(l, tf.keras.Model):
            reinitialize(l)
            continue
        if hasattr(l,"kernel_initializer"):
            l.kernel.assign(l.kernel_initializer(tf.shape(l.kernel)))
        if hasattr(l,"bias_initializer"):
            l.bias.assign(l.bias_initializer(tf.shape(l.bias)))
        if hasattr(l,"recurrent_initializer"):
            l.recurrent_kernel.assign(l.recurrent_initializer(tf.shape(l.recurrent_kernel)))
            
    
def restart_univariate_gaussian_EDA(objective_function, n, lower_bound, upper_bound, population_size, truncation_value, replacement,
                                    max_generations, diversity, proportional, trigger_No_Improvement, diversity_threshold):
  
    not_improved = 0
    best_fitness = 10**10
    to_take = 2
    
    for generation in range(max_generations):     
        if generation==0:               
            population = initialize_population(population_size, n, lower_bound, upper_bound)
            fitness = np.array([objective_function(individual) for individual in population])
        else:
            sel_psize = selected_population.shape[0]                             
            sel_div = np.std(selected_fitness[to_take:])
            if  ( (sel_div<diversity_threshold) or (not_improved==trigger_No_Improvement)):
                not_improved = 0
                
                
                population = initialize_population(population_size, n, lower_bound, upper_bound)
                #aux_population = initialize_population(population_size-int(sel_psize/2), n, lower_bound, upper_bound)
                #population[:population_size-int(sel_psize/2),:] = aux_population
                fitness = np.array([objective_function(individual) for individual in population])                    
                population[:to_take,:] = selected_population[:to_take,:]
                fitness[:to_take] = selected_fitness[:to_take]
            else:                
                new_population = sample_from_multivariate_gaussian_model(mean, cov, population_size, diversity)               
                #for i in range(0,ranges.shape[1]):
                #    new_population[:,i] = ranges[0,i] + (ranges[1,i]-ranges[0,i]+10**(-16))*new_population[:, i]
                
                new_fitness = np.array([objective_function(individual) for individual in new_population])
                population, fitness = merge_populations(population, fitness, new_population, new_fitness,
                                                            population_size)
                    
        selected_population, selected_fitness = truncation_selection(population, fitness, truncation_value)        
        #ranges=np.vstack((np.min(selected_population,0),np.max(selected_population,0)))
        #normalized_selected_population = (selected_population - ranges[0,:]) / (10**(-16)+ (ranges[1,:]-ranges[0,:]))
        
        # Learn multivariate Gaussian model
        if proportional==0:
            mean, cov = learn_univariate_gaussian_model(selected_population)
        else:
            mean, cov = learn_weighted_univariate_gaussian_model(selected_population, selected_fitness)


        pop_best_fitness = selected_fitness[0] 

        if pop_best_fitness<best_fitness:
            #print("difference", best_fitness-pop_best_fitness,  diversity_threshold)
            best_index = np.argmin(fitness)
            best_solution = population[best_index]
            best_fitness = fitness[best_index]
            not_improved = 0
        else:
            not_improved = not_improved + 1
            
        #if generation==(max_generations-1):
        print(f"Generation {generation}: Best Fitness = {best_fitness}, MeanFitness = {np.mean(fitness)}, STD = {np.std(selected_fitness[2:])}, IMP = {not_improved}")          
    
    return best_solution, best_fitness


def restart_BO_univariate_gaussian_EDA(objective_function, n, lower_bound, upper_bound, population_size, truncation_value, replacement,
                                    max_generations, diversity, proportional, trigger_No_Improvement, diversity_threshold):
  
    not_improved = 0
    best_fitness = 10**10
    to_take = 2
    
    
    for generation in range(max_generations):     
        if generation==0:               
            population = initialize_population(population_size, n, lower_bound, upper_bound)
            fitness = np.array([objective_function(individual) for individual in population])
            archive = np.hstack((population[:to_take,:],fitness[:to_take].reshape(to_take,1)))
        else:
            sel_psize = selected_population.shape[0]                             
            sel_div = np.std(selected_fitness[to_take:])
            if  ( (sel_div<diversity_threshold) or (not_improved==trigger_No_Improvement)):
                not_improved = 0
                
                archive = np.vstack((archive,np.hstack((population[2:to_take+2,:],fitness[2:to_take+2].reshape(to_take,1)))))
                #print(archive[:,:-1],archive[:,-1],lower_bound,upper_bound, population_size, population_size*5, 1.0)
                population = BO_initialize_population(archive[:,:-1],archive[:,-1],lower_bound,upper_bound,
                                                      population_size, population_size*10, 1, 0)
    
              
                fitness = np.array([objective_function(individual) for individual in population])                    
                population[:to_take,:] = selected_population[:to_take,:]
                fitness[:to_take] = selected_fitness[:to_take]
            else:                
                new_population = sample_from_multivariate_gaussian_model(mean, cov, population_size, diversity)               
              
                
                new_fitness = np.array([objective_function(individual) for individual in new_population])
                population, fitness = merge_populations(population, fitness, new_population, new_fitness,
                                                            population_size)
                    
        selected_population, selected_fitness = truncation_selection(population, fitness, truncation_value)               
       
        # Learn multivariate Gaussian model
        if proportional==0:
            mean, cov = learn_univariate_gaussian_model(selected_population)
        else:
            mean, cov = learn_weighted_univariate_gaussian_model(selected_population, selected_fitness)


        pop_best_fitness = selected_fitness[0] 

        #if  best_fitness-pop_best_fitness > 100*diversity_threshold: 
        if pop_best_fitness<best_fitness:
            #print("difference", best_fitness-pop_best_fitness,  diversity_threshold)
            best_index = np.argmin(fitness)
            best_solution = population[best_index]
            best_fitness = fitness[best_index]
            not_improved = 0
        else:
            not_improved = not_improved + 1
            
        #if generation==(max_generations-1):
        print(f"Generation {generation}: Best Fitness = {best_fitness}, MeanFitness = {np.mean(fitness)}, STD = {np.std(selected_fitness[2:])}, IMP = {not_improved}")          
    
    return best_solution, best_fitness

def restart_PROPBO_univariate_gaussian_EDA(objective_function, n, lower_bound, upper_bound, population_size, truncation_value, replacement,
                                    max_generations, diversity, proportional, trigger_No_Improvement, diversity_threshold):
  
    not_improved = 0
    best_fitness = 10**10
    to_take = 2
    
    
    for generation in range(max_generations):     
        if generation==0:               
            population = initialize_population(population_size, n, lower_bound, upper_bound)
            fitness = np.array([objective_function(individual) for individual in population])
            archive = np.hstack((population[:to_take,:],fitness[:to_take].reshape(to_take,1)))
        else:
            sel_psize = selected_population.shape[0]                             
            sel_div = np.std(selected_fitness[to_take:])
            if  ( (sel_div<diversity_threshold) or (not_improved==trigger_No_Improvement)):
                not_improved = 0
                
                
                #population = initialize_population(population_size, n, lower_bound, upper_bound)

                archive = np.vstack((archive,np.hstack((population[2:to_take+2,:],fitness[2:to_take+2].reshape(to_take,1)))))
                #print(archive[:,:-1],archive[:,-1],lower_bound,upper_bound, population_size, population_size*5, 1.0)
                population = BO_initialize_population(archive[:,:-1],archive[:,-1],lower_bound,upper_bound,
                                                      population_size, population_size*10, 0.01, 1)
    
                #aux_population = initialize_population(population_size-int(sel_psize/2), n, lower_bound, upper_bound)
                #population[:population_size-int(sel_psize/2),:] = aux_population
                fitness = np.array([objective_function(individual) for individual in population])                    
                population[:to_take,:] = selected_population[:to_take,:]
                fitness[:to_take] = selected_fitness[:to_take]
            else:                
                new_population = sample_from_multivariate_gaussian_model(mean, cov, population_size, diversity)               
                #for i in range(0,ranges.shape[1]):
                #    new_population[:,i] = ranges[0,i] + (ranges[1,i]-ranges[0,i]+10**(-16))*new_population[:, i]
                
                new_fitness = np.array([objective_function(individual) for individual in new_population])
                population, fitness = merge_populations(population, fitness, new_population, new_fitness,
                                                            population_size)
                    
        selected_population, selected_fitness = truncation_selection(population, fitness, truncation_value)        
        #ranges=np.vstack((np.min(selected_population,0),np.max(selected_population,0)))
        #normalized_selected_population = (selected_population - ranges[0,:]) / (10**(-16)+ (ranges[1,:]-ranges[0,:]))

     
        # Learn multivariate Gaussian model
        if proportional==0:
            mean, cov = learn_univariate_gaussian_model(selected_population)
        else:
            mean, cov = learn_weighted_univariate_gaussian_model(selected_population, selected_fitness)


        pop_best_fitness = selected_fitness[0] 

        #if  best_fitness-pop_best_fitness > 100*diversity_threshold: 
        if pop_best_fitness<best_fitness:
            #print("difference", best_fitness-pop_best_fitness,  diversity_threshold)
            best_index = np.argmin(fitness)
            best_solution = population[best_index]
            best_fitness = fitness[best_index]
            not_improved = 0
        else:
            not_improved = not_improved + 1
            
        #if generation==(max_generations-1):
        print(f"Generation {generation}: Best Fitness = {best_fitness}, MeanFitness = {np.mean(fitness)}, STD = {np.std(selected_fitness[2:])}, IMP = {not_improved}")          
    
    return best_solution, best_fitness

def restart_POOL_univariate_gaussian_EDA(objective_function, n, lower_bound, upper_bound, population_size, truncation_value, replacement,
                                    max_generations, diversity, proportional, trigger_No_Improvement, diversity_threshold):
  
    not_improved = 0
    best_fitness = 10**10
    to_take = 2
    
    
    for generation in range(max_generations):     
        if generation==0:               
            population = initialize_population(population_size, n, lower_bound, upper_bound)
            fitness = np.array([objective_function(individual) for individual in population])
           
            archive = np.hstack((population[:to_take:],fitness[:to_take].reshape(to_take,1)))
        else:
            sel_psize = selected_population.shape[0]                             
            sel_div = np.std(selected_fitness[to_take:])
            if  ( (sel_div<diversity_threshold) or (not_improved==trigger_No_Improvement)):
                not_improved = 0
                
                
                aux_population = initialize_population(population_size-2*to_take, n, lower_bound, upper_bound)

                archive = np.vstack((archive,np.hstack((population[2:to_take+2,:],fitness[2:to_take+2].reshape(to_take,1)))))
                #print(archive[:,:-1],archive[:,-1],lower_bound,upper_bound, population_size, population_size*5, 1.0)
                #population = BO_initialize_population(archive[:,:-1],archive[:,-1],lower_bound,upper_bound, population_size, population_size*10, 1, 0)
                
                
                #aux_population = initialize_population(population_size-int(sel_psize/2), n, lower_bound, upper_bound)
                #population[:population_size-int(sel_psize/2),:] = aux_population
                aux_fitness = np.array([objective_function(individual) for individual in aux_population])                    
                population[:to_take,:] = selected_population[:to_take,:]
                fitness[:to_take] = selected_fitness[:to_take]
                to_select = np.random.choice(np.arange(archive.shape[0]), size=to_take)
                population[to_take:2*to_take,:] = archive[to_select,:-1]
                fitness[to_take:2*to_take] = archive[to_select,-1]
                population[2*to_take:,:] = aux_population
                fitness[2*to_take:] = aux_fitness
            else:                
                new_population = sample_from_multivariate_gaussian_model(mean, cov, population_size, diversity)               
                #for i in range(0,ranges.shape[1]):
                #    new_population[:,i] = ranges[0,i] + (ranges[1,i]-ranges[0,i]+10**(-16))*new_population[:, i]
                
                new_fitness = np.array([objective_function(individual) for individual in new_population])
                population, fitness = merge_populations(population, fitness, new_population, new_fitness,
                                                            population_size)
                    
        selected_population, selected_fitness = truncation_selection(population, fitness, truncation_value)        
        #ranges=np.vstack((np.min(selected_population,0),np.max(selected_population,0)))
        #normalized_selected_population = (selected_population - ranges[0,:]) / (10**(-16)+ (ranges[1,:]-ranges[0,:]))
        
        # Learn multivariate Gaussian model
        if proportional==0:
            mean, cov = learn_univariate_gaussian_model(selected_population)
        else:
            mean, cov = learn_weighted_univariate_gaussian_model(selected_population, selected_fitness)


        pop_best_fitness = selected_fitness[0] 

        if pop_best_fitness<best_fitness:
            #print("difference", best_fitness-pop_best_fitness,  diversity_threshold)
            best_index = np.argmin(fitness)
            best_solution = population[best_index]
            best_fitness = fitness[best_index]
            not_improved = 0
        else:
            not_improved = not_improved + 1
            
        #if generation==(max_generations-1):
        print(f"Generation {generation}: Best Fitness = {best_fitness}, MeanFitness = {np.mean(fitness)}, STD = {np.std(selected_fitness[2:])}, IMP = {not_improved}")          
    
    return best_solution, best_fitness




def restart_gaussian_EDA(objective_function, n, lower_bound, upper_bound, population_size, truncation_value, replacement,
                                    max_generations, diversity, proportional, trigger_No_Improvement, diversity_threshold):
  
    not_improved = 0
    best_fitness = 10**10
    to_take = 2
    
    for generation in range(max_generations):     
        if generation==0:               
            population = initialize_population(population_size, n, lower_bound, upper_bound)
            fitness = np.array([objective_function(individual) for individual in population])
        else:
            sel_psize = selected_population.shape[0]                             
            sel_div = np.std(selected_fitness[to_take:])
            if  ( (sel_div<diversity_threshold) or (not_improved==trigger_No_Improvement)):
                not_improved = 0
                
                
                population = initialize_population(population_size, n, lower_bound, upper_bound)
                #aux_population = initialize_population(population_size-int(sel_psize/2), n, lower_bound, upper_bound)
                #population[:population_size-int(sel_psize/2),:] = aux_population
                fitness = np.array([objective_function(individual) for individual in population])                    
                population[:to_take,:] = selected_population[:to_take,:]
                fitness[:to_take] = selected_fitness[:to_take]
            else:                
                new_population = sample_from_multivariate_gaussian_model(mean, cov, population_size, diversity)
                #for i in range(0,ranges.shape[1]):
                #    new_population[:,i] = ranges[0,i] + (ranges[1,i]-ranges[0,i]+10**(-16))*new_population[:, i]
                new_fitness = np.array([objective_function(individual) for individual in new_population])
                population, fitness = merge_populations(population, fitness, new_population, new_fitness,
                                                            population_size)
                    
        selected_population, selected_fitness = truncation_selection(population, fitness, truncation_value)
        #ranges=np.vstack((np.min(selected_population,0),np.max(selected_population,0)))
        #normalized_selected_population = (selected_population - ranges[0,:]) / (10**(-16)+ (ranges[1,:]-ranges[0,:]))


        # Learn multivariate Gaussian model
        if proportional==0:
            mean, cov = learn_multivariate_gaussian_model(selected_population)
        else:
            mean, cov = learn_weighted_multivariate_gaussian_model(selected_population,selected_fitness)


        pop_best_fitness = selected_fitness[0] 

        if pop_best_fitness<best_fitness:
            best_index = np.argmin(fitness)
            best_solution = population[best_index]
            best_fitness = fitness[best_index]
            not_improved = 0
        else:
            not_improved = not_improved + 1
            
        #if generation==(max_generations-1):
        print(f"Generation {generation}: Best Fitness = {best_fitness}, MeanFitness = {np.mean(fitness)}, STD = {np.std(selected_fitness[2:])}, IMP = {not_improved}")          
    
    return best_solution, best_fitness


def restart_BO_gaussian_EDA(objective_function, n, lower_bound, upper_bound, population_size, truncation_value, replacement,
                                    max_generations, diversity, proportional, trigger_No_Improvement, diversity_threshold):
  
    not_improved = 0
    best_fitness = 10**10
    to_take = 2
    
    for generation in range(max_generations):     
        if generation==0:               
            population = initialize_population(population_size, n, lower_bound, upper_bound)
            fitness = np.array([objective_function(individual) for individual in population])
            archive = np.hstack((population[:to_take,:],fitness[:to_take].reshape(to_take,1)))
        else:
            sel_psize = selected_population.shape[0]                             
            sel_div = np.std(selected_fitness[to_take:])
            if  ( (sel_div<diversity_threshold) or (not_improved==trigger_No_Improvement) ):
                not_improved = 0
                
                
                #population = initialize_population(population_size, n, lower_bound, upper_bound)

                archive = np.vstack((archive,np.hstack((population[2:to_take+2,:],fitness[2:to_take+2].reshape(to_take,1)))))
                #print(archive[:,:-1],archive[:,-1],lower_bound,upper_bound, population_size, population_size*5, 1.0)
                population = BO_initialize_population(archive[:,:-1],archive[:,-1],lower_bound,upper_bound,
                                                      population_size, population_size*10, 1, 0)
                
                #aux_population = initialize_population(population_size-int(sel_psize/2), n, lower_bound, upper_bound)
                #population[:population_size-int(sel_psize/2),:] = aux_population
                fitness = np.array([objective_function(individual) for individual in population])                    
                population[:to_take,:] = selected_population[:to_take,:]
                fitness[:to_take] = selected_fitness[:to_take]
            else:                
                new_population = sample_from_multivariate_gaussian_model(mean, cov, population_size, diversity)
                #for i in range(0,ranges.shape[1]):
                #    new_population[:,i] = ranges[0,i] + (ranges[1,i]-ranges[0,i]+10**(-16))*new_population[:, i]
                new_fitness = np.array([objective_function(individual) for individual in new_population])
                population, fitness = merge_populations(population, fitness, new_population, new_fitness,
                                                            population_size)
                    
        selected_population, selected_fitness = truncation_selection(population, fitness, truncation_value)
        #ranges=np.vstack((np.min(selected_population,0),np.max(selected_population,0)))
        #normalized_selected_population = (selected_population - ranges[0,:]) / (10**(-16)+ (ranges[1,:]-ranges[0,:]))

        proportional = 1
        # Learn multivariate Gaussian model
        if proportional==0:
            mean, cov = learn_multivariate_gaussian_model(selected_population)
        else:
            mean, cov = learn_weighted_multivariate_gaussian_model(selected_population,selected_fitness)


        pop_best_fitness = selected_fitness[0] 

        
        if pop_best_fitness<best_fitness:
            best_index = np.argmin(fitness)
            best_solution = population[best_index]
            best_fitness = fitness[best_index]
            not_improved = 0
        else:
            not_improved = not_improved + 1
            
        #if generation==(max_generations-1):
        print(f"Generation {generation}: Best Fitness = {best_fitness}, MeanFitness = {np.mean(fitness)}, STD = {np.std(selected_fitness[2:])}, IMP = {not_improved}")          
    
    return best_solution, best_fitness


def restart_PROPBO_gaussian_EDA(objective_function, n, lower_bound, upper_bound, population_size, truncation_value, replacement,
                                    max_generations, diversity, proportional, trigger_No_Improvement, diversity_threshold):
  
    not_improved = 0
    best_fitness = 10**10
    to_take = 2
    
    for generation in range(max_generations):     
        if generation==0:               
            population = initialize_population(population_size, n, lower_bound, upper_bound)
            fitness = np.array([objective_function(individual) for individual in population])
            archive = np.hstack((population[:to_take,:],fitness[:to_take].reshape(to_take,1)))
        else:
            sel_psize = selected_population.shape[0]                             
            sel_div = np.std(selected_fitness[to_take:])
            if  ( (sel_div<diversity_threshold) or (not_improved==trigger_No_Improvement) ):
                not_improved = 0
                
                
                #population = initialize_population(population_size, n, lower_bound, upper_bound)

                archive = np.vstack((archive,np.hstack((population[2:to_take+2,:],fitness[2:to_take+2].reshape(to_take,1)))))
                #print(archive[:,:-1],archive[:,-1],lower_bound,upper_bound, population_size, population_size*5, 1.0)
                population = BO_initialize_population(archive[:,:-1],archive[:,-1],lower_bound,upper_bound,
                                                      population_size, population_size*10, 0.01, 1)
                
                #aux_population = initialize_population(population_size-int(sel_psize/2), n, lower_bound, upper_bound)
                #population[:population_size-int(sel_psize/2),:] = aux_population
                fitness = np.array([objective_function(individual) for individual in population])                    
                population[:to_take,:] = selected_population[:to_take,:]
                fitness[:to_take] = selected_fitness[:to_take]
            else:                
                new_population = sample_from_multivariate_gaussian_model(mean, cov, population_size, diversity)
                #for i in range(0,ranges.shape[1]):
                #    new_population[:,i] = ranges[0,i] + (ranges[1,i]-ranges[0,i]+10**(-16))*new_population[:, i]
                new_fitness = np.array([objective_function(individual) for individual in new_population])
                population, fitness = merge_populations(population, fitness, new_population, new_fitness,
                                                            population_size)
                    
        selected_population, selected_fitness = truncation_selection(population, fitness, truncation_value)
        #ranges=np.vstack((np.min(selected_population,0),np.max(selected_population,0)))
        #normalized_selected_population = (selected_population - ranges[0,:]) / (10**(-16)+ (ranges[1,:]-ranges[0,:]))

        
        # Learn multivariate Gaussian model
        if proportional==0:
            mean, cov = learn_multivariate_gaussian_model(selected_population)
        else:
            mean, cov = learn_weighted_multivariate_gaussian_model(selected_population,selected_fitness)


        pop_best_fitness = selected_fitness[0] 


        if pop_best_fitness<best_fitness:
            best_index = np.argmin(fitness)
            best_solution = population[best_index]
            best_fitness = fitness[best_index]
            not_improved = 0
        else:
            not_improved = not_improved + 1
            
        #if generation==(max_generations-1):
        print(f"Generation {generation}: Best Fitness = {best_fitness}, MeanFitness = {np.mean(fitness)}, STD = {np.std(selected_fitness[2:])}, IMP = {not_improved}")          
    
    return best_solution, best_fitness


def restart_adaptive_gaussian_UMDA(objective_function, n, lower_bound, upper_bound, population_size, truncation_value, replacement,
                                   max_generations, diversity,  trigger_No_Improvement, diversity_threshold):
    best_solution, best_fitness = restart_univariate_gaussian_EDA(objective_function, n, lower_bound, upper_bound, population_size, truncation_value,
                                                          replacement,max_generations, diversity,0, trigger_No_Improvement, diversity_threshold)
    
    return best_solution, best_fitness



def restart_adaptive_gaussian_EDA(objective_function, n, lower_bound, upper_bound, population_size, truncation_value, replacement,
                                  max_generations, diversity,  trigger_No_Improvement, diversity_threshold):
    best_solution, best_fitness = restart_gaussian_EDA(objective_function, n, lower_bound, upper_bound, population_size, truncation_value,
                                                       replacement,max_generations, diversity,0, trigger_No_Improvement, diversity_threshold)
    return best_solution, best_fitness

 
    



def restart_gaussian_mixture_EM_EDA(objective_function, n, lower_bound, upper_bound, population_size, truncation_value, replacement,
                                    max_generations, n_components,trigger_No_Improvement, diversity_threshold):   
        
    not_improved = 0
    best_fitness = 10**10
    to_take = 2
    
    for generation in range(max_generations):     
        if generation==0:               
            population = initialize_population(population_size, n, lower_bound, upper_bound)
            fitness = np.array([objective_function(individual) for individual in population])
        else:
            sel_psize = selected_population.shape[0]                             
            sel_div = np.std(selected_fitness[to_take:])
            if  ( (sel_div<diversity_threshold) or (not_improved==trigger_No_Improvement)):
                not_improved = 0
                population = initialize_population(population_size, n, lower_bound, upper_bound)
                fitness = np.array([objective_function(individual) for individual in population])                    
                population[:to_take,:] = selected_population[:to_take,:]
                fitness[:to_take] = selected_fitness[:to_take]
            else:               
                
                new_population = sample_from_gaussian_mixture_model_EM(gm, population_size)[0]
                #for i in range(0,ranges.shape[1]):
                #    new_population[:,i] = ranges[0,i] + (ranges[1,i]-ranges[0,i]+10**(-16))*new_population[:, i]
                
                new_fitness = np.array([objective_function(individual) for individual in new_population])
                population, fitness = merge_populations(population, fitness, new_population, new_fitness,
                                                            population_size)
                    
        selected_population, selected_fitness = truncation_selection(population, fitness, truncation_value)        
        #ranges=np.vstack((np.min(selected_population,0),np.max(selected_population,0)))
        #normalized_selected_population = (selected_population - ranges[0,:]) / (10**(-16)+ (ranges[1,:]-ranges[0,:]))
        
        #gm = learn_gaussian_mixture_model_EM(normalized_selected_population, n_components)
        gm = learn_gaussian_mixture_model_EM(selected_population, n_components)
        pop_best_fitness = selected_fitness[0] 

        if pop_best_fitness<best_fitness:
            best_index = np.argmin(fitness)
            best_solution = population[best_index]
            best_fitness = fitness[best_index]
            not_improved = 0
        else:
            not_improved = not_improved + 1
            
        #if generation==(max_generations-1):
        print(f"Generation {generation}: Best Fitness = {best_fitness}, MeanFitness = {np.mean(fitness)}, STD = {np.std(selected_fitness[2:])}, IMP = {not_improved}")          
    
    return best_solution, best_fitness


def restart_focused_diffusion_EDA(objective_function, n, lower_bound, upper_bound, population_size, truncation_value, replacement, max_generations, num_epochs, batch_size, num_iterations, nas_nn_index, expand_factor,trigger_No_Improvement, diversity_threshold):
        
    not_improved = 0
    best_fitness = 10**10
    to_take = 2

    deblending_model = initialize_diffusion_model(n, nas_nn_index, expand_factor)
    
    for generation in range(max_generations):     
        if generation==0:               
            population = initialize_population(population_size, n, lower_bound, upper_bound)
            fitness = np.array([objective_function(individual) for individual in population])
        else:
            sel_psize = selected_population.shape[0]                             
            sel_div = np.std(selected_fitness[to_take:])
            if  ( (sel_div<diversity_threshold) or (not_improved==trigger_No_Improvement)):
                not_improved = 0
                population = initialize_population(population_size, n, lower_bound, upper_bound)
                fitness = np.array([objective_function(individual) for individual in population])                    
                population[:to_take,:] = selected_population[:to_take,:]
                fitness[:to_take] = selected_fitness[:to_take]
            else:               
                
               
                new_population = sample_from_diffusion_model(deblending_model, ranges, population_size, population, num_iterations)
                #for i in range(0,ranges.shape[1]):
                #    new_population[:,i] = ranges[0,i] + (ranges[1,i]-ranges[0,i]+10**(-16))*new_population[:, i]
                
                new_fitness = np.array([objective_function(individual) for individual in new_population])
                population, fitness = merge_populations(population, fitness, new_population, new_fitness,
                                                            population_size)
                    
        selected_population, selected_fitness = truncation_selection(population, fitness, truncation_value)        
        #ranges=np.vstack((np.min(selected_population,0),np.max(selected_population,0)))
        #normalized_selected_population = (selected_population - ranges[0,:]) / (10**(-16)+ (ranges[1,:]-ranges[0,:]))
        
        deblending_model, ranges = learn_focused_diffusion_model(population,selected_population,num_iterations,num_epochs,batch_size, deblending_model)
       

        pop_best_fitness = selected_fitness[0] 

        if pop_best_fitness<best_fitness:
            best_index = np.argmin(fitness)
            best_solution = population[best_index]
            best_fitness = fitness[best_index]
            not_improved = 0
        else:
            not_improved = not_improved + 1
            
        #if generation==(max_generations-1):
        print(f"Generation {generation}: Best Fitness = {best_fitness}, MeanFitness = {np.mean(fitness)}, STD = {np.std(selected_fitness[2:])}, IMP = {not_improved}")          
    
    return best_solution, best_fitness




def restart_p0_univariate_diffusion_EDA(objective_function, n, lower_bound, upper_bound, population_size, truncation_value, replacement, max_generations, num_epochs, batch_size, num_iterations, nas_nn_index, expand_factor, trigger_No_Improvement, diversity_threshold):

    not_improved = 0
    best_fitness = 10**10
    to_take = 2

    deblending_model = initialize_diffusion_model(n, nas_nn_index, expand_factor)
    
    for generation in range(max_generations):     
        if generation==0:               
            population = initialize_population(population_size, n, lower_bound, upper_bound)
            fitness = np.array([objective_function(individual) for individual in population])
        else:
            sel_psize = selected_population.shape[0]                             
            sel_div = np.std(selected_fitness[to_take:])
            if  ( (sel_div<diversity_threshold) or (not_improved==trigger_No_Improvement)):
                not_improved = 0
                population = initialize_population(population_size, n, lower_bound, upper_bound)
                fitness = np.array([objective_function(individual) for individual in population])                    
                population[:to_take,:] = selected_population[:to_take,:]
                fitness[:to_take] = selected_fitness[:to_take]
            else:               
                
                new_population = sample_p1_univariate_diffusion_model(deblending_model, ranges, population_size, population, num_iterations)
                #for i in range(0,ranges.shape[1]):
                #    new_population[:,i] = ranges[0,i] + (ranges[1,i]-ranges[0,i]+10**(-16))*new_population[:, i]
                
                new_fitness = np.array([objective_function(individual) for individual in new_population])
                population, fitness = merge_populations(population, fitness, new_population, new_fitness,
                                                            population_size)
                    
        selected_population, selected_fitness = truncation_selection(population, fitness, truncation_value)        
        #ranges=np.vstack((np.min(selected_population,0),np.max(selected_population,0)))
        #normalized_selected_population = (selected_population - ranges[0,:]) / (10**(-16)+ (ranges[1,:]-ranges[0,:]))
        

        deblending_model, ranges = learn_p0_univariate_diffusion_model(population,selected_population,num_iterations, num_epochs,batch_size, deblending_model)

        pop_best_fitness = selected_fitness[0] 

        if pop_best_fitness<best_fitness:
            best_index = np.argmin(fitness)
            best_solution = population[best_index]
            best_fitness = fitness[best_index]
            not_improved = 0
        else:
            not_improved = not_improved + 1
            
        #if generation==(max_generations-1):
        print(f"Generation {generation}: Best Fitness = {best_fitness}, MeanFitness = {np.mean(fitness)}, STD = {np.std(selected_fitness[2:])}, IMP = {not_improved}")          
    
    return best_solution, best_fitness    
    




def restart_p1_univariate_diffusion_EDA(objective_function, n, lower_bound, upper_bound, population_size, truncation_value, replacement, max_generations, num_epochs, batch_size, num_iterations, nas_nn_index, expand_factor, trigger_No_Improvement, diversity_threshold):

    not_improved = 0
    best_fitness = 10**10
    to_take = 2

    deblending_model = initialize_diffusion_model(n, nas_nn_index, expand_factor)
    
    for generation in range(max_generations):     
        if generation==0:               
            population = initialize_population(population_size, n, lower_bound, upper_bound)
            fitness = np.array([objective_function(individual) for individual in population])
        else:
            sel_psize = selected_population.shape[0]                             
            sel_div = np.std(selected_fitness[to_take:])
            if  ( (sel_div<diversity_threshold) or (not_improved==trigger_No_Improvement)):
                not_improved = 0
                population = initialize_population(population_size, n, lower_bound, upper_bound)
                fitness = np.array([objective_function(individual) for individual in population])                    
                population[:to_take,:] = selected_population[:to_take,:]
                fitness[:to_take] = selected_fitness[:to_take]
            else:                               
                
                new_population = sample_from_diffusion_model(deblending_model, ranges, population_size, population, num_iterations)
                #for i in range(0,ranges.shape[1]):
                #    new_population[:,i] = ranges[0,i] + (ranges[1,i]-ranges[0,i]+10**(-16))*new_population[:, i]
                
                new_fitness = np.array([objective_function(individual) for individual in new_population])
                population, fitness = merge_populations(population, fitness, new_population, new_fitness,
                                                            population_size)
                    
        selected_population, selected_fitness = truncation_selection(population, fitness, truncation_value)        
        #ranges=np.vstack((np.min(selected_population,0),np.max(selected_population,0)))
        #normalized_selected_population = (selected_population - ranges[0,:]) / (10**(-16)+ (ranges[1,:]-ranges[0,:]))
        

        deblending_model, ranges = learn_p1_univariate_diffusion_model(population,selected_population,num_iterations, num_epochs,batch_size, deblending_model)

        pop_best_fitness = selected_fitness[0] 

        if pop_best_fitness<best_fitness:
            best_index = np.argmin(fitness)
            best_solution = population[best_index]
            best_fitness = fitness[best_index]
            not_improved = 0
        else:
            not_improved = not_improved + 1
            
        #if generation==(max_generations-1):
        print(f"Generation {generation}: Best Fitness = {best_fitness}, MeanFitness = {np.mean(fitness)}, STD = {np.std(selected_fitness[2:])}, IMP = {not_improved}")          
    
    return best_solution, best_fitness    



def restart_p01_univariate_diffusion_EDA(objective_function, n, lower_bound, upper_bound, population_size, truncation_value, replacement, max_generations, num_epochs, batch_size, num_iterations, nas_nn_index, expand_factor, trigger_No_Improvement, diversity_threshold):

    not_improved = 0
    best_fitness = 10**10
    to_take = 2

    deblending_model = initialize_diffusion_model(n, nas_nn_index, expand_factor)
    
    for generation in range(max_generations):     
        if generation==0:               
            population = initialize_population(population_size, n, lower_bound, upper_bound)
            fitness = np.array([objective_function(individual) for individual in population])
        else:
            sel_psize = selected_population.shape[0]                             
            sel_div = np.std(selected_fitness[to_take:])
            if  ( (sel_div<diversity_threshold) or (not_improved==trigger_No_Improvement)):
                not_improved = 0
                population = initialize_population(population_size, n, lower_bound, upper_bound)
                fitness = np.array([objective_function(individual) for individual in population])                    
                population[:to_take,:] = selected_population[:to_take,:]
                fitness[:to_take] = selected_fitness[:to_take]
            else:               
                
                new_population = sample_p1_univariate_diffusion_model(deblending_model, ranges, population_size, population, num_iterations)
                #for i in range(0,ranges.shape[1]):
                #    new_population[:,i] = ranges[0,i] + (ranges[1,i]-ranges[0,i]+10**(-16))*new_population[:, i]
                
                new_fitness = np.array([objective_function(individual) for individual in new_population])
                population, fitness = merge_populations(population, fitness, new_population, new_fitness,
                                                            population_size)
                    
        selected_population, selected_fitness = truncation_selection(population, fitness, truncation_value)        
        #ranges=np.vstack((np.min(selected_population,0),np.max(selected_population,0)))
        #normalized_selected_population = (selected_population - ranges[0,:]) / (10**(-16)+ (ranges[1,:]-ranges[0,:]))
        

        deblending_model, ranges = learn_p01_univariate_diffusion_model(population,selected_population, num_iterations, num_epochs,batch_size, deblending_model)

        pop_best_fitness = selected_fitness[0] 

        if pop_best_fitness<best_fitness:
            best_index = np.argmin(fitness)
            best_solution = population[best_index]
            best_fitness = fitness[best_index]
            not_improved = 0
        else:
            not_improved = not_improved + 1
            
        #if generation==(max_generations-1):
        print(f"Generation {generation}: Best Fitness = {best_fitness}, MeanFitness = {np.mean(fitness)}, STD = {np.std(selected_fitness[2:])}, IMP = {not_improved}")          
    
    return best_solution, best_fitness    



def restart_PROPBO_p01_univariate_diffusion_EDA(objective_function, n, lower_bound, upper_bound, population_size, truncation_value, replacement, max_generations, num_epochs, batch_size, num_iterations, nas_nn_index, expand_factor, trigger_No_Improvement, diversity_threshold):

    not_improved = 0
    best_fitness = 10**10
    to_take = 2

    deblending_model = initialize_diffusion_model(n, nas_nn_index, expand_factor)
    
    for generation in range(max_generations):     
        if generation==0:               
            population = initialize_population(population_size, n, lower_bound, upper_bound)
            fitness = np.array([objective_function(individual) for individual in population])
            archive = np.hstack((population[:to_take,:],fitness[:to_take].reshape(to_take,1)))
        else:
            sel_psize = selected_population.shape[0]                             
            sel_div = np.std(selected_fitness[to_take:])
            if  ( (sel_div<diversity_threshold) or (not_improved==trigger_No_Improvement)):
                not_improved = 0
                archive = np.vstack((archive,np.hstack((population[2:to_take+2,:],fitness[2:to_take+2].reshape(to_take,1)))))
                aux_population = BO_initialize_population(archive[:,:-1],archive[:,-1],lower_bound,upper_bound,
                                                      population_size-to_take, population_size*10, 0.01, 1)
    
                aux_fitness = np.array([objective_function(individual) for individual in aux_population])                    
                population[:to_take,:] = selected_population[:to_take,:]
                fitness[:to_take] = selected_fitness[:to_take]
                population[to_take:,:] = aux_population 
                fitness[to_take:] = aux_fitness                

            else:               
                
                new_population = sample_p1_univariate_diffusion_model(deblending_model, ranges, population_size, population, num_iterations)
                #for i in range(0,ranges.shape[1]):
                #    new_population[:,i] = ranges[0,i] + (ranges[1,i]-ranges[0,i]+10**(-16))*new_population[:, i]
                
                new_fitness = np.array([objective_function(individual) for individual in new_population])
                population, fitness = merge_populations(population, fitness, new_population, new_fitness,
                                                            population_size)
                    
        selected_population, selected_fitness = truncation_selection(population, fitness, truncation_value)        
        #ranges=np.vstack((np.min(selected_population,0),np.max(selected_population,0)))
        #normalized_selected_population = (selected_population - ranges[0,:]) / (10**(-16)+ (ranges[1,:]-ranges[0,:]))
        

        deblending_model, ranges = learn_p01_univariate_diffusion_model(population,selected_population, num_iterations, num_epochs,batch_size, deblending_model)

        pop_best_fitness = selected_fitness[0] 

        
        if pop_best_fitness<best_fitness:
            best_index = np.argmin(fitness)
            best_solution = population[best_index]
            best_fitness = fitness[best_index]
            not_improved = 0
        else:
            not_improved = not_improved + 1
            
        #if generation==(max_generations-1):
        print(f"Generation {generation}: Best Fitness = {best_fitness}, MeanFitness = {np.mean(fitness)}, STD = {np.std(selected_fitness[2:])}, IMP = {not_improved}")          
    
    return best_solution, best_fitness    


def restart_elm_PROPBO_p01_univariate_diffusion_EDA(objective_function, n, lower_bound, upper_bound, population_size, truncation_value, replacement, max_generations, num_epochs, batch_size, num_iterations, nas_nn_index, expand_factor, trigger_No_Improvement, diversity_threshold):

    not_improved = 0
    best_fitness = 10**10
    to_take = 2

    hidden_layer_n_neurons = 25
    elm_deblending_model = initialize_elm_diffusion_model(hidden_layer_n_neurons, nas_nn_index, expand_factor)
    
    for generation in range(max_generations):     
        if generation==0:               
            population = initialize_population(population_size, n, lower_bound, upper_bound)
            fitness = np.array([objective_function(individual) for individual in population])
            archive = np.hstack((population[:to_take,:],fitness[:to_take].reshape(to_take,1)))
        else:
            sel_psize = selected_population.shape[0]                             
            sel_div = np.std(selected_fitness[to_take:])
            if  ( (sel_div<diversity_threshold) or (not_improved==trigger_No_Improvement)):
                not_improved = 0
                archive = np.vstack((archive,np.hstack((population[2:to_take+2,:],fitness[2:to_take+2].reshape(to_take,1)))))
                aux_population = BO_initialize_population(archive[:,:-1],archive[:,-1],lower_bound,upper_bound,
                                                      population_size-to_take, population_size*10, 0.01, 1)
    
                aux_fitness = np.array([objective_function(individual) for individual in aux_population])                    
                population[:to_take,:] = selected_population[:to_take,:]
                fitness[:to_take] = selected_fitness[:to_take]
                population[to_take:,:] = aux_population 
                fitness[to_take:] = aux_fitness                

            else:               
                
                new_population = sample_elm_p1_univariate_diffusion_model(elm_deblending_model, ranges, population_size, population, num_iterations)
                #for i in range(0,ranges.shape[1]):
                #    new_population[:,i] = ranges[0,i] + (ranges[1,i]-ranges[0,i]+10**(-16))*new_population[:, i]
                
                new_fitness = np.array([objective_function(individual) for individual in new_population])
                population, fitness = merge_populations(population, fitness, new_population, new_fitness,
                                                            population_size)
                    
        selected_population, selected_fitness = truncation_selection(population, fitness, truncation_value)        
        #ranges=np.vstack((np.min(selected_population,0),np.max(selected_population,0)))
        #normalized_selected_population = (selected_population - ranges[0,:]) / (10**(-16)+ (ranges[1,:]-ranges[0,:]))
        

        elm_deblending_model, ranges = learn_elm_p01_univariate_diffusion_model(population,selected_population, num_iterations, num_epochs,batch_size, elm_deblending_model)

        pop_best_fitness = selected_fitness[0] 

        #if pop_best_fitness<best_fitness:
        if np.abs((best_fitness-pop_best_fitness)/pop_best_fitness) > 10**(-8):                    
            best_index = np.argmin(fitness)
            best_solution = population[best_index]
            best_fitness = fitness[best_index]
            not_improved = 0
        else:
            not_improved = not_improved + 1
            
        #if generation==(max_generations-1):
        print(f"Generation {generation}: Best Fitness = {best_fitness}, MeanFitness = {np.mean(fitness)}, STD = {np.std(selected_fitness[2:])}, IMP = {not_improved}")          
    
    return best_solution, best_fitness




def restart_p11_univariate_diffusion_EDA(objective_function, n, lower_bound, upper_bound, population_size, truncation_value, replacement, max_generations, num_epochs, batch_size, num_iterations, nas_nn_index, expand_factor, trigger_No_Improvement, diversity_threshold):

    not_improved = 0
    best_fitness = 10**10
    to_take = 2

    deblending_model = initialize_diffusion_model(n, nas_nn_index, expand_factor)
    
    for generation in range(max_generations):     
        if generation==0:               
            population = initialize_population(population_size, n, lower_bound, upper_bound)
            fitness = np.array([objective_function(individual) for individual in population])
        else:
            sel_psize = selected_population.shape[0]                             
            sel_div = np.std(selected_fitness[to_take:])
            if  ( (sel_div<diversity_threshold) or (not_improved==trigger_No_Improvement)):
                not_improved = 0
                population = initialize_population(population_size, n, lower_bound, upper_bound)
                fitness = np.array([objective_function(individual) for individual in population])                    
                population[:to_take,:] = selected_population[:to_take,:]
                fitness[:to_take] = selected_fitness[:to_take]
            else:               
                
                new_population = sample_p1_univariate_diffusion_model(deblending_model, ranges, population_size, population, num_iterations)
                #for i in range(0,ranges.shape[1]):
                #    new_population[:,i] = ranges[0,i] + (ranges[1,i]-ranges[0,i]+10**(-16))*new_population[:, i]
                
                new_fitness = np.array([objective_function(individual) for individual in new_population])
                population, fitness = merge_populations(population, fitness, new_population, new_fitness,
                                                            population_size)
                    
        selected_population, selected_fitness = truncation_selection(population, fitness, truncation_value)        
        #ranges=np.vstack((np.min(selected_population,0),np.max(selected_population,0)))
        #normalized_selected_population = (selected_population - ranges[0,:]) / (10**(-16)+ (ranges[1,:]-ranges[0,:]))
        


        deblending_model, ranges = learn_p11_univariate_diffusion_model(population,selected_population, num_iterations, num_epochs,batch_size, deblending_model)

        pop_best_fitness = selected_fitness[0] 

        if pop_best_fitness<best_fitness:
            best_index = np.argmin(fitness)
            best_solution = population[best_index]
            best_fitness = fitness[best_index]
            not_improved = 0
        else:
            not_improved = not_improved + 1
            
        #if generation==(max_generations-1):
        print(f"Generation {generation}: Best Fitness = {best_fitness}, MeanFitness = {np.mean(fitness)}, STD = {np.std(selected_fitness[2:])}, IMP = {not_improved}")          
    
    return best_solution, best_fitness    
    


def restart_PROPBO_p11_univariate_diffusion_EDA(objective_function, n, lower_bound, upper_bound, population_size, truncation_value, replacement, max_generations, num_epochs, batch_size, num_iterations, nas_nn_index, expand_factor, trigger_No_Improvement, diversity_threshold):

    not_improved = 0
    best_fitness = 10**10
    to_take = 2

    deblending_model = initialize_diffusion_model(n, nas_nn_index, expand_factor)
    
    for generation in range(max_generations):     
        if generation==0:               
            population = initialize_population(population_size, n, lower_bound, upper_bound)
            fitness = np.array([objective_function(individual) for individual in population])
            archive = np.hstack((population[:to_take,:],fitness[:to_take].reshape(to_take,1)))
        else:
            sel_psize = selected_population.shape[0]                             
            sel_div = np.std(selected_fitness[to_take:])
            if  ( (sel_div<diversity_threshold) or (not_improved==trigger_No_Improvement)):
                not_improved = 0
                archive = np.vstack((archive,np.hstack((population[2:to_take+2,:],fitness[2:to_take+2].reshape(to_take,1)))))
                aux_population = BO_initialize_population(archive[:,:-1],archive[:,-1],lower_bound,upper_bound,
                                                      population_size-to_take, population_size*10, 0.01, 1)
    
                aux_fitness = np.array([objective_function(individual) for individual in aux_population])                    
                population[:to_take,:] = selected_population[:to_take,:]
                fitness[:to_take] = selected_fitness[:to_take]
                population[to_take:,:] = aux_population 
                fitness[to_take:] = aux_fitness                

            else:               
                
                new_population = sample_p1_univariate_diffusion_model(deblending_model, ranges, population_size, population, num_iterations)

                #for i in range(0,ranges.shape[1]):
                #    new_population[:,i] = ranges[0,i] + (ranges[1,i]-ranges[0,i]+10**(-16))*new_population[:, i]
                
                new_fitness = np.array([objective_function(individual) for individual in new_population])
                population, fitness = merge_populations(population, fitness, new_population, new_fitness,
                                                            population_size)
                    
        selected_population, selected_fitness = truncation_selection(population, fitness, truncation_value)        
        #ranges=np.vstack((np.min(selected_population,0),np.max(selected_population,0)))
        #normalized_selected_population = (selected_population - ranges[0,:]) / (10**(-16)+ (ranges[1,:]-ranges[0,:]))
        

        deblending_model, ranges = learn_p01_univariate_diffusion_model(population,selected_population, num_iterations, num_epochs,batch_size, deblending_model)

        
        pop_best_fitness = selected_fitness[0] 

        if pop_best_fitness<best_fitness:
            best_index = np.argmin(fitness)
            best_solution = population[best_index]
            best_fitness = fitness[best_index]
            not_improved = 0
        else:
            not_improved = not_improved + 1
            
        #if generation==(max_generations-1):
        print(f"Generation {generation}: Best Fitness = {best_fitness}, MeanFitness = {np.mean(fitness)}, STD = {np.std(selected_fitness[2:])}, IMP = {not_improved}")          
    
    return best_solution, best_fitness    



def restart_elm_PROPBO_p11_univariate_diffusion_EDA(objective_function, n, lower_bound, upper_bound, population_size, truncation_value, replacement, max_generations, num_epochs, batch_size, num_iterations, nas_nn_index, expand_factor, trigger_No_Improvement, diversity_threshold):

    not_improved = 0
    best_fitness = 10**10
    to_take = 2

    hidden_layer_n_neurons = 25
    elm_deblending_model = initialize_elm_diffusion_model(hidden_layer_n_neurons, nas_nn_index, expand_factor)

    
    for generation in range(max_generations):     
        if generation==0:               
            population = initialize_population(population_size, n, lower_bound, upper_bound)
            fitness = np.array([objective_function(individual) for individual in population])
            archive = np.hstack((population[:to_take,:],fitness[:to_take].reshape(to_take,1)))
        else:
            sel_psize = selected_population.shape[0]                             
            sel_div = np.std(selected_fitness[to_take:])
            if  ( (sel_div<diversity_threshold) or (not_improved==trigger_No_Improvement)):
                not_improved = 0
                archive = np.vstack((archive,np.hstack((population[2:to_take+2,:],fitness[2:to_take+2].reshape(to_take,1)))))
                aux_population = BO_initialize_population(archive[:,:-1],archive[:,-1],lower_bound,upper_bound,
                                                      population_size-to_take, population_size*10, 0.01, 1)
    
                aux_fitness = np.array([objective_function(individual) for individual in aux_population])                    
                population[:to_take,:] = selected_population[:to_take,:]
                fitness[:to_take] = selected_fitness[:to_take]
                population[to_take:,:] = aux_population 
                fitness[to_take:] = aux_fitness                

            else:               
                
                new_population = sample_elm_p1_univariate_diffusion_model(elm_deblending_model, ranges, population_size, population, num_iterations)
                #for i in range(0,ranges.shape[1]):
                #    new_population[:,i] = ranges[0,i] + (ranges[1,i]-ranges[0,i]+10**(-16))*new_population[:, i]
                
                new_fitness = np.array([objective_function(individual) for individual in new_population])
                population, fitness = merge_populations(population, fitness, new_population, new_fitness,
                                                            population_size)
                    
        selected_population, selected_fitness = truncation_selection(population, fitness, truncation_value)        
        #ranges=np.vstack((np.min(selected_population,0),np.max(selected_population,0)))
        #normalized_selected_population = (selected_population - ranges[0,:]) / (10**(-16)+ (ranges[1,:]-ranges[0,:]))
        

        elm_deblending_model, ranges = learn_elm_p11_univariate_diffusion_model(population,selected_population, num_iterations, elm_deblending_model)


        
        pop_best_fitness = selected_fitness[0] 

        #if pop_best_fitness<best_fitness:
        if np.abs((best_fitness-pop_best_fitness)/pop_best_fitness) > 10**(-8):             
            best_index = np.argmin(fitness)
            best_solution = population[best_index]
            best_fitness = fitness[best_index]             
            not_improved = 0
        else:
            not_improved = not_improved + 1
            
        #if generation==(max_generations-1):
        print(f"Generation {generation}: Best Fitness = {best_fitness}, MeanFitness = {np.mean(fitness)}, STD = {np.std(selected_fitness[2:])}, IMP = {not_improved}")          
    
    return best_solution, best_fitness    



def restart_p00_univariate_diffusion_EDA(objective_function, n, lower_bound, upper_bound, population_size, truncation_value, replacement, max_generations, num_epochs, batch_size, num_iterations, nas_nn_index, expand_factor, trigger_No_Improvement, diversity_threshold):
    not_improved = 0
    best_fitness = 10**10
    to_take = 2

    deblending_model = initialize_diffusion_model(n, nas_nn_index, expand_factor)
    
    for generation in range(max_generations):     
        if generation==0:               
            population = initialize_population(population_size, n, lower_bound, upper_bound)
            fitness = np.array([objective_function(individual) for individual in population])
        else:
            sel_psize = selected_population.shape[0]                             
            sel_div = np.std(selected_fitness[to_take:])
            if  ( (sel_div<diversity_threshold) or (not_improved==trigger_No_Improvement)):
                not_improved = 0
                population = initialize_population(population_size, n, lower_bound, upper_bound)
                fitness = np.array([objective_function(individual) for individual in population])                    
                population[:to_take,:] = selected_population[:to_take,:]
                fitness[:to_take] = selected_fitness[:to_take]
            else:               
                
                new_population = sample_p1_univariate_diffusion_model(deblending_model, ranges, population_size, population, num_iterations)
                #for i in range(0,ranges.shape[1]):
                #    new_population[:,i] = ranges[0,i] + (ranges[1,i]-ranges[0,i]+10**(-16))*new_population[:, i]
                
                new_fitness = np.array([objective_function(individual) for individual in new_population])
                population, fitness = merge_populations(population, fitness, new_population, new_fitness,
                                                            population_size)
                    
        selected_population, selected_fitness = truncation_selection(population, fitness, truncation_value)        
        #ranges=np.vstack((np.min(selected_population,0),np.max(selected_population,0)))
        #normalized_selected_population = (selected_population - ranges[0,:]) / (10**(-16)+ (ranges[1,:]-ranges[0,:]))
        
        deblending_model, ranges = learn_p00_univariate_diffusion_model(population,selected_population, num_iterations, num_epochs,batch_size, deblending_model)

        pop_best_fitness = selected_fitness[0] 

        if pop_best_fitness<best_fitness:
            best_index = np.argmin(fitness)
            best_solution = population[best_index]
            best_fitness = fitness[best_index]
            not_improved = 0
        else:
            not_improved = not_improved + 1
            
        #if generation==(max_generations-1):
        print(f"Generation {generation}: Best Fitness = {best_fitness}, MeanFitness = {np.mean(fitness)}, STD = {np.std(selected_fitness[2:])}, IMP = {not_improved}")          
    
    return best_solution, best_fitness    




def restart_elm_PROPBO_p00_univariate_diffusion_EDA(objective_function, n, lower_bound, upper_bound, population_size, truncation_value, replacement, max_generations, num_epochs, batch_size, num_iterations, nas_nn_index, expand_factor, trigger_No_Improvement, diversity_threshold):

    not_improved = 0
    best_fitness = 10**10
    to_take = 2

    hidden_layer_n_neurons = 25
    elm_deblending_model = initialize_elm_diffusion_model(hidden_layer_n_neurons, nas_nn_index, expand_factor)
    
    for generation in range(max_generations):     
        if generation==0:               
            population = initialize_population(population_size, n, lower_bound, upper_bound)
            fitness = np.array([objective_function(individual) for individual in population])
            archive = np.hstack((population[:to_take,:],fitness[:to_take].reshape(to_take,1)))
        else:
            sel_psize = selected_population.shape[0]                             
            sel_div = np.std(selected_fitness[to_take:])
            if  ( (sel_div<diversity_threshold) or (not_improved==trigger_No_Improvement)):
                not_improved = 0
                archive = np.vstack((archive,np.hstack((population[2:to_take+2,:],fitness[2:to_take+2].reshape(to_take,1)))))
                aux_population = BO_initialize_population(archive[:,:-1],archive[:,-1],lower_bound,upper_bound,
                                                      population_size-to_take, population_size*10, 0.01, 1)
    
                aux_fitness = np.array([objective_function(individual) for individual in aux_population])                    
                population[:to_take,:] = selected_population[:to_take,:]
                fitness[:to_take] = selected_fitness[:to_take]
                population[to_take:,:] = aux_population 
                fitness[to_take:] = aux_fitness                

            else:               
                
                new_population = sample_elm_p1_univariate_diffusion_model(elm_deblending_model, ranges, population_size, population, num_iterations)
     
                #for i in range(0,ranges.shape[1]):
                #    new_population[:,i] = ranges[0,i] + (ranges[1,i]-ranges[0,i]+10**(-16))*new_population[:, i]
                
                new_fitness = np.array([objective_function(individual) for individual in new_population])
                population, fitness = merge_populations(population, fitness, new_population, new_fitness,
                                                            population_size)
                    
        selected_population, selected_fitness = truncation_selection(population, fitness, truncation_value)        
        #ranges=np.vstack((np.min(selected_population,0),np.max(selected_population,0)))
        #normalized_selected_population = (selected_population - ranges[0,:]) / (10**(-16)+ (ranges[1,:]-ranges[0,:]))
        

        elm_deblending_model, ranges = learn_elm_p00_univariate_diffusion_model(population,selected_population, num_iterations, elm_deblending_model)

        pop_best_fitness = selected_fitness[0] 

        #if pop_best_fitness<best_fitness:
        if np.abs((best_fitness-pop_best_fitness)/pop_best_fitness) > 10**(-8): 
            
            best_index = np.argmin(fitness)
            best_solution = population[best_index]
            best_fitness = fitness[best_index]
            not_improved = 0
        else:
            not_improved = not_improved + 1
            
        #if generation==(max_generations-1):
        print(f"Generation {generation}: Best Fitness = {best_fitness}, MeanFitness = {np.mean(fitness)}, STD = {np.std(selected_fitness[2:])}, IMP = {not_improved}")          
    
    return best_solution, best_fitness






def restart_expand_diffusion_EDA(objective_function, n, lower_bound, upper_bound, population_size, truncation_value, replacement, max_generations, num_epochs, batch_size, num_iterations, nas_nn_index, expand_factor, trigger_No_Improvement, diversity_threshold):
    
    not_improved = 0
    best_fitness = 10**10
    to_take = 2

    deblending_model = initialize_diffusion_model(n, nas_nn_index, expand_factor)
    
    for generation in range(max_generations):     
        if generation==0:               
            population = initialize_population(population_size, n, lower_bound, upper_bound)
            fitness = np.array([objective_function(individual) for individual in population])
        else:
            sel_psize = selected_population.shape[0]                             
            sel_div = np.std(selected_fitness[to_take:])
            if  ( (sel_div<diversity_threshold) or (not_improved==trigger_No_Improvement)):
                not_improved = 0
                population = initialize_population(population_size, n, lower_bound, upper_bound)
                fitness = np.array([objective_function(individual) for individual in population])                    
                population[:to_take,:] = selected_population[:to_take,:]
                fitness[:to_take] = selected_fitness[:to_take]
            else:               
                
                new_population = sample_from_diffusion_model(deblending_model, ranges, population_size, population, num_iterations)
                #for i in range(0,ranges.shape[1]):
                #    new_population[:,i] = ranges[0,i] + (ranges[1,i]-ranges[0,i]+10**(-16))*new_population[:, i]
                
                new_fitness = np.array([objective_function(individual) for individual in new_population])
                population, fitness = merge_populations(population, fitness, new_population, new_fitness,
                                                            population_size)
                    
        selected_population, selected_fitness = truncation_selection(population, fitness, truncation_value)        
        #ranges=np.vstack((np.min(selected_population,0),np.max(selected_population,0)))
        #normalized_selected_population = (selected_population - ranges[0,:]) / (10**(-16)+ (ranges[1,:]-ranges[0,:]))
        
        deblending_model, ranges = learn_diffusion_model(population,selected_population, num_iterations, num_epochs,batch_size, deblending_model)

        pop_best_fitness = selected_fitness[0] 

        if pop_best_fitness<best_fitness:
            best_index = np.argmin(fitness)
            best_solution = population[best_index]
            best_fitness = fitness[best_index]
            not_improved = 0
        else:
            not_improved = not_improved + 1
            
        #if generation==(max_generations-1):
        print(f"Generation {generation}: Best Fitness = {best_fitness}, MeanFitness = {np.mean(fitness)}, STD = {np.std(selected_fitness[2:])}, IMP = {not_improved}")          
    
    return best_solution, best_fitness


def restart_elm_expand_diffusion_EDA(objective_function, n, lower_bound, upper_bound, population_size, truncation_value, replacement, max_generations, num_epochs, batch_size, num_iterations, nas_nn_index, expand_factor, trigger_No_Improvement, diversity_threshold):
    
    not_improved = 0
    best_fitness = 10**10
    to_take = 2

   
    hidden_layer_n_neurons = 25
    elm_deblending_model = initialize_elm_diffusion_model(hidden_layer_n_neurons, nas_nn_index, expand_factor)

    
    for generation in range(max_generations):     
        if generation==0:               
            population = initialize_population(population_size, n, lower_bound, upper_bound)
            fitness = np.array([objective_function(individual) for individual in population])
        else:
            sel_psize = selected_population.shape[0]                             
            sel_div = np.std(selected_fitness[to_take:])
            if  ( (sel_div<diversity_threshold) or (not_improved==trigger_No_Improvement)):
                not_improved = 0
                population = initialize_population(population_size, n, lower_bound, upper_bound)
                fitness = np.array([objective_function(individual) for individual in population])                    
                population[:to_take,:] = selected_population[:to_take,:]
                fitness[:to_take] = selected_fitness[:to_take]
            else:               
                
                new_population = sample_from_elm_diffusion_model(elm_deblending_model, ranges, population_size, population, num_iterations)
                #for i in range(0,ranges.shape[1]):
                #    new_population[:,i] = ranges[0,i] + (ranges[1,i]-ranges[0,i]+10**(-16))*new_population[:, i]
                
                new_fitness = np.array([objective_function(individual) for individual in new_population])
                population, fitness = merge_populations(population, fitness, new_population, new_fitness,
                                                            population_size)
                    
        selected_population, selected_fitness = truncation_selection(population, fitness, truncation_value)        
        #ranges=np.vstack((np.min(selected_population,0),np.max(selected_population,0)))
        #normalized_selected_population = (selected_population - ranges[0,:]) / (10**(-16)+ (ranges[1,:]-ranges[0,:]))
        
        elm_deblending_model, ranges = learn_elm_diffusion_model(population,selected_population, num_iterations, elm_deblending_model)

        pop_best_fitness = selected_fitness[0] 

        #if pop_best_fitness<best_fitness:
        if np.abs((best_fitness-pop_best_fitness)/pop_best_fitness) > 10**(-8):             
            best_index = np.argmin(fitness)
            best_solution = population[best_index]
            best_fitness = fitness[best_index]
            not_improved = 0
        else:
            not_improved = not_improved + 1
            
        #if generation==(max_generations-1):
        print(f"Generation {generation}: Best Fitness = {best_fitness}, MeanFitness = {np.mean(fitness)}, STD = {np.std(selected_fitness[2:])}, IMP = {not_improved}")          
    
    return best_solution, best_fitness



def restart_expand_backdrive_EDA(objective_function, n, lower_bound, upper_bound, population_size, truncation_value, replacement,
                                 max_generations, num_epochs, batch_size, opt_method, nas_nn_index, expand_factor, trigger_No_Improvement,
                                 diversity_threshold):   
    
    not_improved = 0
    best_fitness = 10**10
    to_take = 2

    backdrive_model = initialize_backdrive_model(n, nas_nn_index, expand_factor)
    
    for generation in range(max_generations):     
        if generation==0:               
            population = initialize_population(population_size, n, lower_bound, upper_bound)
            fitness = np.array([objective_function(individual) for individual in population])
        else:
            sel_psize = selected_population.shape[0]                             
            sel_div = np.std(selected_fitness[to_take:])
            if  ( (sel_div<diversity_threshold) or (not_improved==trigger_No_Improvement)):
                not_improved = 0
                population = initialize_population(population_size, n, lower_bound, upper_bound)
                fitness = np.array([objective_function(individual) for individual in population])                    
                population[:to_take,:] = selected_population[:to_take,:]
                fitness[:to_take] = selected_fitness[:to_take]
            else:               
                new_population = sample_from_backdrive_model(backdrive_model, mean, cov, ranges, population_size)
                
                #for i in range(0,ranges.shape[1]):
                #    new_population[:,i] = ranges[0,i] + (ranges[1,i]-ranges[0,i]+10**(-16))*new_population[:, i]
                
                new_fitness = np.array([objective_function(individual) for individual in new_population])
                population, fitness = merge_populations(population, fitness, new_population, new_fitness,
                                                            population_size)
                    
        selected_population, selected_fitness = truncation_selection(population, fitness, truncation_value)        
        #ranges=np.vstack((np.min(selected_population,0),np.max(selected_population,0)))
        #normalized_selected_population = (selected_population - ranges[0,:]) / (10**(-16)+ (ranges[1,:]-ranges[0,:]))
        
        
        backdrive_model, mean, cov, ranges = learn_backdrive_model(selected_population, selected_fitness, num_epochs, batch_size, backdrive_model, opt_method)
        
        pop_best_fitness = selected_fitness[0] 

        if pop_best_fitness<best_fitness:
            best_index = np.argmin(fitness)
            best_solution = population[best_index]
            best_fitness = fitness[best_index]
            not_improved = 0
        else:
            not_improved = not_improved + 1
            
        #if generation==(max_generations-1):
        print(f"Generation {generation}: Best Fitness = {best_fitness}, MeanFitness = {np.mean(fitness)}, STD = {np.std(selected_fitness[2:])}, IMP = {not_improved}")          
    
    return best_solution, best_fitness


def restart_expand_backdrive_EDA(objective_function, n, lower_bound, upper_bound, population_size, truncation_value, replacement,
                                 max_generations, num_epochs, batch_size, opt_method, nas_nn_index, expand_factor, trigger_No_Improvement,
                                 diversity_threshold):   
    
    not_improved = 0
    best_fitness = 10**10
    to_take = 2

    backdrive_model = initialize_backdrive_model(n, nas_nn_index, expand_factor)
    
    for generation in range(max_generations):     
        if generation==0:               
            population = initialize_population(population_size, n, lower_bound, upper_bound)
            fitness = np.array([objective_function(individual) for individual in population])
        else:
            sel_psize = selected_population.shape[0]                             
            sel_div = np.std(selected_fitness[to_take:])
            if  ( (sel_div<diversity_threshold) or (not_improved==trigger_No_Improvement)):
                not_improved = 0
                population = initialize_population(population_size, n, lower_bound, upper_bound)
                fitness = np.array([objective_function(individual) for individual in population])                    
                population[:to_take,:] = selected_population[:to_take,:]
                fitness[:to_take] = selected_fitness[:to_take]
            else:               
                new_population = sample_from_backdrive_model(backdrive_model, mean, cov, ranges, population_size)
                
                #for i in range(0,ranges.shape[1]):
                #    new_population[:,i] = ranges[0,i] + (ranges[1,i]-ranges[0,i]+10**(-16))*new_population[:, i]
                
                new_fitness = np.array([objective_function(individual) for individual in new_population])
                population, fitness = merge_populations(population, fitness, new_population, new_fitness,
                                                            population_size)
                    
        selected_population, selected_fitness = truncation_selection(population, fitness, truncation_value)        
        #ranges=np.vstack((np.min(selected_population,0),np.max(selected_population,0)))
        #normalized_selected_population = (selected_population - ranges[0,:]) / (10**(-16)+ (ranges[1,:]-ranges[0,:]))
        
        
        backdrive_model, mean, cov, ranges = learn_backdrive_model(selected_population, selected_fitness, num_epochs, batch_size, backdrive_model, opt_method)
        
        pop_best_fitness = selected_fitness[0] 

        if pop_best_fitness<best_fitness:
            best_index = np.argmin(fitness)
            best_solution = population[best_index]
            best_fitness = fitness[best_index]
            not_improved = 0
        else:
            not_improved = not_improved + 1
            
        #if generation==(max_generations-1):
        print(f"Generation {generation}: Best Fitness = {best_fitness}, MeanFitness = {np.mean(fitness)}, STD = {np.std(selected_fitness[2:])}, IMP = {not_improved}")          
    
    return best_solution, best_fitness



def restart_elm_expand_backdrive_EDA(objective_function, n, lower_bound, upper_bound, population_size, truncation_value, replacement,
                                 max_generations, num_epochs, batch_size, opt_method, nas_nn_index, expand_factor, trigger_No_Improvement,
                                 diversity_threshold):   
    
    not_improved = 0
    best_fitness = 10**10
    to_take = 2

    elm_backdrive_model = initialize_backdrive_model(n, nas_nn_index, expand_factor)

    hidden_layer_n_neurons = 25
    elm_backdrive_model = initialize_elm_diffusion_model(hidden_layer_n_neurons, nas_nn_index, expand_factor)
    
    for generation in range(max_generations):     
        if generation==0:               
            population = initialize_population(population_size, n, lower_bound, upper_bound)
            fitness = np.array([objective_function(individual) for individual in population])
        else:
            sel_psize = selected_population.shape[0]                             
            sel_div = np.std(selected_fitness[to_take:])
            if  ( (sel_div<diversity_threshold) or (not_improved==trigger_No_Improvement)):
                not_improved = 0
                population = initialize_population(population_size, n, lower_bound, upper_bound)
                fitness = np.array([objective_function(individual) for individual in population])                    
                population[:to_take,:] = selected_population[:to_take,:]
                fitness[:to_take] = selected_fitness[:to_take]
            else:               
                new_population = sample_from_backdrive_model(elm_backdrive_model, mean, cov, ranges, population_size)
                
                #for i in range(0,ranges.shape[1]):
                #    new_population[:,i] = ranges[0,i] + (ranges[1,i]-ranges[0,i]+10**(-16))*new_population[:, i]
                
                new_fitness = np.array([objective_function(individual) for individual in new_population])
                population, fitness = merge_populations(population, fitness, new_population, new_fitness,
                                                            population_size)
                    
        selected_population, selected_fitness = truncation_selection(population, fitness, truncation_value)        
        #ranges=np.vstack((np.min(selected_population,0),np.max(selected_population,0)))
        #normalized_selected_population = (selected_population - ranges[0,:]) / (10**(-16)+ (ranges[1,:]-ranges[0,:]))
        
        
        elm_backdrive_model, mean, cov, ranges = learn_elm_backdrive_model(selected_population, selected_fitness, elm_backdrive_model)
        
        pop_best_fitness = selected_fitness[0] 

        if pop_best_fitness<best_fitness:
            best_index = np.argmin(fitness)
            best_solution = population[best_index]
            best_fitness = fitness[best_index]
            not_improved = 0
        else:
            not_improved = not_improved + 1
            
        #if generation==(max_generations-1):
        print(f"Generation {generation}: Best Fitness = {best_fitness}, MeanFitness = {np.mean(fitness)}, STD = {np.std(selected_fitness[2:])}, IMP = {not_improved}")          
    
    return best_solution, best_fitness


def backdrive_EDA(objective_function, n, lower_bound, upper_bound, population_size, truncation_value, replacement,
                  max_generations, num_epochs, batch_size, opt_method):
    """
    Backdrive EDA for optimizing an n-dimensional continuous function.
    
    :param objective_function: The objective function to minimize.
    :param n: Number of dimensions (variables) in each solution.
    :param lower_bound: Lower bound of the search space.
    :param upper_bound: Upper bound of the search space.
    :param population_size: Number of individuals in the population.
    :param selection_size: Number of individuals to select in each generation.
    :param max_generations: Maximum number of generations.
    :param num_epochs: Number of epochs for the neural network diffusion model
    :param batch_size: Batchsize for training the neural network diffusion model
    :
    :return: The best solution found and its fitness.
    """
    # Step 1: Initialize population
    

    
    model = create_network((4,),n)
    for generation in range(max_generations):        

        if generation==0:
            #  Initialize population
            population = initialize_population(population_size, n, lower_bound, upper_bound)
            fitness = np.array([objective_function(individual) for individual in population])
        else:
            #  Sample new population
            #aux_population = initialize_population(population_size, n, lower_bound, upper_bound)
            population = sample_from_backdrive_model(model, mean, cov, ranges, population_size)
            fitness = np.array([objective_function(individual) for individual in population])
            if replacement==1:
                population, fitness = merge_populations(population, fitness, selected_population, selected_fitness, population_size)
            

        # Truncation Selection        
        selected_population, selected_fitness = truncation_selection(population, fitness, truncation_value)       

        # Learning the model
        reinitialize(model)
        model, mean, cov, ranges = learn_backdrive_model(selected_population, selected_fitness, num_epochs, batch_size,
                                                         model, opt_method)
        
        
        # Print best fitness in current generation
        best_fitness = np.min(fitness)
        #print(population)
        print(f"Generation {generation}: Best Fitness = {best_fitness}, MeanFitness = {np.mean(fitness)}")
    
    # Return the best solution found
    best_index = np.argmin(fitness)
    best_solution = population[best_index]
    best_fitness = fitness[best_index]
    
    return best_solution, best_fitness



def vae_EDA(objective_function, n, lower_bound, upper_bound, population_size, truncation_value, replacement,
                  max_generations, num_epochs, batch_size, latent_dim):
    """
    Variational Autoencoder EDA for optimizing an n-dimensional continuous function.
    
    :param objective_function: The objective function to minimize.
    :param n: Number of dimensions (variables) in each solution.
    :param lower_bound: Lower bound of the search space.
    :param upper_bound: Upper bound of the search space.
    :param population_size: Number of individuals in the population.
    :param selection_size: Number of individuals to select in each generation.
    :param max_generations: Maximum number of generations.
    :param num_epochs: Number of epochs for the neural network diffusion model
    :param batch_size: Batchsize for training the neural network diffusion model
    :param num_iterations: Number of iterations for the diffusion sampling process [steps for bridging distributions]
    :
    :return: The best solution found and its fitness.
    """
    # Step 1: Initialize population
    

    
    deblending_model = initialize_diffusion_model(n, nas_nn_index, expand_factor)
    
    for generation in range(max_generations):

        if generation==0:
            #  Initialize population
            population = initialize_population(population_size, n, lower_bound, upper_bound)
            fitness = np.array([objective_function(individual) for individual in population])
        else:
            #  Sample new population
            #aux_population = initialize_population(population_size, n, lower_bound, upper_bound)
            population = sample_from_vae_model(vae_decoder, ranges, population_size, latent_dim)
            fitness = np.array([objective_function(individual) for individual in population])
            if replacement==1:
                population, fitness = merge_populations(population, fitness, selected_population, selected_fitness, population_size)
            

        # Truncation Selection        
        selected_population, selected_fitness = truncation_selection(population, fitness, truncation_value)       

        # Learning the model        
        vae_decoder, ranges = learn_vae_model(selected_population,num_epochs,batch_size, latent_dim)
        
        
        # Print best fitness in current generation
        best_fitness = np.min(fitness)
        #print(population)
        print(f"Generation {generation}: Best Fitness = {best_fitness}, MeanFitness = {np.mean(fitness)}")
    
    # Return the best solution found
    best_index = np.argmin(fitness)
    best_solution = population[best_index]
    best_fitness = fitness[best_index]
    
    return best_solution, best_fitness



def copula_EDA(objective_function, n, lower_bound, upper_bound, population_size, truncation_value, replacement,
                 max_generations, copula_function, vine_level):
    """
    Gaussian EDA for optimizing an n-dimensional continuous function.
    
    :param objective_function: The objective function to minimize.
    :param n: Number of dimensions (variables) in each solution.
    :param lower_bound: Lower bound of the search space.
    :param upper_bound: Upper bound of the search space.
    :param population_size: Number of individuals in the population.
    :param selection_size: Number of individuals to select in each generation.
    :param max_generations: Maximum number of generations.
    :return: The best solution found and its fitness.
    """
        
    for generation in range(max_generations):
        if generation==0:
            #  Initialize population
            population = initialize_population(population_size, n, lower_bound, upper_bound)
            fitness = np.array([objective_function(individual) for individual in population])
        else:
            #  Sample new population
           
            population = sample_from_vine_model(cop, population_size, ranges)
            fitness = np.array([objective_function(individual) for individual in population])
            if replacement==1:
                population, fitness = merge_populations(population, fitness, selected_population, selected_fitness, population_size)

        # Truncation Selection        
        selected_population, selected_fitness = truncation_selection(population, fitness, truncation_value)       
               
        
        # Learn Vine-Copula model
        cop, ranges = learn_vine_model(selected_population, copula_function, vine_level)       
        

        # Print best fitness in current generation
        best_fitness = np.min(fitness)
        #print(f"Generation {generation}: Best Fitness = {best_fitness}, MeanFitness = {np.mean(fitness)}")
    
    # Return the best solution found
    best_index = np.argmin(fitness)
    best_solution = population[best_index]
    best_fitness = fitness[best_index]
    
    return best_solution, best_fitness

def diffusion_EDA(objective_function, n, lower_bound, upper_bound, population_size, truncation_value, replacement,
                  max_generations, num_epochs, batch_size, num_iterations):
    """
    Diffusion EDA for optimizing an n-dimensional continuous function.
    
    :param objective_function: The objective function to minimize.
    :param n: Number of dimensions (variables) in each solution.
    :param lower_bound: Lower bound of the search space.
    :param upper_bound: Upper bound of the search space.
    :param population_size: Number of individuals in the population.
    :param selection_size: Number of individuals to select in each generation.
    :param max_generations: Maximum number of generations.
    :param num_epochs: Number of epochs for the neural network diffusion model
    :param batch_size: Batchsize for training the neural network diffusion model
    :param num_iterations: Number of iterations for the diffusion sampling process [steps for bridging distributions]
    :
    :return: The best solution found and its fitness.
    """
    # Step 1: Initialize population
    

    
    deblending_model = initialize_diffusion_model(n, nas_nn_index, expand_factor)
    
    for generation in range(max_generations):

        if generation==0:
            #  Initialize population
            population = initialize_population(population_size, n, lower_bound, upper_bound)
            fitness = np.array([objective_function(individual) for individual in population])
        else:
            #  Sample new population
            #aux_population = initialize_population(population_size, n, lower_bound, upper_bound)
            population = sample_from_diffusion_model(deblending_model, ranges, population_size, population, num_iterations)
            fitness = np.array([objective_function(individual) for individual in population])
            if replacement==1:
                population, fitness = merge_populations(population, fitness, selected_population, selected_fitness, population_size)
            

        # Truncation Selection        
        selected_population, selected_fitness = truncation_selection(population, fitness, truncation_value)       

        # Learning the model        
        deblending_model, ranges = learn_diffusion_model(population,selected_population,num_iterations, num_epochs,batch_size, deblending_model)
        
        
        # Print best fitness in current generation
        best_fitness = np.min(fitness)
        #print(population)
        print(f"Generation {generation}: Best Fitness = {best_fitness}, MeanFitness = {np.mean(fitness)}")
    
    # Return the best solution found
    best_index = np.argmin(fitness)
    best_solution = population[best_index]
    best_fitness = fitness[best_index]
    
    return best_solution, best_fitness



def expand_diffusion_EDA(objective_function, n, lower_bound, upper_bound, population_size, truncation_value, replacement, max_generations, num_epochs, batch_size, num_iterations, nas_nn_index, expand_factor):
    """
    Diffusion EDA for optimizing an n-dimensional continuous function.
    
    :param objective_function: The objective function to minimize.
    :param n: Number of dimensions (variables) in each solution.
    :param lower_bound: Lower bound of the search space.
    :param upper_bound: Upper bound of the search space.
    :param population_size: Number of individuals in the population.
    :param selection_size: Number of individuals to select in each generation.
    :param max_generations: Maximum number of generations.
    :param num_epochs: Number of epochs for the neural network diffusion model
    :param batch_size: Batchsize for training the neural network diffusion model
    :param num_iterations: Number of iterations for the diffusion sampling process [steps for bridging distributions]
    :
    :return: The best solution found and its fitness.
    """
    # Step 1: Initialize population
    
    deblending_model = initialize_diffusion_model(n, nas_nn_index, expand_factor)
    
    for generation in range(max_generations):

        if generation==0:
            #  Initialize population
            population = initialize_population(population_size, n, lower_bound, upper_bound)
            fitness = np.array([objective_function(individual) for individual in population])
        else:
            #  Sample new population
            #aux_population = initialize_population(population_size, n, lower_bound, upper_bound)
            population = sample_from_diffusion_model(deblending_model, ranges, population_size, population, num_iterations)
            fitness = np.array([objective_function(individual) for individual in population])
            if replacement==1:
                population, fitness = merge_populations(population, fitness, selected_population, selected_fitness, population_size)
            

        # Truncation Selection        
        selected_population, selected_fitness = truncation_selection(population, fitness, truncation_value)       

        # Learning the model
        #reinitialize(deblending_model)
        deblending_model, ranges = learn_diffusion_model(population,selected_population,num_iterations, num_epochs,batch_size, deblending_model)
        
        
        # Print best fitness in current generation
        best_fitness = np.min(fitness)
        #print(population)
        print(f"Generation {generation}: Best Fitness = {best_fitness}, MeanFitness = {np.mean(fitness)}")
    
    # Return the best solution found
    best_index = np.argmin(fitness)
    best_solution = population[best_index]
    best_fitness = fitness[best_index]
    
    return best_solution, best_fitness            


def focused_diffusion_EDA(objective_function, n, lower_bound, upper_bound, population_size, truncation_value, replacement, max_generations, num_epochs, batch_size, num_iterations, nas_nn_index, expand_factor):
    """
    Diffusion EDA for optimizing an n-dimensional continuous function.
    
    :param objective_function: The objective function to minimize.
    :param n: Number of dimensions (variables) in each solution.
    :param lower_bound: Lower bound of the search space.
    :param upper_bound: Upper bound of the search space.
    :param population_size: Number of individuals in the population.
    :param selection_size: Number of individuals to select in each generation.
    :param max_generations: Maximum number of generations.
    :param num_epochs: Number of epochs for the neural network diffusion model
    :param batch_size: Batchsize for training the neural network diffusion model
    :param num_iterations: Number of iterations for the diffusion sampling process [steps for bridging distributions]
    :
    :return: The best solution found and its fitness.
    """
    # Step 1: Initialize population
    
    
    deblending_model = initialize_diffusion_model(n, nas_nn_index, expand_factor)
    
    #print(deblending_model.summary())
    
    for generation in range(max_generations):

        if generation==0:
            #  Initialize population
            population = initialize_population(population_size, n, lower_bound, upper_bound)
            fitness = np.array([objective_function(individual) for individual in population])
        else:
            #  Sample new population
            #aux_population = initialize_population(population_size, n, lower_bound, upper_bound)
            population = sample_from_diffusion_model(deblending_model, ranges, population_size, population, num_iterations)
            fitness = np.array([objective_function(individual) for individual in population])
            if replacement==1:
                population, fitness = merge_populations(population, fitness, selected_population, selected_fitness, population_size)
            

        # Truncation Selection        
        selected_population, selected_fitness = truncation_selection(population, fitness, truncation_value)       

        # Learning the model
        #reinitialize(deblending_model)
        deblending_model, ranges = learn_focused_diffusion_model(population,selected_population,num_epochs,batch_size, deblending_model)
        
        
        # Print best fitness in current generation
        best_fitness = np.min(fitness)
        #print(population)
        print(f"Generation {generation}: Best Fitness = {best_fitness}, MeanFitness = {np.mean(fitness)}")
    
    # Return the best solution found
    best_index = np.argmin(fitness)
    best_solution = population[best_index]
    best_fitness = fitness[best_index]
    
    return best_solution, best_fitness


def p0_univariate_diffusion_EDA(objective_function, n, lower_bound, upper_bound, population_size, truncation_value, replacement, max_generations, num_epochs, batch_size, num_iterations, nas_nn_index, expand_factor):
    """
    Diffusion EDA for optimizing an n-dimensional continuous function.
    
    :param objective_function: The objective function to minimize.
    :param n: Number of dimensions (variables) in each solution.
    :param lower_bound: Lower bound of the search space.
    :param upper_bound: Upper bound of the search space.
    :param population_size: Number of individuals in the population.
    :param selection_size: Number of individuals to select in each generation.
    :param max_generations: Maximum number of generations.
    :param num_epochs: Number of epochs for the neural network diffusion model
    :param batch_size: Batchsize for training the neural network diffusion model
    :param num_iterations: Number of iterations for the diffusion sampling process [steps for bridging distributions]
    :
    :return: The best solution found and its fitness.
    """
    # Step 1: Initialize population
    

    deblending_model = initialize_diffusion_model(n, nas_nn_index, expand_factor)
    #print(deblending_model.summary())
    
    for generation in range(max_generations):

        if generation==0:
            #  Initialize population
            population = initialize_population(population_size, n, lower_bound, upper_bound)
            fitness = np.array([objective_function(individual) for individual in population])
        else:
            #  Sample new population
            #aux_population = initialize_population(population_size, n, lower_bound, upper_bound)
            population = sample_p1_univariate_diffusion_model(deblending_model, ranges, population_size, population, num_iterations)
            fitness = np.array([objective_function(individual) for individual in population])
            if replacement==1:
                population, fitness = merge_populations(population, fitness, selected_population, selected_fitness, population_size)
            

        # Truncation Selection        
        selected_population, selected_fitness = truncation_selection(population, fitness, truncation_value)       

        # Learning the model
        #reinitialize(deblending_model)
        deblending_model, ranges = learn_p0_univariate_diffusion_model(population,selected_population, num_iterations, num_epochs,batch_size, deblending_model)
        
        
        # Print best fitness in current generation
        best_fitness = np.min(fitness)
        #print(population)
        print(f"Generation {generation}: Best Fitness = {best_fitness}, MeanFitness = {np.mean(fitness)}")
    
    # Return the best solution found
    best_index = np.argmin(fitness)
    best_solution = population[best_index]
    best_fitness = fitness[best_index]
    
    return best_solution, best_fitness





def p1_univariate_diffusion_EDA(objective_function, n, lower_bound, upper_bound, population_size, truncation_value, replacement, max_generations, num_epochs, batch_size, num_iterations, nas_nn_index, expand_factor):
    """
    Diffusion EDA for optimizing an n-dimensional continuous function.
    
    :param objective_function: The objective function to minimize.
    :param n: Number of dimensions (variables) in each solution.
    :param lower_bound: Lower bound of the search space.
    :param upper_bound: Upper bound of the search space.
    :param population_size: Number of individuals in the population.
    :param selection_size: Number of individuals to select in each generation.
    :param max_generations: Maximum number of generations.
    :param num_epochs: Number of epochs for the neural network diffusion model
    :param batch_size: Batchsize for training the neural network diffusion model
    :param num_iterations: Number of iterations for the diffusion sampling process [steps for bridging distributions]
    :
    :return: The best solution found and its fitness.
    """
    # Step 1: Initialize population
    

    deblending_model = initialize_diffusion_model(n, nas_nn_index, expand_factor)
    #print(deblending_model.summary())
    
    for generation in range(max_generations):

        if generation==0:
            #  Initialize population
            population = initialize_population(population_size, n, lower_bound, upper_bound)
            fitness = np.array([objective_function(individual) for individual in population])
        else:
            #  Sample new population
            #aux_population = initialize_population(population_size, n, lower_bound, upper_bound)
            population = sample_from_diffusion_model(deblending_model, ranges, population_size, population, num_iterations)
            fitness = np.array([objective_function(individual) for individual in population])
            if replacement==1:
                population, fitness = merge_populations(population, fitness, selected_population, selected_fitness, population_size)
            

        # Truncation Selection        
        selected_population, selected_fitness = truncation_selection(population, fitness, truncation_value)       

        # Learning the model
        #reinitialize(deblending_model)
        deblending_model, ranges = learn_p1_univariate_diffusion_model(population,selected_population,num_iterations, num_epochs,batch_size, deblending_model)
        
        
        # Print best fitness in current generation
        best_fitness = np.min(fitness)
        #print(population)
        print(f"Generation {generation}: Best Fitness = {best_fitness}, MeanFitness = {np.mean(fitness)}")
    
    # Return the best solution found
    best_index = np.argmin(fitness)
    best_solution = population[best_index]
    best_fitness = fitness[best_index]
    
    return best_solution, best_fitness



def p01_univariate_diffusion_EDA(objective_function, n, lower_bound, upper_bound, population_size, truncation_value, replacement, max_generations, num_epochs, batch_size, num_iterations, nas_nn_index, expand_factor):
    """
    Diffusion EDA for optimizing an n-dimensional continuous function.
    
    :param objective_function: The objective function to minimize.
    :param n: Number of dimensions (variables) in each solution.
    :param lower_bound: Lower bound of the search space.
    :param upper_bound: Upper bound of the search space.
    :param population_size: Number of individuals in the population.
    :param selection_size: Number of individuals to select in each generation.
    :param max_generations: Maximum number of generations.
    :param num_epochs: Number of epochs for the neural network diffusion model
    :param batch_size: Batchsize for training the neural network diffusion model
    :param num_iterations: Number of iterations for the diffusion sampling process [steps for bridging distributions]
    :
    :return: The best solution found and its fitness.
    """
    # Step 1: Initialize population
    

    deblending_model = initialize_diffusion_model(n, nas_nn_index, expand_factor)
    #print(deblending_model.summary())
    
    for generation in range(max_generations):

        if generation==0:
            #  Initialize population
            population = initialize_population(population_size, n, lower_bound, upper_bound)
            fitness = np.array([objective_function(individual) for individual in population])
        else:
            #  Sample new population
            #aux_population = initialize_population(population_size, n, lower_bound, upper_bound)
            population = sample_p1_univariate_diffusion_model(deblending_model, ranges, population_size, population, num_iterations)
            fitness = np.array([objective_function(individual) for individual in population])
            if replacement==1:
                population, fitness = merge_populations(population, fitness, selected_population, selected_fitness, population_size)
            

        # Truncation Selection        
        selected_population, selected_fitness = truncation_selection(population, fitness, truncation_value)       

        # Learning the model
        #reinitialize(deblending_model)
        deblending_model, ranges = learn_p01_univariate_diffusion_model(population,selected_population, num_iterations, num_epochs,batch_size, deblending_model)

        ranges=np.vstack((np.min(population,0),np.max(population,0)))
        
        # Print best fitness in current generation
        best_fitness = np.min(fitness)
        #print(population)
        print(f"Generation {generation}: Best Fitness = {best_fitness}, MeanFitness = {np.mean(fitness)}")
    
    # Return the best solution found
    best_index = np.argmin(fitness)
    best_solution = population[best_index]
    best_fitness = fitness[best_index]
    
    return best_solution, best_fitness




def p11_univariate_diffusion_EDA(objective_function, n, lower_bound, upper_bound, population_size, truncation_value, replacement, max_generations, num_epochs, batch_size, num_iterations, nas_nn_index, expand_factor):
    """
    Diffusion EDA for optimizing an n-dimensional continuous function.
    
    :param objective_function: The objective function to minimize.
    :param n: Number of dimensions (variables) in each solution.
    :param lower_bound: Lower bound of the search space.
    :param upper_bound: Upper bound of the search space.
    :param population_size: Number of individuals in the population.
    :param selection_size: Number of individuals to select in each generation.
    :param max_generations: Maximum number of generations.
    :param num_epochs: Number of epochs for the neural network diffusion model
    :param batch_size: Batchsize for training the neural network diffusion model
    :param num_iterations: Number of iterations for the diffusion sampling process [steps for bridging distributions]
    :
    :return: The best solution found and its fitness.
    """
    # Step 1: Initialize population
    

    deblending_model = initialize_diffusion_model(n, nas_nn_index, expand_factor)
    #print(deblending_model.summary())
    
    for generation in range(max_generations):

        if generation==0:
            #  Initialize population
            population = initialize_population(population_size, n, lower_bound, upper_bound)
            fitness = np.array([objective_function(individual) for individual in population])
        else:
            #  Sample new population
            #aux_population = initialize_population(population_size, n, lower_bound, upper_bound)
            population = sample_p1_univariate_diffusion_model(deblending_model, ranges, population_size, population, num_iterations)
            fitness = np.array([objective_function(individual) for individual in population])
            if replacement==1:
                population, fitness = merge_populations(population, fitness, selected_population, selected_fitness, population_size)
            

        # Truncation Selection        
        selected_population, selected_fitness = truncation_selection(population, fitness, truncation_value)       

        # Learning the model
        #reinitialize(deblending_model)
        deblending_model, ranges = learn_p11_univariate_diffusion_model(population,selected_population, num_iterations, num_epochs,batch_size, deblending_model)
        
        
        # Print best fitness in current generation
        best_fitness = np.min(fitness)
        #print(population)
        print(f"Generation {generation}: Best Fitness = {best_fitness}, MeanFitness = {np.mean(fitness)}")
    
    # Return the best solution found
    best_index = np.argmin(fitness)
    best_solution = population[best_index]
    best_fitness = fitness[best_index]



def p00_univariate_diffusion_EDA(objective_function, n, lower_bound, upper_bound, population_size, truncation_value, replacement, max_generations, num_epochs, batch_size, num_iterations, nas_nn_index, expand_factor):
    """
    Diffusion EDA for optimizing an n-dimensional continuous function.
    
    :param objective_function: The objective function to minimize.
    :param n: Number of dimensions (variables) in each solution.
    :param lower_bound: Lower bound of the search space.
    :param upper_bound: Upper bound of the search space.
    :param population_size: Number of individuals in the population.
    :param selection_size: Number of individuals to select in each generation.
    :param max_generations: Maximum number of generations.
    :param num_epochs: Number of epochs for the neural network diffusion model
    :param batch_size: Batchsize for training the neural network diffusion model
    :param num_iterations: Number of iterations for the diffusion sampling process [steps for bridging distributions]
    :
    :return: The best solution found and its fitness.
    """
    # Step 1: Initialize population
    

    deblending_model = initialize_diffusion_model(n, nas_nn_index, expand_factor)
    #print(deblending_model.summary())
    
    for generation in range(max_generations):

        if generation==0:
            #  Initialize population
            population = initialize_population(population_size, n, lower_bound, upper_bound)
            fitness = np.array([objective_function(individual) for individual in population])
        else:
            #  Sample new population
            #aux_population = initialize_population(population_size, n, lower_bound, upper_bound)
            population = sample_p1_univariate_diffusion_model(deblending_model, ranges, population_size, population, num_iterations)
            fitness = np.array([objective_function(individual) for individual in population])
            if replacement==1:
                population, fitness = merge_populations(population, fitness, selected_population, selected_fitness, population_size)
            

        # Truncation Selection        
        selected_population, selected_fitness = truncation_selection(population, fitness, truncation_value)       

        # Learning the model
        #reinitialize(deblending_model)
        deblending_model, ranges = learn_p00_univariate_diffusion_model(population,selected_population, num_iterations, num_epochs,batch_size, deblending_model)
        
        
        # Print best fitness in current generation
        best_fitness = np.min(fitness)
        #print(population)
        print(f"Generation {generation}: Best Fitness = {best_fitness}, MeanFitness = {np.mean(fitness)}")
    
    # Return the best solution found
    best_index = np.argmin(fitness)







def expand_large_backdrive_EDA(objective_function, n, lower_bound, upper_bound, population_size, truncation_value, replacement,
                         max_generations, num_epochs, batch_size, opt_method, nas_nn_index, expand_factor):
    """
    Backdrive EDA for optimizing an n-dimensional continuous function.
    
    :param objective_function: The objective function to minimize.
    :param n: Number of dimensions (variables) in each solution.
    :param lower_bound: Lower bound of the search space.
    :param upper_bound: Upper bound of the search space.
    :param population_size: Number of individuals in the population.
    :param selection_size: Number of individuals to select in each generation.
    :param max_generations: Maximum number of generations.
    :param num_epochs: Number of epochs for the neural network diffusion model
    :param batch_size: Batchsize for training the neural network diffusion model
    :
    :return: The best solution found and its fitness.
    """
    # Step 1: Initialize population

      

    the_nas_function = nas_select_function(nas_nn_index)
    model = the_nas_function((5,), (n,), expand_factor)
    learning_rate = 0.0001
    mse_loss = tf.keras.losses.MeanSquaredError()
    optimizer = get_optimizer(opt_method, learning_rate)
    
    
    model.compile(optimizer=optimizer, loss=mse_loss)

    
    for generation in range(max_generations):
        with tf.device('cpu'):
            if generation==0:
                #  Initialize population
                population = initialize_population(population_size, n, lower_bound, upper_bound)
                fitness = np.array([objective_function(individual) for individual in population])
            else:
                #  Sample new population
                sel_psize = selected_population.shape[0]
                small_set = int(sel_psize)
                #small_set = population_size
            
             
                new_population = sample_from_backdrive_model(model, mean, cov, ranges, small_set)
                new_fitness = np.array([objective_function(individual) for individual in new_population])
            
                #print(new_fitness)   
                if replacement==1:
                    population, fitness = merge_populations(population, fitness, new_population[0:small_set,:], new_fitness[0:small_set],
                                                            population_size)
                else:
                    #print("Data", new_population.shape, len(new_fitness), small_set)                
                    population = new_population
                    fitness = new_fitness
                    population[0,:] = best_solution
                    fitness[0] = best_fitness
                    #print(selected_fitness)                

                # Truncation Selection        
            selected_population, selected_fitness = truncation_selection(population, fitness, truncation_value)       

        #Learning the model
        #reinitialize(model)
        tf.keras.backend.clear_session()
        model, mean, cov, ranges = learn_large_backdrive_model(selected_population, selected_fitness, num_epochs, batch_size,
                                                         model, opt_method)

        with tf.device('cpu'):
            ranges=np.vstack((np.min(population,0),np.max(population,0)))
        
            # Print best fitness in current generation

            # Return the best solution found
            best_index = np.argmin(fitness)
            best_solution = population[best_index]
            best_fitness = fitness[best_index]

            #print(population)
            print(f"Generation {generation}: Best Fitness = {best_fitness}, MeanFitness = {np.mean(fitness)}")
    
        
    
    return best_solution, best_fitness


def expand_referential_backdrive_EDA(objective_function, n, lower_bound, upper_bound, population_size, truncation_value, replacement,
                         max_generations, num_epochs, batch_size, opt_method, nas_nn_index, expand_factor):
    """
    Backdrive EDA for optimizing an n-dimensional continuous function.
    
    :param objective_function: The objective function to minimize.
    :param n: Number of dimensions (variables) in each solution.
    :param lower_bound: Lower bound of the search space.
    :param upper_bound: Upper bound of the search space.
    :param population_size: Number of individuals in the population.
    :param selection_size: Number of individuals to select in each generation.
    :param max_generations: Maximum number of generations.
    :param num_epochs: Number of epochs for the neural network diffusion model
    :param batch_size: Batchsize for training the neural network diffusion model
    :
    :return: The best solution found and its fitness.
    """
    # Step 1: Initialize population

      
    
    the_nas_function = nas_select_function(nas_nn_index)
    model = the_nas_function((7,), (n,), expand_factor)
    learning_rate = 0.0001
    mse_loss = tf.keras.losses.MeanSquaredError()
    optimizer = get_optimizer(opt_method, learning_rate)
    
    
    model.compile(optimizer=optimizer, loss=mse_loss)

    diversity_threshold = 5*10**(-6)
    not_improved = 0
    best_fitness = 10**8
    trigger_No_Improvement = 40
    for generation in range(max_generations):
        with tf.device('cpu'):
            if generation==0:
                #  Initialize population
                population = initialize_population(population_size, n, lower_bound, upper_bound)
                fitness = np.array([objective_function(individual) for individual in population])
            else:
                #  Sample new population
                sel_psize = selected_population.shape[0]
                small_set = int(sel_psize*0.8)
              
            
             
                new_population = sample_from_backdrive_model(model, mean, cov, ranges, small_set)
                new_fitness = np.array([objective_function(individual) for individual in new_population])
            
                #print(new_fitness)   
                if replacement==1:
                    population, fitness = merge_populations(population, fitness, new_population[0:small_set,:], new_fitness[0:small_set],
                                                            population_size)
                else:
                    #print("Data", new_population.shape, len(new_fitness), small_set)                
                    population = new_population
                    fitness = new_fitness
                    population[0,:] = best_solution
                    fitness[0] = best_fitness
                    #print(selected_fitness)                

                # Truncation Selection        
            selected_population, selected_fitness = truncation_selection(population, fitness, truncation_value)

            sel_div = np.std(selected_fitness[2:])
            #print(sel_div,diversity_threshold)
            #print(selected_fitness)
            if  ( (sel_div<diversity_threshold) or (not_improved==trigger_No_Improvement)):
                not_improved = 0
                population = initialize_population(population_size, n, lower_bound, upper_bound)
                fitness = np.array([objective_function(individual) for individual in population])
                population[:2,:] = selected_population[:2,:]
                fitness[:2] = selected_fitness[:2]
                selected_population, selected_fitness = truncation_selection(population, fitness, truncation_value)
                

        #Learning the model
        #reinitialize(model)
        tf.keras.backend.clear_session()
        model, mean, cov, ranges = learn_referential_backdrive_model(selected_population, selected_fitness, num_epochs, batch_size,
                                                         model, opt_method)

        with tf.device('cpu'):
            ranges=np.vstack((np.min(population,0),np.max(population,0)))
        
            # Print best fitness in current generation

            # Return the best solution found
            
            pop_best_fitness = selected_fitness[0] 

            if pop_best_fitness<best_fitness:
                best_index = np.argmin(fitness)
                best_solution = population[best_index]
                best_fitness = fitness[best_index]
                not_improved = 0
            else:
                not_improved = not_improved + 1


            #print(population)
            print(f"Generation {generation}: Best Fitness = {best_fitness}, MeanFitness = {np.mean(fitness)}, STD = {np.std(selected_fitness[2:])}, IMP = {not_improved}")
    
         
    
    return best_solution, best_fitness       



def expand_copula_EDA(objective_function, n, lower_bound, upper_bound, population_size, truncation_value, replacement,
                      max_generations, copula_function, vine_level,trigger_No_Improvement, diversity_threshold):


    not_improved = 0
    best_fitness = 10**10
    to_take = 2
    
    for generation in range(max_generations):     
        if generation==0:               
            population = initialize_population(population_size, n, lower_bound, upper_bound)
            fitness = np.array([objective_function(individual) for individual in population])
        else:
            sel_psize = selected_population.shape[0]                             
            sel_div = np.std(selected_fitness[to_take:])
            if  ( (sel_div<diversity_threshold) or (not_improved==trigger_No_Improvement)):
                not_improved = 0
                
                aux_population = initialize_population(population_size-int(sel_psize/2), n, lower_bound, upper_bound)
                #population = initialize_population(population_size, n, lower_bound, upper_bound)
                population[:population_size-int(sel_psize/2),:] = aux_population
                fitness = np.array([objective_function(individual) for individual in population])                    
                population[:to_take,:] = selected_population[:to_take,:]
                fitness[:to_take] = selected_fitness[:to_take]
            else:
                new_population = sample_from_vine_model(cop, population_size, ranges)
                new_fitness = np.array([objective_function(individual) for individual in new_population])
                population, fitness = merge_populations(population, fitness, new_population, new_fitness,
                                                            population_size)
                    
        selected_population, selected_fitness = truncation_selection(population, fitness, truncation_value)        
        cop, ranges = learn_vine_model(selected_population, copula_function, vine_level) 

        pop_best_fitness = selected_fitness[0] 

        if pop_best_fitness<best_fitness:
            best_index = np.argmin(fitness)
            best_solution = population[best_index]
            best_fitness = fitness[best_index]
            not_improved = 0
        else:
            not_improved = not_improved + 1
            
        #if generation==(max_generations-1):
        print(f"Generation {generation}: Best Fitness = {best_fitness}, MeanFitness = {np.mean(fitness)}, STD = {np.std(selected_fitness[2:])}, IMP = {not_improved}")          
    
    return best_solution, best_fitness

    

def obj_expand_copula_EDA(objective_function, n, lower_bound, upper_bound, population_size, truncation_value, replacement,
                 max_generations, copula_function, vine_level,trigger_No_Improvement, diversity_threshold):
  
    not_improved = 0
    best_fitness = 10**10
    to_take = 2
    
    for generation in range(max_generations):     
        if generation==0:               
            population = initialize_population(population_size, n, lower_bound, upper_bound)
            fitness = np.array([objective_function(individual) for individual in population])
        else:
            sel_psize = selected_population.shape[0]                             
            sel_div = np.std(selected_fitness[to_take:])
            if  ( (sel_div<diversity_threshold) or (not_improved==trigger_No_Improvement)):
                not_improved = 0
                population = initialize_population(population_size, n, lower_bound, upper_bound)
                fitness = np.array([objective_function(individual) for individual in population])                    
                population[:to_take,:] = selected_population[:to_take,:]
                fitness[:to_take] = selected_fitness[:to_take]
            else:
                new_population = sample_from_vine_model(cop, population_size, ranges)[:,1:]               
                new_fitness = np.array([objective_function(individual) for individual in new_population])
                population, fitness = merge_populations(population, fitness, new_population, new_fitness,
                                                            population_size)
                    
        selected_population, selected_fitness = truncation_selection(population, fitness, truncation_value)        
        obj_enlarged_pop = np.hstack((selected_fitness.reshape(-1,1),selected_population))
        cop, ranges = learn_vine_model(obj_enlarged_pop, copula_function, vine_level) 

        pop_best_fitness = selected_fitness[0] 

        if pop_best_fitness<best_fitness:
            best_index = np.argmin(fitness)
            best_solution = population[best_index]
            best_fitness = fitness[best_index]
            not_improved = 0
        else:
            not_improved = not_improved + 1
            
        #if generation==(max_generations-1):
        print(f"Generation {generation}: Best Fitness = {best_fitness}, MeanFitness = {np.mean(fitness)}, STD = {np.std(selected_fitness[2:])}, IMP = {not_improved}")          
    
    return best_solution, best_fitness   
