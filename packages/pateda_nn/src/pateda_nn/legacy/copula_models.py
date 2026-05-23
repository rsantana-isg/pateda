import numpy as np
import pyvinecopulib as pv
import copy

copulas = [pv.BicopFamily.gaussian,
               pv.BicopFamily.gumbel,
               pv.BicopFamily.frank,
               pv.BicopFamily.joe,
               pv.BicopFamily.indep,
               pv.BicopFamily.clayton,
               pv.BicopFamily.bb1,
               pv.BicopFamily.bb6,
               pv.BicopFamily.bb7,
               pv.BicopFamily.bb8,
               pv.BicopFamily.tll
               ]
           

def learn_vine_model(selected_population, copula_function, vine_level):
    """
    Learn a vine copula model from the selected population.
    
    :param selected_population: The selected individuals.
    :return: Mean vector and covariance matrix of the Gaussian model.
    """

    

    if  copula_function<=10: 
         copula_family = copulas[copula_function]  # cvines  single family         
    elif copula_function<=21 :
         copula_family = copulas[copula_function-11] # normal vines  single family


    ranges=np.vstack((np.min(selected_population,0),np.max(selected_population,0)))
    u = pv.to_pseudo_obs(selected_population)
    n = selected_population.shape[1]
    
    
    #controls = pv.FitControlsVinecop(family_set=[pv.BicopFamily.gaussian])
    #self.cop = pv.Vinecop(self.u, controls=controls)
    
    if  copula_function<=10:
        cvine = pv.CVineStructure([i+1 for i  in range(n) ])
        #cop = pv.Vinecop(data=u,structure=cvine,controls=pv.FitControlsVinecop(family_set=[copula_family]))    
        cop = pv.Vinecop.from_data(data=u,structure=cvine,controls=pv.FitControlsVinecop(family_set=[copula_family]))                 
    elif copula_function<22:  
        cop = pv.Vinecop.from_data(data=u,controls=pv.FitControlsVinecop(family_set=[copula_family]))               
    elif copula_function==22:
        cop = pv.Vinecop.from_data(data=u,controls=pv.FitControlsVinecop(family_set=copulas))


    cop.truncate(vine_level)

    
          
    #if verbosity>1:
      #print("cop:",cop)
    
    return cop, ranges


def sample_from_vine_model(cop, population_size, ranges):
    """
    Sample new solutions from the learned vine copula model.
    
    :param cop: Vine copula learned from data
    :return: A numpy array containing the new population.
    """

    #print("Simulating uniform", pv.simulate_uniform(10, 2, False, [1, 2]))
    

    n = len(cop.order)
    u = np.random.random_sample((population_size,n))      
    #u_sim = cop.inverse_rosenblatt(u)
    u_sim = cop.simulate(n=population_size)
        
    
    new_population = copy.copy(u_sim)

    for i in range(0,n):
       new_population[:,i] = ranges[0,i] + (ranges[1,i]-ranges[0,i])*u_sim[:, i]
    
              
    return new_population        


def biased_sample_from_vine_model(cop, population_size, ranges, exploit):
    """
    Sample new solutions from the learned vine copula model.
    
    :param cop: Vine copula learned from data
    :return: A numpy array containing the new population.
    """

    #print("Simulating uniform", pv.simulate_uniform(10, 2, False, [1, 2]))
    

    n = len(cop.order)
    u = np.random.random_sample((population_size,n))
    u[:,0] = u[:,0]/10.0  
    u_sim = cop.inverse_rosenblatt(u)
    #u_sim = cop.simulate(n=population_size)
        
    
    new_population = copy.copy(u_sim)

    for i in range(0,n):
       new_population[:,i] = ranges[0,i] + (ranges[1,i]-ranges[0,i])*u_sim[:, i]
    
              
    return new_population       
    
