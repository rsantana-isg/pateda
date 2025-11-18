import numpy as np
import sys
from selection_methods import *
from initialization_methods import *
from gaussian_models import *
from diffusion_models import *
from functions import *
from restart_continuous_eda import *
from GNBG_class import GNBG
import scipy
from scipy.io import loadmat
#import matplotlib.pyplot as plt



def fitness_function(x):
    x_as_matrix = x.reshape(1,len(x))
    f = gnbg.fitness(x_as_matrix)
    #print("f shape", f.shape)
    return f[0]

if __name__ == '__main__':
    seed = int(sys.argv[1])  
    n = int(sys.argv[2])      # Number of variables  
    funct  = int(sys.argv[3])       # Function to optimize       
    type_eda  = int(sys.argv[4])     # Type of EDA
                                     #  0: "Gaussian"
                                     #  1: "Diffusion"
    population_size = int(sys.argv[5]) 
    truncation_value = int(sys.argv[6])    # Percent of truncation
    max_generations = int(sys.argv[7]) # Number of the EA generations
    diversity = int(sys.argv[8])      # Whether the standard deviation is expanded to infuse diversity
    copula_function = int(sys.argv[9]) # Copula function
    trigger_No_Improvement = int(sys.argv[10])       # Id of the network architecture to use for expand_diffusion 

    

    #all_functions = [sphere_function, ellipsoidal_function, schaffers_f7_function]     
    #the_function = all_functions[funct]
    
    np.random.seed(seed)
    replacement = 1  
    n_expe = 1
    vine_level = diversity
    #nas_index = 12
    nas_index = 16    
    #diversity_threshold = 5*10**(-9)
    #diversity_threshold = 5*(10**(-3)) 
    diversity_threshold = 1*(10**(-8))
    
    if type_eda==0:
        fmin = univariate_gaussian_EDA
        alg_name = 'Gaussian_UMDA'
    elif type_eda==1:
        fmin = gaussian_EDA
        alg_name = 'Gaussian_EDA'
    elif type_eda==2:
        fmin = adaptive_gaussian_UMDA
        alg_name = 'Adapt_G_UMDA'
    elif type_eda==3:
        fmin = adaptive_gaussian_EDA
        alg_name = 'Adapt_G_EDA'
    elif type_eda==4:
        fmin = diffusion_EDA
        alg_name = 'diffusion_EDA_I'+str(copula_function)
    elif type_eda==5:
        fmin = copula_EDA        
        alg_name = 'copula_EDA_'+str(copula_function)
    elif type_eda==6:
        fmin = gaussian_mixture_EM_EDA        
        alg_name = 'GaussianMixture_EDA_'+str(copula_function)
    elif type_eda==7:
        fmin = univariate_gaussian_EDA
        alg_name = 'Proportional_Gaussian_UMDA'
    elif type_eda==8:
        fmin = gaussian_EDA
        alg_name = 'Proportional_Gaussian_EDA'
    elif type_eda==9:
        fmin = vae_EDA
        alg_name = 'vae_EDA'
    elif type_eda==10:
        fmin = backdrive_EDA
        alg_name = 'backdrive_EDA_O'+str(copula_function)
    elif type_eda==11:
        fmin = expand_diffusion_EDA
        alg_name = 'expand_diffusion_EDA_I'+str(copula_function)+'_X'+str(nas_index)
    elif type_eda==12:
        fmin = focused_diffusion_EDA
        alg_name = 'focused_diffusion_EDA_I'+str(copula_function)+'_X'+str(nas_index)
    elif type_eda==13:
        fmin = p0_univariate_diffusion_EDA
        alg_name = 'p0_univariate_diffusion_EDA_I'+str(copula_function)+'_X'+str(nas_index)
    elif type_eda==14:
        fmin = p1_univariate_diffusion_EDA
        alg_name = 'p1_univariate_diffusion_EDA_I'+str(copula_function)+'_X'+str(nas_index)
    elif type_eda==15:
        fmin = p01_univariate_diffusion_EDA
        alg_name = 'p01_univariate_diffusion_EDA_I'+str(copula_function)+'_X'+str(nas_index)
    elif type_eda==16:
        fmin = p11_univariate_diffusion_EDA
        alg_name = 'p11_univariate_diffusion_EDA_I'+str(copula_function)+'_X'+str(nas_index)         
    elif type_eda==17:
        fmin = p00_univariate_diffusion_EDA
        alg_name = 'p00_univariate_diffusion_EDA_I'+str(copula_function)+'_X'+str(nas_index)
    elif type_eda==20:
        fmin = expand_backdrive_EDA
        alg_name = 'expand_backdrive_EDA_I'+str(copula_function)+'_X'+str(nas_index)
    elif type_eda==21:
        fmin = expand_large_backdrive_EDA
        alg_name = 'expand_large_backdrive_EDA_I'+str(copula_function)+'_X'+str(nas_index)
    elif type_eda==22:
        fmin = expand_referential_backdrive_EDA
        alg_name = 'expand_referential_backdrive_EDA_I'+str(copula_function)+'_X'+str(nas_index)
    elif type_eda==30:
        fmin = expand_copula_EDA        
        alg_name = 'expand_copula_EDA_'+str(copula_function)+'_X'+str(nas_index)
    elif type_eda==31:
        fmin = obj_expand_copula_EDA        
        alg_name = 'obj_expand_copula_EDA_'+str(copula_function)+'_X'+str(nas_index)
    elif type_eda==100:
        fmin = restart_univariate_gaussian_EDA
        alg_name = 'restart_Gaussian_UMDA'
    elif type_eda==101:
        fmin = restart_gaussian_EDA
        alg_name = 'restart_Gaussian_EDA'
    elif type_eda==102:
        fmin = restart_adaptive_gaussian_UMDA
        alg_name = 'restart_Adapt_G_UMDA'
    elif type_eda==103:
        fmin = restart_adaptive_gaussian_EDA
        alg_name = 'restart_Adapt_G_EDA'
    elif type_eda==104:
        fmin = restart_diffusion_EDA
        alg_name = 'restart_diffusion_EDA_I'+str(copula_function)
    elif type_eda==105:
        fmin = restart_copula_EDA        
        alg_name = 'restart_copula_EDA_'+str(copula_function)
    elif type_eda==106:
        fmin = restart_gaussian_mixture_EM_EDA        
        alg_name = 'restart_GaussianMixture_EDA_'+str(copula_function)
    elif type_eda==107:
        fmin = restart_univariate_gaussian_EDA
        alg_name = 'restart_Proportional_Gaussian_UMDA'
    elif type_eda==108:
        fmin = restart_gaussian_EDA
        alg_name = 'restart_Proportional_Gaussian_EDA'
    elif type_eda==109:
        fmin = restart_vae_EDA
        alg_name = 'restart_vae_EDA'
    elif type_eda==110:
        fmin = restart_backdrive_EDA
        alg_name = 'restart_backdrive_EDA_O'+str(copula_function)
    elif type_eda==111:
        fmin = restart_expand_diffusion_EDA
        alg_name = 'restart_expand_diffusion_EDA_I'+str(copula_function)+'_X'+str(nas_index)
    elif type_eda==112:
        fmin = restart_focused_diffusion_EDA
        alg_name = 'restart_focused_diffusion_EDA_I'+str(copula_function)+'_X'+str(nas_index)
    elif type_eda==113:
        fmin = restart_p0_univariate_diffusion_EDA
        alg_name = 'restart_p0_univariate_diffusion_EDA_I'+str(copula_function)+'_X'+str(nas_index)
    elif type_eda==114:
        fmin = restart_p1_univariate_diffusion_EDA
        alg_name = 'restart_p1_univariate_diffusion_EDA_I'+str(copula_function)+'_X'+str(nas_index)
    elif type_eda==115:
        fmin = restart_p01_univariate_diffusion_EDA
        alg_name = 'restart_p01_univariate_diffusion_EDA_I'+str(copula_function)+'_X'+str(nas_index)
    elif type_eda==116:
        fmin = restart_p11_univariate_diffusion_EDA
        alg_name = 'restart_p11_univariate_diffusion_EDA_I'+str(copula_function)+'_X'+str(nas_index)         
    elif type_eda==117:
        fmin = restart_p00_univariate_diffusion_EDA
        alg_name = 'restart_p00_univariate_diffusion_EDA_I'+str(copula_function)+'_X'+str(nas_index)
    elif type_eda==120:
        fmin = restart_expand_backdrive_EDA
        alg_name = 'restart_expand_backdrive_EDA_I'+str(copula_function)+'_X'+str(nas_index)
    elif type_eda==121:
        fmin = restart_expand_large_backdrive_EDA
        alg_name = 'restart_expand_large_backdrive_EDA_I'+str(copula_function)+'_X'+str(nas_index)
    elif type_eda==122:
        fmin = restart_expand_referential_backdrive_EDA
        alg_name = 'restart_expand_referential_backdrive_EDA_I'+str(copula_function)+'_X'+str(nas_index)
    elif type_eda==130:
        fmin = restart_expand_backdrive_EDA
        alg_name = 'restart_expand_backdrive_EDA_I'+str(copula_function)+'_X'+str(nas_index)        
    elif type_eda==150:
        fmin = restart_BO_univariate_gaussian_EDA
        alg_name = 'restart_BO_Gaussian_UMDA'
    elif type_eda==151:
        fmin = restart_BO_gaussian_EDA
        alg_name = 'restart_BO_Gaussian_EDA'   
    elif type_eda==152:
        fmin = restart_BO_univariate_gaussian_EDA
        alg_name = 'restart_POOL_Gaussian_UMDA'
    elif type_eda==153:
        fmin = restart_BO_gaussian_EDA
        alg_name = 'restart_POOL_Gaussian_EDA'
    elif type_eda==154:
        fmin = restart_PROPBO_univariate_gaussian_EDA
        alg_name = 'restart_PROPBO_Gaussian_UMDA'
    elif type_eda==155:
        fmin = restart_PROPBO_gaussian_EDA
        alg_name = 'restart_PROPBO_Gaussian_EDA'
    elif type_eda==156:
        fmin = restart_BO_univariate_gaussian_EDA
        alg_name = 'restart_BO_Gaussian_UMDA_DT-10'
    elif type_eda==157:
        fmin = restart_BO_gaussian_EDA
        alg_name = 'restart_BO_Gaussian_EDA_DT-10'
    elif type_eda==164:
        fmin = restart_PROPBO_p01_univariate_diffusion_EDA
        alg_name = 'restart_PROPBO_p01_univariate_diffusion_EDA_I'+str(copula_function)+'_X'+str(nas_index)
    elif type_eda==165:
        fmin = restart_elm_PROPBO_p01_univariate_diffusion_EDA
        alg_name = 'restart_elm_PROPBO_p01_univariate_diffusion_EDA_I'+str(copula_function)+'_X'+str(nas_index)
    elif type_eda==166:
        fmin = restart_elm_PROPBO_p11_univariate_diffusion_EDA
        alg_name = 'restart_elm_PROPBO_p11_univariate_diffusion_EDA_I'+str(copula_function)+'_X'+str(nas_index)
    elif type_eda==167:
        fmin = restart_elm_PROPBO_p00_univariate_diffusion_EDA
        alg_name = 'restart_elm_PROPBO_p00_univariate_diffusion_EDA_I'+str(copula_function)+'_X'+str(nas_index)         
    elif type_eda==180:
        fmin = restart_elm_expand_backdrive_EDA
        alg_name = 'restart_elm_expand_backdrive_EDA_I'+str(copula_function)+'_X'+str(nas_index)
    elif type_eda==181:
        fmin = restart_elm_expand_diffusion_EDA
        alg_name = 'restart_elm_expand_diffusion_EDA_I'+str(copula_function)+'_X'+str(nas_index)

        
        
    alg_name = alg_name+'_n'+str(funct)+'_N'+str(population_size)+'_T'+str(truncation_value)+'_D'+str(diversity)+'_Seed_'+str(seed)
    print(alg_name)
    #output_folder = alg_name
   
    # Get the current script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the path to the folder where you want to read/write files
    input_folder_path = os.path.join(current_dir+'/GNBG_Instances.Python-main/')
    #output_folder_path = os.path.join(current_dir+'/'+ output_folder)
    # Initialization
    ProblemIndex = funct   # Choose a problem instance range from f1 to f24

    # Preparation and loading of the GNBG parameters based on the chosen problem instance
    if 1 <= ProblemIndex <= 24:
        filename = f'f{ProblemIndex}.mat'
        GNBG_tmp = loadmat(os.path.join(input_folder_path, filename))['GNBG']
        MaxEvals = np.array([item[0] for item in GNBG_tmp['MaxEvals'].flatten()])[0, 0]
        AcceptanceThreshold = np.array([item[0] for item in GNBG_tmp['AcceptanceThreshold'].flatten()])[0, 0]
        Dimension = np.array([item[0] for item in GNBG_tmp['Dimension'].flatten()])[0, 0]
        CompNum = np.array([item[0] for item in GNBG_tmp['o'].flatten()])[0, 0]  # Number of components
        MinCoordinate = np.array([item[0] for item in GNBG_tmp['MinCoordinate'].flatten()])[0, 0]
        MaxCoordinate = np.array([item[0] for item in GNBG_tmp['MaxCoordinate'].flatten()])[0, 0]
        CompMinPos = np.array(GNBG_tmp['Component_MinimumPosition'][0, 0])
        CompSigma = np.array(GNBG_tmp['ComponentSigma'][0, 0], dtype=np.float64)
        CompH = np.array(GNBG_tmp['Component_H'][0, 0])
        Mu = np.array(GNBG_tmp['Mu'][0, 0])
        Omega = np.array(GNBG_tmp['Omega'][0, 0])
        Lambda = np.array(GNBG_tmp['lambda'][0, 0])
        RotationMatrix = np.array(GNBG_tmp['RotationMatrix'][0, 0])
        OptimumValue = np.array([item[0] for item in GNBG_tmp['OptimumValue'].flatten()])[0, 0]
        OptimumPosition = np.array(GNBG_tmp['OptimumPosition'][0, 0])
    else:
        raise ValueError('ProblemIndex must be between 1 and 24.')

    
    gnbg = GNBG(MaxEvals, AcceptanceThreshold, Dimension, CompNum, MinCoordinate, MaxCoordinate, CompMinPos, CompSigma, CompH, Mu, Omega, Lambda, RotationMatrix, OptimumValue, OptimumPosition)


    # The following code is an example of how a GNBG's problem instance can be solved using an optimizer
    # The Differential Evolution (DE) optimizer is used here as an example. You can replace it with any other optimizer of your choice.

    # Define the bounds for the optimizer based on the bounds of the problem instance    
    lb = -100*np.ones(Dimension)
    ub = 100*np.ones(Dimension)
    lower_bound = lb
    upper_bound = ub    
    bounds = []
    for i in range(0,Dimension):
        bounds.append(tuple((lb[i],ub[i])))

    # Run the optimizer where the fitness function is gnbg.fitness    

    
    max_generations = MaxEvals // population_size
    #max_generations = 5
    print("The maximum number of generations is ", max_generations)

    #results = differential_evolution(gnbg.fitness, bounds=bounds, disp=True, polish=False, popsize=popsize, maxiter=maxiter)
    
    n_runs = 1
    for run in range(n_runs):             
                if type_eda==0:
                    xopt,_ = univariate_gaussian_EDA(fitness_function, Dimension, lower_bound, upper_bound, population_size,
                                                     truncation_value, replacement, max_generations, -1, 0)
                elif type_eda==1:
                    xopt,_= gaussian_EDA(fitness_function, Dimension, lower_bound, upper_bound, population_size,
                                         truncation_value, replacement, max_generations, -1, 0)
                elif type_eda==2:
                    xopt,_ = adaptive_gaussian_UMDA(fitness_function, Dimension, lower_bound, upper_bound, population_size,
                                                    truncation_value, replacement, max_generations, diversity)
                elif type_eda==3:
                    xopt,_= adaptive_gaussian_EDA(fitness_function, Dimension, lower_bound, upper_bound, population_size,
                                                  truncation_value, replacement, max_generations, diversity)                                
                elif type_eda==4:         
                    num_epochs = 15
                    batch_size = 10
                    num_iterations = copula_function  # Number of iterations for sampling
                    xopt,_ = diffusion_EDA(fitness_function, Dimension, lower_bound, upper_bound, population_size,
                                           truncation_value, replacement, max_generations, num_epochs, batch_size, num_iterations)

                elif type_eda==5:
                    xopt,_= copula_EDA(fitness_function, Dimension, lower_bound, upper_bound, population_size,
                                       truncation_value, replacement, max_generations, copula_function, vine_level)
                elif type_eda==6:
                    n_components = copula_function  # We reinterpret the copula function number as number of components for Gaussian Mixture
                    xopt,_= gaussian_mixture_EM_EDA(fitness_function, Dimension, lower_bound, upper_bound, population_size,
                                                    truncation_value, replacement, max_generations,n_components)
                elif type_eda==7:
                    proportional = 1   # A univariate gaussian model with exponential selection over the selected population
                    xopt,_ = univariate_gaussian_EDA(fitness_function, Dimension, lower_bound, upper_bound, population_size,
                                                     truncation_value, replacement, max_generations, -1, 1)
                elif type_eda==8:
                    proportional = 1   # A gaussian model with exponential selection over the full population
                    xopt,_ = gaussian_EDA(fitness_function, Dimension, lower_bound, upper_bound, population_size,
                                                     truncation_value, replacement, max_generations, -1, 1)
                elif type_eda==9:         
                    num_epochs = 100
                    batch_size = 8
                    latent_dim = 2  # Latent dimension for VAE
                    xopt,_ = vae_EDA(fitness_function, Dimension, lower_bound, upper_bound, population_size,
                                           truncation_value, replacement, max_generations, num_epochs, batch_size, latent_dim)
                elif type_eda==10:         
                    num_epochs = 75
                    batch_size = 4
                    opt_method = copula_function  # We reinterpret the copula function number as the gradient opt. method
                    xopt,_ = backdrive_EDA(fitness_function, Dimension, lower_bound, upper_bound,
                                                    population_size, truncation_value, replacement,
                                                    max_generations, num_epochs, batch_size,opt_method)
                    
                elif type_eda==11:
                    num_epochs = 15
                    batch_size = 10
                    num_iterations = copula_function  # Number of iterations for sampling
                    nas_nn_index = nas_index
                    expand_factor = diversity  # How the width of layers are expanded
                    
                    xopt,_ = expand_diffusion_EDA(fitness_function, Dimension, lower_bound, upper_bound,
                                                  population_size, truncation_value, replacement,
                                                  max_generations, num_epochs, batch_size, num_iterations,
                                                  nas_nn_index, expand_factor)
                elif type_eda==12:
                    num_epochs = 15
                    batch_size = 10
                    num_iterations = copula_function  # Number of iterations for sampling
                    nas_nn_index = nas_index
                    expand_factor = diversity  # How the width of layers are expanded
                    
                    xopt,_ = focused_diffusion_EDA(fitness_function, Dimension, lower_bound, upper_bound,
                                                  population_size, truncation_value, replacement,
                                                  max_generations, num_epochs, batch_size, num_iterations,
                                                  nas_nn_index, expand_factor)
                elif type_eda==13:
                    num_epochs = 15
                    batch_size = 10
                    num_iterations = copula_function  # Number of iterations for sampling
                    nas_nn_index = nas_index
                    expand_factor = diversity  # How the width of layers are expanded
                    
                    xopt,_ = p0_univariate_diffusion_EDA(fitness_function, Dimension, lower_bound, upper_bound,
                                                  population_size, truncation_value, replacement,
                                                  max_generations, num_epochs, batch_size, num_iterations,
                                                  nas_nn_index, expand_factor)
                elif type_eda==14:
                    num_epochs = 15
                    batch_size = 10
                    num_iterations = copula_function  # Number of iterations for sampling
                    nas_nn_index = nas_index
                    expand_factor = diversity  # How the width of layers are expanded
                    
                    xopt,_ = p1_univariate_diffusion_EDA(fitness_function, Dimension, lower_bound, upper_bound,
                                                  population_size, truncation_value, replacement,
                                                  max_generations, num_epochs, batch_size, num_iterations,
                                                  nas_nn_index, expand_factor)
                elif type_eda==15:
                    num_epochs = 15
                    batch_size = 10
                    num_iterations = copula_function  # Number of iterations for sampling
                    nas_nn_index = nas_index
                    expand_factor = diversity  # How the width of layers are expanded
                    
                    xopt,_ = p01_univariate_diffusion_EDA(fitness_function, Dimension, lower_bound, upper_bound,
                                                  population_size, truncation_value, replacement,
                                                  max_generations, num_epochs, batch_size, num_iterations,
                                                  nas_nn_index, expand_factor)
                elif type_eda==16:
                    num_epochs = 15
                    batch_size = 10
                    num_iterations = copula_function  # Number of iterations for sampling
                    nas_nn_index = nas_index
                    expand_factor = diversity  # How the width of layers are expanded
                    
                    xopt,_ = p11_univariate_diffusion_EDA(fitness_function, Dimension, lower_bound, upper_bound,
                                                  population_size, truncation_value, replacement,
                                                  max_generations, num_epochs, batch_size, num_iterations,
                                                  nas_nn_index, expand_factor)                          
                   
                elif type_eda==17:
                    num_epochs = 15
                    batch_size = 10
                    num_iterations = copula_function  # Number of iterations for sampling
                    nas_nn_index = nas_index
                    expand_factor = diversity  # How the width of layers are expanded
                    
                    xopt,_ = p00_univariate_diffusion_EDA(fitness_function, Dimension, lower_bound, upper_bound,
                                                  population_size, truncation_value, replacement,
                                                  max_generations, num_epochs, batch_size, num_iterations,
                                                  nas_nn_index, expand_factor)
                elif type_eda==20:
                    replacement = 1
                    num_epochs =150
                    batch_size = 16
                    opt_method = copula_function  # We reinterpret the copula function number as the gradient opt. method
                    nas_nn_index = nas_index
                    expand_factor = diversity  # How the width of layers are expanded
                    xopt,_ = expand_backdrive_EDA(fitness_function, Dimension, lower_bound, upper_bound,
                                                  population_size, truncation_value, replacement,
                                                  max_generations, num_epochs, batch_size, opt_method,
                                                  nas_nn_index, expand_factor)

                elif type_eda==21:
                    replacement = 1
                    num_epochs =150
                    batch_size = 16
                    opt_method = copula_function  # We reinterpret the copula function number as the gradient opt. method
                    nas_nn_index = nas_index
                    expand_factor = diversity  # How the width of layers are expanded
                    xopt,_ = expand_large_backdrive_EDA(fitness_function, Dimension, lower_bound, upper_bound,
                                                  population_size, truncation_value, replacement,
                                                  max_generations, num_epochs, batch_size, opt_method,
                                                  nas_nn_index, expand_factor)                    
                    

                elif type_eda==22:
                    replacement = 1
                    num_epochs = 50
                    batch_size = 4
                    opt_method = copula_function  # We reinterpret the copula function number as the gradient opt. method
                    nas_nn_index = nas_index
                    expand_factor = diversity  # How the width of layers are expanded
                    xopt,_ = expand_referential_backdrive_EDA(fitness_function, Dimension, lower_bound, upper_bound,
                                                  population_size, truncation_value, replacement,
                                                  max_generations, num_epochs, batch_size, opt_method,
                                                  nas_nn_index, expand_factor)
                elif type_eda==30:                                       
                    xopt,_= expand_copula_EDA(fitness_function, Dimension, lower_bound, upper_bound, population_size,
                                              truncation_value, replacement, max_generations, copula_function, vine_level,
                                              trigger_No_Improvement, diversity_threshold)                    
                elif type_eda==31:                    
                    xopt,_= obj_expand_copula_EDA(fitness_function, Dimension, lower_bound, upper_bound, population_size,
                                                  truncation_value, replacement, max_generations, copula_function, vine_level,
                                                  trigger_No_Improvement, diversity_threshold)          
                    
                elif type_eda==100:
                    proportional = 0
                    diversity = -1
                    xopt,_ = restart_univariate_gaussian_EDA(fitness_function, Dimension, lower_bound, upper_bound, population_size,
                                                             truncation_value, replacement, max_generations, diversity, proportional,
                                                             trigger_No_Improvement, diversity_threshold)
                elif type_eda==101:
                    proportional = 0
                    diversity = -1
                    xopt,_= restart_gaussian_EDA(fitness_function, Dimension, lower_bound, upper_bound, population_size,
                                                 truncation_value, replacement, max_generations, diversity, proportional,
                                                 trigger_No_Improvement, diversity_threshold)
                elif type_eda==102:
                    xopt,_ = restart_adaptive_gaussian_UMDA(fitness_function, Dimension, lower_bound, upper_bound, population_size,
                                                            truncation_value, replacement, max_generations, diversity,
                                                            trigger_No_Improvement, diversity_threshold)
                elif type_eda==103:
                    xopt,_= restart_adaptive_gaussian_EDA(fitness_function, Dimension, lower_bound, upper_bound, population_size,
                                                          truncation_value, replacement, max_generations, diversity,
                                                          trigger_No_Improvement, diversity_threshold)
                elif type_eda==106:
                    n_components = copula_function  # We reinterpret the copula function number as number of components for Gaussian Mixture
                    xopt,_= restart_gaussian_mixture_EM_EDA(fitness_function, Dimension, lower_bound, upper_bound, population_size,
                                                    truncation_value, replacement, max_generations,n_components,
                                                    trigger_No_Improvement, diversity_threshold)

                elif type_eda==111:
                    num_epochs = 15
                    batch_size = 10
                    num_iterations = copula_function  # Number of iterations for sampling
                    nas_nn_index = nas_index
                    expand_factor = diversity  # How the width of layers are expanded
                    
                    xopt,_ = restart_expand_diffusion_EDA(fitness_function, Dimension, lower_bound, upper_bound,
                                                  population_size, truncation_value, replacement,
                                                  max_generations, num_epochs, batch_size, num_iterations,
                                                          nas_nn_index, expand_factor,
                                                          trigger_No_Improvement, diversity_threshold)                    
                    
                elif type_eda==112:
                    num_epochs = 15
                    batch_size = 10
                    num_iterations = copula_function  # Number of iterations for sampling
                    nas_nn_index = nas_index
                    expand_factor = diversity  # How the width of layers are expanded
                    
                    xopt,_ = restart_focused_diffusion_EDA(fitness_function, Dimension, lower_bound, upper_bound,
                                                  population_size, truncation_value, replacement,
                                                  max_generations, num_epochs, batch_size, num_iterations,
                                                           nas_nn_index, expand_factor,
                                                           trigger_No_Improvement, diversity_threshold)
                elif type_eda==113:
                    num_epochs = 15
                    batch_size = 10
                    num_iterations = copula_function
                    nas_nn_index = nas_index
                    expand_factor = diversity  # How the width of layers are expanded
                    
                    xopt,_ = restart_p0_univariate_diffusion_EDA(fitness_function, Dimension, lower_bound, upper_bound,
                                                  population_size, truncation_value, replacement,
                                                  max_generations, num_epochs, batch_size, num_iterations,
                                                  nas_nn_index, expand_factor,
                                                  trigger_No_Improvement, diversity_threshold)

                elif type_eda==114:
                    num_epochs = 15
                    batch_size = 10
                    num_iterations = copula_function  # Number of iterations for sampling
                    nas_nn_index = nas_index
                    expand_factor = diversity  # How the width of layers are expanded
                    
                    xopt,_ = restart_p1_univariate_diffusion_EDA(fitness_function, Dimension, lower_bound, upper_bound,
                                                  population_size, truncation_value, replacement,
                                                  max_generations, num_epochs, batch_size, num_iterations,
                                                                 nas_nn_index, expand_factor,
                                                                 trigger_No_Improvement, diversity_threshold)
                elif type_eda==115:
                    num_epochs = 15
                    batch_size = 10
                    num_iterations = copula_function  # Number of iterations for sampling
                    nas_nn_index = nas_index
                    expand_factor = diversity  # How the width of layers are expanded
                    
                    xopt,_ = restart_p01_univariate_diffusion_EDA(fitness_function, Dimension, lower_bound, upper_bound,
                                                  population_size, truncation_value, replacement,
                                                  max_generations, num_epochs, batch_size, num_iterations,
                                                                  nas_nn_index, expand_factor,
                                                                  trigger_No_Improvement, diversity_threshold)
                elif type_eda==116:
                    num_epochs = 15
                    batch_size = 10
                    num_iterations = copula_function  # Number of iterations for sampling
                    nas_nn_index = nas_index
                    expand_factor = diversity  # How the width of layers are expanded
                    
                    xopt,_ = restart_p11_univariate_diffusion_EDA(fitness_function, Dimension, lower_bound, upper_bound,
                                                  population_size, truncation_value, replacement,
                                                  max_generations, num_epochs, batch_size, num_iterations,
                                                                  nas_nn_index, expand_factor,
                                                                  trigger_No_Improvement, diversity_threshold)                          
                   
                elif type_eda==117:
                    num_epochs = 15
                    batch_size = 10
                    num_iterations = copula_function  # Number of iterations for sampling
                    nas_nn_index = nas_index
                    expand_factor = diversity  # How the width of layers are expanded
                    
                    xopt,_ = restart_p00_univariate_diffusion_EDA(fitness_function, Dimension, lower_bound, upper_bound,
                                                  population_size, truncation_value, replacement,
                                                  max_generations, num_epochs, batch_size, num_iterations,
                                                                  nas_nn_index, expand_factor,
                                                                  trigger_No_Improvement, diversity_threshold)

                elif type_eda==130:
                    replacement = 1
                    num_epochs = 15
                    batch_size = 10
                    opt_method = copula_function  # We reinterpret the copula function number as the gradient opt. method
                    nas_nn_index = nas_index
                    expand_factor = diversity  # How the width of layers are expanded
                    xopt,_ = restart_expand_backdrive_EDA(fitness_function, Dimension, lower_bound, upper_bound,
                                                  population_size, truncation_value, replacement,
                                                  max_generations, num_epochs, batch_size, opt_method,
                                                          nas_nn_index, expand_factor, trigger_No_Improvement, diversity_threshold)
                    
                elif type_eda==150:
                    proportional = 0
                    diversity = -1
                    xopt,_ = restart_BO_univariate_gaussian_EDA(fitness_function, Dimension, lower_bound, upper_bound, population_size,
                                                             truncation_value, replacement, max_generations, diversity, proportional,
                                                             trigger_No_Improvement, diversity_threshold)
                elif type_eda==151:
                    proportional = 0
                    diversity = -1
                    xopt,_= restart_BO_gaussian_EDA(fitness_function, Dimension, lower_bound, upper_bound, population_size,
                                                 truncation_value, replacement, max_generations, diversity, proportional,
                                                 trigger_No_Improvement, diversity_threshold)
  
                elif type_eda==152:
                    proportional = 0
                    diversity = -1
                    xopt,_ = restart_POOL_univariate_gaussian_EDA(fitness_function, Dimension, lower_bound, upper_bound, population_size,
                                                             truncation_value, replacement, max_generations, diversity, proportional,
                                                             trigger_No_Improvement, diversity_threshold)
                elif type_eda==153:
                    proportional = 0
                    diversity = -1
                    xopt,_= restart_POOL_gaussian_EDA(fitness_function, Dimension, lower_bound, upper_bound, population_size,
                                                 truncation_value, replacement, max_generations, diversity, proportional,
                                                 trigger_No_Improvement, diversity_threshold)
                elif type_eda==154:
                    proportional = 1
                    diversity = 1
                    xopt,_ = restart_PROPBO_univariate_gaussian_EDA(fitness_function, Dimension, lower_bound, upper_bound, population_size,
                                                             truncation_value, replacement, max_generations, diversity, proportional,
                                                             trigger_No_Improvement, diversity_threshold)
                elif type_eda==155:
                    proportional = 1
                    diversity = 1
                    xopt,_= restart_PROPBO_gaussian_EDA(fitness_function, Dimension, lower_bound, upper_bound, population_size,
                                                 truncation_value, replacement, max_generations, diversity, proportional,
                                                 trigger_No_Improvement, diversity_threshold)
                elif type_eda==156:
                    proportional = 0
                    diversity = -1
                    diversity_threshold = 5*10**(-10)
                    xopt,_ = restart_BO_univariate_gaussian_EDA(fitness_function, Dimension, lower_bound, upper_bound, population_size,
                                                             truncation_value, replacement, max_generations, diversity, proportional,
                                                             trigger_No_Improvement, diversity_threshold)
                elif type_eda==157:
                    proportional = 0
                    diversity = -1
                    diversity_threshold = 5*10**(-10)
                    xopt,_= restart_BO_gaussian_EDA(fitness_function, Dimension, lower_bound, upper_bound, population_size,
                                                 truncation_value, replacement, max_generations, diversity, proportional,
                                                 trigger_No_Improvement, diversity_threshold)
                elif type_eda==164:                                     
                    num_epochs = 15
                    batch_size = 10
                    num_iterations = copula_function  # Number of iterations for sampling
                    nas_nn_index = nas_index
                    expand_factor = diversity  # How the width of layers are expanded
                    
                    xopt,_ = restart_PROPBO_p01_univariate_diffusion_EDA(fitness_function, Dimension, lower_bound, upper_bound,
                                                  population_size, truncation_value, replacement,
                                                  max_generations, num_epochs, batch_size, num_iterations,
                                                                  nas_nn_index, expand_factor,
                                                                  trigger_No_Improvement, diversity_threshold)


                elif type_eda==165:                                     
                    num_epochs = 15
                    batch_size = 10
                    num_iterations = copula_function  # Number of iterations for sampling
                    nas_nn_index = nas_index
                    expand_factor = diversity  # How the width of layers are expanded
                    
                    xopt,_ = restart_elm_PROPBO_p01_univariate_diffusion_EDA(fitness_function, Dimension, lower_bound, upper_bound,
                                                  population_size, truncation_value, replacement,
                                                  max_generations, num_epochs, batch_size, num_iterations,
                                                                  nas_nn_index, expand_factor,
                                                                  trigger_No_Improvement, diversity_threshold)
                elif type_eda==166:                                     
                    num_epochs = 15
                    batch_size = 10
                    num_iterations = copula_function  # Number of iterations for sampling
                    nas_nn_index = nas_index
                    expand_factor = diversity  # How the width of layers are expanded
                    
                    xopt,_ = restart_elm_PROPBO_p11_univariate_diffusion_EDA(fitness_function, Dimension, lower_bound, upper_bound,
                                                  population_size, truncation_value, replacement,
                                                  max_generations, num_epochs, batch_size, num_iterations,
                                                                  nas_nn_index, expand_factor,
                                                                  trigger_No_Improvement, diversity_threshold)
                elif type_eda==167:                                     
                    num_epochs = 15
                    batch_size = 10
                    num_iterations = copula_function  # Number of iterations for sampling
                    nas_nn_index = nas_index
                    expand_factor = diversity  # How the width of layers are expanded
                    
                    xopt,_ = restart_elm_PROPBO_p00_univariate_diffusion_EDA(fitness_function, Dimension, lower_bound, upper_bound,
                                                  population_size, truncation_value, replacement,
                                                  max_generations, num_epochs, batch_size, num_iterations,
                                                                  nas_nn_index, expand_factor,
                                                                  trigger_No_Improvement, diversity_threshold)                    
                    
                elif type_eda==180:
                    replacement = 1
                    num_epochs = 15
                    batch_size = 10
                    opt_method = copula_function  # We reinterpret the copula function number as the gradient opt. method
                    nas_nn_index = nas_index
                    expand_factor = diversity  # How the width of layers are expanded
                    xopt,_ = restart_elm_expand_backdrive_EDA(fitness_function, Dimension, lower_bound, upper_bound,
                                                  population_size, truncation_value, replacement,
                                                  max_generations, num_epochs, batch_size, opt_method,
                                                          nas_nn_index, expand_factor, trigger_No_Improvement, diversity_threshold)

                    
                elif type_eda==181:
                    num_epochs = 15
                    batch_size = 10
                    num_iterations = copula_function  # Number of iterations for sampling
                    nas_nn_index = nas_index
                    expand_factor = diversity  # How the width of layers are expanded
                    
                    xopt,_ = restart_elm_expand_diffusion_EDA(fitness_function, Dimension, lower_bound, upper_bound,
                                                  population_size, truncation_value, replacement,
                                                  max_generations, num_epochs, batch_size, num_iterations,
                                                          nas_nn_index, expand_factor,
                                                          trigger_No_Improvement, diversity_threshold)                    

    #rint(len(gnbg.FEhistory))
    np.save(alg_name,gnbg,allow_pickle=True)


    '''
    convergence = []
    best_error = float('inf')
    for value in gnbg.FEhistory:
        error = abs(value - OptimumValue)
        if error < best_error:
            best_error = error
            convergence.append(best_error)

    
    plt.plot(range(1, len(convergence) + 1), convergence)
    plt.xlabel('Function Evaluation Number (FE)')
    plt.ylabel('Error')
    plt.title('Convergence Plot')
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.show()
    '''    
    
    # python3.11 gnbg_benchmarking.py 111 5 1  30 100 80 10 2 0 350
    # python3 gnbg_benchmarking.py 111 5 1 164 100 20 25 1 10 150
