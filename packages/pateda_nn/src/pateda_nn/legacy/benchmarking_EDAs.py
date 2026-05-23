import numpy as np
import sys
from selection_methods import *
from initialization_methods import *
from gaussian_models import *
from diffusion_models import *
from functions import *
from continuous_eda import *
import cocopp
import cocoex
import scipy



if __name__ == '__main__':
    seed = int(sys.argv[1])  
    n = int(sys.argv[2])      # Number of variables
    #lower_bound = int(sys.argv[3])      # The same for all variables
    #upper_bound = int(sys.argv[4])      # The same for all variables
    funct  = int(sys.argv[3])       # Function to optimize   
    
    type_eda  = int(sys.argv[4])     # Type of EDA
                                     #  0: "Gaussian"
                                     #  1: "Diffusion"
    population_size = int(sys.argv[5]) 
    truncation_value = int(sys.argv[6])    # Percent of truncation
    max_generations = int(sys.argv[7]) # Number of the EA generations
    diversity = int(sys.argv[8])      # Whether the standard deviation is expanded to infuse diversity
    copula_function = int(sys.argv[9]) # Copula function
    nas_index = int(sys.argv[10])       # Id of the network architecture to use for expand_diffusion 
    

    #all_functions = [sphere_function, ellipsoidal_function, schaffers_f7_function]     
    #the_function = all_functions[funct]
    
    np.random.seed(seed)
    lower_bound = -5
    upper_bound = 5
    replacement = 1  
    n_expe = 1
    vine_level = diversity

   
    #if tf.test.gpu_device_name():
    #    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    #else:
    #    print("Please install GPU version of TF")
    #print("tensorflow version: {0} ".format(tf.__version__))
    #print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    #print("CUDA: {0} | CUDnn: {1}".format(tf_build_info.cuda_version_number,  tf_build_info.cudnn_version_number))
    
    ### input: define suite and solver (see also "input" below where fmin is called)

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
        
    alg_name = alg_name+'_N'+str(population_size)+'_T'+str(truncation_value)+'_D'+str(diversity)
    
    budget_multiplier = 1  # increase to 3, 10, 30, ... x dimension

    ### prepare
    suite_name = "bbob"
    #suite = cocoex.Suite(suite_name, "", "")  # see https://numbbo.github.io/coco-doc/C/#suite-parameters
    #suite = cocoex.Suite(suite_name, 
    #               "instances: 1-24", 
    #               "dimensions: 5,10,20 instance_indices:1-15")

    suite = cocoex.Suite(suite_name, 
                   "instances: 1-24 ",     
                   "dimensions: "+str(n)+" instance_indices: 1-15");
    
    output_folder = '{}_of_{}_{}D_on_{}'.format(alg_name, fmin.__module__ or '', int(budget_multiplier+0.499), suite_name)
    #output_folder = '{}_of_{}_{}D_on_{}'.format(fmin.__name__, fmin.__module__ or '', int(budget_multiplier+0.499), suite_name)

    observer = cocoex.Observer(suite_name, "result_folder: " + output_folder)
    repeater = cocoex.ExperimentRepeater(budget_multiplier)  # x dimension
    minimal_print = cocoex.utilities.MiniPrint()

    ### go
    while not repeater.done():  # while budget is left and successes are few
        for problem in suite:  # loop takes 2-3 minutes x budget_multiplier
            if  (funct>30) or (problem.id_function==funct):
                if repeater.done(problem):
                    continue
                problem.observe_with(observer)  # generate data for cocopp
                problem(problem.dimension * [0])  # for better comparability
                
            
                if type_eda==0:
                    xopt,_ = univariate_gaussian_EDA(problem, problem.dimension, lower_bound, upper_bound, population_size,
                                                     truncation_value, replacement, max_generations, -1, 0)
                elif type_eda==1:
                    xopt,_= gaussian_EDA(problem, problem.dimension, lower_bound, upper_bound, population_size,
                                         truncation_value, replacement, max_generations, -1, 0)
                elif type_eda==2:
                    xopt,_ = adaptive_gaussian_UMDA(problem, problem.dimension, lower_bound, upper_bound, population_size,
                                                    truncation_value, replacement, max_generations, diversity)
                elif type_eda==3:
                    xopt,_= adaptive_gaussian_EDA(problem, problem.dimension, lower_bound, upper_bound, population_size,
                                                  truncation_value, replacement, max_generations, diversity)                                
                elif type_eda==4:         
                    num_epochs = 15
                    batch_size = 10
                    num_iterations = copula_function  # Number of iterations for sampling
                    xopt,_ = diffusion_EDA(problem, problem.dimension, lower_bound, upper_bound, population_size,
                                           truncation_value, replacement, max_generations, num_epochs, batch_size, num_iterations)

                elif type_eda==5:
                    xopt,_= copula_EDA(problem, problem.dimension, lower_bound, upper_bound, population_size,
                                       truncation_value, replacement, max_generations, copula_function, vine_level)
                elif type_eda==6:
                    n_components = copula_function  # We reinterpret the copula function number as number of components for Gaussian Mixture
                    xopt,_= gaussian_mixture_EM_EDA(problem, problem.dimension, lower_bound, upper_bound, population_size,
                                                    truncation_value, replacement, max_generations,n_components)
                elif type_eda==7:
                    proportional = 1   # A univariate gaussian model with exponential selection over the selected population
                    xopt,_ = univariate_gaussian_EDA(problem, problem.dimension, lower_bound, upper_bound, population_size,
                                                     truncation_value, replacement, max_generations, -1, 1)
                elif type_eda==8:
                    proportional = 1   # A gaussian model with exponential selection over the full population
                    xopt,_ = gaussian_EDA(problem, problem.dimension, lower_bound, upper_bound, population_size,
                                                     truncation_value, replacement, max_generations, -1, 1)
                elif type_eda==9:         
                    num_epochs = 100
                    batch_size = 8
                    latent_dim = 2  # Latent dimension for VAE
                    xopt,_ = vae_EDA(problem, problem.dimension, lower_bound, upper_bound, population_size,
                                           truncation_value, replacement, max_generations, num_epochs, batch_size, latent_dim)
                elif type_eda==10:         
                    num_epochs = 75
                    batch_size = 4
                    opt_method = copula_function  # We reinterpret the copula function number as the gradient opt. method
                    xopt,_ = backdrive_EDA(problem, problem.dimension, lower_bound, upper_bound,
                                                    population_size, truncation_value, replacement,
                                                    max_generations, num_epochs, batch_size,opt_method)
                    
                elif type_eda==11:
                    num_epochs = 15
                    batch_size = 10
                    num_iterations = copula_function  # Number of iterations for sampling
                    nas_nn_index = nas_index
                    expand_factor = diversity  # How the width of layers are expanded
                    
                    xopt,_ = expand_diffusion_EDA(problem, problem.dimension, lower_bound, upper_bound,
                                                  population_size, truncation_value, replacement,
                                                  max_generations, num_epochs, batch_size, num_iterations,
                                                  nas_nn_index, expand_factor)
                elif type_eda==12:
                    num_epochs = 15
                    batch_size = 10
                    num_iterations = copula_function  # Number of iterations for sampling
                    nas_nn_index = nas_index
                    expand_factor = diversity  # How the width of layers are expanded
                    
                    xopt,_ = focused_diffusion_EDA(problem, problem.dimension, lower_bound, upper_bound,
                                                  population_size, truncation_value, replacement,
                                                  max_generations, num_epochs, batch_size, num_iterations,
                                                  nas_nn_index, expand_factor)
                elif type_eda==13:
                    num_epochs = 15
                    batch_size = 10
                    num_iterations = copula_function  # Number of iterations for sampling
                    nas_nn_index = nas_index
                    expand_factor = diversity  # How the width of layers are expanded
                    
                    xopt,_ = p0_univariate_diffusion_EDA(problem, problem.dimension, lower_bound, upper_bound,
                                                  population_size, truncation_value, replacement,
                                                  max_generations, num_epochs, batch_size, num_iterations,
                                                  nas_nn_index, expand_factor)
                elif type_eda==14:
                    num_epochs = 15
                    batch_size = 10
                    num_iterations = copula_function  # Number of iterations for sampling
                    nas_nn_index = nas_index
                    expand_factor = diversity  # How the width of layers are expanded
                    
                    xopt,_ = p1_univariate_diffusion_EDA(problem, problem.dimension, lower_bound, upper_bound,
                                                  population_size, truncation_value, replacement,
                                                  max_generations, num_epochs, batch_size, num_iterations,
                                                  nas_nn_index, expand_factor)
                elif type_eda==15:
                    num_epochs = 15
                    batch_size = 10
                    num_iterations = copula_function  # Number of iterations for sampling
                    nas_nn_index = nas_index
                    expand_factor = diversity  # How the width of layers are expanded
                    
                    xopt,_ = p01_univariate_diffusion_EDA(problem, problem.dimension, lower_bound, upper_bound,
                                                  population_size, truncation_value, replacement,
                                                  max_generations, num_epochs, batch_size, num_iterations,
                                                  nas_nn_index, expand_factor)
                elif type_eda==16:
                    num_epochs = 15
                    batch_size = 10
                    num_iterations = copula_function  # Number of iterations for sampling
                    nas_nn_index = nas_index
                    expand_factor = diversity  # How the width of layers are expanded
                    
                    xopt,_ = p11_univariate_diffusion_EDA(problem, problem.dimension, lower_bound, upper_bound,
                                                  population_size, truncation_value, replacement,
                                                  max_generations, num_epochs, batch_size, num_iterations,
                                                  nas_nn_index, expand_factor)                          
                   
                elif type_eda==17:
                    num_epochs = 15
                    batch_size = 10
                    num_iterations = copula_function  # Number of iterations for sampling
                    nas_nn_index = nas_index
                    expand_factor = diversity  # How the width of layers are expanded
                    
                    xopt,_ = p00_univariate_diffusion_EDA(problem, problem.dimension, lower_bound, upper_bound,
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
                    xopt,_ = expand_backdrive_EDA(problem, problem.dimension, lower_bound, upper_bound,
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
                    xopt,_ = expand_large_backdrive_EDA(problem, problem.dimension, lower_bound, upper_bound,
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
                    xopt,_ = expand_referential_backdrive_EDA(problem, problem.dimension, lower_bound, upper_bound,
                                                  population_size, truncation_value, replacement,
                                                  max_generations, num_epochs, batch_size, opt_method,
                                                  nas_nn_index, expand_factor)
                elif type_eda==30:
                    trigger_No_Improvement = nas_index
                    xopt,_= expand_copula_EDA(problem, problem.dimension, lower_bound, upper_bound, population_size,
                                              truncation_value, replacement, max_generations, copula_function, vine_level,
                                              trigger_No_Improvement)                    
                elif type_eda==31:
                    trigger_No_Improvement = nas_index
                    xopt,_= obj_expand_copula_EDA(problem, problem.dimension, lower_bound, upper_bound, population_size,
                                                  truncation_value, replacement, max_generations, copula_function, vine_level,
                                                  trigger_No_Improvement)          
                    
                    
            
                problem(xopt)  # make sure the returned solution is evaluated

                repeater.track(problem)  # track evaluations and final_target_hit
                minimal_print(problem)  # show progress

                

        ### post-process data
        #dsl = cocopp.main(observer.result_folder + ' bfgs!')  # re-run folders look like "...-001" etc
        #dsl = cocopp.main(observer.result_folder)  # re-run folders look like "...-001" etc
    
    
    
        #best_solution = select_EDA(type_eda,the_function, n, lower_bound, upper_bound, population_size, truncation_value, replacement, max_generations,  num_epochs, batch_size, num_iterations)

        #print("NExpe ", i) 
        #print(f"Best Solution: {best_solution}")
        #print(f"Best Fitness: {best_fitness}")
    
    
    # python3.11 benchmarking_EDAs.py 111 5  1 0 100 20 25 0 2 0
    # python3.11 benchmarking_EDAs.py 111 2 31 6 200 20 25 0 2 0
    # python3.11 benchmarking_EDAs.py 111 2 1  13 200 20 25 0 5 12
    # python3.11 benchmarking_EDAs.py 111 5 1  30 200 20 25 2 0 12
