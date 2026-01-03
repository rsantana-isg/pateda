import sys
import os
import numpy as np

n_gen = 50
n = 30 
p_size = 150
 
if __name__ == '__main__':
	#obj_functions = ['OneMax', 'KDeceptive3', 'Deceptive3', 'HIFF', 'KDeceptive5', 'FC5']
        obj_functions = ['OneMax']
        NN_EDAs = ['VAE','GAN', 'Backdrive', 'Backdrive-Random', 'Backdrive-PerturbBest', 'Backdrive-PerturbSelected', 'Backdrive-Adaptive', 'DAE', 'RBM', 'DbD', 'DbD-CS', 'DbD-CD', 'DbD-UC', 'DbD-US', 'UMDA', 'TreeEDA', 'EBNA', 'MOA','MN-FDA', 'MN-FDAG', 'MK-EDA1', 'MK-EDA2', 'MK-EDA3','MT-EDA2', 'MT-EDA3']
        for seed in np.arange(1,2):
            for alg in NN_EDAs:
                for i in range(len(obj_functions)):
                    A = "sbatch slurm_pateda.sh examples/discrete_EDA.py " +str(seed)+" "+obj_functions[i]+" "+str(n)+" "+str(p_size)+" "+str(n_gen) +" "+alg
                    print(A)
