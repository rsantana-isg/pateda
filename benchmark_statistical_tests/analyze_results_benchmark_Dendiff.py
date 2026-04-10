import sys
import os
import numpy as np
import csv

# Folder where Benchmark DbD .dat files are stored
folder = 'results_benchmark_Dendiff/'


n_timesteps = 400
n_sampling_steps = 20
temperature = 1.0
beta_start = 0.01
beta_end = 1
trunc = 0.1

problem = 'UBQP'
n = 100
p_size = n * 5
n_gen = 250


if __name__ == '__main__':
    # Fixed parameters from launch_benchmark_dbd_experiments.py
  
    
    # Instance names based on problem type
    if problem == 'SAT':
        instance_names = ['uf100-01', 'uf100-02', 'uf100-03', 'uf100-04', 'uf100-05']
        optimal_fitness_values = [430]*5
       
    elif problem == 'Ising':
        instance_names = ['SG_100_1', 'SG_100_2', 'SG_100_3', 'SG_100_4']
        #optimal_fitness_values = [130,136,138,132]
        optimal_fitness_values = [132,142,142,138]
    elif problem == 'UBQP':
        instance_names = ['bqp100']
        optimal_fitness_values = [3955]
    else:
        instance_names = []
        optimal_fitness_fixed = 0

    dbd_variants = ['dbd_cs', 'dbd_cd']
    activations = ['elu', 'relu', 'tanh']
    loss_functions = ['mse', 'weighted_mse', 'ranking', 'huber']
    fitness_guided_values = [0,1]


    variants = ['dendiff_gumbel', 'dendiff_corruption', 'dendiff_ste', 'dendiff_hard_concrete', 'dendiff_deterministic']
    #variants = ['dendiff_gumbel']
    activations = ['elu', 'relu', 'tanh']
    loss_functions = ['mse', 'weighted_mse', 'ranking', 'huber']
    alpha_values = [0, 0.95]
    seeds = np.arange(1, 21)

    results_list = []

    print(f"Extracting data from .dat files in {folder}...")

    for seed in seeds:
        for i,instance_name in enumerate(instance_names):

            
            optimal_fitness_fixed = optimal_fitness_values[i]
          
            for variant in variants:
                # Set variant-dependent parameters
                if variant == 'dendiff_gumbel':
                    sampling_strategy = 'gumbel'                 
                elif variant == 'dendiff_corruption':
                    sampling_strategy = 'corruption'                 
                elif variant == 'dendiff_hard_concrete':
                    sampling_strategy = 'hard_concrete'                      
                elif variant == 'dendiff_deterministic':
                    sampling_strategy = 'deterministic'
                    
                for activation in activations:
                    for loss in loss_functions:                     
                                    for fg in fitness_guided_values:                                      
                                            for alpha in alpha_values:
                                                fname = (f"results_benchmark_dendiff_{problem}_{instance_name}_{p_size}_{n_gen}_{trunc}_"
                                                         f"{variant}_{sampling_strategy}_{activation}_{loss}_"
                                                         f"{n_timesteps}_{n_sampling_steps}_{fg}_"
                                                         f"{temperature}_{beta_start}_{beta_end}_{alpha}_{seed}.dat")
                                                         

                                                full_path = os.path.join(folder, fname)
                                               

                                                
                                                best_fitness = None
                                                generation = None
                                                elapsed_time = None
                                                
                                                try:
                                                    if seed==1:
                                                        print(full_path)
                                                    with open(full_path, 'r') as file:
                                                        for line in file:
                                                            if 'Best fitness found:' in line:
                                                                best_fitness = float(line.split(': ')[1].strip())
                                                            elif 'at generation' in line:
                                                                parts = line.split(':')
                                                                generation = int(parts[-1].strip()) if len(parts) > 1 else int(line.split()[-1])
                                                            elif 'Elapsed Time:' in line:
                                                                elapsed_time = float(line.split(':')[1].strip().split()[0])

                                                    
                                                    if best_fitness is not None:
                                                        results_list.append({
                                                            'instance': instance_name,
                                                            'variant': variant,
                                                            'activation': activation,
                                                            'loss': loss,
                                                            'alpha': alpha,                                                        
                                                            'fitness_guided': fg,
                                                            'best_fitness': best_fitness,
                                                            'generation': generation,
                                                            'elapsed_time': elapsed_time,
                                                            'success': 1.0 if best_fitness >= optimal_fitness_fixed else 0.0
                                                        })
                                                except FileNotFoundError:
                                                    pass

    # Process and Rank Results
    try:
        import pandas as pd
        df = pd.DataFrame(results_list)
        
        if df.empty:
            print("No results found. Check directory path and filename patterns.")
            sys.exit()

        # Group by configuration parameters
        group_cols = ['instance', 'variant', 'activation', 'loss', 'alpha', 'fitness_guided']

        grouped = df.groupby(group_cols).agg({
            'success': 'mean',
            'best_fitness': 'mean',
            'generation': 'mean',
            'elapsed_time': 'mean'
        }).reset_index()

        print(f"\n=== TOP 20 CONFIGURATIONS PER INSTANCE ({problem}) ===")
        print("Criteria: 1. Max Success, 2. Max Fitness, 3. Min Generation, 4. Min Time\n")

        for inst in instance_names:
            inst_df = grouped[grouped['instance'] == inst].copy()
            if inst_df.empty: continue

            ranked = inst_df.sort_values(
                by=['success', 'best_fitness', 'generation', 'elapsed_time'],
                ascending=[False, False, True, True]
            )

            print(f"--- Instance: {inst} ---")
            top_configs = ranked.head(40)

            cols_to_show = ['variant', 'activation', 'loss', 'alpha', 'fitness_guided', 'success', 'best_fitness', 'generation', 'elapsed_time']
            print(top_configs[cols_to_show].to_string(index=False))
            print("\n")

        # Save to CSV
        output_file = f'dendiff_benchmark_{problem.lower()}_results.csv'
        grouped.to_csv(output_file, index=False)
        print(f"Detailed ranked results saved to '{output_file}'")

    except ImportError:
        print("Pandas is required for ranking. Install with: pip install pandas")

        
