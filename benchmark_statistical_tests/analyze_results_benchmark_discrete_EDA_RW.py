import sys
import os
import numpy as np
import csv

# Folder where Benchmark DbD .dat files are stored
folder = 'results_benchmark_EDA_RW/'


if __name__ == '__main__':
    # Fixed parameters from launch_benchmark_dbd_experiments.py
    n_gen = 250  
    problem = 'SAT'
    n = 100
    p_size = n * 5   
    truncation_values = [0.1,0.5]
    alpha_values = [0,0.95]  # Standard threshold for mutation
    discrete_EDAs = ['UMDA', 'TreeEDA', 'EBNA', 'MN-FDA', 'MN-FDAG', 'MK-EDA1', 'MK-EDA2', 'MK-EDA3']    
    
    # Instance names based on problem type
    if problem == 'SAT':
        instance_names = ['uf100-01', 'uf100-02', 'uf100-03', 'uf100-04', 'uf100-05']
        optimal_fitness_values = [430]*5
       
    elif problem == 'Ising':
        instance_names = ['SG_100_1', 'SG_100_2', 'SG_100_3', 'SG_100_4']
        #optimal_fitness_values = [130,136,136,130]
        optimal_fitness_values = [132,142,142,138]
    elif problem == 'UBQP':
        instance_names = ['bqp100']
        optimal_fitness_values = [3955]
    else:
        instance_names = []
        optimal_fitness_fixed = 0
 
    seeds = np.arange(1, 21)
    results_list = []

    print(f"Extracting data from .dat files in {folder}...")

    for seed in seeds:
        for i,instance_name in enumerate(instance_names):
            optimal_fitness_fixed = optimal_fitness_values[i]
            for alg in discrete_EDAs:
                for alpha in alpha_values:
                    for truncation_ratio in truncation_values:
                                                fname = f"results_benchmark_EDA_RW_{problem}_{instance_name}_{p_size}_{n_gen}_{alg}_{alpha}_{truncation_ratio}_{seed}.dat"                                           

                                                full_path = os.path.join(folder, fname)                                               
                                                best_fitness = None
                                                generation = None
                                                elapsed_time = None
                                                
                                                try:
                                                    #if seed==1:
                                                    #    print(full_path)
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
                                                            'alg': alg,                                                       
                                                            'alpha': alpha,                                                            
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
        group_cols = ['instance', 'alg', 'alpha']

        grouped = df.groupby(group_cols).agg({
            'success': 'mean',
            'best_fitness': 'mean',
            'generation': 'mean',
            'elapsed_time': 'mean'
        }).reset_index()

        print(f"\n=== TOP 10 CONFIGURATIONS PER INSTANCE ({problem}) ===")
        print("Criteria: 1. Max Success, 2. Max Fitness, 3. Min Generation, 4. Min Time\n")

        for inst in instance_names:
            inst_df = grouped[grouped['instance'] == inst].copy()
            if inst_df.empty: continue

            ranked = inst_df.sort_values(
                by=['success', 'best_fitness', 'generation', 'elapsed_time'],
                ascending=[False, False, True, True]
            )

            print(f"--- Instance: {inst} ---")
            top_configs = ranked.head(10)

            cols_to_show = ['alg', 'alpha', 'success', 'best_fitness', 'generation', 'elapsed_time']
            print(top_configs[cols_to_show].to_string(index=False))
            print("\n")

        # Save to CSV
        output_file = f'discrete_EDA_RW_benchmark_{problem.lower()}_results.csv'
        grouped.to_csv(output_file, index=False)
        print(f"Detailed ranked results saved to '{output_file}'")

    except ImportError:
        print("Pandas is required for ranking. Install with: pip install pandas")

        
