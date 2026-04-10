import sys
import os
import numpy as np
import csv

# Folder where DbD .dat files are stored
folder = 'results_Dendiff_n30_N15n_new/'
n_small = 30
n_large = 64

n_timesteps = 400
n_sampling_steps = 20
temperature = 1.0
beta_start = 0.01
beta_end = 1
alpha = 0.95

if __name__ == '__main__':
    # Define parameter ranges from launch_dbd_experiments.py
    n_gen = 250
    trunc = 0.5
    alpha_smooth = 0.1  
    fitness_guided_values = [0, 1]
    
    obj_functions = ['OneMax', 'KDeceptive3', 'Deceptive3', 'HIFF', 'KDeceptive5', 'FC5']
    variants = ['dendiff_gumbel', 'dendiff_corruption', 'dendiff_ste', 'dendiff_hard_concrete', 'dendiff_deterministic']
    variants = ['dendiff_gumbel']
    activations = ['elu', 'relu', 'tanh']
    loss_functions = ['mse', 'weighted_mse', 'ranking', 'huber']
    seeds = np.arange(1, 21)

    results_list = []

    print(f"Extracting data from .dat files in {folder}...")

    for seed in seeds:
        for obj_func in obj_functions:
            n = n_large if obj_func == 'HIFF' else n_small
            p_size = n * 15
            
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
                                                                                      
                                            fname = (f"results_dendiff_{obj_func}_{n}_{p_size}_{n_gen}_{trunc}_"
                                                     f"{variant}_{sampling_strategy}_{activation}_{loss}_"
                                                     f"{n_timesteps}_{n_sampling_steps}_{fg}_"
                                                     f"{temperature}_{beta_start}_{beta_end}_{alpha}_{seed}.dat")

                                           
                                            
                                            full_path = os.path.join(folder, fname)
                                            
                                            best_fitness = None
                                            generation = None
                                            elapsed_time = None
                                            optimal_fitness = None
                                            
                                            try:
                                                if seed==1:
                                                        print(full_path)
                                                with open(full_path, 'r') as file:
                                                    for line in file:
                                                        if 'Best fitness found:' in line:
                                                            best_fitness = float(line.split(': ')[1].strip())
                                                        elif 'at generation' in line:
                                                            # Split by ':' then take last part, or split by space
                                                            parts = line.split(':')
                                                            generation = int(parts[-1].strip()) if len(parts) > 1 else int(line.split()[-1])
                                                        elif 'Elapsed Time:' in line:
                                                            elapsed_time = float(line.split(':')[1].strip().split()[0])
                                                        elif 'Optimal Fitness:' in line:
                                                            optimal_fitness = float(line.split(':')[1].strip())
                                                
                                                if best_fitness is not None:
                                                    results_list.append({
                                                        'objective': obj_func,
                                                        'variant': variant,
                                                        'activation': activation,
                                                        'loss': loss,                                                       
                                                        'fitness_guided': fg,
                                                        'best_fitness': best_fitness,
                                                        'generation': generation,
                                                        'elapsed_time': elapsed_time,
                                                        'success': 1.0 if best_fitness == optimal_fitness else 0.0
                                                    })
                                            except FileNotFoundError:
                                                pass

    # Process and Rank Results
    try:
        import pandas as pd
        df = pd.DataFrame(results_list)
        
        if df.empty:
            print("No results found to analyze. Check if the directory path and filenames are correct.")
            sys.exit()

        # Group by all configuration parameters
        group_cols = ['objective', 'variant', 'activation', 'loss', 'fitness_guided']
        grouped = df.groupby(group_cols).agg({
            'success': 'mean',
            'best_fitness': 'mean',
            'generation': 'mean',
            'elapsed_time': 'mean'
        }).reset_index()

        print("\n=== TOP 20 CONFIGURATIONS PER OBJECTIVE FUNCTION (DbD) ===")
        print("Criteria: 1. Max Success, 2. Max Fitness, 3. Min Generation, 4. Min Time\n")

        for obj in obj_functions:
            obj_df = grouped[grouped['objective'] == obj].copy()
            if obj_df.empty: continue

            ranked = obj_df.sort_values(
                by=['success', 'best_fitness', 'generation', 'elapsed_time'],
                ascending=[False, False, True, True]
            )

            print(f"--- Objective: {obj} ---")
            top_configs = ranked.head(20)
            
            # Display relevant columns
            cols_to_show = ['variant', 'activation', 'loss', 'fitness_guided', 'success', 'best_fitness', 'generation', 'elapsed_time']
            print(top_configs[cols_to_show].to_string(index=False))
            print("\n")

        # Save to CSV
        output_file = 'dendiff_results_ranked_n30.csv'
        grouped.to_csv(output_file, index=False)
        print(f"Detailed ranked results saved to '{output_file}'")

    except ImportError:
        print("Pandas is required for ranking. Install with: pip install pandas")

        
