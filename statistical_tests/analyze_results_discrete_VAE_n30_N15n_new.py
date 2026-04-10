import sys
import os
import numpy as np
import json
import csv

# Folder where .dat files are stored
folder = 'results_VAE_450_n30_0.95/' 
n_small = 30
n_large = 64

if __name__ == '__main__':
    # Define parameter ranges from launch_vae_experiments.py
    n_gen = 250
    trunc = 0.5
    obj_functions = ['OneMax', 'KDeceptive3', 'Deceptive3', 'HIFF', 'KDeceptive5', 'FC5']
    variants = ['VAE', 'E-VAE', 'C-VAE']
    activation_enc_options = ['elu', 'relu', 'tanh']
    activation_dec_options = ['elu', 'relu', 'tanh']
    beta_start_options = [0.0]
    beta_end_options = [1.0]
    latent_dim_options = [0]
    epochs_options = [50]
    mi_layer_options = [0]
    alpha = 0.95
    seeds = np.arange(1, 21)
    
    results_list = []

    print("Extracting data from .dat files...")

    for seed in seeds:
        for obj_func in obj_functions:
            n = n_large if obj_func == 'HIFF' else n_small
            p_size = n * 15
            
            for variant in variants:
                for act_enc in activation_enc_options:
                    for act_dec in activation_dec_options:
                        for b_start in beta_start_options:
                            for b_end in beta_end_options:
                                for l_dim in latent_dim_options:
                                    for eps in epochs_options:
                                      for mi_layer in mi_layer_options:
                                        # Filename pattern from slurm_vae.sh
                                        fname = (f"results_discrete_vae_{obj_func}_{n}_{p_size}_{n_gen}_"
                                                 f"{trunc}_{seed}_{variant}_{act_enc}_{act_dec}_"
                                                 f"{b_start}_{b_end}_{l_dim}_{eps}_{mi_layer}_{alpha}.dat")
                                        
                                        full_path = os.path.join(folder, fname)
                                        #print(full_path)
                                        best_fitness = None
                                        generation = None
                                        elapsed_time = None
                                        optimal_fitness = None
                                        
                                        try:
                                            with open(full_path, 'r') as file:
                                                # Pattern matching logic from analyze_results_discrete_EDA.py
                                                for line in file:
                                                    if 'Best fitness found:' in line:
                                                        best_fitness = float(line.split(': ')[1].strip())
                                                    elif 'at generation' in line:
                                                        generation = int(line.split(':')[-1].strip() if ':' in line else line.split(' ')[-1].strip())
                                                    elif 'Elapsed Time:' in line:
                                                        elapsed_time = float(line.split(':')[1].strip().split()[0])
                                                    elif 'Optimal Fitness:' in line:
                                                        optimal_fitness = float(line.split(':')[1].strip())
                                            
                                            if best_fitness is not None:
                                                results_list.append({
                                                    'objective': obj_func,
                                                    'variant': variant,
                                                    'enc': act_enc,
                                                    'dec': act_dec,
                                                    'beta_start': b_start,
                                                    'beta_end': b_end,
                                                    'latent_dim': l_dim,
                                                    'epochs': eps,
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
            print("No results found to analyze.")
            sys.exit()

        # Group by configuration parameters and calculate means
        grouped = df.groupby(['objective', 'variant', 'enc', 'dec', 'beta_start', 'beta_end', 'latent_dim', 'epochs']).agg({
            'success': 'mean',
            'best_fitness': 'mean',
            'generation': 'mean',
            'elapsed_time': 'mean'
        }).reset_index()

        print("\n=== TOP 10 CONFIGURATIONS PER OBJECTIVE FUNCTION ===")
        print("Criteria: 1. Max Success, 2. Max Fitness, 3. Min Generation, 4. Min Time\n")

        for obj in obj_functions:
            # Filter for specific objective
            obj_df = grouped[grouped['objective'] == obj].copy()
            
            if obj_df.empty:
                continue

            # Sort by user criteria
            # ascending=[False, False, True, True] matches your priority list
            ranked = obj_df.sort_values(
                by=['success', 'best_fitness', 'generation', 'elapsed_time'],
                ascending=[False, False, True, True]
            )

            print(f"--- Objective: {obj} ---")
            top_10 = ranked.head(10)
            
            # Print formatted results
            cols_to_show = ['variant', 'enc', 'dec', 'success', 'best_fitness', 'generation', 'elapsed_time']
            print(top_10[cols_to_show].to_string(index=False))
            print("\n")

        # Save to CSV
        grouped.to_csv('vae_results_ranked_n30.csv', index=False)
        print("Full ranked results saved to vae_results_rankedd_n30_N15n_milayer0.csv")

    except ImportError:
        print("Pandas is required for ranking. Install with: pip install pandas")
        
