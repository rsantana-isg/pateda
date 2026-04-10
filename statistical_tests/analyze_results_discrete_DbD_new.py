import sys
import os
import numpy as np
import csv

# Folder where DbD .dat files are stored
folder = 'results_DbD_n30_N15n_new/'
n_small = 30
n_large = 64
alpha = 0.95

if __name__ == '__main__':
    # Define parameter ranges from launch_dbd_experiments.py
    n_gen = 250
    trunc = 0.5
    alpha_smooth = 0.1
    use_markov_init_values = [0,1]
    num_alpha_samples_list = [100]
    n_steps_list = [20]
    k_values = [1,2]
    fitness_guided_values = [0, 1]
    
    obj_functions = ['OneMax', 'KDeceptive3', 'Deceptive3', 'HIFF', 'KDeceptive5', 'FC5']
    dbd_variants = ['dbd_cs', 'dbd_cd','dbd_cs_t', 'dbd_cd_t']
    activations = ['elu', 'relu', 'tanh']
    loss_functions = ['mse', 'weighted_mse', 'ranking', 'huber']
    seeds = np.arange(1, 21)

    results_list = []

    print(f"Extracting data from .dat files in {folder}...")

    for seed in seeds:
        for obj_func in obj_functions:
            n = n_large if obj_func == 'HIFF' else n_small
            p_size = n * 15
            
            for variant in dbd_variants:
                for activation in activations:
                    for loss in loss_functions:
                        for alpha_s in num_alpha_samples_list:
                            for n_steps in n_steps_list:
                                # Logic for k based on variant suffix '_t'
                                k_list = k_values if '_t' in variant else [0]
                                
                                for k in k_list:
                                    for fg in fitness_guided_values:
                                        for m_init in use_markov_init_values:
                                            
                                            # Construct filename based on slurm_dbd.sh redirect pattern:
                                            # results_dbd_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}_{11}_{12}_{13}_{14}_{15}_{16}_{2}.dat
                                            # {3}=obj, {4}=n, {5}=p_size, {6}=n_gen, {7}=trunc, {8}=variant, {9}=act, 
                                            # {10}=loss, {11}=num_alpha, {12}=n_steps, {13}=k, {14}=alpha_s, {15}=fg, {16}=m_init, {2}=seed
                                            fname = (f"results_dbd_{obj_func}_{n}_{p_size}_{n_gen}_{trunc}_"
                                                     f"{variant}_{activation}_{loss}_{alpha_s}_{n_steps}_"
                                                     f"{k}_{alpha_smooth}_{fg}_{m_init}_{alpha}_{seed}.dat")
                                            
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
                                                        'alpha': alpha_s,
                                                        'n_steps': n_steps,
                                                        'k': k,
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
        group_cols = ['objective', 'variant', 'activation', 'loss', 'alpha', 'n_steps', 'k', 'fitness_guided']
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
            top_configs = ranked.head(30)
            
            # Display relevant columns
            cols_to_show = ['variant', 'activation', 'loss', 'fitness_guided', 'alpha', 'success', 'best_fitness', 'generation', 'elapsed_time']
            print(top_configs[cols_to_show].to_string(index=False))
            print("\n")

        # Save to CSV
        output_file = 'dbd_results_ranked_n30.csv'
        grouped.to_csv(output_file, index=False)
        print(f"Detailed ranked results saved to '{output_file}'")

    except ImportError:
        print("Pandas is required for ranking. Install with: pip install pandas")

        
