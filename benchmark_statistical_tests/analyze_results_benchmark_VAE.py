import sys
import os
import numpy as np
import csv

# Folder where Benchmark VAE .dat files are stored
folder = 'results_benchmark_VAE/'

if __name__ == '__main__':
    # Fixed parameters from launch_benchmark_selected_vae_experiments.py
    n_gen = 250
    trunc = 0.1
    # You can change this to 'Ising' or 'UBQP' as needed
    problem = 'UBQP' 
    n = 100
    p_size = n * 5
    
    # Instance names based on problem type
    if problem == 'SAT':
        instance_names = ['uf100-01', 'uf100-02', 'uf100-03', 'uf100-04', 'uf100-05']         
        optimal_fitness_values = [430]*5
        
    elif problem == 'Ising':
        instance_names = ['SG_100_1', 'SG_100_2', 'SG_100_3', 'SG_100_4']      
        optimal_fitness_values = [132,142,142,138]
    elif problem == 'UBQP':
        instance_names = ['bqp100']
        optimal_fitness_values = [3955]
      
    else:
        instance_names = []
        optimal_fitness_fixed = 0

    # VAE Specific Parameters
    variants = ['E-VAE', 'C-VAE']
    activation_enc_options = ['elu', 'relu', 'tanh']
    activation_dec_options = ['elu', 'relu', 'tanh']
    beta_start_options = [0.0]
    beta_end_options = [1.0]
    latent_dim_options = [0]
    epochs_options = [50]
    mi_layer_options = [0]
    alpha_values = [0.0, 0.95]
    seeds = np.arange(1, 21)

    results_list = []

    print(f"Extracting data from .dat files in {folder}...")

    for seed in seeds:
           for i,instance_name in enumerate(instance_names):

            
            optimal_fitness_fixed = optimal_fitness_values[i]    
            for variant in variants:
                for alpha in alpha_values:
                    for act_enc in activation_enc_options:
                        for act_dec in activation_dec_options:
                            for b_start in beta_start_options:
                                for b_end in beta_end_options:
                                    for l_dim in latent_dim_options:
                                        for epochs in epochs_options:
                                            for mi in mi_layer_options:
                                                
                                                # Filename pattern from slurm_benchmark_vae.sh:
                                                # results_benchmark_vae_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}_{11}_{12}_{13}_{14}_{15}_{16}_{2}.dat
                                                # {3}=problem, {4}=instance, {5}=pop, {6}=gen, {7}=trunc, {8}=alg, {9}=act_enc, 
                                                # {10}=act_dec, {11}=b_start, {12}=b_end, {13}=l_dim, {14}=epochs, {15}=mi, {16}=alpha, {2}=seed
                                                fname = (f"results_benchmark_vae_{problem}_{instance_name}_{p_size}_{n_gen}_{trunc}_"
                                                         f"{variant}_{act_enc}_{act_dec}_{b_start}_{b_end}_{l_dim}_"
                                                         f"{epochs}_{mi}_{alpha}_{seed}.dat")

                                                full_path = os.path.join(folder, fname)
                                                
                                                best_fitness = None
                                                generation = None
                                                elapsed_time = None
                                                
                                                try:
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
                                                            'act_enc': act_enc,
                                                            'act_dec': act_dec,
                                                            'alpha': alpha,
                                                            'latent_dim': l_dim,
                                                            'mi_layer': mi,
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
        group_cols = ['instance', 'variant', 'act_enc', 'act_dec', 'alpha', 'latent_dim', 'mi_layer']

        grouped = df.groupby(group_cols).agg({
            'success': 'mean',
            'best_fitness': 'mean',
            'generation': 'mean',
            'elapsed_time': 'mean'
        }).reset_index()

        print(f"\n=== TOP CONFIGURATIONS PER INSTANCE ({problem}) ===")
        print("Criteria: 1. Max Success, 2. Max Fitness, 3. Min Gen, 4. Min Time\n")

        for inst in instance_names:
            inst_df = grouped[grouped['instance'] == inst].copy()
            if inst_df.empty: continue

            ranked = inst_df.sort_values(
                by=['success', 'best_fitness', 'generation', 'elapsed_time'],
                ascending=[False, False, True, True]
            )

            print(f"--- Instance: {inst} ---")
            top_configs = ranked.head(10) # Showing top 10 for brevity

            cols_to_show = ['variant', 'act_enc', 'act_dec', 'alpha', 'success', 'best_fitness', 'generation']
            print(top_configs[cols_to_show].to_string(index=False))
            print("\n")

        # Save to CSV
        output_file = f'vae_benchmark_{problem.lower()}_results.csv'
        grouped.to_csv(output_file, index=False)
        print(f"Detailed ranked results saved to '{output_file}'")

    except ImportError:
        print("Pandas is required for ranking. Install with: pip install pandas")
