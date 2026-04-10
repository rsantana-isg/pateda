import sys
import os
import numpy as np
import json
import csv

# Directory settings
#folder = 'results_discrete_vae_N15n/'
folder = 'results_discrete_eda_v2_450_n30/'
n1 = 30
n2 = 64

if __name__ == '__main__':
    obj_functions = ['OneMax', 'KDeceptive3', 'Deceptive3', 'HIFF', 'KDeceptive5']
    NN_EDAs = ['UMDA', 'TreeEDA', 'EBNA', 'MN-FDA', 'MN-FDAG', 'MK-EDA1', 'MK-EDA2', 'MK-EDA3']
    n_gen = 250 
    results_list = []

    print(f"Extracting data from {folder}...")

    # Data Extraction Loop
    for seed in np.arange(1, 21):
        for alg in NN_EDAs:
            for i in range(len(obj_functions)):
                fn = obj_functions[i]
                n = n2 if fn == 'HIFF' else n1
                p_size = n * 15  
                fname = f"results_discrete_eda_v2_{fn}_{n}_{p_size}_{n_gen}_{alg}_{seed}.dat"
                
                try:
                    with open(os.path.join(folder, fname), 'r') as file:
                        lines = file.readlines()
                        best_fitness = None
                        generation = None
                        elapsed_time = None
                        optimal_fitness = None
                        
                        for line in lines:
                            if 'Best fitness found:' in line:
                                best_fitness = float(line.split(': ')[1].strip())
                            elif 'at generation' in line:
                                val = line.split(':')[-1].strip() if ':' in line else line.split(' ')[-1].strip()
                                generation = int(val)
                            elif 'Elapsed Time:' in line:
                                elapsed_time = float(line.split(':')[1].strip().split()[0])
                            elif 'Optimal Fitness:' in line:
                                optimal_fitness = float(line.split(':')[1].strip())
                        
                        if best_fitness is not None:
                            results_list.append({
                                'objective_function': fn,
                                'algorithm': alg,
                                'seed': seed,
                                'best_fitness': best_fitness,
                                'generation': generation,
                                'elapsed_time': elapsed_time,
                                'optimal_fitness': optimal_fitness,
                                'success': 1.0 if best_fitness == optimal_fitness else 0.0
                            })
                except FileNotFoundError:
                    continue

    # Ranking and Display Logic (Inspired by analyze_results_discrete_Backdrive.py)
    try:
        import pandas as pd
        df = pd.DataFrame(results_list)
        
        if df.empty:
            print("No data found to analyze.")
            sys.exit()

        # Group by objective and algorithm to calculate averages
        grouped = df.groupby(['objective_function', 'algorithm']).agg({
            'success': 'mean',
            'best_fitness': 'mean',
            'generation': 'mean',
            'elapsed_time': 'mean'
        }).reset_index()

        print("\n" + "="*80)
        print("TOP 15 CONFIGURATIONS PER OBJECTIVE FUNCTION")
        print("Criteria: 1. Max Success, 2. Max Fitness, 3. Min Generation, 4. Min Time")
        print("="*80 + "\n")

        for obj in obj_functions:
            obj_df = grouped[grouped['objective_function'] == obj].copy()
            if obj_df.empty:
                continue

            # Sorting based on the requested criteria
            ranked = obj_df.sort_values(
                by=['success', 'best_fitness', 'generation', 'elapsed_time'],
                ascending=[False, False, True, True]
            )

            print(f"--- Objective: {obj} ---")
            top_15 = ranked.head(15)
            
            # Clean display formatting
            display_df = top_15.rename(columns={
                'algorithm': 'Algorithm',
                'success': 'Success Rate',
                'best_fitness': 'Avg Fitness',
                'generation': 'Avg Gen',
                'elapsed_time': 'Avg Time'
            })
            
            # Print only relevant columns
            cols = ['Algorithm', 'Success Rate', 'Avg Fitness', 'Avg Gen', 'Avg Time']
            print(display_df[cols].to_string(index=False))
            print("\n")

        # Save aggregated results to CSV
        grouped.to_csv('eda_results_ranked_n30.csv', index=False)
        print("Full ranked results saved to eda_results_ranked_n30.csv")

    except ImportError:
        print("Pandas is required for ranking. Install with: pip install pandas")
