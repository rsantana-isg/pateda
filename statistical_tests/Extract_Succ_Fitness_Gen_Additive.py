import pandas as pd

def generate_multi_column_table():
    # File mapping for the ranked results
    files = {
        "dbd": "dbd_results_ranked_n30.csv",
        "dendiff": "dendiff_results_ranked_n30.csv",
        "eda": "eda_results_ranked_n30.csv",
        "vae": "vae_results_ranked_n30.csv"
    }

    # Targeted problem instances and algorithms
    objectives = ['OneMax', 'KDeceptive3', 'Deceptive3', 'HIFF', 'KDeceptive5']
    algorithm_names = ['Diff-EDA', 'DbD-CS', 'UMDA', 'TreeEDA', 'EBNA', 'MN-FDAG', 'C-VAE']

    # Data structure to store Succ, Fitness, and Gen
    results = {obj: {alg: {'success': 'N/A', 'best_fitness': 'N/A', 'generation': 'N/A'} 
               for alg in algorithm_names} for obj in objectives}

    def fill_res(obj, alg, row):
        if obj in objectives:
            results[obj][alg] = {
                'success': f"{row['success']:.2f}",
                'best_fitness': f"{row['best_fitness']:.2f}",
                'generation': f"{row['generation']:.2f}"
            }

    # 1. Filter Diff-EDA: variant=dendiff_gumbel, activation=relu, loss=ranking, fitness_guided=0
    df_den = pd.read_csv(files['dendiff'])
    f_den = df_den[(df_den['variant'] == 'dendiff_gumbel') & (df_den['activation'] == 'relu') & 
                   (df_den['loss'] == 'ranking') & (df_den['fitness_guided'] == 0)]
    for _, row in f_den.iterrows():
        fill_res(row['objective'], 'Diff-EDA', row)

    # 2. Filter DbD-CS: variant=dbd_cs, activation=relu, loss=huber, fitness_guided=1
    df_dbd = pd.read_csv(files['dbd'])
    f_dbd = df_dbd[(df_dbd['variant'] == 'dbd_cs') & (df_dbd['activation'] == 'relu') & 
                   (df_dbd['loss'] == 'huber') & (df_dbd['fitness_guided'] == 1)]
    for _, row in f_dbd.iterrows():
        fill_res(row['objective'], 'DbD-CS', row)

    # 3. Filter EDAs: algorithm in [UMDA, TreeEDA, EBNA, MN-FDAG]
    df_eda = pd.read_csv(files['eda'])
    eda_algs = ['UMDA', 'TreeEDA', 'EBNA', 'MN-FDAG']
    f_eda = df_eda[df_eda['algorithm'].isin(eda_algs)]
    for _, row in f_eda.iterrows():
        fill_res(row['objective_function'], row['algorithm'], row)

    # 4. Filter C-VAE: variant=C-VAE, enc=relu, dec=relu
    df_vae = pd.read_csv(files['vae'])
    f_vae = df_vae[(df_vae['variant'] == 'C-VAE') & (df_vae['enc'] == 'relu') & (df_vae['dec'] == 'relu')]
    for _, row in f_vae.iterrows():
        fill_res(row['objective'], 'C-VAE', row)

    # Construct LaTeX Table with Multi-columns
    latex = "\\begin{table*}[t]\n\\centering\n\\small\n"
    latex += "\\caption{Comparison of Algorithm Performance (Success Rate, Best Fitness, and Generation)}\n"
    col_def = "l" + "ccc" * len(algorithm_names)
    latex += "\\begin{tabular}{" + col_def + "}\n\\toprule\n"
    
    # Header 1: Algorithm names spanning 3 columns each
    header1 = "Instance"
    for alg in algorithm_names:
        header1 += f" & \\multicolumn{{3}}{{c}}{{{alg}}}"
    latex += header1 + " \\\\\n"
    
    # Header 2: Stat names
    header2 = ""
    for _ in range(len(algorithm_names)):
        header2 += " & Succ. & Fitness & Gen."
    latex += header2 + " \\\\\n\\midrule\n"
    
    # Data Rows
    for obj in objectives:
        row_str = f"{obj}"
        for alg in algorithm_names:
            res = results[obj][alg]
            row_str += f" & {res['success']} & {res['best_fitness']} & {res['generation']}"
        latex += row_str + " \\\\\n"
    
    latex += "\\bottomrule\n\\end{tabular}\n\\end{table*}"
    return latex

print(generate_multi_column_table())
