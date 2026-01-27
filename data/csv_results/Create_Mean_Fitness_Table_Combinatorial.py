import pandas as pd

def generate_full_comparison_table():
    all_results = []
    problems = ['sat', 'ising', 'ubqp']
    # Specific order requested for the columns
    algo_order = ['DiffEDA', 'DbD-CS', 'DbD-CD', 'UMDA', 'TreeEDA', 'EBNA', 'MN-FDAG', 'C-VAE']

    for p in problems:
        # 1. VAE Results (C-VAE with relu/relu)
        try:
            df = pd.read_csv(f"vae_benchmark_{p}_results.csv")
            f = df[(df['alpha'] == 0.95) & (df['variant'] == 'C-VAE') & 
                   (df['act_enc'] == 'relu') & (df['act_dec'] == 'relu')].copy()
            f['algorithm'] = 'C-VAE'
            all_results.append(f[['instance', 'algorithm', 'best_fitness']])
        except FileNotFoundError: pass

        # 2. Discrete EDA Results (UMDA, TreeEDA, EBNA, MN-FDAG)
        try:
            df = pd.read_csv(f"discrete_EDA_RW_benchmark_{p}_results.csv")
            f = df[(df['alpha'] == 0.95) & (df['alg'].isin(['UMDA', 'TreeEDA', 'EBNA', 'MN-FDAG']))].copy()
            f['algorithm'] = f['alg']
            all_results.append(f[['instance', 'algorithm', 'best_fitness']])
        except FileNotFoundError: pass

        # 3. DBD Results (DbD-CS, DbD-CD with relu/mse/no-fg)
        try:
            df = pd.read_csv(f"dbd_benchmark_{p}_results.csv")
            f = df[(df['alpha'] == 0.95) & (df['activation'] == 'relu') & 
                   (df['loss'] == 'mse') & (df['fitness_guided'] == 0) &
                   (df['variant'].isin(['dbd_cs', 'dbd_cd']))].copy()
            f['algorithm'] = f['variant'].str.replace('dbd_cs', 'DbD-CS').str.replace('dbd_cd', 'DbD-CD')
            all_results.append(f[['instance', 'algorithm', 'best_fitness']])
        except FileNotFoundError: pass

        # 4. Dendiff Results (DiffEDA with relu/mse/no-fg)
        try:
            df = pd.read_csv(f"dendiff_benchmark_{p}_results.csv")
            f = df[(df['alpha'] == 0.95) & (df['activation'] == 'relu') & 
                   (df['loss'] == 'mse') & (df['fitness_guided'] == 0)].copy()
            f['algorithm'] = 'DiffEDA' # Mapped from internal variant name
            all_results.append(f[['instance', 'algorithm', 'best_fitness']])
        except FileNotFoundError: pass

    # Combine and Pivot
    final_df = pd.concat(all_results, ignore_index=True)
    pivot_df = final_df.pivot_table(index='instance', columns='algorithm', values='best_fitness')

    # Sort columns by requested order
    available_cols = [a for a in algo_order if a in pivot_df.columns]
    pivot_df = pivot_df[available_cols]

    # Custom row sort to group by problem type (SAT -> Ising -> UBQP)
    def row_priority(inst):
        if inst.startswith('uf'): return (0, inst)
        if inst.startswith('SG'): return (1, inst)
        if inst.startswith('bqp'): return (2, inst)
        return (3, inst)
    
    pivot_df = pivot_df.loc[sorted(pivot_df.index, key=row_priority)]

    # Generate LaTeX
    return pivot_df.to_latex(
        caption="Comparison of Best Fitness across SAT, Ising, and UBQP ($\\\\alpha=0.95$)",
        label="tab:all_results_fitness",
        column_format="l" + "c" * len(available_cols),
        float_format="%.2f",
        escape=False
    )

print(generate_full_comparison_table())
