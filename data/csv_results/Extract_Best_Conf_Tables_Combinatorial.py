 
import pandas as pd

def map_act(a):
    m = {'relu': 'R', 'elu': 'E', 'tanh': 'T'}
    return m.get(str(a).lower(), str(a))

def map_loss(l):
    m = {'mse': 'M', 'weighted_mse': 'WM', 'ranking': 'RK', 'huber': 'H'}
    return m.get(str(l).lower(), '-')

def map_fg(f):
    if pd.isna(f): return '-'
    return 'Y' if float(f) > 0 else 'N'

def generate_best_config_latex():
    problems = ['sat', 'ising', 'ubqp']
    algos = ['DiffEDA', 'DbD-CS', 'DbD-CD', 'C-VAE']
    all_data = []

    # Extract all relevant algorithm data from files
    for p in problems:
        # DiffEDA (Dendiff)
        try:
            df = pd.read_csv(f'dendiff_benchmark_{p}_results.csv')
            df = df[df['variant'] == 'dendiff_gumbel'].copy()
            df['alg'] = 'DiffEDA'
            all_data.append(df)
        except: pass
        
        # DbD Variants
        try:
            df = pd.read_csv(f'dbd_benchmark_{p}_results.csv')
            for v, label in [('dbd_cs', 'DbD-CS'), ('dbd_cd', 'DbD-CD')]:
                subset = df[df['variant'] == v].copy()
                subset['alg'] = label
                all_data.append(subset)
        except: pass

        # C-VAE
        try:
            df = pd.read_csv(f'vae_benchmark_{p}_results.csv')
            vae = df[df['variant'] == 'C-VAE'].copy()
            vae['alg'] = 'C-VAE'
            all_data.append(vae)
        except: pass

    combined = pd.concat(all_data, ignore_index=True, sort=False)
    
    # Instance ordering (SAT -> Ising -> UBQP)
    def inst_sort(x):
        if x.startswith('uf'): return (0, x)
        if x.startswith('SG'): return (1, x)
        return (2, x)

    sorted_instances = sorted(combined['instance'].unique(), key=inst_sort)
    final_rows = []

    for inst in sorted_instances:
        config_row = {'Instance': inst, 'Type': 'Config'}
        fitness_row = {'Instance': '', 'Type': 'Fitness'}
        
        for alg in algos:
            subset = combined[(combined['instance'] == inst) & (combined['alg'] == alg)]
            if subset.empty:
                config_row[alg] = 'N/A'; fitness_row[alg] = 'N/A'
            else:
                best_row = subset.loc[subset['best_fitness'].idxmax()]
                # Compact configuration formatting
                if alg == 'C-VAE':
                    act = f"{map_act(best_row['act_enc'])}/{map_act(best_row['act_dec'])}"
                    loss = 'VAE'
                    fg = map_fg(best_row.get('mi_layer', 0))
                else:
                    act = map_act(best_row['activation'])
                    loss = map_loss(best_row['loss'])
                    fg = map_fg(best_row['fitness_guided'])
                
                config_row[alg] = f"[{act}, {loss}, {fg}]"
                fitness_row[alg] = f"{best_row['best_fitness']:.2f}"
                
        final_rows.append(config_row)
        final_rows.append(fitness_row)

    # Convert to LaTeX
    df_final = pd.DataFrame(final_rows)
    caption = (
        "Best fitness and corresponding configurations [Activation, Loss, Fitness Guided]. "
        "Acronyms: R (ReLU), E (ELU), T (Tanh), M (MSE), WM (Weighted MSE), RK (Ranking), "
        "H (Huber), Y/N (Yes/No for Fitness Guided), VAE (Standard VAE Loss)."
    )
    return df_final.to_latex(index=False, caption=caption, label="tab:best_configs", column_format="llcccc")

print(generate_best_config_latex())
 
