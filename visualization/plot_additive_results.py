#!/usr/bin/env python3
"""
Generate figures illustrating EDA algorithm performance on additive functions.

This script creates publication-quality EPS figures showing:
- Best fitness comparison across algorithms
- Success rate comparison
- Impact of loss and activation functions
- Average generations needed
- Average elapsed time

Author: PATEDA Team
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set publication style
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 300

# Define paths
DATA_DIR = Path(__file__).parent.parent / "data" / "csv_results" / "additive"
OUTPUT_DIR = Path(__file__).parent.parent / "figures" / "additive"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load all CSV files with results."""
    eda_df = pd.read_csv(DATA_DIR / "eda_results_ranked_n30.csv")
    dbd_df = pd.read_csv(DATA_DIR / "dbd_results_ranked_n30.csv")
    dendiff_df = pd.read_csv(DATA_DIR / "dendiff_results_ranked_n30.csv")
    vae_df = pd.read_csv(DATA_DIR / "vae_results_ranked_n30.csv")
    
    return eda_df, dbd_df, dendiff_df, vae_df


def prepare_eda_data(eda_df):
    """Standardize EDA dataframe columns."""
    eda_df = eda_df.copy()
    eda_df.rename(columns={'objective_function': 'objective', 'algorithm': 'alg'}, inplace=True)
    return eda_df


def plot_best_fitness_comparison(eda_df, dbd_df, dendiff_df, vae_df):
    """Create bar plot comparing best fitness across algorithms for each problem."""
    print("Generating best fitness comparison plot...")
    
    # Prepare data
    eda_df = prepare_eda_data(eda_df)
    
    # Get unique objectives
    objectives = sorted(eda_df['objective'].unique())
    
    fig, axes = plt.subplots(1, len(objectives), figsize=(15, 4))
    if len(objectives) == 1:
        axes = [axes]
    
    for idx, obj in enumerate(objectives):
        ax = axes[idx]
        
        # Traditional EDAs
        eda_data = eda_df[eda_df['objective'] == obj].groupby('alg')['best_fitness'].mean()
        
        # DbD EDAs - average across all configurations
        dbd_data = dbd_df[dbd_df['objective'] == obj].groupby('variant')['best_fitness'].mean()
        
        # Dendiff EDAs - average across all configurations
        dendiff_data = dendiff_df[dendiff_df['objective'] == obj].groupby('variant')['best_fitness'].mean()
        
        # VAE EDAs - average across all configurations
        vae_data = vae_df[vae_df['objective'] == obj].groupby('variant')['best_fitness'].mean()
        
        # Combine all data
        all_algs = []
        all_fitness = []
        all_types = []
        
        for alg, val in eda_data.items():
            all_algs.append(alg)
            all_fitness.append(val)
            all_types.append('Traditional EDA')
        
        for alg, val in dbd_data.items():
            all_algs.append(alg)
            all_fitness.append(val)
            all_types.append('DbD-EDA')
        
        for alg, val in dendiff_data.items():
            all_algs.append(alg)
            all_fitness.append(val)
            all_types.append('Diff-EDA')
        
        for alg, val in vae_data.items():
            all_algs.append(alg)
            all_fitness.append(val)
            all_types.append('VAE-EDA')
        
        # Create dataframe for plotting
        plot_df = pd.DataFrame({
            'Algorithm': all_algs,
            'Best Fitness': all_fitness,
            'Type': all_types
        })
        
        # Sort by fitness
        plot_df = plot_df.sort_values('Best Fitness', ascending=False)
        
        # Define colors
        type_colors = {
            'Traditional EDA': '#1f77b4',
            'DbD-EDA': '#ff7f0e',
            'Diff-EDA': '#2ca02c',
            'VAE-EDA': '#d62728'
        }
        
        colors = [type_colors[t] for t in plot_df['Type']]
        
        # Plot
        ax.barh(range(len(plot_df)), plot_df['Best Fitness'], color=colors)
        ax.set_yticks(range(len(plot_df)))
        ax.set_yticklabels(plot_df['Algorithm'], fontsize=8)
        ax.set_xlabel('Average Best Fitness', fontsize=10)
        ax.set_title(obj, fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add legend only to first subplot
        if idx == 0:
            handles = [plt.Rectangle((0,0),1,1, color=c) for c in type_colors.values()]
            ax.legend(handles, type_colors.keys(), loc='best', fontsize=8)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / "best_fitness_comparison.eps"
    plt.savefig(output_file, format='eps', bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_success_rate_comparison(eda_df, dbd_df, dendiff_df, vae_df):
    """Create bar plot comparing success rates across algorithms."""
    print("Generating success rate comparison plot...")
    
    eda_df = prepare_eda_data(eda_df)
    objectives = sorted(eda_df['objective'].unique())
    
    fig, axes = plt.subplots(1, len(objectives), figsize=(15, 4))
    if len(objectives) == 1:
        axes = [axes]
    
    for idx, obj in enumerate(objectives):
        ax = axes[idx]
        
        # Traditional EDAs
        eda_data = eda_df[eda_df['objective'] == obj].groupby('alg')['success'].mean()
        
        # DbD EDAs
        dbd_data = dbd_df[dbd_df['objective'] == obj].groupby('variant')['success'].mean()
        
        # Dendiff EDAs
        dendiff_data = dendiff_df[dendiff_df['objective'] == obj].groupby('variant')['success'].mean()
        
        # VAE EDAs
        vae_data = vae_df[vae_df['objective'] == obj].groupby('variant')['success'].mean()
        
        # Combine all data
        all_algs = []
        all_success = []
        all_types = []
        
        for alg, val in eda_data.items():
            all_algs.append(alg)
            all_success.append(val)
            all_types.append('Traditional EDA')
        
        for alg, val in dbd_data.items():
            all_algs.append(alg)
            all_success.append(val)
            all_types.append('DbD-EDA')
        
        for alg, val in dendiff_data.items():
            all_algs.append(alg)
            all_success.append(val)
            all_types.append('Diff-EDA')
        
        for alg, val in vae_data.items():
            all_algs.append(alg)
            all_success.append(val)
            all_types.append('VAE-EDA')
        
        # Create dataframe
        plot_df = pd.DataFrame({
            'Algorithm': all_algs,
            'Success Rate': all_success,
            'Type': all_types
        })
        
        # Sort by success rate
        plot_df = plot_df.sort_values('Success Rate', ascending=False)
        
        # Define colors
        type_colors = {
            'Traditional EDA': '#1f77b4',
            'DbD-EDA': '#ff7f0e',
            'Diff-EDA': '#2ca02c',
            'VAE-EDA': '#d62728'
        }
        
        colors = [type_colors[t] for t in plot_df['Type']]
        
        # Plot
        ax.barh(range(len(plot_df)), plot_df['Success Rate'], color=colors)
        ax.set_yticks(range(len(plot_df)))
        ax.set_yticklabels(plot_df['Algorithm'], fontsize=8)
        ax.set_xlabel('Success Rate', fontsize=10)
        ax.set_title(obj, fontsize=11, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.grid(axis='x', alpha=0.3)
        
        if idx == 0:
            handles = [plt.Rectangle((0,0),1,1, color=c) for c in type_colors.values()]
            ax.legend(handles, type_colors.keys(), loc='best', fontsize=8)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / "success_rate_comparison.eps"
    plt.savefig(output_file, format='eps', bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_activation_loss_impact(dbd_df, dendiff_df):
    """Plot heatmaps showing impact of activation and loss functions."""
    print("Generating activation/loss function impact plots...")
    
    # Get unique objectives
    objectives = sorted(dbd_df['objective'].unique())
    
    # DbD-EDA heatmap
    fig, axes = plt.subplots(2, len(objectives), figsize=(15, 8))
    if len(objectives) == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, obj in enumerate(objectives):
        # DbD-EDA
        ax1 = axes[0, idx]
        dbd_obj = dbd_df[dbd_df['objective'] == obj]
        pivot_dbd = dbd_obj.groupby(['activation', 'loss'])['best_fitness'].mean().reset_index()
        pivot_dbd = pivot_dbd.pivot(index='activation', columns='loss', values='best_fitness')
        
        sns.heatmap(pivot_dbd, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax1, cbar_kws={'label': 'Best Fitness'})
        ax1.set_title(f'DbD-EDA: {obj}', fontweight='bold')
        ax1.set_xlabel('Loss Function')
        ax1.set_ylabel('Activation Function')
        
        # Diff-EDA (Dendiff)
        ax2 = axes[1, idx]
        dendiff_obj = dendiff_df[dendiff_df['objective'] == obj]
        pivot_dendiff = dendiff_obj.groupby(['activation', 'loss'])['best_fitness'].mean().reset_index()
        pivot_dendiff = pivot_dendiff.pivot(index='activation', columns='loss', values='best_fitness')
        
        sns.heatmap(pivot_dendiff, annot=True, fmt='.2f', cmap='YlGnBu', ax=ax2, cbar_kws={'label': 'Best Fitness'})
        ax2.set_title(f'Diff-EDA: {obj}', fontweight='bold')
        ax2.set_xlabel('Loss Function')
        ax2.set_ylabel('Activation Function')
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / "activation_loss_impact_fitness.eps"
    plt.savefig(output_file, format='eps', bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    
    # Success rate heatmap
    fig, axes = plt.subplots(2, len(objectives), figsize=(15, 8))
    if len(objectives) == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, obj in enumerate(objectives):
        # DbD-EDA
        ax1 = axes[0, idx]
        dbd_obj = dbd_df[dbd_df['objective'] == obj]
        pivot_dbd = dbd_obj.groupby(['activation', 'loss'])['success'].mean().reset_index()
        pivot_dbd = pivot_dbd.pivot(index='activation', columns='loss', values='success')
        
        sns.heatmap(pivot_dbd, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax1, vmin=0, vmax=1, cbar_kws={'label': 'Success Rate'})
        ax1.set_title(f'DbD-EDA: {obj}', fontweight='bold')
        ax1.set_xlabel('Loss Function')
        ax1.set_ylabel('Activation Function')
        
        # Diff-EDA
        ax2 = axes[1, idx]
        dendiff_obj = dendiff_df[dendiff_df['objective'] == obj]
        pivot_dendiff = dendiff_obj.groupby(['activation', 'loss'])['success'].mean().reset_index()
        pivot_dendiff = pivot_dendiff.pivot(index='activation', columns='loss', values='success')
        
        sns.heatmap(pivot_dendiff, annot=True, fmt='.2f', cmap='YlGnBu', ax=ax2, vmin=0, vmax=1, cbar_kws={'label': 'Success Rate'})
        ax2.set_title(f'Diff-EDA: {obj}', fontweight='bold')
        ax2.set_xlabel('Loss Function')
        ax2.set_ylabel('Activation Function')
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / "activation_loss_impact_success.eps"
    plt.savefig(output_file, format='eps', bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_generations_comparison(eda_df, dbd_df, dendiff_df, vae_df):
    """Create box plots comparing number of generations across algorithms."""
    print("Generating generations comparison plot...")
    
    eda_df = prepare_eda_data(eda_df)
    objectives = sorted(eda_df['objective'].unique())
    
    fig, axes = plt.subplots(1, len(objectives), figsize=(15, 5))
    if len(objectives) == 1:
        axes = [axes]
    
    for idx, obj in enumerate(objectives):
        ax = axes[idx]
        
        # Prepare data for box plot
        data_list = []
        labels_list = []
        colors_list = []
        
        # Traditional EDAs
        for alg in eda_df[eda_df['objective'] == obj]['alg'].unique():
            data = eda_df[(eda_df['objective'] == obj) & (eda_df['alg'] == alg)]['generation']
            if len(data) > 0:
                data_list.append(data.values)
                labels_list.append(alg)
                colors_list.append('#1f77b4')
        
        # DbD EDAs - select representative variants
        for variant in dbd_df[dbd_df['objective'] == obj]['variant'].unique()[:2]:
            data = dbd_df[(dbd_df['objective'] == obj) & (dbd_df['variant'] == variant)]['generation']
            if len(data) > 0:
                data_list.append(data.values)
                labels_list.append(variant)
                colors_list.append('#ff7f0e')
        
        # Dendiff EDAs
        for variant in dendiff_df[dendiff_df['objective'] == obj]['variant'].unique()[:2]:
            data = dendiff_df[(dendiff_df['objective'] == obj) & (dendiff_df['variant'] == variant)]['generation']
            if len(data) > 0:
                data_list.append(data.values)
                labels_list.append(variant)
                colors_list.append('#2ca02c')
        
        # Create box plot
        bp = ax.boxplot(data_list, labels=labels_list, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.set_ylabel('Number of Generations', fontsize=10)
        ax.set_title(obj, fontsize=11, fontweight='bold')
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / "generations_comparison.eps"
    plt.savefig(output_file, format='eps', bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_time_comparison(eda_df, dbd_df, dendiff_df, vae_df):
    """Create bar plot comparing elapsed time across algorithms."""
    print("Generating elapsed time comparison plot...")
    
    eda_df = prepare_eda_data(eda_df)
    objectives = sorted(eda_df['objective'].unique())
    
    fig, axes = plt.subplots(1, len(objectives), figsize=(15, 4))
    if len(objectives) == 1:
        axes = [axes]
    
    for idx, obj in enumerate(objectives):
        ax = axes[idx]
        
        # Traditional EDAs
        eda_data = eda_df[eda_df['objective'] == obj].groupby('alg')['elapsed_time'].mean()
        
        # DbD EDAs
        dbd_data = dbd_df[dbd_df['objective'] == obj].groupby('variant')['elapsed_time'].mean()
        
        # Dendiff EDAs
        dendiff_data = dendiff_df[dendiff_df['objective'] == obj].groupby('variant')['elapsed_time'].mean()
        
        # VAE EDAs
        vae_data = vae_df[vae_df['objective'] == obj].groupby('variant')['elapsed_time'].mean()
        
        # Combine all data
        all_algs = []
        all_times = []
        all_types = []
        
        for alg, val in eda_data.items():
            all_algs.append(alg)
            all_times.append(val)
            all_types.append('Traditional EDA')
        
        for alg, val in dbd_data.items():
            all_algs.append(alg)
            all_times.append(val)
            all_types.append('DbD-EDA')
        
        for alg, val in dendiff_data.items():
            all_algs.append(alg)
            all_times.append(val)
            all_types.append('Diff-EDA')
        
        for alg, val in vae_data.items():
            all_algs.append(alg)
            all_times.append(val)
            all_types.append('VAE-EDA')
        
        # Create dataframe
        plot_df = pd.DataFrame({
            'Algorithm': all_algs,
            'Elapsed Time': all_times,
            'Type': all_types
        })
        
        # Sort by time
        plot_df = plot_df.sort_values('Elapsed Time')
        
        # Define colors
        type_colors = {
            'Traditional EDA': '#1f77b4',
            'DbD-EDA': '#ff7f0e',
            'Diff-EDA': '#2ca02c',
            'VAE-EDA': '#d62728'
        }
        
        colors = [type_colors[t] for t in plot_df['Type']]
        
        # Plot with log scale
        ax.barh(range(len(plot_df)), plot_df['Elapsed Time'], color=colors)
        ax.set_yticks(range(len(plot_df)))
        ax.set_yticklabels(plot_df['Algorithm'], fontsize=8)
        ax.set_xlabel('Average Elapsed Time (seconds, log scale)', fontsize=10)
        ax.set_xscale('log')
        ax.set_title(obj, fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        if idx == 0:
            handles = [plt.Rectangle((0,0),1,1, color=c) for c in type_colors.values()]
            ax.legend(handles, type_colors.keys(), loc='best', fontsize=8)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / "elapsed_time_comparison.eps"
    plt.savefig(output_file, format='eps', bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def generate_summary_table(eda_df, dbd_df, dendiff_df, vae_df):
    """Generate LaTeX table with summary statistics."""
    print("Generating summary table...")
    
    eda_df = prepare_eda_data(eda_df)
    
    # Combine all datasets
    summary_data = []
    
    for obj in sorted(eda_df['objective'].unique()):
        # Traditional EDAs
        eda_obj = eda_df[eda_df['objective'] == obj]
        for alg in eda_obj['alg'].unique():
            alg_data = eda_obj[eda_obj['alg'] == alg]
            summary_data.append({
                'Objective': obj,
                'Algorithm': alg,
                'Type': 'Traditional EDA',
                'Success Rate': alg_data['success'].mean(),
                'Best Fitness': alg_data['best_fitness'].mean(),
                'Avg Generations': alg_data['generation'].mean(),
                'Avg Time (s)': alg_data['elapsed_time'].mean()
            })
        
        # DbD EDAs
        dbd_obj = dbd_df[dbd_df['objective'] == obj]
        for variant in dbd_obj['variant'].unique():
            var_data = dbd_obj[dbd_obj['variant'] == variant]
            summary_data.append({
                'Objective': obj,
                'Algorithm': variant,
                'Type': 'DbD-EDA',
                'Success Rate': var_data['success'].mean(),
                'Best Fitness': var_data['best_fitness'].mean(),
                'Avg Generations': var_data['generation'].mean(),
                'Avg Time (s)': var_data['elapsed_time'].mean()
            })
        
        # Dendiff EDAs
        dendiff_obj = dendiff_df[dendiff_df['objective'] == obj]
        for variant in dendiff_obj['variant'].unique():
            var_data = dendiff_obj[dendiff_obj['variant'] == variant]
            summary_data.append({
                'Objective': obj,
                'Algorithm': variant,
                'Type': 'Diff-EDA',
                'Success Rate': var_data['success'].mean(),
                'Best Fitness': var_data['best_fitness'].mean(),
                'Avg Generations': var_data['generation'].mean(),
                'Avg Time (s)': var_data['elapsed_time'].mean()
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save as CSV
    output_file = OUTPUT_DIR / "summary_table.csv"
    summary_df.to_csv(output_file, index=False, float_format='%.3f')
    print(f"Saved: {output_file}")
    
    # Generate LaTeX table
    latex_output = OUTPUT_DIR / "summary_table.tex"
    with open(latex_output, 'w') as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Summary of EDA Algorithm Performance on Additive Functions}\n")
        f.write("\\label{tab:additive_summary}\n")
        f.write("\\begin{tabular}{lllrrrr}\n")
        f.write("\\hline\n")
        f.write("Objective & Algorithm & Type & Success & Fitness & Generations & Time (s) \\\\\n")
        f.write("\\hline\n")
        
        for _, row in summary_df.iterrows():
            f.write(f"{row['Objective']} & {row['Algorithm']} & {row['Type']} & "
                   f"{row['Success Rate']:.2f} & {row['Best Fitness']:.2f} & "
                   f"{row['Avg Generations']:.1f} & {row['Avg Time (s)']:.1f} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"Saved: {latex_output}")


def main():
    """Main execution function."""
    print("=" * 60)
    print("EDA Algorithm Visualization - Additive Functions")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    eda_df, dbd_df, dendiff_df, vae_df = load_data()
    print(f"  Traditional EDAs: {len(eda_df)} records")
    print(f"  DbD-EDAs: {len(dbd_df)} records")
    print(f"  Diff-EDAs: {len(dendiff_df)} records")
    print(f"  VAE-EDAs: {len(vae_df)} records")
    
    # Generate plots
    print("\nGenerating figures...")
    plot_best_fitness_comparison(eda_df, dbd_df, dendiff_df, vae_df)
    plot_success_rate_comparison(eda_df, dbd_df, dendiff_df, vae_df)
    plot_activation_loss_impact(dbd_df, dendiff_df)
    plot_generations_comparison(eda_df, dbd_df, dendiff_df, vae_df)
    plot_time_comparison(eda_df, dbd_df, dendiff_df, vae_df)
    generate_summary_table(eda_df, dbd_df, dendiff_df, vae_df)
    
    print("\n" + "=" * 60)
    print(f"All figures saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
