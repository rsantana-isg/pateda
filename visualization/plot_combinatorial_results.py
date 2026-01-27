#!/usr/bin/env python3
"""
Generate figures illustrating EDA algorithm performance on combinatorial problems.

This script creates publication-quality EPS figures showing:
- Best fitness comparison across algorithms for Ising, UBQP, SAT
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
DATA_DIR = Path(__file__).parent.parent / "data" / "csv_results" / "combinatorial"
OUTPUT_DIR = Path(__file__).parent.parent / "figures" / "combinatorial"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(problem_type):
    """Load all CSV files for a specific problem type."""
    problem_dir = DATA_DIR / problem_type
    
    eda_file = problem_dir / f"discrete_EDA_RW_benchmark_{problem_type}_results.csv"
    dbd_file = problem_dir / f"dbd_benchmark_{problem_type}_results.csv"
    dendiff_file = problem_dir / f"dendiff_benchmark_{problem_type}_results.csv"
    vae_file = problem_dir / f"vae_benchmark_{problem_type}_results.csv"
    
    eda_df = pd.read_csv(eda_file) if eda_file.exists() else pd.DataFrame()
    dbd_df = pd.read_csv(dbd_file) if dbd_file.exists() else pd.DataFrame()
    dendiff_df = pd.read_csv(dendiff_file) if dendiff_file.exists() else pd.DataFrame()
    vae_df = pd.read_csv(vae_file) if vae_file.exists() else pd.DataFrame()
    
    return eda_df, dbd_df, dendiff_df, vae_df


def plot_best_fitness_by_instance(problem_type, eda_df, dbd_df, dendiff_df, vae_df):
    """Create grouped bar plot comparing best fitness across instances."""
    print(f"Generating best fitness plot for {problem_type}...")
    
    # Get unique instances
    if 'instance' in eda_df.columns:
        instances = sorted(eda_df['instance'].unique())[:10]  # Limit to first 10 for readability
    else:
        print(f"  Warning: No instance column found for {problem_type}")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(instances))
    width = 0.2
    
    # Calculate average best fitness per instance and algorithm type
    eda_means = []
    dbd_means = []
    dendiff_means = []
    vae_means = []
    
    for inst in instances:
        # Traditional EDAs (average across algorithms and alpha values)
        eda_val = eda_df[eda_df['instance'] == inst]['best_fitness'].mean()
        eda_means.append(eda_val)
        
        # DbD EDAs
        dbd_val = dbd_df[dbd_df['instance'] == inst]['best_fitness'].mean() if 'instance' in dbd_df.columns else 0
        dbd_means.append(dbd_val)
        
        # Dendiff EDAs
        dendiff_val = dendiff_df[dendiff_df['instance'] == inst]['best_fitness'].mean() if 'instance' in dendiff_df.columns else 0
        dendiff_means.append(dendiff_val)
        
        # VAE EDAs
        vae_val = vae_df[vae_df['instance'] == inst]['best_fitness'].mean() if 'instance' in vae_df.columns else 0
        vae_means.append(vae_val)
    
    # Create bars
    ax.bar(x - 1.5*width, eda_means, width, label='Traditional EDA', color='#1f77b4', alpha=0.8)
    ax.bar(x - 0.5*width, dbd_means, width, label='DbD-EDA', color='#ff7f0e', alpha=0.8)
    ax.bar(x + 0.5*width, dendiff_means, width, label='Diff-EDA', color='#2ca02c', alpha=0.8)
    ax.bar(x + 1.5*width, vae_means, width, label='VAE-EDA', color='#d62728', alpha=0.8)
    
    ax.set_xlabel('Problem Instance', fontsize=12)
    ax.set_ylabel('Average Best Fitness', fontsize=12)
    ax.set_title(f'{problem_type.upper()}: Best Fitness by Instance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([inst.split('_')[-1] for inst in instances], rotation=45, ha='right')
    ax.legend(loc='best')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / f"{problem_type}_best_fitness_by_instance.eps"
    plt.savefig(output_file, format='eps', bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_success_rate_by_algorithm(problem_type, eda_df, dbd_df, dendiff_df, vae_df):
    """Create bar plot comparing success rates across algorithms."""
    print(f"Generating success rate plot for {problem_type}...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Collect data
    all_algs = []
    all_success = []
    all_types = []
    
    # Traditional EDAs
    if len(eda_df) > 0:
        eda_grouped = eda_df.groupby('alg')['success'].mean()
        for alg, val in eda_grouped.items():
            all_algs.append(alg)
            all_success.append(val)
            all_types.append('Traditional EDA')
    
    # DbD EDAs
    if len(dbd_df) > 0:
        if 'variant' in dbd_df.columns:
            dbd_grouped = dbd_df.groupby('variant')['success'].mean()
            for alg, val in dbd_grouped.items():
                all_algs.append(alg)
                all_success.append(val)
                all_types.append('DbD-EDA')
    
    # Dendiff EDAs
    if len(dendiff_df) > 0:
        if 'variant' in dendiff_df.columns:
            dendiff_grouped = dendiff_df.groupby('variant')['success'].mean()
            for alg, val in dendiff_grouped.items():
                all_algs.append(alg)
                all_success.append(val)
                all_types.append('Diff-EDA')
    
    # VAE EDAs
    if len(vae_df) > 0:
        if 'variant' in vae_df.columns:
            vae_grouped = vae_df.groupby('variant')['success'].mean()
            for alg, val in vae_grouped.items():
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
    ax.barh(range(len(plot_df)), plot_df['Success Rate'], color=colors, alpha=0.8)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df['Algorithm'], fontsize=9)
    ax.set_xlabel('Success Rate', fontsize=12)
    ax.set_title(f'{problem_type.upper()}: Success Rate Comparison', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.grid(axis='x', alpha=0.3)
    
    # Add legend
    handles = [plt.Rectangle((0,0),1,1, color=c, alpha=0.8) for c in type_colors.values()]
    ax.legend(handles, type_colors.keys(), loc='best', fontsize=9)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / f"{problem_type}_success_rate_comparison.eps"
    plt.savefig(output_file, format='eps', bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_activation_loss_impact(problem_type, dbd_df, dendiff_df):
    """Plot heatmaps showing impact of activation and loss functions."""
    print(f"Generating activation/loss impact plots for {problem_type}...")
    
    # Check if we have the required columns
    if 'activation' not in dbd_df.columns or 'loss' not in dbd_df.columns:
        print(f"  Warning: Missing activation/loss columns for {problem_type}")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # DbD-EDA - Best Fitness
    ax1 = axes[0, 0]
    if len(dbd_df) > 0:
        pivot_dbd = dbd_df.groupby(['activation', 'loss'])['best_fitness'].mean().reset_index()
        pivot_dbd = pivot_dbd.pivot(index='activation', columns='loss', values='best_fitness')
        
        sns.heatmap(pivot_dbd, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax1, cbar_kws={'label': 'Best Fitness'})
        ax1.set_title(f'DbD-EDA: Best Fitness ({problem_type.upper()})', fontweight='bold')
        ax1.set_xlabel('Loss Function')
        ax1.set_ylabel('Activation Function')
    
    # DbD-EDA - Success Rate
    ax2 = axes[0, 1]
    if len(dbd_df) > 0:
        pivot_dbd_success = dbd_df.groupby(['activation', 'loss'])['success'].mean().reset_index()
        pivot_dbd_success = pivot_dbd_success.pivot(index='activation', columns='loss', values='success')
        
        sns.heatmap(pivot_dbd_success, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax2, vmin=0, vmax=1, cbar_kws={'label': 'Success Rate'})
        ax2.set_title(f'DbD-EDA: Success Rate ({problem_type.upper()})', fontweight='bold')
        ax2.set_xlabel('Loss Function')
        ax2.set_ylabel('Activation Function')
    
    # Diff-EDA - Best Fitness
    ax3 = axes[1, 0]
    if len(dendiff_df) > 0 and 'activation' in dendiff_df.columns:
        pivot_dendiff = dendiff_df.groupby(['activation', 'loss'])['best_fitness'].mean().reset_index()
        pivot_dendiff = pivot_dendiff.pivot(index='activation', columns='loss', values='best_fitness')
        
        sns.heatmap(pivot_dendiff, annot=True, fmt='.1f', cmap='YlGnBu', ax=ax3, cbar_kws={'label': 'Best Fitness'})
        ax3.set_title(f'Diff-EDA: Best Fitness ({problem_type.upper()})', fontweight='bold')
        ax3.set_xlabel('Loss Function')
        ax3.set_ylabel('Activation Function')
    
    # Diff-EDA - Success Rate
    ax4 = axes[1, 1]
    if len(dendiff_df) > 0 and 'activation' in dendiff_df.columns:
        pivot_dendiff_success = dendiff_df.groupby(['activation', 'loss'])['success'].mean().reset_index()
        pivot_dendiff_success = pivot_dendiff_success.pivot(index='activation', columns='loss', values='success')
        
        sns.heatmap(pivot_dendiff_success, annot=True, fmt='.2f', cmap='YlGnBu', ax=ax4, vmin=0, vmax=1, cbar_kws={'label': 'Success Rate'})
        ax4.set_title(f'Diff-EDA: Success Rate ({problem_type.upper()})', fontweight='bold')
        ax4.set_xlabel('Loss Function')
        ax4.set_ylabel('Activation Function')
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / f"{problem_type}_activation_loss_impact.eps"
    plt.savefig(output_file, format='eps', bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_generations_violin(problem_type, eda_df, dbd_df, dendiff_df, vae_df):
    """Create violin plots for number of generations."""
    print(f"Generating generations violin plot for {problem_type}...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data
    data_list = []
    labels_list = []
    
    # Traditional EDAs - select top algorithms by success rate
    if len(eda_df) > 0:
        top_edas = eda_df.groupby('alg')['success'].mean().nlargest(3).index
        for alg in top_edas:
            data = eda_df[eda_df['alg'] == alg]['generation']
            data_list.append(data.values)
            labels_list.append(f"{alg}\n(EDA)")
    
    # DbD EDAs
    if len(dbd_df) > 0 and 'variant' in dbd_df.columns:
        variants = dbd_df['variant'].unique()[:2]
        for variant in variants:
            data = dbd_df[dbd_df['variant'] == variant]['generation']
            data_list.append(data.values)
            labels_list.append(f"{variant}\n(DbD)")
    
    # Dendiff EDAs
    if len(dendiff_df) > 0 and 'variant' in dendiff_df.columns:
        variants = dendiff_df['variant'].unique()[:2]
        for variant in variants:
            data = dendiff_df[dendiff_df['variant'] == variant]['generation']
            data_list.append(data.values)
            labels_list.append(f"{variant}\n(Diff)")
    
    # Create violin plot
    parts = ax.violinplot(data_list, positions=range(len(data_list)), showmeans=True, showmedians=True)
    
    # Color the violins
    colors = ['#1f77b4'] * 3 + ['#ff7f0e'] * 2 + ['#2ca02c'] * 2
    for i, pc in enumerate(parts['bodies']):
        if i < len(colors):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.6)
    
    ax.set_xticks(range(len(labels_list)))
    ax.set_xticklabels(labels_list, fontsize=9)
    ax.set_ylabel('Number of Generations', fontsize=12)
    ax.set_title(f'{problem_type.upper()}: Distribution of Generations', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / f"{problem_type}_generations_violin.eps"
    plt.savefig(output_file, format='eps', bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_time_comparison(problem_type, eda_df, dbd_df, dendiff_df, vae_df):
    """Create bar plot comparing elapsed time across algorithms."""
    print(f"Generating time comparison plot for {problem_type}...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Collect data
    all_algs = []
    all_times = []
    all_types = []
    
    # Traditional EDAs
    if len(eda_df) > 0:
        eda_grouped = eda_df.groupby('alg')['elapsed_time'].mean()
        for alg, val in eda_grouped.items():
            all_algs.append(alg)
            all_times.append(val)
            all_types.append('Traditional EDA')
    
    # DbD EDAs
    if len(dbd_df) > 0 and 'variant' in dbd_df.columns:
        dbd_grouped = dbd_df.groupby('variant')['elapsed_time'].mean()
        for alg, val in dbd_grouped.items():
            all_algs.append(alg)
            all_times.append(val)
            all_types.append('DbD-EDA')
    
    # Dendiff EDAs
    if len(dendiff_df) > 0 and 'variant' in dendiff_df.columns:
        dendiff_grouped = dendiff_df.groupby('variant')['elapsed_time'].mean()
        for alg, val in dendiff_grouped.items():
            all_algs.append(alg)
            all_times.append(val)
            all_types.append('Diff-EDA')
    
    # VAE EDAs
    if len(vae_df) > 0 and 'variant' in vae_df.columns:
        vae_grouped = vae_df.groupby('variant')['elapsed_time'].mean()
        for alg, val in vae_grouped.items():
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
    ax.barh(range(len(plot_df)), plot_df['Elapsed Time'], color=colors, alpha=0.8)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df['Algorithm'], fontsize=9)
    ax.set_xlabel('Average Elapsed Time (seconds, log scale)', fontsize=12)
    ax.set_xscale('log')
    ax.set_title(f'{problem_type.upper()}: Computational Time Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add legend
    handles = [plt.Rectangle((0,0),1,1, color=c, alpha=0.8) for c in type_colors.values()]
    ax.legend(handles, type_colors.keys(), loc='best', fontsize=9)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / f"{problem_type}_time_comparison.eps"
    plt.savefig(output_file, format='eps', bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def generate_summary_table(problem_type, eda_df, dbd_df, dendiff_df, vae_df):
    """Generate CSV and LaTeX table with summary statistics."""
    print(f"Generating summary table for {problem_type}...")
    
    summary_data = []
    
    # Traditional EDAs
    if len(eda_df) > 0:
        for alg in eda_df['alg'].unique():
            alg_data = eda_df[eda_df['alg'] == alg]
            summary_data.append({
                'Problem': problem_type.upper(),
                'Algorithm': alg,
                'Type': 'Traditional EDA',
                'Success Rate': alg_data['success'].mean(),
                'Best Fitness': alg_data['best_fitness'].mean(),
                'Avg Generations': alg_data['generation'].mean(),
                'Avg Time (s)': alg_data['elapsed_time'].mean()
            })
    
    # DbD EDAs
    if len(dbd_df) > 0 and 'variant' in dbd_df.columns:
        for variant in dbd_df['variant'].unique():
            var_data = dbd_df[dbd_df['variant'] == variant]
            summary_data.append({
                'Problem': problem_type.upper(),
                'Algorithm': variant,
                'Type': 'DbD-EDA',
                'Success Rate': var_data['success'].mean(),
                'Best Fitness': var_data['best_fitness'].mean(),
                'Avg Generations': var_data['generation'].mean(),
                'Avg Time (s)': var_data['elapsed_time'].mean()
            })
    
    # Dendiff EDAs
    if len(dendiff_df) > 0 and 'variant' in dendiff_df.columns:
        for variant in dendiff_df['variant'].unique():
            var_data = dendiff_df[dendiff_df['variant'] == variant]
            summary_data.append({
                'Problem': problem_type.upper(),
                'Algorithm': variant,
                'Type': 'Diff-EDA',
                'Success Rate': var_data['success'].mean(),
                'Best Fitness': var_data['best_fitness'].mean(),
                'Avg Generations': var_data['generation'].mean(),
                'Avg Time (s)': var_data['elapsed_time'].mean()
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save as CSV
    output_file = OUTPUT_DIR / f"{problem_type}_summary_table.csv"
    summary_df.to_csv(output_file, index=False, float_format='%.3f')
    print(f"Saved: {output_file}")
    
    return summary_df


def main():
    """Main execution function."""
    print("=" * 60)
    print("EDA Algorithm Visualization - Combinatorial Problems")
    print("=" * 60)
    
    problem_types = ['ising', 'ubqp', 'sat']
    
    for problem_type in problem_types:
        print(f"\n{'='*60}")
        print(f"Processing {problem_type.upper()} problems...")
        print('='*60)
        
        # Load data
        print(f"\nLoading {problem_type} data...")
        eda_df, dbd_df, dendiff_df, vae_df = load_data(problem_type)
        print(f"  Traditional EDAs: {len(eda_df)} records")
        print(f"  DbD-EDAs: {len(dbd_df)} records")
        print(f"  Diff-EDAs: {len(dendiff_df)} records")
        print(f"  VAE-EDAs: {len(vae_df)} records")
        
        # Generate plots
        print(f"\nGenerating {problem_type} figures...")
        plot_best_fitness_by_instance(problem_type, eda_df, dbd_df, dendiff_df, vae_df)
        plot_success_rate_by_algorithm(problem_type, eda_df, dbd_df, dendiff_df, vae_df)
        plot_activation_loss_impact(problem_type, dbd_df, dendiff_df)
        plot_generations_violin(problem_type, eda_df, dbd_df, dendiff_df, vae_df)
        plot_time_comparison(problem_type, eda_df, dbd_df, dendiff_df, vae_df)
        generate_summary_table(problem_type, eda_df, dbd_df, dendiff_df, vae_df)
    
    print("\n" + "=" * 60)
    print(f"All figures saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
