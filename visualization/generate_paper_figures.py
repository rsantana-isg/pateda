#!/usr/bin/env python3
"""
Generate specific figures for paper illustration.

This script creates individual heatmaps and time comparison figures as specified:
1. Heatmaps for activation vs loss function data for:
   - Deceptive3 problem (DbD-EDA and Diff-EDA)
   - SAT problem (DbD-EDA and Diff-EDA)
   - Ising problem (DbD-EDA and Diff-EDA)
   - UBQP problem (DbD-EDA and Diff-EDA)
2. Time comparison figures for combinatorial problems with specific algorithm subset

Author: PATEDA Team
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set publication style with larger fonts for readability
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 300

# Define paths
DATA_DIR = Path(__file__).parent.parent / "data" / "csv_results"
OUTPUT_DIR = Path(__file__).parent.parent / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_deceptive3_heatmaps():
    """Generate individual heatmaps for Deceptive3 problem."""
    print("Generating Deceptive3 heatmaps...")
    
    # Load data
    dbd_df = pd.read_csv(DATA_DIR / "additive" / "dbd_results_ranked_n30.csv")
    dendiff_df = pd.read_csv(DATA_DIR / "additive" / "dendiff_results_ranked_n30.csv")
    
    # Filter for Deceptive3
    dbd_dec3 = dbd_df[dbd_df['objective'] == 'Deceptive3']
    dendiff_dec3 = dendiff_df[dendiff_df['objective'] == 'Deceptive3']
    
    # DbD-EDA heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    pivot_dbd = dbd_dec3.groupby(['activation', 'loss'])['best_fitness'].mean().reset_index()
    pivot_dbd = pivot_dbd.pivot(index='activation', columns='loss', values='best_fitness')
    
    sns.heatmap(pivot_dbd, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax, 
                cbar_kws={'label': 'Best Fitness'}, annot_kws={'size': 12})
    ax.set_title('DbD-EDA: Deceptive3 - Activation vs Loss Functions', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Loss Function', fontsize=14, fontweight='bold')
    ax.set_ylabel('Activation Function', fontsize=14, fontweight='bold')
    ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / "additive" / "deceptive3_dbd_activation_loss_heatmap.eps"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, format='eps', bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()
    
    # Diff-EDA heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    pivot_dendiff = dendiff_dec3.groupby(['activation', 'loss'])['best_fitness'].mean().reset_index()
    pivot_dendiff = pivot_dendiff.pivot(index='activation', columns='loss', values='best_fitness')
    
    sns.heatmap(pivot_dendiff, annot=True, fmt='.2f', cmap='YlGnBu', ax=ax, 
                cbar_kws={'label': 'Best Fitness'}, annot_kws={'size': 12})
    ax.set_title('Diff-EDA: Deceptive3 - Activation vs Loss Functions', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Loss Function', fontsize=14, fontweight='bold')
    ax.set_ylabel('Activation Function', fontsize=14, fontweight='bold')
    ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / "additive" / "deceptive3_diff_activation_loss_heatmap.eps"
    plt.savefig(output_file, format='eps', bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def generate_combinatorial_heatmaps(problem_type):
    """Generate individual heatmaps for a combinatorial problem."""
    print(f"Generating {problem_type.upper()} heatmaps...")
    
    # Load data
    problem_dir = DATA_DIR / "combinatorial" / problem_type
    dbd_file = problem_dir / f"dbd_benchmark_{problem_type}_results.csv"
    dendiff_file = problem_dir / f"dendiff_benchmark_{problem_type}_results.csv"
    
    if not dbd_file.exists() or not dendiff_file.exists():
        print(f"  Warning: Data files not found for {problem_type}")
        return
    
    dbd_df = pd.read_csv(dbd_file)
    dendiff_df = pd.read_csv(dendiff_file)
    
    # Check if we have the required columns
    if 'activation' not in dbd_df.columns or 'loss' not in dbd_df.columns:
        print(f"  Warning: Missing activation/loss columns for {problem_type}")
        return
    
    # DbD-EDA heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    pivot_dbd = dbd_df.groupby(['activation', 'loss'])['best_fitness'].mean().reset_index()
    pivot_dbd = pivot_dbd.pivot(index='activation', columns='loss', values='best_fitness')
    
    sns.heatmap(pivot_dbd, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax, 
                cbar_kws={'label': 'Best Fitness'}, annot_kws={'size': 12})
    ax.set_title(f'DbD-EDA: {problem_type.upper()} - Activation vs Loss Functions', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Loss Function', fontsize=14, fontweight='bold')
    ax.set_ylabel('Activation Function', fontsize=14, fontweight='bold')
    ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / "combinatorial" / f"{problem_type}_dbd_activation_loss_heatmap.eps"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, format='eps', bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()
    
    # Diff-EDA heatmap
    if 'activation' in dendiff_df.columns and 'loss' in dendiff_df.columns:
        fig, ax = plt.subplots(figsize=(10, 8))
        pivot_dendiff = dendiff_df.groupby(['activation', 'loss'])['best_fitness'].mean().reset_index()
        pivot_dendiff = pivot_dendiff.pivot(index='activation', columns='loss', values='best_fitness')
        
        sns.heatmap(pivot_dendiff, annot=True, fmt='.1f', cmap='YlGnBu', ax=ax, 
                    cbar_kws={'label': 'Best Fitness'}, annot_kws={'size': 12})
        ax.set_title(f'Diff-EDA: {problem_type.upper()} - Activation vs Loss Functions', 
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Loss Function', fontsize=14, fontweight='bold')
        ax.set_ylabel('Activation Function', fontsize=14, fontweight='bold')
        ax.tick_params(labelsize=12)
        
        plt.tight_layout()
        output_file = OUTPUT_DIR / "combinatorial" / f"{problem_type}_diff_activation_loss_heatmap.eps"
        plt.savefig(output_file, format='eps', bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()


def generate_time_comparison_figures():
    """Generate time comparison figures for combinatorial problems with specific algorithm subset."""
    print("Generating time comparison figures for combinatorial problems...")
    
    # Define algorithm mapping for renaming
    algorithm_mapping = {
        'dbs_cs': 'DbD-EDA',
        'dendiff_gumbel': 'Diff-EDA',
        'C-VAE': 'C-VAE-EDA',
        'EBNA': 'EBNA',
        'MN-FDAG': 'MN-FDAG',
        'TreeEDA': 'Tree-EDA',
        'UMDA': 'UMDA'
    }
    
    # Algorithms to include (original names from data)
    # For traditional EDAs: EBNA, MN-FDAG, TreeEDA, UMDA
    # For DbD: dbd_cs
    # For Dendiff: dendiff_gumbel
    # For VAE: C-VAE
    included_algorithms = ['dbd_cs', 'dendiff_gumbel', 'C-VAE', 'EBNA', 'MN-FDAG', 'TreeEDA', 'UMDA']
    
    problem_types = ['ising', 'ubqp', 'sat']
    
    for problem_type in problem_types:
        print(f"  Processing {problem_type.upper()}...")
        
        # Load data
        problem_dir = DATA_DIR / "combinatorial" / problem_type
        eda_file = problem_dir / f"discrete_EDA_RW_benchmark_{problem_type}_results.csv"
        dbd_file = problem_dir / f"dbd_benchmark_{problem_type}_results.csv"
        dendiff_file = problem_dir / f"dendiff_benchmark_{problem_type}_results.csv"
        vae_file = problem_dir / f"vae_benchmark_{problem_type}_results.csv"
        
        # Collect data for plot
        all_algs = []
        all_times = []
        all_types = []
        
        # Load traditional EDAs
        if eda_file.exists():
            eda_df = pd.read_csv(eda_file)
            if 'alg' in eda_df.columns and 'elapsed_time' in eda_df.columns:
                eda_grouped = eda_df.groupby('alg')['elapsed_time'].mean()
                for alg, val in eda_grouped.items():
                    if alg in included_algorithms:
                        display_name = algorithm_mapping.get(alg, alg)
                        all_algs.append(display_name)
                        all_times.append(val)
                        all_types.append('Traditional EDA')
        
        # Load DbD EDAs
        if dbd_file.exists():
            dbd_df = pd.read_csv(dbd_file)
            if 'variant' in dbd_df.columns and 'elapsed_time' in dbd_df.columns:
                dbd_grouped = dbd_df.groupby('variant')['elapsed_time'].mean()
                for variant, val in dbd_grouped.items():
                    if variant in included_algorithms:
                        display_name = algorithm_mapping.get(variant, variant)
                        all_algs.append(display_name)
                        all_times.append(val)
                        all_types.append('DbD-EDA')
        
        # Load Dendiff EDAs
        if dendiff_file.exists():
            dendiff_df = pd.read_csv(dendiff_file)
            if 'variant' in dendiff_df.columns and 'elapsed_time' in dendiff_df.columns:
                dendiff_grouped = dendiff_df.groupby('variant')['elapsed_time'].mean()
                for variant, val in dendiff_grouped.items():
                    if variant in included_algorithms:
                        display_name = algorithm_mapping.get(variant, variant)
                        all_algs.append(display_name)
                        all_times.append(val)
                        all_types.append('Diff-EDA')
        
        # Load VAE EDAs
        if vae_file.exists():
            vae_df = pd.read_csv(vae_file)
            if 'variant' in vae_df.columns and 'elapsed_time' in vae_df.columns:
                vae_grouped = vae_df.groupby('variant')['elapsed_time'].mean()
                for variant, val in vae_grouped.items():
                    if variant in included_algorithms:
                        display_name = algorithm_mapping.get(variant, variant)
                        all_algs.append(display_name)
                        all_times.append(val)
                        all_types.append('VAE-EDA')
        
        if not all_algs:
            print(f"    Warning: No data found for {problem_type}")
            continue
        
        # Create dataframe for plotting
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
        
        colors = [type_colors.get(t, '#888888') for t in plot_df['Type']]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.barh(range(len(plot_df)), plot_df['Elapsed Time'], color=colors, alpha=0.8)
        ax.set_yticks(range(len(plot_df)))
        ax.set_yticklabels(plot_df['Algorithm'], fontsize=12)
        ax.set_xlabel('Average Elapsed Time (seconds, log scale)', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.set_title(f'{problem_type.upper()}: Computational Time Comparison', 
                     fontsize=16, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3)
        
        # Add legend
        handles = [plt.Rectangle((0,0),1,1, color=c, alpha=0.8) for c in type_colors.values()]
        ax.legend(handles, type_colors.keys(), loc='best', fontsize=11)
        
        plt.tight_layout()
        output_file = OUTPUT_DIR / "combinatorial" / f"{problem_type}_time_comparison_selected.eps"
        plt.savefig(output_file, format='eps', bbox_inches='tight')
        print(f"    Saved: {output_file}")
        plt.close()


def main():
    """Main execution function."""
    print("=" * 70)
    print("PATEDA Paper Figure Generation")
    print("=" * 70)
    print()
    
    # Generate Deceptive3 heatmaps
    print("\n1. Generating Deceptive3 heatmaps...")
    print("-" * 70)
    generate_deceptive3_heatmaps()
    
    # Generate combinatorial problem heatmaps
    print("\n2. Generating combinatorial problem heatmaps...")
    print("-" * 70)
    for problem_type in ['sat', 'ising', 'ubqp']:
        generate_combinatorial_heatmaps(problem_type)
    
    # Generate time comparison figures
    print("\n3. Generating time comparison figures...")
    print("-" * 70)
    generate_time_comparison_figures()
    
    print("\n" + "=" * 70)
    print(f"All figures saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
