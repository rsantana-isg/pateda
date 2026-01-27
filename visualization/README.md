# EDA Algorithm Visualization Tools

This directory contains Python scripts for generating publication-quality figures illustrating the performance of different Estimation of Distribution Algorithm (EDA) variants.

## Overview

The visualization tools analyze experimental results from CSV files and generate:
- **Comparison plots**: Bar charts comparing best fitness and success rates across algorithms
- **Heatmaps**: Showing the impact of activation and loss function choices
- **Distribution plots**: Box plots and violin plots for generations and time analysis
- **Summary tables**: CSV and LaTeX formatted tables for easy inclusion in papers

## Generated Figures

### Additive Functions
Located in `figures/additive/`:
1. `best_fitness_comparison.eps` - Comparison of best fitness achieved by each algorithm
2. `success_rate_comparison.eps` - Success rates across algorithms
3. `activation_loss_impact_fitness.eps` - Heatmap showing impact of activation/loss functions on fitness
4. `activation_loss_impact_success.eps` - Heatmap showing impact of activation/loss functions on success rate
5. `generations_comparison.eps` - Box plots comparing number of generations
6. `elapsed_time_comparison.eps` - Bar chart comparing computational time
7. `summary_table.csv` - Summary statistics in CSV format
8. `summary_table.tex` - LaTeX formatted table for direct inclusion in papers

### Combinatorial Problems (Ising, UBQP, SAT)
Located in `figures/combinatorial/`:

For each problem type (ising, ubqp, sat):
1. `{problem}_best_fitness_by_instance.eps` - Grouped bar chart by problem instance
2. `{problem}_success_rate_comparison.eps` - Success rate comparison
3. `{problem}_activation_loss_impact.eps` - Heatmaps for activation/loss function impact
4. `{problem}_generations_violin.eps` - Violin plots for generation distributions
5. `{problem}_time_comparison.eps` - Computational time comparison
6. `{problem}_summary_table.csv` - Summary statistics

## Scripts

### `plot_additive_results.py`
Generates all figures for additive functions (Deceptive3, HIFF, KDeceptive3, etc.).

**Usage:**
```bash
python3 visualization/plot_additive_results.py
```

**Data sources:**
- `data/csv_results/additive/eda_results_ranked_n30.csv`
- `data/csv_results/additive/dbd_results_ranked_n30.csv`
- `data/csv_results/additive/dendiff_results_ranked_n30.csv`
- `data/csv_results/additive/vae_results_ranked_n30.csv`

### `plot_combinatorial_results.py`
Generates all figures for combinatorial problems (Ising, UBQP, SAT).

**Usage:**
```bash
python3 visualization/plot_combinatorial_results.py
```

**Data sources:**
- `data/csv_results/combinatorial/ising/*.csv`
- `data/csv_results/combinatorial/ubqp/*.csv`
- `data/csv_results/combinatorial/sat/*.csv`

### `generate_all_figures.py`
Master script that runs all visualization programs.

**Usage:**
```bash
python3 visualization/generate_all_figures.py
```

This will generate all figures for both additive and combinatorial problems in one run.

## Requirements

All required packages are specified in `requirements.txt`:
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- numpy >= 1.21.0

Install with:
```bash
pip install -r requirements.txt
```

## Algorithm Types

The visualizations compare four types of EDA algorithms:

1. **Traditional EDAs**: EBNA, MK-EDA1/2/3, MN-FDA, MN-FDAG, TreeEDA, UMDA
2. **DbD-EDA**: Denoising by Denoising EDAs with various activation and loss functions
3. **Diff-EDA**: Diffusion-based EDAs (Dendiff variants)
4. **VAE-EDA**: Variational Autoencoder based EDAs

## Key Metrics Visualized

- **Success Rate**: Proportion of runs that found the optimal solution
- **Best Fitness**: Average best fitness value achieved
- **Generations**: Number of generations until convergence or termination
- **Elapsed Time**: Computational time in seconds

## Hyperparameter Analysis

For DbD-EDA and Diff-EDA, the scripts generate heatmaps showing the impact of:
- **Activation functions**: elu, relu, tanh
- **Loss functions**: mse, huber, ranking, weighted_mse

These help identify the best configurations for each problem type.

## Output Format

All figures are saved in **EPS (Encapsulated PostScript)** format, which is:
- Vector-based (scalable without quality loss)
- Widely accepted by academic journals
- Compatible with LaTeX documents
- Suitable for publication-quality output

## Using Figures in LaTeX

To include a figure in your LaTeX document:

```latex
\begin{figure}[ht]
\centering
\includegraphics[width=0.8\textwidth]{figures/additive/best_fitness_comparison.eps}
\caption{Comparison of best fitness achieved by different EDA algorithms on additive functions.}
\label{fig:best_fitness}
\end{figure}
```

For tables:
```latex
\input{figures/additive/summary_table.tex}
```

## Customization

To modify the visualization style:

1. **Colors**: Edit the `type_colors` dictionary in each script
2. **Figure size**: Modify the `figsize` parameter in `plt.subplots()`
3. **Font settings**: Adjust `plt.rcParams` at the top of each script
4. **DPI**: Change `plt.rcParams['figure.dpi']` for higher/lower resolution

## Notes

- The PostScript backend does not support transparency, so you may see warnings about transparent artists being rendered opaque. This is normal and does not affect the quality of the output.
- Log scale is used for time comparisons to better visualize the wide range of computational costs.
- For large datasets, some plots may show only representative subsets (e.g., top algorithms by performance) to maintain readability.

## Authors

PATEDA Team

## License

See the main repository LICENSE file.
