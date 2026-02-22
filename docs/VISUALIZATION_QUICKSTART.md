# Quick Start Guide: Generating EDA Figures

This guide shows you how to quickly generate all publication-quality figures for your EDA experiments.

## Prerequisites

Make sure you have the required Python packages installed:

```bash
pip install pandas matplotlib seaborn numpy
```

Or install all dependencies from the main repository:

```bash
pip install -r requirements.txt
```

## Quick Start

### Generate All Figures (Recommended)

To generate all figures in one command:

```bash
cd /path/to/pateda
python3 visualization/generate_all_figures.py
```

This will create:
- 8 figures for additive functions in `figures/additive/`
- 18 figures for combinatorial problems in `figures/combinatorial/`

### Generate Specific Figures

**For additive functions only:**
```bash
python3 visualization/plot_additive_results.py
```

**For combinatorial problems only:**
```bash
python3 visualization/plot_combinatorial_results.py
```

## What Gets Generated

### Additive Functions
- Comparison plots showing performance across Deceptive3, HIFF, KDeceptive3, KDeceptive5, OneMax
- Heatmaps for activation/loss function impact analysis
- Statistical distributions and timing analysis
- LaTeX-ready summary table

### Combinatorial Problems
- Instance-level performance analysis for Ising, UBQP, and SAT
- Algorithm comparison across different problem types
- Hyperparameter impact visualization
- Detailed performance metrics

## Using Figures in Your Paper

All figures are saved as `.eps` (Encapsulated PostScript) files, which are:
- Vector-based (perfect quality at any scale)
- Accepted by most academic journals
- Easy to include in LaTeX documents

### LaTeX Example

```latex
\begin{figure}[ht]
\centering
\includegraphics[width=0.8\textwidth]{figures/additive/best_fitness_comparison.eps}
\caption{Comparison of best fitness achieved by different EDA algorithms.}
\label{fig:fitness}
\end{figure}
```

### Include Summary Table

```latex
\input{figures/additive/summary_table.tex}
```

## Customization

To customize the figures, edit the respective Python scripts:
- Modify colors in the `type_colors` dictionary
- Adjust figure sizes with the `figsize` parameter
- Change font settings in `plt.rcParams`

## Troubleshooting

**Problem:** "ModuleNotFoundError: No module named 'pandas'"
**Solution:** Install required packages: `pip install pandas matplotlib seaborn numpy`

**Problem:** "FileNotFoundError: [Errno 2] No such file or directory: 'data/csv_results/...'"
**Solution:** Make sure you're running the script from the repository root directory

**Problem:** Warnings about transparency in PostScript backend
**Solution:** This is normal and doesn't affect output quality. EPS format doesn't support transparency.

## Performance Tips

- Scripts process hundreds of data points and create multiple figures
- Expected runtime: 1-2 minutes for all figures
- Output directory is automatically created if it doesn't exist
- Regenerating figures will overwrite existing ones

## Data Sources

The scripts read from:
- `data/csv_results/additive/*.csv` - Results for additive functions
- `data/csv_results/combinatorial/{ising,ubqp,sat}/*.csv` - Results for combinatorial problems

Make sure these files exist before running the visualization scripts.

## Need Help?

See `visualization/README.md` for detailed documentation, or check the inline comments in the Python scripts.
