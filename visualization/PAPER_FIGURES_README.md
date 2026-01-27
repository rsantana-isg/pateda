# Paper Figure Generation Script

This script generates specific figures for paper illustration as requested.

## Usage

```bash
cd /home/runner/work/pateda/pateda
python3 visualization/generate_paper_figures.py
```

## Generated Figures

The script generates the following figures in the `figures/` directory:

### 1. Deceptive3 Heatmaps (Additive Problems)
- `figures/additive/deceptive3_dbd_activation_loss_heatmap.eps` - DbD-EDA activation vs loss function heatmap
- `figures/additive/deceptive3_diff_activation_loss_heatmap.eps` - Diff-EDA activation vs loss function heatmap

### 2. Combinatorial Problem Heatmaps (SAT, Ising, UBQP)
Each problem has two heatmaps:

**SAT:**
- `figures/combinatorial/sat_dbd_activation_loss_heatmap.eps` - DbD-EDA heatmap
- `figures/combinatorial/sat_diff_activation_loss_heatmap.eps` - Diff-EDA heatmap

**Ising:**
- `figures/combinatorial/ising_dbd_activation_loss_heatmap.eps` - DbD-EDA heatmap
- `figures/combinatorial/ising_diff_activation_loss_heatmap.eps` - Diff-EDA heatmap

**UBQP:**
- `figures/combinatorial/ubqp_dbd_activation_loss_heatmap.eps` - DbD-EDA heatmap
- `figures/combinatorial/ubqp_diff_activation_loss_heatmap.eps` - Diff-EDA heatmap

### 3. Time Comparison Figures (Combinatorial Problems)
- `figures/combinatorial/ising_time_comparison_selected.eps`
- `figures/combinatorial/ubqp_time_comparison_selected.eps`
- `figures/combinatorial/sat_time_comparison_selected.eps`

These time comparison figures include only the following algorithms with renamed labels:
- **dbd_cs** → **DbD-EDA**
- **dendiff_gumbel** → **Diff-EDA**
- **C-VAE** → **C-VAE-EDA**
- **EBNA** → **EBNA**
- **MN-FDAG** → **MN-FDAG**
- **TreeEDA** → **Tree-EDA**
- **UMDA** → **UMDA**

## Key Features

1. **Individual Heatmaps**: Each heatmap is generated as a separate figure with:
   - Larger font sizes (14pt) for better readability
   - Clear cell annotations showing numeric values
   - Publication-quality EPS format
   - Appropriate color schemes (YlOrRd for DbD-EDA, YlGnBu for Diff-EDA)

2. **Selective Algorithm Inclusion**: Time comparison figures only include the specified subset of algorithms, excluding others from the original datasets.

3. **Algorithm Renaming**: Algorithms are automatically renamed according to the specified mapping for consistency with paper terminology.

## Requirements

The script requires the following Python packages:
- pandas
- numpy
- matplotlib
- seaborn

These can be installed using:
```bash
pip install pandas numpy matplotlib seaborn
```

## Data Sources

The script reads data from:
- `data/csv_results/additive/` - Additive problem results
- `data/csv_results/combinatorial/{ising,ubqp,sat}/` - Combinatorial problem results

Each problem type requires the following CSV files:
- `dbd_benchmark_{problem}_results.csv` - DbD-EDA results
- `dendiff_benchmark_{problem}_results.csv` - Dendiff (Diff-EDA) results
- `discrete_EDA_RW_benchmark_{problem}_results.csv` - Traditional EDA results
- `vae_benchmark_{problem}_results.csv` - VAE-EDA results

## Notes

- The script automatically creates output directories if they don't exist.
- All figures are saved in EPS format for publication quality.
- Transparency warnings for EPS format are expected and do not affect the output quality.
- The heatmaps show average best fitness values across all experimental runs.
- Time comparison figures use log scale for better visualization of the wide range of execution times.
