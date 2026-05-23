#!/usr/bin/env python
"""
Example script demonstrating the new enhanced VAE EDA variants

This script shows how to use each of the 6 new VAE variants on different problems.
Run individual examples by uncommenting the desired variant below.
"""

import sys
import os

# Example commands for each variant:

# 1. BA-VAE (Beta-Annealed VAE) - Good for preventing posterior collapse
# Recommended for: All problems, especially with small populations
print("=" * 80)
print("BA-VAE Example - Beta-Annealed VAE")
print("=" * 80)
print("Command:")
print("  python examples/discrete_VAE_EDA.py 0 OneMax 30 100 50 0.5 BA-VAE --epochs 100")
print("\nDescription:")
print("  Uses cyclical beta annealing to prevent KL divergence vanishing.")
print("  Good default choice for most problems.")
print()

# 2. AA-VAE (Adaptive-Architecture VAE) - Good for small populations
# Recommended for: Problems with population < 100
print("=" * 80)
print("AA-VAE Example - Adaptive-Architecture VAE")
print("=" * 80)
print("Command:")
print("  python examples/discrete_VAE_EDA.py 1 Deceptive3 30 80 40 0.5 AA-VAE --epochs 150")
print("\nDescription:")
print("  Uses ultra-small networks with dropout to prevent overfitting.")
print("  Best for small populations where overfitting is a concern.")
print()

# 3. FW-VAE (Fitness-Weighted VAE) - Good for hard problems
# Recommended for: Deceptive and hierarchical problems
print("=" * 80)
print("FW-VAE Example - Fitness-Weighted VAE")
print("=" * 80)
print("Command:")
print("  python examples/discrete_VAE_EDA.py 2 HIFF 64 200 50 0.5 FW-VAE \\")
print("    --beta-start 0.0 --beta-end 1.0 --epochs 100")
print("\nDescription:")
print("  Weights reconstruction by fitness to focus on good solutions.")
print("  Excellent for hard deceptive and hierarchical problems.")
print()

# 4. GS-VAE (Greedy-Sampling VAE) - Good for quick convergence
# Recommended for: Problems where exploitation is more important
print("=" * 80)
print("GS-VAE Example - Greedy-Sampling VAE")
print("=" * 80)
print("Command:")
print("  python examples/discrete_VAE_EDA.py 3 FHTrap1 81 150 40 0.5 GS-VAE --epochs 100")
print("\nDescription:")
print("  Uses deterministic (greedy) sampling for better exploitation.")
print("  Reduces variance and speeds up convergence.")
print()

# 5. HS-VAE (Hybrid-Sampling VAE) - Good for balanced search
# Recommended for: When you want both exploration and exploitation
print("=" * 80)
print("HS-VAE Example - Hybrid-Sampling VAE")
print("=" * 80)
print("Command:")
print("  python examples/discrete_VAE_EDA.py 4 KDeceptive3 30 100 50 0.5 HS-VAE --epochs 100")
print("\nDescription:")
print("  Combines 70% deterministic (exploitation) + 30% stochastic (exploration).")
print("  Good balance for most optimization scenarios.")
print()

# 6. TC-VAE (Temperature-Controlled VAE) - Good for long runs
# Recommended for: Problems requiring gradual shift from exploration to exploitation
print("=" * 80)
print("TC-VAE Example - Temperature-Controlled VAE")
print("=" * 80)
print("Command:")
print("  python examples/discrete_VAE_EDA.py 5 Polytree3 30 120 60 0.5 TC-VAE --epochs 100")
print("\nDescription:")
print("  Temperature decreases over generations for adaptive search.")
print("  Starts with exploration, gradually shifts to exploitation.")
print()

print("=" * 80)
print("Tips:")
print("=" * 80)
print("1. For small populations (< 100): Use AA-VAE")
print("2. For hard deceptive problems: Use FW-VAE or BA-VAE")
print("3. For quick convergence: Use GS-VAE")
print("4. For robust search: Use HS-VAE or TC-VAE")
print("5. General purpose: BA-VAE is a good default")
print()
print("All variants support:")
print("  --activation-enc: relu, tanh, sigmoid, leakyrelu, etc.")
print("  --activation-dec: relu, tanh, sigmoid, leakyrelu, etc.")
print("  --beta-start: Initial KL weight (default: 0.0)")
print("  --beta-end: Final KL weight (default: 1.0)")
print("  --latent-dim: Latent space dimension")
print("  --epochs: Training epochs per generation (default: 30)")
print()

# Comparison table
print("=" * 80)
print("Quick Comparison")
print("=" * 80)
print("Variant | Main Feature           | Best For")
print("--------|------------------------|----------------------------------")
print("BA-VAE  | Cyclical beta          | General purpose, all problems")
print("AA-VAE  | Small networks         | Small populations")
print("FW-VAE  | Fitness weighting      | Hard deceptive problems")
print("GS-VAE  | Deterministic sampling | Quick convergence")
print("HS-VAE  | Hybrid sampling        | Balanced search")
print("TC-VAE  | Adaptive temperature   | Long optimization runs")
print()

print("For detailed documentation, see ENHANCED_VAE_VARIANTS.md")
print("For analysis and motivation, see DISCRETE_VAE_ANALYSIS.md")
print()
