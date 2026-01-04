#!/usr/bin/env python3
"""
Quick demonstration of discrete_Backdrive_EDA.py command-line interface
Shows the help and parameter structure without running actual algorithms
"""

import subprocess
import sys

def run_command(cmd, description):
    """Run a command and display its output"""
    print("=" * 80)
    print(f"DEMO: {description}")
    print("=" * 80)
    print(f"Command: {' '.join(cmd)}")
    print("-" * 80)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        print()
    except subprocess.TimeoutExpired:
        print("(Command timed out - this is expected for actual runs)")
        print()
    except Exception as e:
        print(f"Error: {e}")
        print()

def main():
    script_path = "examples/discrete_Backdrive_EDA.py"
    
    print("\n")
    print("*" * 80)
    print("*" + " " * 78 + "*")
    print("*" + "  discrete_Backdrive_EDA.py - Command-Line Interface Demo".center(78) + "*")
    print("*" + " " * 78 + "*")
    print("*" * 80)
    print("\n")
    
    # Demo 1: Help message
    run_command(
        ["python", script_path, "--help"],
        "Display help message"
    )
    
    # Demo 2: Show parameter structure (will fail without deps, but shows structure)
    print("=" * 80)
    print("DEMO: Example command structures (syntax demonstration)")
    print("=" * 80)
    print()
    
    examples = [
        "# Basic usage with default settings",
        "python discrete_Backdrive_EDA.py 0 OneMax 20 80 20 0.5",
        "",
        "# With early stopping and weight transfer",
        "python discrete_Backdrive_EDA.py 0 OneMax 20 80 20 0.5 --weight-transfer --early-stopping",
        "",
        "# Custom initialization and loss function",
        "python discrete_Backdrive_EDA.py 0 Deceptive3 30 100 30 0.5 --init perturb_best --loss weighted_mse",
        "",
        "# Adaptive variant with custom settings",
        "python discrete_Backdrive_EDA.py 0 HIFF 64 200 50 0.5 --variant backdrive_adaptive --activation tanh",
        "",
        "# With surrogate filtering",
        "python discrete_Backdrive_EDA.py 0 OneMax 20 80 20 0.5 --surrogate-filtering",
        "",
        "# Testing different loss functions",
        "python discrete_Backdrive_EDA.py 0 Deceptive3 30 100 30 0.5 --loss mse",
        "python discrete_Backdrive_EDA.py 0 Deceptive3 30 100 30 0.5 --loss weighted_mse",
        "python discrete_Backdrive_EDA.py 0 Deceptive3 30 100 30 0.5 --loss ranking",
        "python discrete_Backdrive_EDA.py 0 Deceptive3 30 100 30 0.5 --loss huber",
        "",
        "# Testing different initializations",
        "python discrete_Backdrive_EDA.py 0 OneMax 20 80 20 0.5 --init random",
        "python discrete_Backdrive_EDA.py 0 OneMax 20 80 20 0.5 --init perturb_best",
        "python discrete_Backdrive_EDA.py 0 OneMax 20 80 20 0.5 --init perturb_selected",
        "",
        "# Different truncation percentages",
        "python discrete_Backdrive_EDA.py 0 OneMax 20 80 20 0.3  # 30% selection",
        "python discrete_Backdrive_EDA.py 0 OneMax 20 80 20 0.5  # 50% selection",
        "python discrete_Backdrive_EDA.py 0 OneMax 20 80 20 0.7  # 70% selection",
    ]
    
    for line in examples:
        print(line)
    
    print()
    print("=" * 80)
    print()
    
    print("*" * 80)
    print("*" + " " * 78 + "*")
    print("*" + "  For full functionality, install dependencies:".center(78) + "*")
    print("*" + "  pip install numpy torch scipy".center(78) + "*")
    print("*" + " " * 78 + "*")
    print("*" + "  See README_discrete_Backdrive_EDA.md for complete documentation".center(78) + "*")
    print("*" + " " * 78 + "*")
    print("*" * 80)
    print()

if __name__ == "__main__":
    main()
