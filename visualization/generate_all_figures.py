#!/usr/bin/env python3
"""
Master script to generate all EDA algorithm figures.

This script runs all visualization programs to generate comprehensive
figures for both additive functions and combinatorial problems.

Author: PATEDA Team
"""

import sys
import subprocess
from pathlib import Path


def run_script(script_path):
    """Run a Python script using subprocess."""
    result = subprocess.run([sys.executable, script_path], 
                          capture_output=True, 
                          text=True)
    # Print output for visibility
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Script {script_path} failed with exit code {result.returncode}")


def main():
    """Run all visualization scripts."""
    print("\n" + "=" * 80)
    print("PATEDA - Generating All EDA Algorithm Figures")
    print("=" * 80)
    
    base_dir = Path(__file__).parent
    
    try:
        # Generate additive function figures
        print("\n[1/2] Generating additive function figures...")
        print("-" * 80)
        run_script(base_dir / "plot_additive_results.py")
        
        # Generate combinatorial problem figures
        print("\n[2/2] Generating combinatorial problem figures...")
        print("-" * 80)
        run_script(base_dir / "plot_combinatorial_results.py")
        
        print("\n" + "=" * 80)
        print("SUCCESS: All figures generated successfully!")
        print("=" * 80)
        print("\nFigures are saved in:")
        print("  - figures/additive/")
        print("  - figures/combinatorial/")
        print("\nYou can use these .eps files directly in your LaTeX documents.")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("ERROR: Failed to generate figures")
        print("=" * 80)
        print(f"\nError details: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
