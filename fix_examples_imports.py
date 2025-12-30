#!/usr/bin/env python3
"""
Script to add sys.path manipulation to example files so they can be run
without installing the pateda package.
"""

import os
import re
from pathlib import Path


def has_sys_path_insert(content):
    """Check if file already has sys.path manipulation"""
    return 'sys.path.insert' in content or 'sys.path.append' in content


def get_first_import_line(lines):
    """Find the line number of the first import statement"""
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('import ') or stripped.startswith('from '):
            return i
    return None


def fix_example_file(filepath):
    """Add sys.path manipulation to an example file if needed"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.splitlines(keepends=True)

    # Check if already has sys.path manipulation
    if has_sys_path_insert(content):
        print(f"  ✓ {filepath.name} - already has sys.path manipulation")
        return False

    # Check if it imports pateda
    if 'from pateda' not in content and 'import pateda' not in content:
        print(f"  ⊘ {filepath.name} - doesn't import pateda")
        return False

    # Find first import line
    first_import_idx = get_first_import_line(lines)
    if first_import_idx is None:
        print(f"  ✗ {filepath.name} - no imports found")
        return False

    # Insert sys.path manipulation before first import
    insertion = [
        "import sys\n",
        "import os\n",
        "\n",
        "# Add parent directory to path for running examples without installation\n",
        "sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n",
        "\n",
    ]

    # Insert the lines
    new_lines = lines[:first_import_idx] + insertion + lines[first_import_idx:]

    # Write back to file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    print(f"  ✓ {filepath.name} - fixed")
    return True


def main():
    """Fix all example files"""
    examples_dir = Path(__file__).parent / 'examples'

    print("Fixing example files to allow running without installation...\n")

    fixed_count = 0
    for filepath in sorted(examples_dir.glob('*.py')):
        if filepath.name == '__init__.py':
            continue

        if fix_example_file(filepath):
            fixed_count += 1

    print(f"\n✓ Fixed {fixed_count} example files")


if __name__ == '__main__':
    main()
