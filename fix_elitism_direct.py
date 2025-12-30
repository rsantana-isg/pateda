#!/usr/bin/env python3
"""
Direct script to add ElitistReplacement to EDAComponents.
"""

import re
from pathlib import Path


def fix_file(filepath):
    """Fix a single file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    modified = False
    has_elitist_import = False
    has_generational_import = False

    # First pass: check what imports exist
    for line in lines:
        if 'ElitistReplacement' in line:
            has_elitist_import = True
        if 'GenerationalReplacement' in line:
            has_generational_import = True

    # Skip if doesn't use EDAComponents
    content = ''.join(lines)
    if 'EDAComponents(' not in content:
        print(f"  ⊘ {filepath.name} - doesn't use EDAComponents")
        return False

    # Skip if already has replacement=ElitistReplacement
    if 'replacement=ElitistReplacement' in content:
        print(f"  ✓ {filepath.name} - already uses ElitistReplacement correctly")
        return False

    new_lines = []
    in_components = False
    components_indent = ""
    last_import_line = -1

    for i, line in enumerate(lines):
        # Track last import line for adding new import
        if line.strip().startswith('from pateda') or line.strip().startswith('import pateda'):
            last_import_line = len(new_lines)

        # Replace GenerationalReplacement with ElitistReplacement in imports
        if 'from pateda.replacement import GenerationalReplacement' in line:
            line = line.replace('GenerationalReplacement', 'ElitistReplacement')
            modified = True
            has_elitist_import = True

        # Replace GenerationalReplacement() usage
        if 'replacement=GenerationalReplacement()' in line:
            line = line.replace('GenerationalReplacement()', 'ElitistReplacement()')
            modified = True

        # Detect EDAComponents( start
        if 'EDAComponents(' in line:
            in_components = True
            components_indent = line[:len(line) - len(line.lstrip())]

        # If in EDAComponents and we hit stop_condition without seeing replacement=
        if in_components and 'stop_condition=' in line and 'replacement=' not in ''.join(lines[max(0,i-10):i+1]):
            # Add replacement before stop_condition
            indent = line[:len(line) - len(line.lstrip())]
            new_lines.append(f"{indent}replacement=ElitistReplacement(),\n")
            modified = True

        # Detect EDAComponents end
        if in_components and line.strip().startswith(')'):
            in_components = False

        new_lines.append(line)

    # Add import if needed and modified
    if modified and not has_elitist_import and last_import_line >= 0:
        new_lines.insert(last_import_line + 1, 'from pateda.replacement import ElitistReplacement\n')

    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f"  ✓ {filepath.name} - added ElitistReplacement")
        return True

    return False


def main():
    """Fix all example files"""
    examples_dir = Path(__file__).parent / 'examples'

    print("Adding elitism to EDA examples...\n")

    fixed_count = 0
    for filepath in sorted(examples_dir.glob('*.py')):
        if filepath.name == '__init__.py':
            continue

        if fix_file(filepath):
            fixed_count += 1

    print(f"\n✓ Fixed {fixed_count} files")


if __name__ == '__main__':
    main()
