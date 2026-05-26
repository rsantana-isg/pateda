#!/usr/bin/env python3
"""
Script to add ElitistReplacement to all EDA examples to guarantee elitism.
"""

import os
import re
from pathlib import Path


def has_elitist_replacement(content):
    """Check if file already uses ElitistReplacement"""
    return 'ElitistReplacement' in content


def has_generational_replacement(content):
    """Check if file uses GenerationalReplacement"""
    return 'GenerationalReplacement()' in content


def has_replacement_import(content):
    """Check if file imports any replacement class"""
    return bool(re.search(r'from pateda\.replacement import|from replacement import', content))


def has_eda_components(content):
    """Check if file uses EDAComponents"""
    return 'EDAComponents(' in content


def add_elitist_import(content):
    """Add ElitistReplacement to imports"""
    # Check if there's already a replacement import
    if 'from pateda.replacement import' in content:
        # Add ElitistReplacement to existing import
        content = re.sub(
            r'from pateda\.replacement import ([^\n]+)',
            lambda m: f"from pateda.replacement import {m.group(1)}, ElitistReplacement" if 'ElitistReplacement' not in m.group(1) else m.group(0),
            content
        )
    elif 'from replacement import' in content:
        # Add ElitistReplacement to existing import (direct import)
        content = re.sub(
            r'from replacement import ([^\n]+)',
            lambda m: f"from replacement import {m.group(1)}, ElitistReplacement" if 'ElitistReplacement' not in m.group(1) else m.group(0),
            content
        )
    else:
        # Add new import after other pateda imports
        # Find last pateda import
        lines = content.split('\n')
        last_import_idx = -1
        for i, line in enumerate(lines):
            if line.strip().startswith('from pateda') or line.strip().startswith('import pateda'):
                last_import_idx = i

        if last_import_idx >= 0:
            lines.insert(last_import_idx + 1, 'from pateda.replacement import ElitistReplacement')
            content = '\n'.join(lines)

    return content


def replace_generational_with_elitist(content):
    """Replace GenerationalReplacement() with ElitistReplacement()"""
    content = re.sub(
        r'replacement=GenerationalReplacement\(\)',
        'replacement=ElitistReplacement()',
        content
    )
    # Also remove the import of GenerationalReplacement if it's the only one
    content = re.sub(
        r'from pateda\.replacement import GenerationalReplacement\n',
        'from pateda.replacement import ElitistReplacement\n',
        content
    )
    content = re.sub(
        r'from replacement import GenerationalReplacement\n',
        'from replacement import ElitistReplacement\n',
        content
    )
    # If it's in a multi-import, replace it
    content = re.sub(
        r'(\bfrom pateda\.replacement import [^,\n]*)(GenerationalReplacement)([,\s])',
        r'\1ElitistReplacement\3',
        content
    )

    return content


def add_replacement_to_components(content):
    """Add replacement=ElitistReplacement() to EDAComponents that don't have it"""
    # Find EDAComponents( blocks and add replacement if missing

    # Pattern to match EDAComponents(...) allowing for multiline
    pattern = r'(EDAComponents\s*\([^)]*?)(stop_condition=[^,\)]+)'

    def replacer(match):
        components_start = match.group(1)
        stop_condition = match.group(2)

        # Check if replacement is already specified
        if 'replacement=' in components_start + stop_condition:
            return match.group(0)

        # Add replacement before stop_condition
        return f"{components_start}replacement=ElitistReplacement(),\n        {stop_condition}"

    content = re.sub(pattern, replacer, content, flags=re.DOTALL)

    return content


def fix_example_file(filepath):
    """Add ElitistReplacement to an example file if needed"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Skip if doesn't use EDAComponents
    if not has_eda_components(content):
        print(f"  ⊘ {filepath.name} - doesn't use EDAComponents")
        return False

    # Skip if already uses ElitistReplacement
    if has_elitist_replacement(content):
        print(f"  ✓ {filepath.name} - already uses ElitistReplacement")
        return False

    original_content = content

    # Replace GenerationalReplacement with ElitistReplacement
    if has_generational_replacement(content):
        content = replace_generational_with_elitist(content)
        print(f"  ✓ {filepath.name} - replaced GenerationalReplacement with ElitistReplacement")
    else:
        # Add ElitistReplacement import if not present
        if not has_replacement_import(content):
            content = add_elitist_import(content)

        # Add replacement to EDAComponents
        content = add_replacement_to_components(content)
        print(f"  ✓ {filepath.name} - added ElitistReplacement")

    # Only write if content changed
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True

    return False


def main():
    """Fix all example files to use elitism"""
    examples_dir = Path(__file__).parent.parent / 'packages' / 'pateda' / 'examples'

    print("Adding ElitistReplacement to EDA examples to guarantee elitism...\n")

    fixed_count = 0
    for filepath in sorted(examples_dir.glob('*.py')):
        if filepath.name == '__init__.py':
            continue

        if fix_example_file(filepath):
            fixed_count += 1

    print(f"\n✓ Updated {fixed_count} example files with elitism")


if __name__ == '__main__':
    main()
