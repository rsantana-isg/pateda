#!/usr/bin/env python3
"""
Script to properly add ElitistReplacement to all EDA examples.
This version handles all cases correctly.
"""

import re
from pathlib import Path


def fix_eda_components_file(filepath):
    """Fix a file that uses EDAComponents"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # Skip if already uses ElitistReplacement
    if 'ElitistReplacement' in content:
        print(f"  ✓ {filepath.name} - already uses ElitistReplacement")
        return False

    # Skip if doesn't use EDAComponents
    if 'EDAComponents(' not in content:
        print(f"  ⊘ {filepath.name} - doesn't use EDAComponents")
        return False

    # 1. Replace GenerationalReplacement with ElitistReplacement in imports
    content = re.sub(
        r'from pateda\.replacement import GenerationalReplacement',
        'from pateda.replacement import ElitistReplacement',
        content
    )

    # 2. Replace GenerationalReplacement() with ElitistReplacement() in code
    content = re.sub(
        r'\breplacement=GenerationalReplacement\(\)',
        'replacement=ElitistReplacement()',
        content
    )

    # 3. Add import if not present and file uses EDAComponents but has no replacement
    if 'from pateda.replacement import' not in content and 'from replacement import' not in content:
        # Find the last 'from pateda' import line
        lines = content.split('\n')
        last_pateda_import = -1
        for i, line in enumerate(lines):
            if line.strip().startswith('from pateda') and 'import' in line:
                last_pateda_import = i

        if last_pateda_import >= 0:
            lines.insert(last_pateda_import + 1, 'from pateda.replacement import ElitistReplacement')
            content = '\n'.join(lines)

    # 4. Add replacement= to EDAComponents if not present
    # Find all EDAComponents( blocks
    def add_replacement_to_components(match):
        full_match = match.group(0)

        # Skip if already has replacement=
        if 'replacement=' in full_match:
            return full_match

        # Find stop_condition= line and add replacement= before it
        if 'stop_condition=' in full_match:
            # Add replacement before stop_condition
            result = re.sub(
                r'(\s+)(stop_condition=)',
                r'\1replacement=ElitistReplacement(),\n\1\2',
                full_match
            )
            return result

        # If no stop_condition, add at the end before closing paren
        result = re.sub(
            r'(\s+)(\))\s*$',
            r',\1replacement=ElitistReplacement()\1\2',
            full_match,
            flags=re.MULTILINE
        )
        return result

    # Match EDAComponents(...) allowing multiline
    content = re.sub(
        r'EDAComponents\s*\([^)]*\)',
        add_replacement_to_components,
        content,
        flags=re.DOTALL
    )

    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✓ {filepath.name} - added ElitistReplacement")
        return True

    return False


def main():
    """Fix all example files"""
    examples_dir = Path(__file__).parent / 'examples'

    print("Fixing elitism in all EDA examples...\n")

    fixed_count = 0
    for filepath in sorted(examples_dir.glob('*.py')):
        if filepath.name == '__init__.py':
            continue

        if fix_eda_components_file(filepath):
            fixed_count += 1

    print(f"\n✓ Fixed {fixed_count} files")


if __name__ == '__main__':
    main()
