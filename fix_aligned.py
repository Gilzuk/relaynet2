"""Fix the remaining \begin{aligned} block and 2>&1 remnants."""
with open('thesis.md', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f'Total lines: {len(lines)}')

# Find and fix the aligned block
fixed = 0
i = 0
while i < len(lines):
    # Replace '2>&1\n' that precedes \begin{aligned}
    if lines[i] == '2>&1\n' and i + 1 < len(lines) and '\\begin{aligned}' in lines[i+1]:
        lines[i] = '$$\n'
        fixed += 1
        print(f'Fixed line {i+1}: 2>&1 -> $$ (before aligned)')
    # Replace '2>&1\n' that follows \end{aligned}
    elif '\\end{aligned}' in lines[i] and i + 1 < len(lines) and lines[i+1] == '2>&1\n':
        lines[i+1] = '$$\n'
        fixed += 1
        print(f'Fixed line {i+2}: 2>&1 -> $$ (after aligned)')
    # Remove any remaining standalone 2>&1 lines
    elif lines[i].strip() == '2>&1':
        lines[i] = '\n'
        fixed += 1
        print(f'Fixed line {i+1}: removed standalone 2>&1')
    i += 1

print(f'Total fixes: {fixed}')

content = ''.join(lines)
with open('thesis.md', 'w', encoding='utf-8') as f:
    f.write(content)

import re
remaining = len(re.findall(r'2>&1', content))
print(f'Remaining 2>&1: {remaining}')
print(f'File size: {len(content):,} chars')