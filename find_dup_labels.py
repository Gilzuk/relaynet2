import re

with open('chapters/ch09_appendices.tex', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find all labels and their line numbers
labels = {}
for i, line in enumerate(lines):
    for m in re.finditer(r'\\label\{([^}]+)\}', line):
        label = m.group(1)
        if label not in labels:
            labels[label] = []
        labels[label].append(i+1)

# Print duplicate labels with line numbers
dup = {k: v for k, v in labels.items() if len(v) > 1}
with open('dup_labels.txt', 'w', encoding='utf-8') as f:
    for k, v in dup.items():
        f.write(f'{k}: lines {v}\n')
        # Show context around each occurrence
        for line_num in v:
            f.write(f'  Line {line_num}: {lines[line_num-1].strip()}\n')