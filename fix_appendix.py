"""
Fix ch09_appendices.tex by removing the accidentally embedded ch05_experiments.tex content.

The corruption:
- Line 22: '\(\mathbf{W}\) & Neural network weight matrix \\%% ============================================================'
  (ch05_experiments.tex content starts here, embedded in the middle of the table)
- Lines 22-550: Two copies of ch05_experiments.tex content
- Line 551: '\(\mathbf{b}\) & Bias vector \\' (original table resumes here)

Fix: Remove the embedded content and restore the table.
"""

with open('chapters/ch09_appendices.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# The corruption starts at the end of line 21 (after the \\ of the \mathbf{W} row)
# and ends just before line 551 (\(\mathbf{b}\) & Bias vector \\)

# Find the exact corruption boundaries
CORRUPT_START = r'\(\mathbf{W}\) & Neural network weight matrix \\%% ============================================================'
CORRUPT_END_MARKER = r'\(\mathbf{b}\) & Bias vector \\'

# Find the start of corruption
start_idx = content.find(CORRUPT_START)
if start_idx == -1:
    print('ERROR: Could not find corruption start marker!')
    exit(1)

# Find the end of corruption (the \mathbf{b} row)
end_idx = content.find(CORRUPT_END_MARKER, start_idx)
if end_idx == -1:
    print('ERROR: Could not find corruption end marker!')
    exit(1)

print(f'Corruption starts at char {start_idx}')
print(f'Corruption ends at char {end_idx}')
print(f'Removing {end_idx - start_idx - len(CORRUPT_START)} chars of embedded content')

# Fix: Replace the corrupted section with just the correct table row
fixed_content = (
    content[:start_idx] +
    r'\(\mathbf{W}\) & Neural network weight matrix \\' + '\n' +
    content[end_idx:]
)

with open('chapters/ch09_appendices.tex', 'w', encoding='utf-8') as f:
    f.write(fixed_content)

print('Fixed ch09_appendices.tex')

# Verify
with open('chapters/ch09_appendices.tex', 'r', encoding='utf-8') as f:
    lines = f.readlines()
print(f'New line count: {len(lines)}')

# Check for duplicate labels
import re
labels = {}
for i, line in enumerate(lines):
    for m in re.finditer(r'\\label\{([^}]+)\}', line):
        label = m.group(1)
        if label not in labels:
            labels[label] = []
        labels[label].append(i+1)

dup = {k: v for k, v in labels.items() if len(v) > 1}
print(f'Duplicate labels after fix: {len(dup)}')
if dup:
    for k, v in dup.items():
        print(f'  {k}: lines {v}')