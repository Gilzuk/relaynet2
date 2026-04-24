with open('chapters/ch09_appendices.tex', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the \chapter{Appendices} line and add the label
for i, line in enumerate(lines):
    if r'\chapter{Appendices}' in line and 'chap:app-experiments' not in line:
        # Add the label to this line
        lines[i] = line.rstrip() + r'\label{chap:app-experiments}' + '\n'
        print(f'Added label to line {i+1}: {lines[i].strip()}')
        break

with open('chapters/ch09_appendices.tex', 'w', encoding='utf-8') as f:
    f.writelines(lines)

# Verify
with open('chapters/ch09_appendices.tex', 'r', encoding='utf-8') as f:
    text = f.read()
if 'chap:app-experiments' in text:
    print('Verification: Label found in file')
else:
    print('Verification: ERROR - Label NOT found')