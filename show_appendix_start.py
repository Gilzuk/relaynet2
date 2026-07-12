with open('chapters/ch09_appendices.tex', 'r', encoding='utf-8') as f:
    lines = f.readlines()

with open('appendix_start.txt', 'w', encoding='utf-8') as f:
    f.write(f'Total lines: {len(lines)}\n')
    f.write('Lines 1-35:\n')
    for i, line in enumerate(lines[:35]):
        f.write(f'{i+1}: {line.rstrip()}\n')
    f.write('\nLines 280-300:\n')
    for i, line in enumerate(lines[279:300]):
        f.write(f'{279+i+1}: {line.rstrip()}\n')