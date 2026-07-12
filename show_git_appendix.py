with open('ch09_appendices_backup.tex', 'r', encoding='utf-8') as f:
    lines = f.readlines()

with open('git_appendix_start.txt', 'w', encoding='utf-8') as f:
    f.write(f'Lines in git version: {len(lines)}\n')
    f.write('Lines 1-30:\n')
    for i, line in enumerate(lines[:30]):
        f.write(f'{i+1}: {line.rstrip()}\n')