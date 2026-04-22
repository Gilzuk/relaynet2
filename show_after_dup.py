with open('chapters/ch09_appendices.tex', 'r', encoding='utf-8') as f:
    lines = f.readlines()

with open('appendix_after_dup.txt', 'w', encoding='utf-8') as f:
    f.write(f'Total lines: {len(lines)}\n')
    f.write('Lines 525-545:\n')
    for i, line in enumerate(lines[524:545]):
        f.write(f'{524+i+1}: {line.rstrip()}\n')
    f.write('\nLines 540-560:\n')
    for i, line in enumerate(lines[539:560]):
        f.write(f'{539+i+1}: {line.rstrip()}\n')