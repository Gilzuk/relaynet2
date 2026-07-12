import re

with open('chapters/ch09_appendices.tex', 'r', encoding='utf-8') as f:
    text = f.read()

figs = len(re.findall(r'\\begin\{figure\}', text))
longtables = len(re.findall(r'\\begin\{longtable\}', text))
includegraphics = len(re.findall(r'\\includegraphics', text))

with open('appendix_analysis.txt', 'w', encoding='utf-8') as f:
    f.write(f'Figures: {figs}\n')
    f.write(f'Longtables: {longtables}\n')
    f.write(f'Includegraphics: {includegraphics}\n')
    f.write(f'File size: {len(text)} chars\n')

    # Check for any potential issues
    long_lines = [i+1 for i, line in enumerate(text.split('\n')) if len(line) > 500]
    f.write(f'Lines > 500 chars: {len(long_lines)}\n')
    if long_lines:
        f.write(f'First few: {long_lines[:5]}\n')

    # Check for duplicate labels
    labels = re.findall(r'\\label\{([^}]+)\}', text)
    from collections import Counter
    dup_labels = {k: v for k, v in Counter(labels).items() if v > 1}
    f.write(f'\nDuplicate labels: {len(dup_labels)}\n')
    for k, v in dup_labels.items():
        f.write(f'  {k}: {v} times\n')

    # Check for missing image files
    imgs = re.findall(r'\\includegraphics\[.*?\]\{([^}]+)\}', text)
    import os
    missing = [img for img in imgs if not os.path.exists(img)]
    f.write(f'\nMissing image files: {len(missing)}\n')
    for m in missing[:10]:
        f.write(f'  {m}\n')