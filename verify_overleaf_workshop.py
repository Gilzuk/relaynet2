import os

# Check if references.bib was copied
src = os.path.join('overleaf_thesis', 'references.bib')
dst = os.path.join('__overleaf_thesis (1)', 'references.bib')
print(f'references.bib in overleaf_thesis/: {os.path.exists(src)}')
print(f'references.bib in __overleaf_thesis (1)/: {os.path.exists(dst)}')

# List what's in __overleaf_thesis (1)/
print('\nContents of __overleaf_thesis (1)/:')
for f in sorted(os.listdir('__overleaf_thesis (1)')):
    if not f.startswith('.'):
        print(f'  {f}')

# Count chapters
chapters_dir = os.path.join('__overleaf_thesis (1)', 'chapters')
if os.path.exists(chapters_dir):
    chapters = [f for f in os.listdir(chapters_dir) if f.endswith('.tex')]
    print(f'\nChapters: {len(chapters)}')
    for c in sorted(chapters):
        print(f'  {c}')

# Count figures
results_dir = os.path.join('__overleaf_thesis (1)', 'results')
if os.path.exists(results_dir):
    figs = []
    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if f.endswith('.png'):
                figs.append(f)
    print(f'\nFigures: {len(figs)}')