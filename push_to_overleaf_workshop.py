"""
Copy updated files from overleaf_thesis/ to __overleaf_thesis (1)/
Preserves the .overleaf metadata folder (Overleaf Workshop sync data).
"""
import os, shutil, glob, re

src = 'overleaf_thesis'
dst = '__overleaf_thesis (1)'

print(f'Copying from {src}/ to {dst}/')

# 1. Copy main.tex
shutil.copy(os.path.join(src, 'main.tex'), os.path.join(dst, 'main.tex'))
print('Copied: main.tex')

# 2. Copy hebrewcal.sty
shutil.copy(os.path.join(src, 'hebrewcal.sty'), os.path.join(dst, 'hebrewcal.sty'))
print('Copied: hebrewcal.sty')

# 3. Copy references.bib (if it exists in src)
if os.path.exists(os.path.join(src, 'references.bib')):
    shutil.copy(os.path.join(src, 'references.bib'), os.path.join(dst, 'references.bib'))
    print('Copied: references.bib')

# 4. Copy all chapter .tex files
src_chapters = os.path.join(src, 'chapters')
dst_chapters = os.path.join(dst, 'chapters')
os.makedirs(dst_chapters, exist_ok=True)

for f in sorted(glob.glob(os.path.join(src_chapters, '*.tex'))):
    fname = os.path.basename(f)
    shutil.copy(f, os.path.join(dst_chapters, fname))
    print(f'Copied: chapters/{fname}')

# 5. Copy results/ folder (only referenced figures)
# Find all figures referenced in the .tex files
referenced = set()
for f in glob.glob('chapters/*.tex'):
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
        matches = re.findall(r'\\includegraphics\[.*?\]\{([^}]+)\}', content)
        for m in matches:
            m = m.strip()
            if m.startswith('results/'):
                m = m[len('results/'):]
            elif m.startswith('results\\'):
                m = m[len('results\\'):]
            referenced.add(m)

print(f'\nCopying {len(referenced)} referenced figures to results/')
dst_results = os.path.join(dst, 'results')
os.makedirs(dst_results, exist_ok=True)

copied_figs = 0
for fig in sorted(referenced):
    src_fig = os.path.join('results', fig.replace('/', os.sep))
    dst_fig = os.path.join(dst_results, fig.replace('/', os.sep))
    
    if os.path.exists(src_fig):
        os.makedirs(os.path.dirname(dst_fig), exist_ok=True)
        shutil.copy(src_fig, dst_fig)
        copied_figs += 1
    else:
        print(f'WARNING: Figure not found: {src_fig}')

print(f'Copied: {copied_figs} figures to results/')

print(f'\nDone! {dst}/ is now up-to-date.')
print('You can now use the Overleaf Workshop extension to push these changes to Overleaf.')