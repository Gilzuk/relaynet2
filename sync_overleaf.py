import os, shutil, zipfile, glob, re

print("Starting overleaf sync (clean version)...")

# 1. Find all figures referenced in the .tex files
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

print(f'Referenced figures: {len(referenced)}')

# 2. Build the zip directly (no intermediate folder copy)
zip_path = 'overleaf_thesis.zip'
if os.path.exists(zip_path):
    os.remove(zip_path)

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    # Add main.tex
    zf.write('thesis_tau.tex', 'main.tex')
    print('Added: main.tex')

    # Add hebrewcal.sty
    zf.write('hebrewcal.sty', 'hebrewcal.sty')
    print('Added: hebrewcal.sty')

    # Add all chapter .tex files
    for f in sorted(glob.glob('chapters/*.tex')):
        arcname = f.replace('\\', '/')
        zf.write(f, arcname)
        print(f'Added: {arcname}')

    # Add only referenced figures from results/
    added_figs = 0
    for fig in sorted(referenced):
        src = os.path.join('results', fig.replace('/', os.sep))
        if os.path.exists(src):
            arcname = 'results/' + fig
            zf.write(src, arcname)
            added_figs += 1
        else:
            print(f'WARNING: Referenced figure not found: {src}')

    print(f'Added: {added_figs} figures from results/')

size = os.path.getsize(zip_path)
print(f'\nDone! overleaf_thesis.zip created: {size/1024/1024:.1f} MB')
print(f'(Only referenced figures included, no PDFs, no unused files)')