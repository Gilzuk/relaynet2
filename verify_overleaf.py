import os, zipfile

size = os.path.getsize('overleaf_thesis.zip')
print(f'overleaf_thesis.zip size: {size/1024/1024:.1f} MB')

with zipfile.ZipFile('overleaf_thesis.zip', 'r') as zf:
    names = zf.namelist()
    tex_files = [n for n in names if n.endswith('.tex')]
    print(f'Total files: {len(names)}, .tex files: {len(tex_files)}')
    for t in sorted(tex_files):
        print(f'  {t}')