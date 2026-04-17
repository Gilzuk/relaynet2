import os, shutil, zipfile, glob

print("Starting overleaf sync...")

# Clear old chapters in overleaf_thesis
for f in glob.glob('overleaf_thesis/chapters/*.tex'):
    os.remove(f)
    print(f'Removed: {f}')

# Copy all chapters to overleaf_thesis
for f in glob.glob('chapters/*.tex'):
    dst = 'overleaf_thesis/' + f
    shutil.copy(f, dst)
    print(f'Copied: {f} -> {dst}')

# Copy updated main.tex
shutil.copy('thesis_tau.tex', 'overleaf_thesis/main.tex')
print('Copied: thesis_tau.tex -> overleaf_thesis/main.tex')

# Copy hebrewcal.sty
shutil.copy('hebrewcal.sty', 'overleaf_thesis/hebrewcal.sty')
print('Copied: hebrewcal.sty')

# Copy results folder
def force_remove(func, path, exc_info):
    import stat
    os.chmod(path, stat.S_IWRITE)
    func(path)

if os.path.exists('overleaf_thesis/results'):
    shutil.rmtree('overleaf_thesis/results', onerror=force_remove)
shutil.copytree('results', 'overleaf_thesis/results')
print('Copied: results/ folder')

# Rebuild zip
zip_path = 'overleaf_thesis.zip'
if os.path.exists(zip_path):
    os.remove(zip_path)

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk('overleaf_thesis'):
        for f in files:
            file_path = os.path.join(root, f)
            arcname = os.path.relpath(file_path, 'overleaf_thesis')
            zf.write(file_path, arcname)

size = os.path.getsize(zip_path)
print(f'\nDone! overleaf_thesis.zip created: {size/1024/1024:.1f} MB')