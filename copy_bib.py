import shutil, os, re

# Copy references.bib from root to __overleaf_thesis (1)/
if os.path.exists('references.bib'):
    shutil.copy('references.bib', '__overleaf_thesis (1)/references.bib')
    print('Copied: references.bib')
else:
    print('references.bib not found in root!')

# Check main.tex for bibliography reference
with open('__overleaf_thesis (1)/main.tex', 'r', encoding='utf-8') as f:
    main = f.read()

bib_refs = re.findall(r'\\bibliography\{[^}]+\}', main)
print(f'Bibliography commands in main.tex: {bib_refs}')

# Verify final state
print('\nFinal contents of __overleaf_thesis (1)/:')
for f in sorted(os.listdir('__overleaf_thesis (1)')):
    if not f.startswith('.'):
        print(f'  {f}')