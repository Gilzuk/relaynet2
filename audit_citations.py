"""
Audit citations: find all \cite{} keys used in .tex files,
check which are defined in references.bib, and which are missing.
Also find bib entries that are never cited.
"""
import re
import os

# Collect all .tex files in chapters/
tex_files = []
for root, dirs, files in os.walk('chapters'):
    for f in files:
        if f.endswith('.tex'):
            tex_files.append(os.path.join(root, f))

# Also check main tex
tex_files.append('thesis_tau.tex')

# Extract all \cite{...} keys from tex files
cited_keys = set()
cite_locations = {}  # key -> list of (file, line)

for tex_file in tex_files:
    try:
        with open(tex_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            # Match \cite{key1,key2,...} or \citep{} or \citet{} etc.
            for m in re.finditer(r'\\cite[tp]?\{([^}]+)\}', line):
                keys = [k.strip() for k in m.group(1).split(',')]
                for key in keys:
                    cited_keys.add(key)
                    if key not in cite_locations:
                        cite_locations[key] = []
                    cite_locations[key].append((tex_file, i+1))
    except Exception as e:
        print(f'Error reading {tex_file}: {e}')

# Extract all @entry{key, ...} from references.bib
bib_keys = set()
try:
    with open('references.bib', 'r', encoding='utf-8') as f:
        bib_text = f.read()
    for m in re.finditer(r'@\w+\{([^,\s]+)\s*,', bib_text):
        bib_keys.add(m.group(1).strip())
except Exception as e:
    print(f'Error reading references.bib: {e}')

# Find missing citations (cited but not in bib)
missing = cited_keys - bib_keys
# Find uncited bib entries
uncited = bib_keys - cited_keys

with open('citation_audit.txt', 'w', encoding='utf-8') as f:
    f.write(f'Total cited keys: {len(cited_keys)}\n')
    f.write(f'Total bib entries: {len(bib_keys)}\n')
    f.write(f'Missing from bib (cited but not defined): {len(missing)}\n')
    f.write(f'Uncited bib entries: {len(uncited)}\n\n')

    if missing:
        f.write('=== MISSING FROM BIB (need to add) ===\n')
        for key in sorted(missing):
            locs = cite_locations.get(key, [])
            f.write(f'  {key}\n')
            for loc in locs[:3]:
                f.write(f'    -> {loc[0]}:{loc[1]}\n')
        f.write('\n')

    if uncited:
        f.write('=== UNCITED BIB ENTRIES (never referenced) ===\n')
        for key in sorted(uncited):
            f.write(f'  {key}\n')
        f.write('\n')

    f.write('=== ALL CITED KEYS ===\n')
    for key in sorted(cited_keys):
        status = 'OK' if key in bib_keys else 'MISSING'
        f.write(f'  [{status}] {key}\n')

print('Done. See citation_audit.txt')