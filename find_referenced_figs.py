import glob, re, os

# Find all figures referenced in the .tex files
referenced = set()
for f in glob.glob('chapters/*.tex'):
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
        matches = re.findall(r'\\includegraphics\[.*?\]\{([^}]+)\}', content)
        for m in matches:
            # Normalize: strip leading results/ prefix
            m = m.strip()
            if m.startswith('results/'):
                m = m[len('results/'):]
            elif m.startswith('results\\'):
                m = m[len('results\\'):]
            referenced.add(m)

print(f'Total referenced figures: {len(referenced)}')
for r in sorted(referenced):
    print(f'  {r}')

# Find all files in results/
all_results = set()
for root, dirs, files in os.walk('results'):
    for f in files:
        rel = os.path.relpath(os.path.join(root, f), 'results').replace('\\', '/')
        all_results.add(rel)

print(f'\nTotal files in results/: {len(all_results)}')

# Find unreferenced
unreferenced = all_results - referenced
print(f'\nUnreferenced files ({len(unreferenced)}):')
for u in sorted(unreferenced):
    print(f'  {u}')