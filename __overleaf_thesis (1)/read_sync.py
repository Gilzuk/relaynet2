with open('overleaf_readme.md', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    lower_line = line.lower()
    if 'sync' in lower_line or 'upload' in lower_line or 'push' in lower_line or 'replica' in lower_line:
        print(f'Line {i+1}: {line.strip()}')