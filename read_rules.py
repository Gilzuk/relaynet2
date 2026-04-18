import os

for f in sorted(os.listdir('.clinerules')):
    path = os.path.join('.clinerules', f)
    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()
    print(f'=== {f} ===')
    print(content)
    print()