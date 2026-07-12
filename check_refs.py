import re

with open('thesis.md', 'r', encoding='utf-8') as f:
    content = f.read()

matches = list(re.finditer(r'\b(Section|Chapter|Figure|Table)\b', content, re.IGNORECASE))
print(f'Matches found in thesis.md: {len(matches)}')
for m in matches[:10]:
    start = max(0, m.start() - 30)
    end = min(len(content), m.end() + 30)
    print(repr(content[start:end]))