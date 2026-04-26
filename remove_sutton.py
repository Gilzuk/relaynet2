import re

with open('references.bib', 'r', encoding='utf-8') as f:
    text = f.read()

# Remove the sutton2018reinforcement entry
pattern = r'@\w+\{sutton2018reinforcement[^@]*'
match = re.search(pattern, text, re.DOTALL)
if match:
    print('Found entry:')
    print(repr(match.group()[:200]))
    new_text = text[:match.start()] + text[match.end():]
    with open('references.bib', 'w', encoding='utf-8') as f:
        f.write(new_text.strip() + '\n')
    print('Removed sutton2018reinforcement from references.bib')
else:
    print('Entry not found - checking manually...')
    idx = text.find('sutton2018')
    if idx != -1:
        print(f'Found at {idx}: {repr(text[idx-5:idx+100])}')
    else:
        print('Not found at all')