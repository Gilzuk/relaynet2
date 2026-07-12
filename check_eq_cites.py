import re

for filename in ['chapters/ch01_introduction.tex', 'chapters/ch02_literature_review.tex']:
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()

    # Check if citations were added after equations
    matches = re.findall(r'\\end\{equation\}\\par\{\\small', text)
    print(f'{filename}: {len(matches)} citations added after equations')

    # Check the context around relay-capacity-bound
    idx = text.find('eq:relay-capacity-bound')
    if idx != -1:
        print(repr(text[idx:idx+200]))
        print()

    # Check how many equations have \par{\small after them
    eq_ends = list(re.finditer(r'\\end\{equation\}', text))
    cited = 0
    for m in eq_ends:
        after = text[m.end():m.end()+50]
        if '\\par{\\small' in after or '\\cite' in after:
            cited += 1
    print(f'  Equations with citation after \\end{{equation}}: {cited} / {len(eq_ends)}')