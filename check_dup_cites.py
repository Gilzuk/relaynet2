import re

for filename in ['chapters/ch01_introduction.tex', 'chapters/ch02_literature_review.tex']:
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()

    # Check for duplicate citations (two \par{\small after same equation)
    dups = re.findall(r'\\par\{\\small\\textit\{Source:[^}]+\}\}\s*\\par\{\\small\\textit\{Source:', text)
    print(f'{filename}: {len(dups)} duplicate citations')

    # Check the relay-capacity-bound context
    idx = text.find('relay-capacity-bound')
    if idx != -1:
        print(repr(text[idx:idx+300]))