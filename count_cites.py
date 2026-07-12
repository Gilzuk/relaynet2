import re

chapters = [
    'chapters/ch01_introduction.tex',
    'chapters/ch02_literature_review.tex',
    'chapters/ch03_objectives.tex',
    'chapters/ch04_methods.tex',
]

for ch in chapters:
    try:
        with open(ch, 'r', encoding='utf-8') as f:
            text = f.read()
        cites = re.findall(r'\\cite[tp]?\{[^}]+\}', text)
        lines = text.splitlines()
        print(f'{ch}: {len(cites)} citations, {len(lines)} lines')
    except Exception as e:
        print(f'Error {ch}: {e}')