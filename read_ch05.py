with open('chapters/ch05_experiments.tex', 'r', encoding='utf-8') as f:
    text = f.read()

import re
headings = re.findall(r'\\(?:chapter|section|subsection)\{([^}]+)\}', text)
for h in headings:
    print(h)
print(f'Total length: {len(text)} chars')