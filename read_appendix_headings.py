import re

with open('chapters/ch09_appendices.tex', 'r', encoding='utf-8') as f:
    text = f.read()

headings = re.findall(r'\\(?:chapter|section|subsection)\*?\{([^}]+)\}', text)

with open('appendix_headings.txt', 'w', encoding='utf-8') as f:
    for h in headings:
        f.write(h + '\n')
    f.write(f'\nTotal lines: {len(text.splitlines())}\n')