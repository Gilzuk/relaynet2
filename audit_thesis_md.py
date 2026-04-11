"""Final audit of thesis.md for parsing errors."""
import re

with open('thesis.md', 'r', encoding='utf-8') as f:
    content = f.read()

print(f'File size: {len(content):,} chars')
print(f'Lines: {content.count(chr(10)):,}')

checks = [
    (r'2>&1', 'shell garbage'),
    (r'\\begin\{(?!cases|aligned|bmatrix|pmatrix|vmatrix)', 'LaTeX env (non-math)'),
    (r'\\label\{', 'LaTeX label'),
    (r'\\protect\b', 'LaTeX protect'),
    (r'\\phantomsection', 'LaTeX phantomsection'),
    (r'\\tabularnewline', 'tabularnewline'),
    (r'\\toprule', 'toprule'),
    (r'\\midrule', 'midrule'),
    (r'\\bottomrule', 'bottomrule'),
    (r'\\pandocbounded', 'pandocbounded'),
    (r'\\includegraphics', 'includegraphics'),
    (r'\\section\{', 'LaTeX section'),
    (r'\\chapter\{', 'LaTeX chapter'),
]

all_clean = True
for pattern, desc in checks:
    count = len(re.findall(pattern, content))
    if count:
        m = re.search(pattern, content)
        line = content[:m.start()].count('\n') + 1
        print(f'  {count:3d}x {desc} (first line {line})')
        all_clean = False

if all_clean:
    print('ALL CLEAN - no parsing errors found!')

headings = re.findall(r'^#{1,4} .+', content, re.MULTILINE)
math_blocks = content.count('$$') // 2
figures = len(re.findall(r'!\[', content))
table_rows = len(re.findall(r'^\|.*\|$', content, re.MULTILINE))

print(f'Headings: {len(headings)}')
print(f'Math blocks ($$): {math_blocks}')
print(f'Figures: {figures}')
print(f'Table rows: {table_rows}')