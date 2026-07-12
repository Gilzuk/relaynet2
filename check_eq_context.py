import re

with open('chapters/ch01_introduction.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# Find all \par{\small\textit{Source: ...}} annotations
sources = list(re.finditer(r'\\par\{\\small\\textit\{Source:[^}]*\}[^}]*\}', text))
print(f'Total source annotations: {len(sources)}')
for m in sources[:5]:
    print(repr(m.group(0)))
    print()

# Check if any equation has two source annotations
eq_blocks = re.findall(r'\\end\{equation\}(.{0,200})', text, re.DOTALL)
double = [b for b in eq_blocks if b.count('\\par{\\small') > 1]
print(f'Equations with 2+ source annotations: {len(double)}')
for d in double[:2]:
    print(repr(d[:200]))