"""Fix parsing errors in thesis.md - remove LaTeX remnants and shell garbage."""
import re

with open('thesis.md', 'r', encoding='utf-8') as f:
    content = f.read()

print(f'Input: {len(content):,} chars')

# ── 1. Fix the M_{ij} cases block ────────────────────────────────────────────
# Current broken form: $$M_{ij} = 2>&1\n\n\\begin{cases}...\\end{cases}\n\n
# Target: $$\nM_{ij} = \begin{cases}...\end{cases}\n$$
content = re.sub(
    r'\$\$M_\{ij\}\s*=\s*2>&1\s*\\\\begin\{cases\}(.*?)\\\\end\{cases\}',
    r'$$\nM_{ij} = \\begin{cases}\1\\end{cases}\n$$',
    content, flags=re.DOTALL
)

# ── 2. Fix the aligned block ──────────────────────────────────────────────────
# Current: \n\n2>&1\n\\begin{aligned}...\\end{aligned}\n2>&1
content = re.sub(
    r'\n+2>&1\n\\\\begin\{aligned\}(.*?)\\\\end\{aligned\}\n2>&1',
    r'\n\n$$\n\\begin{aligned}\1\\end{aligned}\n$$',
    content, flags=re.DOTALL
)

# ── 3. Remove all remaining 2>&1 occurrences ─────────────────────────────────
content = re.sub(r'\s*2>&1\s*', ' ', content)
# Clean up spaces that replaced 2>&1 at line boundaries
content = re.sub(r'^ $', '', content, flags=re.MULTILINE)

# ── 4. Fix double-backslash begin/end that remain ────────────────────────────
content = re.sub(r'\\\\begin\{cases\}', r'\\begin{cases}', content)
content = re.sub(r'\\\\end\{cases\}', r'\\end{cases}', content)
content = re.sub(r'\\\\begin\{aligned\}', r'\\begin{aligned}', content)
content = re.sub(r'\\\\end\{aligned\}', r'\\end{aligned}', content)

# ── 5. Fix any remaining LaTeX table remnants ─────────────────────────────────
for cmd in ['toprule', 'midrule', 'bottomrule', 'hline', 'tabularnewline',
            'endfirsthead', 'endhead', 'endlastfoot']:
    content = re.sub(r'\\' + cmd + r'\b', '', content)

# ── 6. Clean up multiple blank lines ─────────────────────────────────────────
content = re.sub(r'\n{4,}', '\n\n\n', content)

with open('thesis.md', 'w', encoding='utf-8') as f:
    f.write(content)

print(f'Output: {len(content):,} chars')

# Final audit
issues = []
for pattern, desc in [
    (r'\\begin\{', 'begin env'),
    (r'\\end\{', 'end env'),
    (r'2>&1', 'shell garbage'),
    (r'\\tabularnewline', 'tabularnewline'),
    (r'\\toprule', 'toprule'),
]:
    count = len(re.findall(pattern, content))
    if count:
        m = re.search(pattern, content)
        line = content[:m.start()].count('\n') + 1
        issues.append(f'{count}x {desc} (first line {line})')

print('Remaining issues:', issues if issues else 'NONE - CLEAN!')