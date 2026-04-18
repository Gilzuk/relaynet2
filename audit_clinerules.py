import re, glob

print("=== AUDIT AGAINST .clinerules ===\n")

# 1. Check ch05 for bold text, figures, longtables, hardcoded refs
with open('chapters/ch05_experiments.tex', 'r', encoding='utf-8') as f:
    ch05 = f.read()

bold = re.findall(r'\\textbf\{[^}]+\}', ch05)
figs = re.findall(r'\\begin\{figure\}', ch05)
tables = re.findall(r'\\begin\{longtable\}', ch05)
hardcoded = re.findall(r'\b(Section|Chapter|Figure|Table)\s+\d+', ch05, re.IGNORECASE)

print(f"ch05 - Bold text: {len(bold)}")
for b in bold:
    print(f"  {b}")
print(f"ch05 - Figures: {len(figs)} (should be 0 except master table)")
print(f"ch05 - Longtables: {len(tables)} (should be 0 except master table)")
print(f"ch05 - Hardcoded refs: {hardcoded}")

# 2. Check all chapters for hardcoded refs
print("\n=== Hardcoded refs in all chapters ===")
for f in sorted(glob.glob('chapters/*.tex')):
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    matches = re.findall(r'\b(Section|Chapter|Figure|Table)\s+\d+(?:\.\d+)?\b', content, re.IGNORECASE)
    if matches:
        print(f"{f}: {matches}")

# 3. Check equations without \cite in ch01 and ch02
print("\n=== Equations without \\cite in ch01/ch02 ===")
for f in ['chapters/ch01_introduction.tex', 'chapters/ch02_literature_review.tex']:
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    # Find equation blocks
    eq_blocks = re.findall(r'\\begin\{equation\}(.*?)\\end\{equation\}', content, re.DOTALL)
    no_cite = [eq for eq in eq_blocks if '\\cite' not in eq and '\\par{\\small' not in eq]
    if no_cite:
        print(f"{f}: {len(no_cite)} equations without citation")
        for eq in no_cite[:3]:
            print(f"  {eq[:80].strip()}")

# 4. Check appendix figures have labels
print("\n=== Appendix figures without labels ===")
with open('chapters/ch09_appendices.tex', 'r', encoding='utf-8') as f:
    ch09 = f.read()

fig_blocks = re.findall(r'\\begin\{figure\}.*?\\end\{figure\}', ch09, re.DOTALL)
no_label = [b for b in fig_blocks if '\\label{' not in b]
print(f"ch09 - Figures without label: {len(no_label)}")
for b in no_label[:3]:
    print(f"  {b[:100].strip()}")