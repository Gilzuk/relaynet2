"""
fix_final_errors.py
-------------------
Fixes remaining LaTeX errors:
  1. Blank line inside aligned environment (line 1218)
  2. \def\LTcaptype{none} causing "No counter 'none' defined"
"""

import re

with open("thesis_tau.tex", "r", encoding="utf-8") as f:
    tex = f.read()

print(f"Input: {len(tex):,} chars")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Fix blank lines inside aligned/align environments
#    LaTeX doesn't allow blank lines inside math environments
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] Fixing blank lines inside math environments ...")

def remove_blank_lines_in_math(tex):
    # Fix blank lines inside aligned environments
    pattern = re.compile(
        r"(\\begin\{aligned\})(.*?)(\\end\{aligned\})",
        re.DOTALL
    )
    def replacer(m):
        content = m.group(2)
        # Remove blank lines (lines with only whitespace)
        content = re.sub(r"\n\s*\n", "\n", content)
        return m.group(1) + content + m.group(3)
    
    new_tex, c = pattern.subn(replacer, tex)
    print(f"  Fixed {c} aligned environments")
    return new_tex

def remove_blank_lines_in_align(tex):
    pattern = re.compile(
        r"(\\begin\{align\*?\})(.*?)(\\end\{align\*?\})",
        re.DOTALL
    )
    def replacer(m):
        content = m.group(2)
        content = re.sub(r"\n\s*\n", "\n", content)
        return m.group(1) + content + m.group(3)
    
    new_tex, c = pattern.subn(replacer, tex)
    print(f"  Fixed {c} align environments")
    return new_tex

tex = remove_blank_lines_in_math(tex)
tex = remove_blank_lines_in_align(tex)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Fix \def\LTcaptype{none} → suppress table counter increment differently
#    pandoc-crossref uses this for unlabeled tables, but 'none' counter doesn't exist
#    Fix: replace with a comment or use a valid approach
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Fixing \\def\\LTcaptype{none} ...")

# Replace \def\LTcaptype{none} with \def\LTcaptype{table}
# This makes unlabeled tables use the normal table counter
tex, c = re.subn(
    r"\\def\\LTcaptype\{none\}",
    r"\\def\\LTcaptype{table}",
    tex
)
print(f"  Fixed {c} \\def\\LTcaptype{{none}} occurrences")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Fix \label{none} and \ref{none} from pandoc-crossref
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Fixing \\label{none} and \\ref{none} ...")

# Remove \label{none} (unlabeled items)
tex, c1 = re.subn(r"\\label\{none\}", "", tex)
print(f"  Removed {c1} \\label{{none}}")

# Replace \ref{none} with ?? (broken reference)
tex, c2 = re.subn(r"\\ref\{none\}", r"??", tex)
print(f"  Replaced {c2} \\ref{{none}}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Write output
# ─────────────────────────────────────────────────────────────────────────────
with open("thesis_tau.tex", "w", encoding="utf-8") as f:
    f.write(tex)

print(f"\nOutput: thesis_tau.tex ({len(tex):,} chars)")