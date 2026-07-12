"""
fix_latex_errors.py
-------------------
Fixes two LaTeX compilation errors in thesis_tau.tex:
  1. \frontmatter undefined: report class doesn't have it → change to book class
  2. bidi/tabular conflict: polyglossia's bidi package conflicts with tabular
     → use bidi=basic option + replace title page tabular with plain text
"""

import re

with open("thesis_tau.tex", "r", encoding="utf-8") as f:
    tex = f.read()

print(f"Input: {len(tex):,} chars")

# ─────────────────────────────────────────────────────────────────────────────
# Fix 1: Change report → book (adds \frontmatter, \mainmatter, \backmatter)
# ─────────────────────────────────────────────────────────────────────────────
tex = tex.replace(
    r"\documentclass[12pt,a4paper,oneside]{report}",
    r"\documentclass[12pt,a4paper,oneside]{book}"
)
print("[1] Changed report → book")

# ─────────────────────────────────────────────────────────────────────────────
# Fix 2: Use bidi=basic to avoid tabular conflict
# ─────────────────────────────────────────────────────────────────────────────
tex = tex.replace(
    r"\usepackage{polyglossia}" + "\n" + r"\setmainlanguage{english}" + "\n" + r"\setotherlanguage{hebrew}",
    r"\usepackage[bidi=basic]{polyglossia}" + "\n" + r"\setmainlanguage{english}" + "\n" + r"\setotherlanguage{hebrew}"
)
print("[2] Added bidi=basic to polyglossia")

# ─────────────────────────────────────────────────────────────────────────────
# Fix 3: Replace title page tabular with plain text (belt-and-suspenders)
# ─────────────────────────────────────────────────────────────────────────────
old_tabular = r"""\vfill
\begin{tabular}{ll}
\textbf{Thesis Advisor:} & [Supervisor Name]\\[4pt]
\textbf{Department:} & Electrical Engineering\\[4pt]
\textbf{Faculty:} & The Iby and Aladar Fleischman Faculty of Engineering\\
\end{tabular}"""

new_tabular = r"""\vfill
\begin{flushleft}
\textbf{Thesis Advisor:} \quad [Supervisor Name]\\[4pt]
\textbf{Department:} \quad Electrical Engineering\\[4pt]
\textbf{Faculty:} \quad The Iby and Aladar Fleischman Faculty of Engineering
\end{flushleft}"""

if old_tabular in tex:
    tex = tex.replace(old_tabular, new_tabular)
    print("[3] Replaced title page tabular with flushleft")
else:
    # Try flexible match
    tex, c = re.subn(
        r"\\vfill\s*\\begin\{tabular\}\{ll\}.*?\\end\{tabular\}",
        new_tabular,
        tex,
        flags=re.DOTALL
    )
    print(f"[3] Replaced title page tabular (flexible match, {c} replacements)")

# ─────────────────────────────────────────────────────────────────────────────
# Fix 4: Add \chapter* to TOC for unnumbered chapters
#         (book class needs explicit \addcontentsline for \chapter*)
# ─────────────────────────────────────────────────────────────────────────────
# Already handled in the template - no change needed

# ─────────────────────────────────────────────────────────────────────────────
# Fix 5: Ensure \tocloft is compatible with book class
#         (tocloft works with book, no change needed)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Fix 6: Fix chapter* headings in book class
#         \chapter*{Equation Reference} needs \addcontentsline
# ─────────────────────────────────────────────────────────────────────────────
# Check if Equation Reference chapter has addcontentsline
if r"\chapter*{Equation Reference}" in tex:
    if r"\addcontentsline{toc}{chapter}{Equation Reference}" not in tex:
        tex = tex.replace(
            r"\chapter*{Equation Reference}",
            r"\chapter*{Equation Reference}" + "\n" +
            r"\addcontentsline{toc}{chapter}{Equation Reference}" + "\n" +
            r"\markboth{Equation Reference}{Equation Reference}"
        )
        print("[6] Added TOC entry for Equation Reference chapter")

# ─────────────────────────────────────────────────────────────────────────────
# Fix 7: Fix \chapter{References} - should be unnumbered in back matter
# ─────────────────────────────────────────────────────────────────────────────
# The References chapter is in \backmatter so it should be \chapter*
# But pandoc generated \chapter{References} - let's keep it numbered
# Actually in book class, \backmatter makes chapters unnumbered automatically

# ─────────────────────────────────────────────────────────────────────────────
# Fix 8: Add \graphicspath to find images
# ─────────────────────────────────────────────────────────────────────────────
if r"\graphicspath" not in tex:
    tex = tex.replace(
        r"\usepackage{graphicx}",
        r"\usepackage{graphicx}" + "\n" + r"\graphicspath{{./}{./results/}}"
    )
    print("[8] Added \\graphicspath")

# ─────────────────────────────────────────────────────────────────────────────
# Write output
# ─────────────────────────────────────────────────────────────────────────────
with open("thesis_tau.tex", "w", encoding="utf-8") as f:
    f.write(tex)

print(f"\nOutput: thesis_tau.tex ({len(tex):,} chars)")

# Verify
print("\nVerification:")
print(f"  documentclass book: {'book' in tex[:200]}")
print(f"  bidi=basic: {'bidi=basic' in tex}")
print(f"  tabular in title page: {'begin{tabular}' in tex[tex.find('titlepage'):tex.find('end{titlepage}')]}")
print(f"  frontmatter: {tex.count(chr(92)+'frontmatter')}")
print(f"  mainmatter: {tex.count(chr(92)+'mainmatter')}")