"""
fix_crossrefs.py
----------------
Fixes cross-references and numbering in thesis_tau.tex:
  1. Convert hardcoded "Figure X" → Figure~\ref{fig:figX}
  2. Add \counterwithin{equation}{chapter} for chapter-based eq numbering
  3. Add \counterwithin{figure}{chapter} and \counterwithin{table}{chapter}
     (book class should do this automatically, but make it explicit)
"""

import re

with open("thesis_tau.tex", "r", encoding="utf-8") as f:
    tex = f.read()

print(f"Input: {len(tex):,} chars")

# ── 1. Fix hardcoded Figure X references ──────────────────────────────────────
print("\n[1] Fixing hardcoded Figure X references ...")

# Map of specific replacements (from context analysis)
# Only replace in body text, not in captions or source citations
replacements = [
    # Figure 4 shows the PDF
    (
        r"Figure 4 shows the probability density function",
        r"Figure~\\ref{fig:fig4} shows the probability density function"
    ),
    # Figure 8c (16-QAM constellation)
    (
        r"\(Figure 8c\)",
        r"(Figure~\\ref{fig:fig8}c)"
    ),
    # Figure 8 presents the constellation diagrams
    (
        r"Figure 8 presents the constellation diagrams",
        r"Figure~\\ref{fig:fig8} presents the constellation diagrams"
    ),
    # Figure 8d (16-PSK constellation)
    (
        r"\(Figure 8d\)",
        r"(Figure~\\ref{fig:fig8}d)"
    ),
]

total_fixes = 0
for old, new in replacements:
    tex, c = re.subn(old, new, tex)
    if c > 0:
        print(f"  Fixed {c}: {old[:60]}")
        total_fixes += c

print(f"  Total figure ref fixes: {total_fixes}")

# ── 2. Add chapter-based equation numbering ───────────────────────────────────
print("\n[2] Adding chapter-based equation numbering ...")

# Check if already set
if "counterwithin{equation}" not in tex and "numberwithin{equation}" not in tex:
    # Add after \usepackage{amsmath} or before \begin{document}
    numbering_cmd = r"""
%% ── Chapter-based numbering ─────────────────────────────────────────────────
\counterwithin{figure}{chapter}
\counterwithin{table}{chapter}
\counterwithin{equation}{chapter}
"""
    # Insert before \begin{document}
    tex = tex.replace(r"\begin{document}", numbering_cmd + "\n" + r"\begin{document}", 1)
    print("  Added \\counterwithin for figure, table, equation")
else:
    print("  Chapter-based numbering already set")

# ── 3. Ensure chngcntr package is loaded (needed for \counterwithin) ──────────
print("\n[3] Checking chngcntr package ...")
if "chngcntr" not in tex:
    # Add after \usepackage{amsmath}
    tex = tex.replace(
        r"\usepackage{amsmath}",
        r"\usepackage{amsmath}" + "\n" + r"\usepackage{chngcntr}"
    )
    print("  Added \\usepackage{chngcntr}")
else:
    print("  chngcntr already loaded")

# ── 4. Fix equation numbering format: (1.1) not (1) ──────────────────────────
# This is handled automatically by \counterwithin{equation}{chapter}

# ── 5. Write output ───────────────────────────────────────────────────────────
with open("thesis_tau.tex", "w", encoding="utf-8") as f:
    f.write(tex)

print(f"\nOutput: {len(tex):,} chars")
print(f"counterwithin equation: {'counterwithin{equation}' in tex}")
print(f"counterwithin figure: {'counterwithin{figure}' in tex}")
print(f"counterwithin table: {'counterwithin{table}' in tex}")