"""
fix_ch00.py
-----------
Removes duplicate \tableofcontents, \listoffigures, \listoftables
from ch00_frontmatter.tex (they are already in the main thesis_tau.tex).
"""

import re

with open("chapters/ch00_frontmatter.tex", "r", encoding="utf-8") as f:
    content = f.read()

print(f"Input: {len(content):,} chars")

# Remove the TOC/LOF/LOT block from ch00_frontmatter.tex
# These are already in the main thesis_tau.tex
toc_block = r"""\clearpage

%% ── Table of Contents ────────────────────────────────────────
\tableofcontents
\clearpage

%% ── List of Figures ──────────────────────────────────────────
\cleardoublepage
\phantomsection
\addcontentsline{toc}{chapter}{\listfigurename}
\listoffigures
\clearpage

%% ── List of Tables ───────────────────────────────────────────
\cleardoublepage
\phantomsection
\addcontentsline{toc}{chapter}{\listtablename}
\listoftables
\clearpage

%% ── Main Body ────────────────────────────────────────────────"""

if toc_block in content:
    content = content.replace(toc_block, "")
    print("[1] Removed duplicate TOC/LOF/LOT block")
else:
    # Try removing just the TOC/LOF/LOT commands individually
    content = re.sub(
        r"\n%% ── Table of Contents.*?%% ── Main Body ────────────────────────────────────────────────",
        "",
        content,
        flags=re.DOTALL
    )
    print("[1] Removed TOC/LOF/LOT block (regex)")

with open("chapters/ch00_frontmatter.tex", "w", encoding="utf-8") as f:
    f.write(content)

print(f"Output: {len(content):,} chars")
print("\nEnd of ch00_frontmatter.tex:")
print(content[-400:])