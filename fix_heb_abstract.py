"""
fix_heb_abstract.py
-------------------
Fixes the Hebrew abstract section structure:
- Correct nesting of flushright and otherlanguage
- Fix remaining markdown bold
"""

import re

with open("thesis_tau.tex", "r", encoding="utf-8") as f:
    tex = f.read()

print(f"Input: {len(tex):,} chars")

# Find the Hebrew abstract section (from \backmatter to \end{document})
backmatter_idx = tex.find(r"\backmatter")
end_doc_idx = tex.find(r"\end{document}")

if backmatter_idx >= 0 and end_doc_idx >= 0:
    before = tex[:backmatter_idx]
    heb_section = tex[backmatter_idx:end_doc_idx]
    after = tex[end_doc_idx:]
    
    print(f"Hebrew section: {len(heb_section)} chars")
    print("Current structure:")
    for i, l in enumerate(heb_section.split("\n")[:30]):
        print(f"  {i+1}: {l[:100]}")
    
    # Extract the Hebrew abstract content (between the flushright blocks)
    # Get the actual Hebrew text content
    content_match = re.search(
        r"\\begin\{flushright\}\s*\\label\{sec:abstract-hebrew\}(.*?)\\end\{flushright\}",
        heb_section,
        re.DOTALL
    )
    
    if content_match:
        heb_content = content_match.group(1).strip()
        # Fix markdown bold **text** → \textbf{text}
        heb_content, c = re.subn(r"\*\*([^*\n]+)\*\*", r"\\textbf{\1}", heb_content)
        print(f"\nFixed {c} bold patterns in content")
        
        # Build correct structure
        new_heb_section = r"""\backmatter
\clearpage
\chapter*{\texthebrew{תקציר}}
\addcontentsline{toc}{chapter}{Abstract (Hebrew)}
\markboth{Abstract (Hebrew)}{Abstract (Hebrew)}

\begin{otherlanguage}{hebrew}
\begin{flushright}
\label{sec:abstract-hebrew}

""" + heb_content + r"""

\end{flushright}
\end{otherlanguage}

"""
        tex = before + new_heb_section + after
        print("\nRebuilt Hebrew abstract section with correct nesting")
    else:
        print("Could not find Hebrew content block - trying alternative approach")
        # Just fix the nesting manually
        # The issue: \begin{flushright} contains \begin{otherlanguage}
        # Fix: move \begin{otherlanguage} before \begin{flushright}
        heb_section_fixed = heb_section
        
        # Remove the outer flushright wrapper
        heb_section_fixed = re.sub(
            r"\\begin\{flushright\}\s*\n\\label\{sec:abstract-hebrew\}\s*\n\s*\n\s*\n\\begin\{flushright\}\s*\n\\begin\{otherlanguage\}\{hebrew\}",
            r"\\begin{otherlanguage}{hebrew}\n\\begin{flushright}\n\\label{sec:abstract-hebrew}\n",
            heb_section_fixed
        )
        # Fix end order: \end{flushright}\n\end{flushright}\n\end{otherlanguage}
        # → \end{flushright}\n\end{otherlanguage}
        heb_section_fixed = re.sub(
            r"\\end\{flushright\}\s*\n\\end\{flushright\}\s*\n\\end\{otherlanguage\}",
            r"\\end{flushright}\n\\end{otherlanguage}",
            heb_section_fixed
        )
        # Fix markdown bold
        heb_section_fixed, c = re.subn(r"\*\*([^*\n]+)\*\*", r"\\textbf{\1}", heb_section_fixed)
        print(f"Fixed {c} bold patterns")
        
        tex = before + heb_section_fixed + after

with open("thesis_tau.tex", "w", encoding="utf-8") as f:
    f.write(tex)

print(f"\nOutput: {len(tex):,} chars")

# Verify structure
with open("thesis_tau.tex", "r", encoding="utf-8") as f:
    lines = f.readlines()
print("\nLines 4020-4050:")
for i, l in enumerate(lines[4019:4055], start=4020):
    print(f"{i:5d}: {l}", end="")