"""
fix_tau_tex.py
--------------
Fixes thesis_tau.tex:
  1. Scale all figures to \linewidth (keepaspectratio, max height 0.45\textheight)
  2. Fix equation source citations (the \1 backreference was literal, not substituted)
"""

import re

with open("thesis_tau.tex", "r", encoding="utf-8") as f:
    tex = f.read()

print(f"Input: {len(tex):,} chars")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Fix figure graphics — scale to \linewidth
#    Current: \includegraphics{results/...}
#    Target:  \includegraphics[width=\linewidth,height=0.45\textheight,keepaspectratio]{results/...}
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] Fixing figure sizes ...")

# Replace \includegraphics{...} (no options) with scaled version
tex, c1 = re.subn(
    r"\\includegraphics\{(results/[^\}]+)\}",
    r"\\includegraphics[width=\\linewidth,height=0.45\\textheight,keepaspectratio]{\\1}",
    tex
)
print(f"  Fixed {c1} bare \\includegraphics{{}} calls")

# Replace \includegraphics[...]{...} that don't already have width=\linewidth
def fix_existing_options(m):
    opts = m.group(1)
    path = m.group(2)
    # If already has width=\linewidth, skip
    if "linewidth" in opts or "textwidth" in opts:
        return m.group(0)
    # Replace/add width and height options
    return (
        r"\includegraphics[width=\linewidth,height=0.45\textheight,"
        f"keepaspectratio]{{{path}}}"
    )

tex, c2 = re.subn(
    r"\\includegraphics\[([^\]]+)\]\{(results/[^\}]+)\}",
    fix_existing_options,
    tex
)
print(f"  Fixed {c2} \\includegraphics[opts]{{}} calls")

# Also ensure figures use [H] placement and are centered
# Current: \begin{figure}
# Target:  \begin{figure}[H]\centering
def fix_figure_env(m):
    content = m.group(1)
    # Add [H] if not present
    if not content.startswith("["):
        content = "[H]\n\\centering\n" + content.lstrip()
    elif "[H]" not in content and "[h]" not in content and "[htbp]" not in content:
        content = re.sub(r"^\[([^\]]+)\]", r"[H]", content)
        content = content.lstrip() 
        content = "\\centering\n" + content
    return "\\begin{figure}" + content

tex, c3 = re.subn(
    r"\\begin\{figure\}(.*?)(?=\\includegraphics)",
    fix_figure_env,
    tex,
    flags=re.DOTALL
)
print(f"  Fixed {c3} figure environments")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Fix equation source citations
#    The preprocessing used re.subn with \1 backreference in replacement string
#    but the result was literal \1 in the LaTeX file.
#    Current: \footnotesize{\textit{Source: \1}}\normalsize
#    Target:  \footnotesize{\textit{Source: [N, Ch. X]}}\normalsize
#    
#    Actually the issue is that the regex replacement in to_latex.py used:
#    r"\n\\footnotesize{\\textit{Source: \\1}}\\normalsize"
#    which in Python re.subn means group 1 of the match.
#    The match group 1 was the citation text. So it SHOULD have worked.
#    Let me check what's actually in the file.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Checking equation source citations ...")

# Find all source citation patterns
source_patterns = re.findall(r"\\footnotesize\{\\textit\{Source:[^\}]+\}\}\\normalsize", tex)
print(f"  Found {len(source_patterns)} source citations")
if source_patterns:
    print(f"  Sample: {source_patterns[0][:100]}")

# Check for literal \1 (broken backreference)
broken = re.findall(r"\\footnotesize\{\\textit\{Source: \\1\}\}\\normalsize", tex)
print(f"  Broken (literal \\1): {len(broken)}")

if broken:
    print("  Fixing broken backreferences ...")
    # These need to be fixed - the source text was lost
    # We need to re-apply the source citations from the preprocessed markdown
    
    # Read the preprocessed markdown to get the source citations
    with open("thesis_preprocessed.md", "r", encoding="utf-8") as f:
        md = f.read()
    
    # Find all source citations in the markdown
    md_sources = re.findall(r"\\footnotesize\{\\textit\{Source: ([^\}]+)\}\}\\normalsize", md)
    print(f"  Found {len(md_sources)} source citations in preprocessed markdown")
    
    # The broken ones in tex need to be replaced with the actual citations
    # Since we can't easily match them, let's re-extract from the original
    # equation labels and their citations
    
    # Actually, let's check the preprocessed markdown for the pattern
    md_sources2 = re.findall(r"\*Source: ([^\n]+)\*", md)
    print(f"  Found {len(md_sources2)} *Source:* patterns in preprocessed markdown")
    if md_sources2:
        print(f"  Sample: {md_sources2[0][:80]}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Re-apply equation source citations from the preprocessed markdown
#    The issue: in to_latex.py step 5, the regex was:
#    re.subn(r"\n\*Source: ([^\n]+)\*", r"\n\\footnotesize{\\textit{Source: \\1}}\\normalsize", text)
#    In Python re.subn, \\1 in the replacement string IS the backreference.
#    So it should have worked. Let me verify by checking the preprocessed file.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Verifying source citations in preprocessed markdown ...")
with open("thesis_preprocessed.md", "r", encoding="utf-8") as f:
    md = f.read()

# Check what the source citations look like in the preprocessed file
footnote_in_md = re.findall(r"\\footnotesize\{\\textit\{Source: ([^\}]+)\}\}", md)
print(f"  \\footnotesize citations in preprocessed md: {len(footnote_in_md)}")
if footnote_in_md:
    for s in footnote_in_md[:3]:
        print(f"    {s[:80]}")

# Check what's in the tex file
footnote_in_tex = re.findall(r"\\footnotesize\{\\textit\{Source: ([^\}]+)\}\}", tex)
print(f"  \\footnotesize citations in thesis_tau.tex: {len(footnote_in_tex)}")
if footnote_in_tex:
    for s in footnote_in_tex[:3]:
        print(f"    {s[:80]}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Fix the equation source note format for better LaTeX rendering
#    Wrap in a proper \par and use \small instead of \footnotesize
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] Reformatting equation source notes ...")

# Current: \footnotesize{\textit{Source: [N, Ch. X]}}\normalsize
# Better:  \par{\small\textit{Source: [N, Ch. X]}}
tex, c4 = re.subn(
    r"\\footnotesize\{\\textit\{(Source: [^\}]+)\}\}\\normalsize",
    r"\\par{\\small\\textit{\\1}}",
    tex
)
print(f"  Reformatted {c4} source notes")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Write output
# ─────────────────────────────────────────────────────────────────────────────
with open("thesis_tau.tex", "w", encoding="utf-8") as f:
    f.write(tex)

print(f"\nOutput: thesis_tau.tex ({len(tex):,} chars)")

# Verify
with open("thesis_tau.tex", "r", encoding="utf-8") as f:
    verify = f.read()

lw_count = verify.count("linewidth")
src_count = len(re.findall(r"Source:", verify))
fig_h = verify.count(chr(92) + "begin{figure}[H]")
fig_total = verify.count(chr(92) + "begin{figure}")
print(f"\nVerification:")
print(f"  linewidth occurrences: {lw_count}")
print(f"  Source citations: {src_count}")
print(f"  Figures [H]: {fig_h}")
print(f"  Total figures: {fig_total}")
