"""
to_latex.py
-----------
Converts thesis_restructured.md to thesis.tex using pandoc.

Pre-processing steps:
  1. Merge *Figure N: caption* text with preceding ![alt](path) image refs
  2. Convert <div dir="rtl">...</div> to LaTeX raw RTL block
  3. Strip {.unnumbered} from headings (handled by template)
  4. Remove \tag{N} from equations (LaTeX equation numbering via pandoc-crossref)
  5. Write preprocessed Markdown, then run pandoc
"""

import re
import subprocess
import sys

PANDOC = r"C:\Users\gzukerma\AppData\Local\Pandoc\pandoc.exe"
PANDOC_CROSSREF = r"C:\Users\gzukerma\AppData\Local\Pandoc\pandoc-crossref.exe"

# ─────────────────────────────────────────────────────────────────────────────
# Read source
# ─────────────────────────────────────────────────────────────────────────────
with open("thesis_restructured.md", "r", encoding="utf-8") as f:
    text = f.read()

print(f"Input: {len(text):,} chars")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Merge figure captions with image references
#    Pattern: ![alt](path)\n\n*Figure N: long caption*
#    → ![long caption](path){#fig:figN}
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] Merging figure captions with images ...")

def merge_fig_caption(text):
    # Pattern: image followed (within 3 blank lines) by *Figure N: caption*
    pattern = re.compile(
        r"(!\[[^\]]*\]\(([^\)]+)\))"   # image ref
        r"\s*\n\n"                       # blank line
        r"\*Figure (\d+): ([^\*]+)\*",  # caption
        re.MULTILINE
    )
    def replacer(m):
        path = m.group(2)
        fig_num = m.group(3)
        caption = m.group(4).strip()
        return f"![{caption}]({path}){{#fig:fig{fig_num}}}"

    new_text, count = pattern.subn(replacer, text)
    print(f"  Merged {count} figure captions")
    return new_text

text = merge_fig_caption(text)

# Also handle: *Figure N: caption*\n\n![alt](path)  (caption before image)
def merge_fig_caption_before(text):
    pattern = re.compile(
        r"\*Figure (\d+): ([^\*]+)\*"   # caption
        r"\s*\n\n"                       # blank line
        r"(!\[[^\]]*\]\(([^\)]+)\))",   # image ref
        re.MULTILINE
    )
    def replacer(m):
        fig_num = m.group(1)
        caption = m.group(2).strip()
        path = m.group(4)
        return f"![{caption}]({path}){{#fig:fig{fig_num}}}"

    new_text, count = pattern.subn(replacer, text)
    print(f"  Merged {count} caption-before-image figures")
    return new_text

text = merge_fig_caption_before(text)

# Remove any remaining standalone *Figure N: caption* lines
# (they've been merged or are duplicates)
remaining_caps = re.findall(r"\*Figure \d+:[^\*]+\*", text)
print(f"  Remaining standalone captions: {len(remaining_caps)}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Convert <div dir="rtl">...</div> to LaTeX raw block
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Converting Hebrew RTL div to LaTeX ...")

def convert_rtl_div(text):
    pattern = re.compile(r'<div dir="rtl">(.*?)</div>', re.DOTALL)
    def replacer(m):
        content = m.group(1).strip()
        # Convert to LaTeX raw block
        latex_block = (
            "\n```{=latex}\n"
            "\\begin{flushright}\n"
            "\\begin{otherlanguage}{hebrew}\n"
            f"{content}\n"
            "\\end{otherlanguage}\n"
            "\\end{flushright}\n"
            "```\n"
        )
        return latex_block
    new_text, count = pattern.subn(replacer, text)
    print(f"  Converted {count} RTL div blocks")
    return new_text

text = convert_rtl_div(text)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Remove \tag{N} from equations
#    (pandoc-crossref handles equation numbering in LaTeX)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Removing \\tag{N} from equations ...")
text, count = re.subn(r" \\tag\{\d+\}", "", text)
print(f"  Removed {count} \\tag{{N}} instances")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Fix section headings: remove manual numbers since LaTeX will number them
#    BUT keep the {.unnumbered} attribute
#    "## 1. Introduction..." → "## Introduction..."
#    "### 1.1 Cooperative..." → "### Cooperative..."
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] Removing manual section numbers from headings ...")

# Remove "N. " prefix from ## headings
text, c1 = re.subn(r"^(#{2,4} )\d+\.\d*\.?\d*\.? ", r"\1", text, flags=re.MULTILINE)
# Remove "N." prefix (chapter level)
text, c2 = re.subn(r"^(## )\d+\. ", r"\1", text, flags=re.MULTILINE)
print(f"  Removed {c1 + c2} manual section numbers")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Fix *Source: ...* equation citation lines
#    Convert to LaTeX small italic note
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] Converting equation source notes ...")
text, count = re.subn(
    r"\n\*Source: ([^\n]+)\*",
    r"\n\\footnotesize{\\textit{Source: \\1}}\\normalsize",
    text
)
print(f"  Converted {count} source notes")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Write preprocessed Markdown
# ─────────────────────────────────────────────────────────────────────────────
with open("thesis_preprocessed.md", "w", encoding="utf-8") as f:
    f.write(text)

print(f"\nPreprocessed: {len(text):,} chars → thesis_preprocessed.md")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Create YAML metadata
# ─────────────────────────────────────────────────────────────────────────────
yaml_meta = r"""---
title: "Deep Learning Architectures for Two-Hop Relay Communication: A Comparative Study of Classical and Neural Network Relay Strategies"
author: "Gil Zukerma"
date: "2026"
lang: en
documentclass: report
classoption:
  - 12pt
  - a4paper
  - oneside
geometry:
  - top=2.5cm
  - bottom=2.5cm
  - left=3cm
  - right=2.5cm
fontsize: 12pt
linestretch: 1.5
toc: true
toc-depth: 3
lof: true
lot: true
numbersections: true
secnumdepth: 3
colorlinks: true
linkcolor: blue
citecolor: blue
urlcolor: blue
header-includes:
  - \usepackage{amsmath}
  - \usepackage{amssymb}
  - \usepackage{booktabs}
  - \usepackage{longtable}
  - \usepackage{array}
  - \usepackage{multirow}
  - \usepackage{graphicx}
  - \usepackage{float}
  - \usepackage{caption}
  - \usepackage{subcaption}
  - \usepackage{hyperref}
  - \usepackage{cleveref}
  - \usepackage{polyglossia}
  - \setmainlanguage{english}
  - \setotherlanguage{hebrew}
  - \newfontfamily\hebrewfont[Script=Hebrew]{Arial}
  - \usepackage{geometry}
  - \usepackage{setspace}
  - \usepackage{fancyhdr}
  - \pagestyle{fancy}
  - \fancyhf{}
  - \fancyhead[L]{\leftmark}
  - \fancyhead[R]{\thepage}
  - \renewcommand{\headrulewidth}{0.4pt}
  - \usepackage{titlesec}
  - \usepackage{microtype}
  - \usepackage{xcolor}
  - \definecolor{darkblue}{RGB}{0,0,139}
  - \usepackage{listings}
  - \usepackage{algorithm}
  - \usepackage{algpseudocode}
crossrefYaml: crossref.yaml
---
"""

with open("thesis_meta.yaml", "w", encoding="utf-8") as f:
    f.write(yaml_meta)

print("Written thesis_meta.yaml")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Create pandoc-crossref config
# ─────────────────────────────────────────────────────────────────────────────
crossref_yaml = """figureTitle: "Figure"
tableTitle: "Table"
listingTitle: "Listing"
figPrefix: "Figure"
eqnPrefix: "Equation"
tblPrefix: "Table"
secPrefix: "Section"
autoSectionLabels: true
numberSections: true
chapters: true
chaptersDepth: 1
"""

with open("crossref.yaml", "w", encoding="utf-8") as f:
    f.write(crossref_yaml)

print("Written crossref.yaml")

# ─────────────────────────────────────────────────────────────────────────────
# 9. Run pandoc
# ─────────────────────────────────────────────────────────────────────────────
print("\n[9] Running pandoc ...")

cmd = [
    PANDOC,
    "thesis_preprocessed.md",
    "--from", "markdown+raw_tex+tex_math_dollars+fenced_code_blocks",
    "--to", "latex",
    "--standalone",
    "--filter", PANDOC_CROSSREF,
    "--metadata-file", "thesis_meta.yaml",
    "--output", "thesis.tex",
    "--wrap", "none",
    "--top-level-division=chapter",
    "--pdf-engine=xelatex",
    "-V", "mainfont=Times New Roman",
    "-V", "sansfont=Arial",
    "-V", "monofont=Courier New",
]

print(f"  Command: {' '.join(cmd[:6])} ...")
result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")

if result.returncode == 0:
    print("  pandoc succeeded!")
else:
    print(f"  pandoc FAILED (exit code {result.returncode})")
    if result.stderr:
        print(f"  STDERR:\n{result.stderr[:2000]}")
    if result.stdout:
        print(f"  STDOUT:\n{result.stdout[:500]}")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# 10. Verify output
# ─────────────────────────────────────────────────────────────────────────────
import os
if os.path.exists("thesis.tex"):
    size = os.path.getsize("thesis.tex")
    print(f"\nOutput: thesis.tex ({size:,} bytes)")
    with open("thesis.tex", "r", encoding="utf-8") as f:
        tex = f.read()
    print(f"  \\chapter: {tex.count(chr(92) + 'chapter')}")
    print(f"  \\section: {tex.count(chr(92) + 'section')}")
    print(f"  \\begin{{figure}}: {tex.count(chr(92) + 'begin{figure}')}")
    print(f"  \\begin{{table}}: {tex.count(chr(92) + 'begin{table}')}")
    print(f"  \\begin{{equation}}: {tex.count(chr(92) + 'begin{equation}')}")
    print(f"  \\label: {tex.count(chr(92) + 'label')}")
else:
    print("ERROR: thesis.tex not created!")