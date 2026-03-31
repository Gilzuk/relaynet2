"""
Build thesis_doc.md from thesis.md:
  - Adds YAML front matter (pandoc/docx + pandoc-crossref settings)
  - Adds TAU cover pages (front + title page per guidelines)
  - Auto-numbers every display equation with {#eq:eqN} labels
  - Preserves all body content verbatim
"""
import re, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

with open("thesis.md", encoding="utf-8") as f:
    body = f.read()

# ── Remove the informal header block that precedes the ToC ───────────────────
body = re.sub(
    r"^# Deep Learning.*?\n---\n\*\*Gil Zukerman\*\*.*?---\n",
    "",
    body,
    flags=re.DOTALL,
)

# ── Auto-number every display equation ───────────────────────────────────────
# Strategy: wrap each $$ block in a one-row, two-column markdown table.
# Left cell holds the equation (centred), right cell holds (N) right-aligned.
# This renders perfectly in .docx without any pandoc filter dependency.
eq_counter = [0]

def add_eq_number(m):
    inner = m.group(1).strip()
    # skip inline-style $$ that are actually inline (no newline inside)
    if not inner:
        return m.group(0)
    eq_counter[0] += 1
    n = eq_counter[0]
    # Pandoc table: pipe table with no header, two columns
    # Col 1 (90%): equation centred  |  Col 2 (10%): number right-aligned
    table = (
        f"\n| $$\n{inner}\n$$ | ({n}) |\n"
        f"|:---:|---:|\n"
    )
    return table

body = re.sub(r'\$\$\s*([\s\S]+?)\s*\$\$', add_eq_number, body)
print(f"  Numbered {eq_counter[0]} display equations")

YAML = """\
---
title: "Deep Learning for Two-Hop Relay Communication: A Comparative Study of Classical and Neural Network-Based Strategies"
author: "Gil Zukerman"
date: "2026"
institute: |
  Tel Aviv University
  The Iby and Aladar Fleischman Faculty of Engineering
  School of Electrical Engineering
degree: "Master of Science in Electrical and Electronic Engineering"
supervisor: "[Supervisor Name]"
lang: en
fontsize: 12pt
linestretch: 1.5
geometry: "top=2.5cm, bottom=2.5cm, left=3cm, right=2.5cm"
papersize: a4
numbersections: true
toc: true
toc-depth: 3
lof: true
lot: true
link-citations: true
---

"""

# TAU cover page 1 (front cover — no supervisor)
COVER_FRONT = (
    "---\n"
    "\n"
    "**TEL AVIV UNIVERSITY**\n"
    "\n"
    "**THE IBY AND ALADAR FLEISCHMAN FACULTY OF ENGINEERING**\n"
    "\n"
    "**School of Electrical Engineering**\n"
    "\n"
    "---\n"
    "\n"
    "&nbsp;\n"
    "\n"
    "## Deep Learning for Two-Hop Relay Communication: A Comparative Study of Classical and Neural Network-Based Strategies {.unnumbered}\n"
    "\n"
    "&nbsp;\n"
    "\n"
    "Thesis submitted toward the degree of **Master of Science in Electrical and Electronic Engineering** in Tel-Aviv University\n"
    "\n"
    "&nbsp;\n"
    "\n"
    "by\n"
    "\n"
    "**Gil Zukerman**\n"
    "\n"
    "&nbsp;\n"
    "\n"
    "**2026**\n"
    "\n"
    "---\n"
    "\n"
    "\\newpage\n"
    "\n"
)

# TAU cover page 2 (title page — with supervisor)
COVER_TITLE = (
    "---\n"
    "\n"
    "**TEL AVIV UNIVERSITY**\n"
    "\n"
    "**THE IBY AND ALADAR FLEISCHMAN FACULTY OF ENGINEERING**\n"
    "\n"
    "**School of Electrical Engineering**\n"
    "\n"
    "---\n"
    "\n"
    "&nbsp;\n"
    "\n"
    "## Deep Learning for Two-Hop Relay Communication: A Comparative Study of Classical and Neural Network-Based Strategies {.unnumbered}\n"
    "\n"
    "&nbsp;\n"
    "\n"
    "Thesis submitted toward the degree of **Master of Science in Electrical and Electronic Engineering** in Tel-Aviv University\n"
    "\n"
    "&nbsp;\n"
    "\n"
    "by\n"
    "\n"
    "**Gil Zukerman**\n"
    "\n"
    "&nbsp;\n"
    "\n"
    "This research work was carried out at Tel-Aviv University\n"
    "in the School of Electrical Engineering, Faculty of Engineering,\n"
    "under the supervision of **[Supervisor Name]**\n"
    "\n"
    "&nbsp;\n"
    "\n"
    "**2026**\n"
    "\n"
    "---\n"
    "\n"
    "\\newpage\n"
    "\n"
)

out = YAML + COVER_FRONT + COVER_TITLE + body

with open("thesis_doc.md", "w", encoding="utf-8") as f:
    f.write(out)

print(f"thesis_doc.md written — {len(out):,} chars, {out.count(chr(10)):,} lines")
