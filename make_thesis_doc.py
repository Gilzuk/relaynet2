"""
Build thesis_doc.md from thesis.md:
  - Adds YAML front matter (pandoc/docx settings)
  - Adds TAU cover pages (front + title page per guidelines)
  - Preserves all body content verbatim
"""
import re, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

with open("thesis.md", encoding="utf-8") as f:
    body = f.read()

# Remove the informal header block that precedes the ToC
body = re.sub(
    r"^# Deep Learning.*?\n---\n\*\*Gil Zukerman\*\*.*?---\n",
    "",
    body,
    flags=re.DOTALL,
)

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
COVER_FRONT = """\
---

**TEL AVIV UNIVERSITY**

**THE IBY AND ALADAR FLEISCHMAN FACULTY OF ENGINEERING**

**School of Electrical Engineering**

---

&nbsp;

&nbsp;

## Deep Learning for Two-Hop Relay Communication: A Comparative Study of Classical and Neural Network-Based Strategies

&nbsp;

Thesis submitted toward the degree of **Master of Science in Electrical and Electronic Engineering** in Tel-Aviv University

&nbsp;

by

&nbsp;

**Gil Zukerman**

&nbsp;

&nbsp;

**2026**

---

\\newpage

"""

# TAU cover page 2 (title page — with supervisor)
COVER_TITLE = """\
---

**TEL AVIV UNIVERSITY**

**THE IBY AND ALADAR FLEISCHMAN FACULTY OF ENGINEERING**

**School of Electrical Engineering**

---

&nbsp;

&nbsp;

## Deep Learning for Two-Hop Relay Communication: A Comparative Study of Classical and Neural Network-Based Strategies

&nbsp;

Thesis submitted toward the degree of **Master of Science in Electrical and Electronic Engineering** in Tel-Aviv University

&nbsp;

by

&nbsp;

**Gil Zukerman**

&nbsp;

This research work was carried out at Tel-Aviv University
in the School of Electrical Engineering, Faculty of Engineering,
under the supervision of **[Supervisor Name]**

&nbsp;

**2026**

---

\\newpage

"""

out = YAML + COVER_FRONT + COVER_TITLE + body

with open("thesis_doc.md", "w", encoding="utf-8") as f:
    f.write(out)

print(f"thesis_doc.md written — {len(out):,} chars, {out.count(chr(10)):,} lines")
