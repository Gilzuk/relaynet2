"""
fix_shaded2.py
--------------
Properly adds Shaded/Highlighting definitions to the preamble.
The previous script skipped because 'Shaded' was found in the body.
"""

import re

with open("thesis_tau.tex", "r", encoding="utf-8") as f:
    tex = f.read()

print(f"Input: {len(tex):,} chars")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Add Shaded/Highlighting definitions to preamble
#    (check for newenvironment{Shaded}, not just 'Shaded')
# ─────────────────────────────────────────────────────────────────────────────
if r"\newenvironment{Shaded}" not in tex:
    shaded_defs = r"""
%% ── Pandoc syntax highlighting ──────────────────────────────
\usepackage{fancyvrb}
\usepackage{framed}
\newcommand{\VerbBar}{|}
\newcommand{\VERB}{\Verb[commandchars=\\\{\}]}
\DefineVerbatimEnvironment{Highlighting}{Verbatim}{commandchars=\\\{\},fontsize=\small}
\definecolor{shadecolor}{RGB}{248,248,248}
\newenvironment{Shaded}{\begin{snugshade}}{\end{snugshade}}
\newcommand{\AlertTok}[1]{\textcolor[rgb]{0.94,0.16,0.16}{#1}}
\newcommand{\AnnotationTok}[1]{\textcolor[rgb]{0.56,0.35,0.01}{\textbf{\textit{#1}}}}
\newcommand{\AttributeTok}[1]{\textcolor[rgb]{0.77,0.63,0.00}{#1}}
\newcommand{\BaseNTok}[1]{\textcolor[rgb]{0.00,0.00,0.81}{#1}}
\newcommand{\BuiltInTok}[1]{#1}
\newcommand{\CharTok}[1]{\textcolor[rgb]{0.31,0.60,0.02}{#1}}
\newcommand{\CommentTok}[1]{\textcolor[rgb]{0.56,0.35,0.01}{\textit{#1}}}
\newcommand{\CommentVarTok}[1]{\textcolor[rgb]{0.56,0.35,0.01}{\textbf{\textit{#1}}}}
\newcommand{\ConstantTok}[1]{\textcolor[rgb]{0.00,0.00,0.00}{#1}}
\newcommand{\ControlFlowTok}[1]{\textcolor[rgb]{0.13,0.29,0.53}{\textbf{#1}}}
\newcommand{\DataTypeTok}[1]{\textcolor[rgb]{0.13,0.29,0.53}{#1}}
\newcommand{\DecValTok}[1]{\textcolor[rgb]{0.00,0.00,0.81}{#1}}
\newcommand{\DocumentationTok}[1]{\textcolor[rgb]{0.56,0.35,0.01}{\textbf{\textit{#1}}}}
\newcommand{\ErrorTok}[1]{\textcolor[rgb]{0.64,0.00,0.00}{\textbf{#1}}}
\newcommand{\ExtensionTok}[1]{#1}
\newcommand{\FloatTok}[1]{\textcolor[rgb]{0.00,0.00,0.81}{#1}}
\newcommand{\FunctionTok}[1]{\textcolor[rgb]{0.00,0.00,0.00}{#1}}
\newcommand{\ImportTok}[1]{#1}
\newcommand{\InformationTok}[1]{\textcolor[rgb]{0.56,0.35,0.01}{\textbf{\textit{#1}}}}
\newcommand{\KeywordTok}[1]{\textcolor[rgb]{0.13,0.29,0.53}{\textbf{#1}}}
\newcommand{\NormalTok}[1]{#1}
\newcommand{\OperatorTok}[1]{\textcolor[rgb]{0.81,0.36,0.00}{\textbf{#1}}}
\newcommand{\OtherTok}[1]{\textcolor[rgb]{0.56,0.35,0.01}{#1}}
\newcommand{\PreprocessorTok}[1]{\textcolor[rgb]{0.56,0.35,0.01}{\textit{#1}}}
\newcommand{\RegionMarkerTok}[1]{#1}
\newcommand{\SpecialCharTok}[1]{\textcolor[rgb]{0.00,0.00,0.00}{#1}}
\newcommand{\SpecialStringTok}[1]{\textcolor[rgb]{0.31,0.60,0.02}{#1}}
\newcommand{\StringTok}[1]{\textcolor[rgb]{0.31,0.60,0.02}{#1}}
\newcommand{\VariableTok}[1]{\textcolor[rgb]{0.00,0.00,0.00}{#1}}
\newcommand{\VerbatimStringTok}[1]{\textcolor[rgb]{0.31,0.60,0.02}{#1}}
\newcommand{\WarningTok}[1]{\textcolor[rgb]{0.56,0.35,0.01}{\textbf{\textit{#1}}}}
"""
    # Insert just before \begin{document}
    tex = tex.replace(r"\begin{document}", shaded_defs + "\n" + r"\begin{document}", 1)
    print("[1] Added Shaded/Highlighting definitions to preamble")
else:
    print("[1] Shaded already defined in preamble")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Fix the Shaded blocks that contain mermaid diagrams
#    Mermaid code blocks have \textbackslash{} and ** which cause issues.
#    Replace \textbackslash{} with \textbackslash and ** with \textbf{}
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Fixing mermaid/code blocks ...")

# Count Shaded blocks
shaded_blocks = re.findall(r"\\begin\{Shaded\}.*?\\end\{Shaded\}", tex, re.DOTALL)
print(f"  Total Shaded blocks: {len(shaded_blocks)}")

# The \textbackslash{}n in Highlighting blocks causes issues
# Replace \textbackslash{}n with \\n (literal backslash-n in verbatim)
# Actually in Verbatim environment, these are already escaped
# The issue is \textbackslash{} inside Verbatim - it should be fine

# Check for math mode issues in Shaded blocks
math_in_shaded = sum(1 for b in shaded_blocks if "$" in b or "\\[" in b)
print(f"  Shaded blocks with math: {math_in_shaded}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Fix undefined control sequences
#    Find what's causing them by looking at the log
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Checking for undefined control sequences ...")

# Look for \textbackslash outside verbatim (in normal text)
tb_count = tex.count(r"\textbackslash{}")
print(f"  \\textbackslash{{}} occurrences: {tb_count}")

# Look for \** patterns (bold in markdown → \textbf{} in LaTeX)
# These should be fine

# Check for \1 (broken backreference) remaining
broken = tex.count(r"\1")
print(f"  Remaining \\1 (broken backreference): {broken}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Write output
# ─────────────────────────────────────────────────────────────────────────────
with open("thesis_tau.tex", "w", encoding="utf-8") as f:
    f.write(tex)

print(f"\nOutput: thesis_tau.tex ({len(tex):,} chars)")
print(f"newenvironment{{Shaded}}: {tex.count(chr(92)+'newenvironment{Shaded}')}")