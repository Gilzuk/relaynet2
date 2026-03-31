"""
fix_shaded.py
-------------
Adds missing pandoc syntax-highlighting environments (Shaded, Highlighting)
and fixes headheight warning in thesis_tau.tex.
"""

import re

with open("thesis_tau.tex", "r", encoding="utf-8") as f:
    tex = f.read()

print(f"Input: {len(tex):,} chars")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Add Shaded / Highlighting / VerbatimEnvironment definitions
#    (pandoc syntax highlighting support)
# ─────────────────────────────────────────────────────────────────────────────
shaded_defs = r"""
%% ── Pandoc syntax highlighting ──────────────────────────────
\usepackage{fancyvrb}
\usepackage{framed}
\newcommand{\VerbBar}{|}
\newcommand{\VERB}{\Verb[commandchars=\\\{\}]}
\DefineVerbatimEnvironment{Highlighting}{Verbatim}{commandchars=\\\{\}}
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

# Insert before \begin{document}
if "Shaded" not in tex:
    tex = tex.replace(
        r"\begin{document}",
        shaded_defs + "\n" + r"\begin{document}"
    )
    print("[1] Added Shaded/Highlighting environments")
else:
    print("[1] Shaded already defined")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Fix headheight (increase from 15pt to 25pt)
# ─────────────────────────────────────────────────────────────────────────────
tex = tex.replace(
    "headheight=15pt",
    "headheight=25pt"
)
print("[2] Fixed headheight to 25pt")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Fix \par{\small\textit{Source: ...}} inside equation environments
#    The \par command is invalid inside math mode.
#    Change to use \text{} or move outside equation.
#    Actually these should be AFTER the equation, not inside it.
#    Let's check if they're inside equation environments.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Checking Source citations placement ...")
# Find Source citations that are inside equation environments
eq_blocks = re.findall(r"\\begin\{equation\}.*?\\end\{equation\}", tex, re.DOTALL)
source_in_eq = sum(1 for b in eq_blocks if "Source:" in b)
print(f"  Source citations inside equation blocks: {source_in_eq}")

# Find Source citations that are inside align environments
align_blocks = re.findall(r"\\begin\{align\*?\}.*?\\end\{align\*?\}", tex, re.DOTALL)
source_in_align = sum(1 for b in align_blocks if "Source:" in b)
print(f"  Source citations inside align blocks: {source_in_align}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Fix undefined control sequences from \par in math mode
#    Move \par{\small\textit{Source:...}} AFTER the equation environment
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] Moving Source citations outside equation environments ...")

def move_source_out_of_eq(tex):
    # Pattern: equation environment containing \par{\small\textit{Source:...}}
    pattern = re.compile(
        r"(\\begin\{equation\}[^}]*?)"
        r"(\\par\{\\small\\textit\{Source: [^\}]+\}\})"
        r"(.*?\\end\{equation\})",
        re.DOTALL
    )
    def replacer(m):
        eq_start = m.group(1)
        source = m.group(2)
        eq_end = m.group(3)
        return eq_start + eq_end + "\n" + source
    
    new_tex, count = pattern.subn(replacer, tex)
    print(f"  Moved {count} Source citations out of equation environments")
    return new_tex

tex = move_source_out_of_eq(tex)

# Also handle align environments
def move_source_out_of_align(tex):
    pattern = re.compile(
        r"(\\begin\{align\*?\}.*?)"
        r"(\\par\{\\small\\textit\{Source: [^\}]+\}\})"
        r"(.*?\\end\{align\*?\})",
        re.DOTALL
    )
    def replacer(m):
        return m.group(1) + m.group(3) + "\n" + m.group(2)
    
    new_tex, count = pattern.subn(replacer, tex)
    print(f"  Moved {count} Source citations out of align environments")
    return new_tex

tex = move_source_out_of_align(tex)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Write output
# ─────────────────────────────────────────────────────────────────────────────
with open("thesis_tau.tex", "w", encoding="utf-8") as f:
    f.write(tex)

print(f"\nOutput: thesis_tau.tex ({len(tex):,} chars)")
print("Shaded defined:", "Shaded" in tex)
print("headheight=25pt:", "headheight=25pt" in tex)