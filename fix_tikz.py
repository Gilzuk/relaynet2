"""Fix the TikZ diagrams in ch03_methods.tex - correct backslash handling."""
import re

with open('chapters/ch03_methods.tex', 'r', encoding='utf-8') as f:
    tex = f.read()

# Find and remove both TikZ figure blocks entirely
pattern = re.compile(
    r'\\begin\{figure\}\[H\]\s*\\centering\s*(?:\\resizebox[^\n]*\n)?\\begin\{tikzpicture\}.*?\\end\{figure\}',
    re.DOTALL
)
matches = list(pattern.finditer(tex))
print(f"Found {len(matches)} TikZ figure blocks to replace")

# Build correct LaTeX strings (no Python raw string tricks needed)
TIKZ1 = (
    "\\begin{figure}[H]\n"
    "\\centering\n"
    "\\begin{tikzpicture}[\n"
    "  node distance=0.5cm and 0.4cm,\n"
    "  box/.style={rectangle, draw, rounded corners=3pt, minimum width=1.7cm,\n"
    "              minimum height=0.85cm, align=center, font=\\small},\n"
    "  arr/.style={->, thick, >=stealth}\n"
    "]\n"
    "\\node[box] (src)   {Source\\\\tx bits};\n"
    "\\node[box, right=1.2cm of src]  (mod)   {BPSK\\\\Mod.};\n"
    "\\node[box, right=1.2cm of mod]  (ch1)   {Hop 1\\\\Channel};\n"
    "\\node[box, right=1.2cm of ch1]  (relay) {Relay\\\\Neural Net};\n"
    "\\node[box, right=1.2cm of relay](ch2)   {Hop 2\\\\Channel};\n"
    "\\node[box, right=1.2cm of ch2]  (dst)   {Dest.\\\\bits};\n"
    "\\draw[arr] (src) -- (mod);\n"
    "\\draw[arr] (mod) -- (ch1);\n"
    "\\draw[arr] (ch1) -- node[above,font=\\tiny]{Hop 1} (relay);\n"
    "\\draw[arr] (relay) -- node[above,font=\\tiny]{Hop 2} (ch2);\n"
    "\\draw[arr] (ch2) -- (dst);\n"
    "\\end{tikzpicture}\n"
    "\\caption{Two-hop relay system model: source transmits through two channels\n"
    "         with a neural network relay in between.}\n"
    "\\label{fig:relay-system-simple}\n"
    "\\end{figure}"
)

TIKZ2 = (
    "\\begin{figure}[H]\n"
    "\\centering\n"
    "\\resizebox{\\textwidth}{!}{%\n"
    "\\begin{tikzpicture}[\n"
    "  node distance=0.5cm and 0.3cm,\n"
    "  src/.style={ellipse, draw, fill=blue!20, minimum width=1.5cm,\n"
    "              minimum height=0.8cm, align=center, font=\\small},\n"
    "  box/.style={rectangle, draw, rounded corners=3pt, minimum width=1.7cm,\n"
    "              minimum height=0.85cm, align=center, font=\\small},\n"
    "  ch/.style={rectangle, draw, fill=orange!20, rounded corners=3pt,\n"
    "             minimum width=2.1cm, minimum height=0.85cm, align=center, font=\\small},\n"
    "  rel/.style={rectangle, draw, fill=green!20, rounded corners=3pt,\n"
    "              minimum width=1.7cm, minimum height=0.85cm, align=center, font=\\small},\n"
    "  eq/.style={rectangle, draw, fill=purple!20, rounded corners=3pt,\n"
    "             minimum width=1.7cm, minimum height=0.85cm, align=center, font=\\small},\n"
    "  arr/.style={->, thick, >=stealth}\n"
    "]\n"
    "\\node[src] (src)   {Source\\\\tx bits};\n"
    "\\node[box,  right=1.1cm of src]  (mod)   {BPSK\\\\Mod.};\n"
    "\\node[ch,   right=1.1cm of mod]  (ch1)   {Hop 1 Ch.\\\\AWGN/Ray./Ric.};\n"
    "\\node[rel,  right=1.1cm of ch1]  (relay) {Relay\\\\Neural Net};\n"
    "\\node[ch,   right=1.1cm of relay](ch2)   {Hop 2 Ch.\\\\$2{\\times}2$ MIMO};\n"
    "\\node[eq,   right=1.1cm of ch2]  (eq)    {Equalizer\\\\ZF/MMSE/SIC};\n"
    "\\node[src,  right=1.1cm of eq]   (dst)   {Dest.\\\\bits};\n"
    "\\draw[arr] (src)   -- node[above,font=\\tiny]{bits} (mod);\n"
    "\\draw[arr] (mod)   -- node[above,font=\\tiny]{$x$} (ch1);\n"
    "\\draw[arr] (ch1)   -- node[above,font=\\tiny]{$y_R$} (relay);\n"
    "\\draw[arr] (relay) -- node[above,font=\\tiny]{$\\hat{x}_R$} (ch2);\n"
    "\\draw[arr] (ch2)   -- node[above,font=\\tiny]{$\\mathbf{y}$} (eq);\n"
    "\\draw[arr] (eq)    -- node[above,font=\\tiny]{bits} (dst);\n"
    "\\end{tikzpicture}}\n"
    "\\caption{Detailed two-hop relay system model with MIMO Hop 2 channel and\n"
    "         equalizer. The relay neural network denoises the Hop 1 received signal\n"
    "         before retransmission.}\n"
    "\\label{fig:relay-system-detailed}\n"
    "\\end{figure}"
)

if len(matches) >= 2:
    tex = tex[:matches[1].start()] + TIKZ2 + tex[matches[1].end():]
    tex = tex[:matches[0].start()] + TIKZ1 + tex[matches[0].end():]
    print("Replaced both TikZ blocks")
elif len(matches) == 1:
    tex = tex[:matches[0].start()] + TIKZ1 + tex[matches[0].end():]
    print("Replaced 1 TikZ block")
else:
    print("No TikZ blocks found!")

with open('chapters/ch03_methods.tex', 'w', encoding='utf-8') as f:
    f.write(tex)
print("Saved ch03_methods.tex")

# Verify
with open('chapters/ch03_methods.tex', 'r', encoding='utf-8') as f:
    lines = f.readlines()
for i, l in enumerate(lines[84:106], start=85):
    print(f"{i:3d}: {l}", end='')