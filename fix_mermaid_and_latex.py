"""
fix_mermaid_and_latex.py
------------------------
1. Replace corrupted Shaded/Highlighting mermaid blocks in ch03_methods.tex
   with proper TikZ flowchart diagrams.
2. Update ch04_experiments.tex to reference the new split figures.
3. Use PIL to split bpsk_activation figures (no JSON available).
"""

import re, os

# ── 1. Split bpsk_activation PNGs using PIL ───────────────────────────────────
print("[BPSK] Splitting bpsk_activation PNGs into 3 panels ...")
try:
    from PIL import Image
    for tag in ['awgn', 'rayleigh']:
        src = f'results/activation_comparison/bpsk_activation_{tag}.png'
        if not os.path.exists(src):
            print(f"  {src} not found, skipping")
            continue
        img = Image.open(src)
        w, h = img.size
        # Split into 3 equal horizontal panels
        panel_w = w // 3
        for i in range(1, 4):
            x0 = (i-1) * panel_w
            x1 = i * panel_w if i < 3 else w
            panel = img.crop((x0, 0, x1, h))
            out = f'results/activation_comparison/bpsk_activation_{tag}_split_{i}.png'
            panel.save(out, dpi=(150, 150))
            print(f"  Saved {out}")
except ImportError:
    print("  PIL not available, skipping bpsk split")

# ── 2. Fix ch03_methods.tex: replace Shaded blocks with TikZ ─────────────────
print("\n[ch03] Fixing mermaid Shaded blocks ...")

TIKZ_DIAGRAM_1 = r"""
\begin{figure}[H]
\centering
\begin{tikzpicture}[
  node distance=1.4cm and 0.6cm,
  box/.style={rectangle, draw, rounded corners=3pt, minimum width=1.8cm,
              minimum height=0.9cm, align=center, font=\small},
  arr/.style={->, thick, >=stealth},
  every node/.style={font=\small}
]
\node[box] (src)   {Source\\tx bits};
\node[box, right=of src]  (mod)   {BPSK\\Modulator};
\node[box, right=of mod]  (ch1)   {Hop 1\\Channel};
\node[box, right=of ch1]  (relay) {Relay\\Neural Net};
\node[box, right=of relay](ch2)   {Hop 2\\Channel};
\node[box, right=of ch2]  (dst)   {Destination\\bits};

\draw[arr] (src)   -- (mod);
\draw[arr] (mod)   -- (ch1);
\draw[arr] (ch1)   -- node[above,font=\tiny]{Hop 1} (relay);
\draw[arr] (relay) -- node[above,font=\tiny]{Hop 2} (ch2);
\draw[arr] (ch2)   -- (dst);
\end{tikzpicture}
\caption{Two-hop relay system model: source transmits through two channels
         with a neural network relay in between.}
\label{fig:relay-system-simple}
\end{figure}
"""

TIKZ_DIAGRAM_2 = r"""
\begin{figure}[H]
\centering
\begin{tikzpicture}[
  node distance=1.2cm and 0.5cm,
  src/.style={ellipse, draw, fill=blue!20, minimum width=1.6cm,
              minimum height=0.8cm, align=center, font=\small},
  box/.style={rectangle, draw, rounded corners=3pt, minimum width=1.8cm,
              minimum height=0.9cm, align=center, font=\small},
  ch/.style={rectangle, draw, fill=orange!20, rounded corners=3pt,
             minimum width=2.0cm, minimum height=0.9cm, align=center, font=\small},
  rel/.style={rectangle, draw, fill=green!20, rounded corners=3pt,
              minimum width=1.8cm, minimum height=0.9cm, align=center, font=\small},
  eq/.style={rectangle, draw, fill=purple!20, rounded corners=3pt,
             minimum width=1.8cm, minimum height=0.9cm, align=center, font=\small},
  arr/.style={->, thick, >=stealth},
]
\node[src] (src)   {Source\\tx bits};
\node[box,  right=of src]  (mod)   {BPSK\\Modulator};
\node[ch,   right=of mod]  (ch1)   {Hop 1 Channel\\AWGN/Rayleigh/Rician};
\node[rel,  right=of ch1]  (relay) {Relay\\Neural Net\\denoise};
\node[ch,   right=of relay](ch2)   {Hop 2 Channel\\$2\times2$ MIMO\\$\mathbf{y}=\mathbf{H}\mathbf{x}_R+\mathbf{n}$};
\node[eq,   right=of ch2]  (eq)    {Equalizer\\ZF/MMSE/SIC};
\node[src,  right=of eq]   (dst)   {Destination\\bits};

\draw[arr] (src)   -- node[above,font=\tiny]{bits} (mod);
\draw[arr] (mod)   -- node[above,font=\tiny]{$x$} (ch1);
\draw[arr] (ch1)   -- node[above,font=\tiny]{Hop 1\\$y_R$} (relay);
\draw[arr] (relay) -- node[above,font=\tiny]{Hop 2\\$\hat{x}_R$} (ch2);
\draw[arr] (ch2)   -- node[above,font=\tiny]{$\mathbf{y}$} (eq);
\draw[arr] (eq)    -- node[above,font=\tiny]{bits} (dst);
\end{tikzpicture}
\caption{Detailed two-hop relay system model with MIMO Hop 2 channel and
         equalizer. The relay neural network denoises the Hop 1 received signal
         before retransmission.}
\label{fig:relay-system-detailed}
\end{figure}
"""

with open('chapters/ch03_methods.tex', 'r', encoding='utf-8') as f:
    tex = f.read()

# Find and replace first Shaded block (simple diagram)
# The first Shaded block starts right after "The system under study is..."
pattern1 = re.compile(
    r'\\begin\{Shaded\}\s*\\begin\{Highlighting\}\[\]\s*\\NormalTok\{flowchart LR\}.*?\\end\{Shaded\}',
    re.DOTALL
)
matches = list(pattern1.finditer(tex))
print(f"  Found {len(matches)} Shaded blocks")

if len(matches) >= 1:
    # Replace first block
    tex = tex[:matches[0].start()] + TIKZ_DIAGRAM_1 + tex[matches[0].end():]
    print("  Replaced first Shaded block with TikZ diagram 1")

# Re-find after first replacement
matches2 = list(pattern1.finditer(tex))
if len(matches2) >= 1:
    tex = tex[:matches2[0].start()] + TIKZ_DIAGRAM_2 + tex[matches2[0].end():]
    print("  Replaced second Shaded block with TikZ diagram 2")

# Add tikz packages to preamble if not already there
# (they should be in thesis_tau.tex preamble)

with open('chapters/ch03_methods.tex', 'w', encoding='utf-8') as f:
    f.write(tex)
print("  Saved ch03_methods.tex")

# ── 3. Add TikZ packages to thesis_tau.tex preamble ──────────────────────────
print("\n[preamble] Adding TikZ packages ...")
with open('thesis_tau.tex', 'r', encoding='utf-8') as f:
    main = f.read()

if 'tikz' not in main:
    tikz_pkg = r"""\usepackage{tikz}
\usetikzlibrary{shapes.geometric,arrows.meta,positioning,fit,calc}"""
    main = main.replace(r'\usepackage{chngcntr}',
                        r'\usepackage{chngcntr}' + '\n' + tikz_pkg)
    with open('thesis_tau.tex', 'w', encoding='utf-8') as f:
        f.write(main)
    print("  Added TikZ packages")
else:
    print("  TikZ already loaded")

# ── 4. Update ch04_experiments.tex: replace single figures with 3 split figs ─
print("\n[ch04] Updating figure references ...")

with open('chapters/ch04_experiments.tex', 'r', encoding='utf-8') as f:
    tex4 = f.read()

# Helper: build 3-figure replacement block
def make_split_figs(label_base, paths, captions, main_caption):
    """Build LaTeX for 3 separate figures replacing one dense figure."""
    blocks = []
    for i, (path, cap) in enumerate(zip(paths, captions), start=1):
        lbl = f"{label_base}_p{i}"
        block = f"""
\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.92\\linewidth]{{{path}}}
\\caption{{{cap}}}
\\label{{{lbl}}}
\\end{{figure}}"""
        blocks.append(block)
    return '\n'.join(blocks)


# ── 4.27: combined_modulation_awgn ───────────────────────────────────────────
old_fig27 = re.search(
    r'\\begin\{figure\}.*?\\label\{fig:fig27\}.*?\\end\{figure\}', tex4, re.DOTALL)
if old_fig27:
    new_fig27 = make_split_figs(
        'fig:fig27',
        ['results/modulation/combined_modulation_awgn_split_1.png',
         'results/modulation/combined_modulation_awgn_split_2.png',
         'results/modulation/combined_modulation_awgn_split_3.png'],
        ['Combined modulation comparison (AWGN) --- BPSK: all nine relay strategies with 95\\% CI. Zoom inset shows low-SNR region (0--8~dB).',
         'Combined modulation comparison (AWGN) --- QPSK: all nine relay strategies with 95\\% CI. Zoom inset shows low-SNR region.',
         'Combined modulation comparison (AWGN) --- 16-QAM: all nine relay strategies with 95\\% CI. Zoom inset shows low-SNR region.'],
        'Combined modulation comparison (AWGN)'
    )
    tex4 = tex4[:old_fig27.start()] + new_fig27 + tex4[old_fig27.end():]
    print("  Updated fig:fig27")

# ── 4.28: qam16_activation_awgn ──────────────────────────────────────────────
old_fig28 = re.search(
    r'\\begin\{figure\}.*?\\label\{fig:fig28\}.*?\\end\{figure\}', tex4, re.DOTALL)
if old_fig28:
    new_fig28 = make_split_figs(
        'fig:fig28',
        ['results/qam16_activation/qam16_activation_awgn_split_1.png',
         'results/qam16_activation/qam16_activation_awgn_split_2.png',
         'results/qam16_activation/qam16_activation_awgn_split_3.png'],
        ['16-QAM activation experiment (AWGN) --- tanh activation: all nine relay strategies. Zoom inset: 0--8~dB.',
         '16-QAM activation experiment (AWGN) --- linear activation: all nine relay strategies. Zoom inset: 0--8~dB.',
         '16-QAM activation experiment (AWGN) --- clipped tanh (hardtanh) activation: all nine relay strategies.'],
        '16-QAM activation experiment (AWGN)'
    )
    tex4 = tex4[:old_fig28.start()] + new_fig28 + tex4[old_fig28.end():]
    print("  Updated fig:fig28")

# ── 4.29: qam16_activation_rayleigh ──────────────────────────────────────────
old_fig29 = re.search(
    r'\\begin\{figure\}.*?\\label\{fig:fig29\}.*?\\end\{figure\}', tex4, re.DOTALL)
if old_fig29:
    new_fig29 = make_split_figs(
        'fig:fig29',
        ['results/qam16_activation/qam16_activation_rayleigh_split_1.png',
         'results/qam16_activation/qam16_activation_rayleigh_split_2.png',
         'results/qam16_activation/qam16_activation_rayleigh_split_3.png'],
        ['16-QAM activation experiment (Rayleigh) --- tanh activation: all nine relay strategies.',
         '16-QAM activation experiment (Rayleigh) --- linear activation: all nine relay strategies.',
         '16-QAM activation experiment (Rayleigh) --- clipped tanh activation: all nine relay strategies.'],
        '16-QAM activation experiment (Rayleigh)'
    )
    tex4 = tex4[:old_fig29.start()] + new_fig29 + tex4[old_fig29.end():]
    print("  Updated fig:fig29")

# ── 4.30: bpsk_activation_awgn ───────────────────────────────────────────────
old_fig30 = re.search(
    r'\\begin\{figure\}.*?\\label\{fig:fig30\}.*?\\end\{figure\}', tex4, re.DOTALL)
if old_fig30:
    # Check if split files exist
    if os.path.exists('results/activation_comparison/bpsk_activation_awgn_split_1.png'):
        new_fig30 = make_split_figs(
            'fig:fig30',
            ['results/activation_comparison/bpsk_activation_awgn_split_1.png',
             'results/activation_comparison/bpsk_activation_awgn_split_2.png',
             'results/activation_comparison/bpsk_activation_awgn_split_3.png'],
            ['BPSK constellation-aware activation comparison (AWGN) --- Part~1.',
             'BPSK constellation-aware activation comparison (AWGN) --- Part~2.',
             'BPSK constellation-aware activation comparison (AWGN) --- Part~3.'],
            'BPSK activation comparison (AWGN)'
        )
        tex4 = tex4[:old_fig30.start()] + new_fig30 + tex4[old_fig30.end():]
        print("  Updated fig:fig30 (PIL split)")
    else:
        print("  fig:fig30 split files not found, keeping original")

# ── 4.31: bpsk_activation_rayleigh ───────────────────────────────────────────
old_fig31 = re.search(
    r'\\begin\{figure\}.*?\\label\{fig:fig31\}.*?\\end\{figure\}', tex4, re.DOTALL)
if old_fig31:
    if os.path.exists('results/activation_comparison/bpsk_activation_rayleigh_split_1.png'):
        new_fig31 = make_split_figs(
            'fig:fig31',
            ['results/activation_comparison/bpsk_activation_rayleigh_split_1.png',
             'results/activation_comparison/bpsk_activation_rayleigh_split_2.png',
             'results/activation_comparison/bpsk_activation_rayleigh_split_3.png'],
            ['BPSK constellation-aware activation comparison (Rayleigh) --- Part~1.',
             'BPSK constellation-aware activation comparison (Rayleigh) --- Part~2.',
             'BPSK constellation-aware activation comparison (Rayleigh) --- Part~3.'],
            'BPSK activation comparison (Rayleigh)'
        )
        tex4 = tex4[:old_fig31.start()] + new_fig31 + tex4[old_fig31.end():]
        print("  Updated fig:fig31 (PIL split)")
    else:
        print("  fig:fig31 split files not found, keeping original")

# ── 4.32: qpsk_activation_awgn ───────────────────────────────────────────────
old_fig32 = re.search(
    r'\\begin\{figure\}.*?\\label\{fig:fig32\}.*?\\end\{figure\}', tex4, re.DOTALL)
if old_fig32:
    new_fig32 = make_split_figs(
        'fig:fig32',
        ['results/activation_comparison/qpsk_activation_awgn_split_1.png',
         'results/activation_comparison/qpsk_activation_awgn_split_2.png',
         'results/activation_comparison/qpsk_activation_awgn_split_3.png'],
        ['QPSK constellation-aware activation comparison (AWGN) --- sigmoid activation.',
         'QPSK constellation-aware activation comparison (AWGN) --- clipped tanh activation.',
         'QPSK constellation-aware activation comparison (AWGN) --- scaled tanh activation.'],
        'QPSK activation comparison (AWGN)'
    )
    tex4 = tex4[:old_fig32.start()] + new_fig32 + tex4[old_fig32.end():]
    print("  Updated fig:fig32")

# ── 4.33: qpsk_activation_rayleigh ───────────────────────────────────────────
old_fig33 = re.search(
    r'\\begin\{figure\}.*?\\label\{fig:fig33\}.*?\\end\{figure\}', tex4, re.DOTALL)
if old_fig33:
    new_fig33 = make_split_figs(
        'fig:fig33',
        ['results/activation_comparison/qpsk_activation_rayleigh_split_1.png',
         'results/activation_comparison/qpsk_activation_rayleigh_split_2.png',
         'results/activation_comparison/qpsk_activation_rayleigh_split_3.png'],
        ['QPSK constellation-aware activation comparison (Rayleigh) --- sigmoid activation.',
         'QPSK constellation-aware activation comparison (Rayleigh) --- clipped tanh activation.',
         'QPSK constellation-aware activation comparison (Rayleigh) --- scaled tanh activation.'],
        'QPSK activation comparison (Rayleigh)'
    )
    tex4 = tex4[:old_fig33.start()] + new_fig33 + tex4[old_fig33.end():]
    print("  Updated fig:fig33")

# ── 4.34: qam16_activation_awgn (activation_comparison) ─────────────────────
old_fig34 = re.search(
    r'\\begin\{figure\}.*?\\label\{fig:fig34\}.*?\\end\{figure\}', tex4, re.DOTALL)
if old_fig34:
    new_fig34 = make_split_figs(
        'fig:fig34',
        ['results/activation_comparison/qam16_activation_awgn_split_1.png',
         'results/activation_comparison/qam16_activation_awgn_split_2.png',
         'results/activation_comparison/qam16_activation_awgn_split_3.png'],
        ['16-QAM constellation-aware activation comparison (AWGN) --- sigmoid activation.',
         '16-QAM constellation-aware activation comparison (AWGN) --- clipped tanh activation.',
         '16-QAM constellation-aware activation comparison (AWGN) --- scaled tanh activation.'],
        '16-QAM activation comparison (AWGN)'
    )
    tex4 = tex4[:old_fig34.start()] + new_fig34 + tex4[old_fig34.end():]
    print("  Updated fig:fig34")

# ── 4.36: various_activation_functions ───────────────────────────────────────
old_fig36 = re.search(
    r'\\begin\{figure\}.*?\\label\{fig:fig36\}.*?\\end\{figure\}', tex4, re.DOTALL)
if old_fig36:
    new_fig36 = make_split_figs(
        'fig:fig36',
        ['results/activation_comparison/various_activation_functions_split_1.png',
         'results/activation_comparison/various_activation_functions_split_2.png',
         'results/activation_comparison/various_activation_functions_split_3.png'],
        ['Activation functions (left) and derivatives (right): tanh and clipped tanh (hardtanh). Hardtanh has a sharp transition at the clip bounds.',
         'Activation functions (left) and derivatives (right): scaled tanh with $A_{\\max}=0.9487$ for 16-QAM.',
         'Activation functions (left) and derivatives (right): sigmoid and scaled sigmoid. Both provide smooth saturation with non-zero gradients throughout.'],
        'Activation function shapes and derivatives'
    )
    tex4 = tex4[:old_fig36.start()] + new_fig36 + tex4[old_fig36.end():]
    print("  Updated fig:fig36")

# ── 4.38: top3_qam16_rayleigh ────────────────────────────────────────────────
old_fig40 = re.search(
    r'\\begin\{figure\}.*?\\label\{fig:fig40\}.*?\\end\{figure\}', tex4, re.DOTALL)
if old_fig40:
    new_fig40 = make_split_figs(
        'fig:fig40',
        ['results/csi/top3_qam16_rayleigh_split_1.png',
         'results/csi/top3_qam16_rayleigh_split_2.png',
         'results/csi/top3_qam16_rayleigh_split_3.png'],
        ['16-QAM Rayleigh --- classical relays (AF, DF) vs.~top-3 neural relay variants. Zoom inset: 0--8~dB.',
         '16-QAM Rayleigh --- top-3 neural relay variants with 95\\% confidence intervals. Zoom inset: 0--8~dB.',
         '16-QAM Rayleigh --- additional neural relay variants for comparison.'],
        'Top-3 neural relay architectures (16-QAM Rayleigh)'
    )
    tex4 = tex4[:old_fig40.start()] + new_fig40 + tex4[old_fig40.end():]
    print("  Updated fig:fig40 (4.38)")

with open('chapters/ch04_experiments.tex', 'w', encoding='utf-8') as f:
    f.write(tex4)
print("\nSaved ch04_experiments.tex")
print("Done.")