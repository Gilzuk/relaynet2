"""
rebuild.py
----------
Rebuilds thesis_restructured.md from thesis_backup.md using
string splitting (no large regex replacements that cause MemoryError).

Changes made vs. original:
  1. TOC updated to reflect new chapter structure
  2. Literature Review: Channel Models + MIMO Equalization theory moved here
  3. Methods: Channel Models + MIMO Equalization sections removed
  4. Results → Experiments chapter with Goal/Trials/Conclusion + charts
"""

import re

print("Reading thesis_backup.md ...")
with open("thesis_backup.md", "r", encoding="utf-8") as f:
    src = f.read()

print(f"  Source size: {len(src):,} chars")

# ─────────────────────────────────────────────────────────────────────────────
# Helper: find the byte-offset of a heading
# ─────────────────────────────────────────────────────────────────────────────
def find_heading(text, heading, start=0):
    """Return the start index of 'heading' in text, or -1."""
    idx = text.find(heading, start)
    return idx

# ─────────────────────────────────────────────────────────────────────────────
# 1. Locate major section boundaries
# ─────────────────────────────────────────────────────────────────────────────
idx_toc        = find_heading(src, "\n## Table of Contents")
idx_intro      = find_heading(src, "\n## Introduction and Literature Review")
idx_objectives = find_heading(src, "\n## Research Objectives")
idx_methods    = find_heading(src, "\n## Methods")
idx_results    = find_heading(src, "\n## Results")
idx_discussion = find_heading(src, "\n## Discussion and Conclusions")
idx_references = find_heading(src, "\n## References")

print(f"  TOC at {idx_toc}, Intro at {idx_intro}, Objectives at {idx_objectives}")
print(f"  Methods at {idx_methods}, Results at {idx_results}")
print(f"  Discussion at {idx_discussion}, References at {idx_references}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Extract raw sections
# ─────────────────────────────────────────────────────────────────────────────
front_matter   = src[:idx_toc]
toc_section    = src[idx_toc:idx_intro]
intro_section  = src[idx_intro:idx_objectives]
obj_section    = src[idx_objectives:idx_methods]
methods_section= src[idx_methods:idx_results]
results_section= src[idx_results:idx_discussion]
discussion_section = src[idx_discussion:idx_references]
back_matter    = src[idx_references:]

# ─────────────────────────────────────────────────────────────────────────────
# 3. New TOC
# ─────────────────────────────────────────────────────────────────────────────
new_toc = """
## Table of Contents {.unnumbered}

**Front Matter**

| | |
|---|---|
| I | List of Abbreviations |
| II | Abstract (Hebrew) |
| III | Keywords |

**Body**

1. [Introduction and Literature Review](#introduction-and-literature-review)
   - 1.1 Cooperative Relay Communication
   - 1.2 Classical Relay Strategies
   - 1.3 Machine Learning in Wireless Communication
   - 1.4 Generative Models for Signal Processing
   - 1.5 Sequence Models: Transformers and State Space Models
   - 1.6 MIMO Systems and Equalization
   - 1.7 Research Gap and Motivation
   - 1.8 Theoretical Foundations: Channel Models
   - 1.9 Theoretical Foundations: MIMO Equalization
2. [Research Objectives](#research-objectives)
   - 2.1 Main Objective
   - 2.2 Research Hypotheses
   - 2.3 Specific Objectives
   - 2.4 Scope and Delimitations
3. [Methods](#methods)
   - 3.1 System Model
   - 3.2 Relay Strategies
   - 3.3 Simulation Framework
   - 3.4 Normalized Parameter Comparison
   - 3.5 Modulation Schemes
4. [Experiments](#experiments)
   - 4.1 Channel Model Validation
   - 4.2 SISO BPSK Performance (Baseline Relay Comparison)
   - 4.3 MIMO 2x2 BPSK Performance
   - 4.4 Parameter Normalization & Complexity Trade-off
   - 4.5 Higher-Order Modulation Scalability (Constellation-Aware Training)
   - 4.6 Input Normalization and CSI Injection
   - 4.7 16-Class 2D Classification for QAM16
   - 4.8 End-to-End Joint Optimization
5. [Discussion and Conclusions](#discussion-and-conclusions)
   - 5.1 Interpretation of Results
   - 5.2 The Less is More Principle
   - 5.3 State Space vs. Attention for Signal Processing
   - 5.4 Practical Deployment Recommendations
   - 5.5 Limitations
   - 5.6 Future Work
   - 5.7 Conclusions
6. [References](#references)
7. [Appendices](#appendices)

**Back Matter**

| | |
|---|---|
| VIII | Abstract (English) |

"""

# ─────────────────────────────────────────────────────────────────────────────
# 4. Extract Channel Models and MIMO Equalization from Methods
#    to append to Literature Review
# ─────────────────────────────────────────────────────────────────────────────
def extract_subsection(text, start_heading, stop_headings):
    """Extract from start_heading up to the first of stop_headings."""
    idx_start = text.find(start_heading)
    if idx_start == -1:
        return ""
    idx_end = len(text)
    for stop in stop_headings:
        idx_s = text.find(stop, idx_start + len(start_heading))
        if idx_s != -1 and idx_s < idx_end:
            idx_end = idx_s
    return text[idx_start:idx_end]

channel_models_theory = extract_subsection(
    methods_section,
    "\n### Channel Models",
    ["\n### Relay Strategies", "\n### Simulation Framework", "\n### System Model"]
)

mimo_eq_theory = extract_subsection(
    methods_section,
    "\n### MIMO Equalization Techniques",
    ["\n### Simulation Framework", "\n### Normalized", "\n### Modulation"]
)

# Append theory to Literature Review
# Strip the original sub-heading from the extracted block to avoid duplication
def strip_first_heading(text):
    """Remove the first ### heading line from a block."""
    return re.sub(r"^\s*###[^\n]*\n", "", text, count=1)

theory_appendix = ""
if channel_models_theory.strip():
    theory_appendix += "\n\n### Theoretical Foundations: Channel Models\n\n"
    theory_appendix += strip_first_heading(channel_models_theory).strip()

if mimo_eq_theory.strip():
    theory_appendix += "\n\n### Theoretical Foundations: MIMO Equalization\n\n"
    theory_appendix += strip_first_heading(mimo_eq_theory).strip()

new_intro_section = intro_section.rstrip() + theory_appendix + "\n"

# ─────────────────────────────────────────────────────────────────────────────
# 5. Trim Methods: remove Channel Models and MIMO Equalization sub-sections
# ─────────────────────────────────────────────────────────────────────────────
def remove_subsection(text, start_heading, stop_headings):
    """Remove from start_heading up to the first of stop_headings."""
    idx_start = text.find(start_heading)
    if idx_start == -1:
        return text
    idx_end = len(text)
    for stop in stop_headings:
        idx_s = text.find(stop, idx_start + len(start_heading))
        if idx_s != -1 and idx_s < idx_end:
            idx_end = idx_s
    return text[:idx_start] + text[idx_end:]

new_methods_section = methods_section
new_methods_section = remove_subsection(
    new_methods_section,
    "\n### Channel Models",
    ["\n### Relay Strategies", "\n### Simulation Framework"]
)
new_methods_section = remove_subsection(
    new_methods_section,
    "\n### MIMO Equalization Techniques",
    ["\n### Simulation Framework", "\n### Normalized", "\n### Modulation"]
)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Build the new Experiments chapter
# ─────────────────────────────────────────────────────────────────────────────

# Helper: extract figures and tables from backup
def find_figure(n):
    m = re.search(r"(!\[Figure " + str(n) + r"[^\]]*\]\([^\)]+\))", src)
    return m.group(1) if m else ""

def find_caption(n):
    m = re.search(r"(\*Figure " + str(n) + r"[^\*]+\*)", src)
    return m.group(1) if m else ""

def find_table(n):
    """Find Table N (exact number, not partial match like 13 for 1).
    Handles prose text between label and table (up to 6000 chars gap).
    Uses word boundary to avoid matching Table 13 when looking for Table 1.
    """
    # Find the label line
    label_m = re.search(r"\*{0,2}Table " + str(n) + r"\b[^\n]*", src)
    if not label_m:
        return ""
    label_text = label_m.group(0).strip("*").strip()

    # Search for the next markdown table within 6000 chars of the label
    search_region = src[label_m.start(): label_m.start() + 6000]
    tbl_m = re.search(r"(\|[^\n]+\|\n\|[-|]+\|(?:\n\|[^\n]+\|)*)", search_region)
    if not tbl_m:
        return ""

    # Return label + table
    return label_text + "\n\n" + tbl_m.group(1).strip()

def figs_block(nums):
    parts = []
    for n in nums:
        img = find_figure(n)
        cap = find_caption(n)
        if img:
            parts.append(img)
        if cap:
            parts.append("\n" + cap)
        if img or cap:
            parts.append("")
    return "\n".join(parts).strip()

def tbls_block(nums):
    parts = []
    for n in nums:
        t = find_table(n)
        if t:
            parts.append(t)
        else:
            print(f"    WARNING: Table {n} not found")
    return "\n\n".join(parts)

# Experiment definitions
experiments = [
    {
        "id": "4.1",
        "title": "Channel Model Validation",
        "goal": "Validate the simulation framework against closed-form theoretical BER expressions to ensure baseline accuracy before evaluating AI relays.",
        "trials": [
            ("Topology", "SISO, MIMO 2x2"),
            ("Modulation scheme", "BPSK"),
            ("Channel", "AWGN, Rayleigh, Rician (K=3)"),
            ("Equalizers (in MIMO only)", "None (SISO), ZF, MMSE, SIC"),
            ("Demod", "Hard decision only"),
        ],
        "conclusion": "Monte Carlo simulations match theoretical predictions within 95% confidence intervals across all channel models and topologies. AWGN follows the expected exponential decay, Rayleigh validates the $1/(4\\bar{\\gamma})$ high-SNR slope, Rician falls between the two, and MIMO equalization correctly exhibits the expected ZF < MMSE < SIC performance hierarchy.",
        "fig_nums": list(range(1, 8)),
        "tbl_nums": [],
    },
    {
        "id": "4.2",
        "title": "SISO BPSK Performance (Baseline Relay Comparison)",
        "goal": "Evaluate baseline classical and AI-based relay strategies on single-antenna configurations across different fading environments.",
        "trials": [
            ("Topology", "SISO"),
            ("Modulation scheme", "BPSK"),
            ("Channel", "AWGN, Rayleigh, Rician (K=3)"),
            ("Equalizers", "None"),
            ("NN architecture", "Supervised (MLP, Hybrid), Generative (VAE, CGAN), Sequence (Transformer, Mamba S6, Mamba-2 SSD)"),
            ("NN activation", "tanh"),
        ],
        "conclusion": "AI relays selectively outperform AF and, on selected channels (AWGN, Rician), DF at low SNR (0–4 dB). However, under Rayleigh fading, classical DF dominates even at low SNR. Across all channels, classical DF remains dominant at medium-to-high SNR ($\\geq 6$ dB), matching or exceeding all AI methods with zero parameters.",
        "fig_nums": list(range(9, 12)),
        "tbl_nums": [1, 2, 3],
    },
    {
        "id": "4.3",
        "title": "MIMO 2x2 BPSK Performance",
        "goal": "Evaluate relay strategies under spatial multiplexing and various interference cancellation techniques.",
        "trials": [
            ("Topology", "MIMO 2x2"),
            ("Modulation scheme", "BPSK"),
            ("Channel", "Rayleigh"),
            ("Equalizers (in MIMO only)", "ZF, MMSE, SIC"),
            ("NN architecture", "Supervised (MLP, Hybrid), Generative (VAE, CGAN), Sequence (Transformer, Mamba S6, Mamba-2 SSD)"),
            ("NN activation", "tanh"),
        ],
        "conclusion": "The MIMO equalization hierarchy (ZF < MMSE < SIC) holds for all relay types. The AI advantage at low SNR is preserved under ZF and MMSE (where Mamba S6 achieves the lowest BER), but under the superior SIC equalization, classical DF provides the lowest BER at all low-to-medium SNR points. Relay processing gains and MIMO equalization gains are additive.",
        "fig_nums": list(range(12, 15)),
        "tbl_nums": [4, 5, 6],
    },
    {
        "id": "4.4",
        "title": "Parameter Normalization & Complexity Trade-off",
        "goal": "Isolate architectural inductive biases from parameter count effects and characterize the complexity-performance trade-off for relay denoising.",
        "trials": [
            ("Topology", "SISO, MIMO 2x2"),
            ("Modulation scheme", "BPSK"),
            ("Channel", "AWGN, Rayleigh, Rician, MIMO (ZF, MMSE, SIC)"),
            ("Equalizers (in MIMO only)", "None, ZF, MMSE, SIC"),
            ("NN architecture", "All normalized to ~3,000 parameters, plus original sizes (169 to 26K)"),
            ("NN activation", "tanh"),
        ],
        "conclusion": "The relay denoising task exhibits an inverted-U complexity relationship: a minimal 169-parameter MLP matches models 140x larger, while excessive parameters (11K+) lead to overfitting. At a normalized scale of 3,000 parameters, the performance gap between feedforward and sequence architectures narrows to ~1% BER, indicating that parameter count rather than architectural choice is the primary performance driver. Generative VAE is a consistent underperformer due to probabilistic overhead.",
        "fig_nums": list(range(15, 21)),
        "tbl_nums": [7, 8, 9, 10, 11, 12, 13],
    },
    {
        "id": "4.5",
        "title": "Higher-Order Modulation Scalability (Constellation-Aware Training)",
        "goal": "Evaluate the generalizability of BPSK-trained relays to complex constellations and resolve the multi-level amplitude bottleneck.",
        "trials": [
            ("Topology", "SISO"),
            ("Modulation scheme", "QPSK, 16-QAM"),
            ("Channel", "AWGN, Rayleigh"),
            ("Equalizers", "None"),
            ("NN architecture", "Supervised (MLP, Hybrid), Generative (VAE, CGAN), Sequence (Transformer, Mamba S6, Mamba-2 SSD)"),
            ("NN activation", "tanh, linear, clipped tanh (hardtanh), scaled tanh, scaled sigmoid"),
            ("Special case", "Constellation Aware training"),
        ],
        "conclusion": "QPSK performance mirrors BPSK perfectly due to I/Q independence. On 16-QAM, standard tanh compression causes a severe, irreducible BER floor (~0.22 at 16 dB). Replacing tanh with constellation-aware bounded activations (hardtanh, scaled tanh) bounded to the precise signal amplitude ($3/\\sqrt{10}$) and retraining eliminates this floor. Sequence models benefit most, reducing their BER floor by 5x, though a gap to classical DF persists due to per-axis error accumulation.",
        "fig_nums": list(range(21, 37)),
        "tbl_nums": [14, 15, 16],
    },
    {
        "id": "4.6",
        "title": "Input Normalization and CSI Injection",
        "goal": "Determine the impact of structural input normalization and explicit channel state information (CSI) injection on higher-order modulations in fading channels.",
        "trials": [
            ("Topology", "SISO"),
            ("Modulation scheme", "16-QAM, 16-PSK"),
            ("Channel", "Rayleigh"),
            ("Equalizers", "None"),
            ("NN architecture", "Transformer, Mamba S6, Mamba-2 SSD"),
            ("NN activation", "tanh, hardtanh, scaled tanh, sigmoid"),
            ("Special case", "CSI Injection, LayerNorm"),
        ],
        "conclusion": "Input LayerNorm universally benefits multi-level constellations like 16-QAM. Explicit CSI injection is highly modulation-dependent: it degrades performance for amplitude-carrying 16-QAM (creating redundant feature confusion) but significantly improves performance for constant-envelope 16-PSK, bringing the best neural models to within 2.5% of AF at high SNR. Across the 48 tested combinatorial variants, Mamba S6 proved the strongest architecture, but no neural relay surpassed classical DF.",
        "fig_nums": list(range(37, 46)),
        "tbl_nums": [17, 18, 19, 20, 21, 22],
    },
    {
        "id": "4.7",
        "title": "16-Class 2D Classification for QAM16",
        "goal": "Eliminate the structural BER floor imposed by per-axis I/Q splitting for 16-QAM by utilizing full 2D decision boundaries.",
        "trials": [
            ("Topology", "SISO"),
            ("Modulation scheme", "16-QAM"),
            ("Channel", "AWGN"),
            ("Equalizers", "None"),
            ("NN architecture", "Supervised (MLP), Generative (VAE), Sequence (Transformer, Mamba S6, Mamba-2 SSD)"),
            ("NN activation", "None (Softmax implicitly via Cross-Entropy loss)"),
            ("Special case", "16-class joint 2D classification"),
        ],
        "conclusion": "Treating the relay as a joint 16-point classifier over the full 2D constellation space completely eliminates the structural 4-class BER floor. For the first time, neural variants (VAE, Transformer, MLP) matched classical DF performance at high SNR, achieving near-zero BER at 20 dB. This proves the previous BER floor was an artifact of I/Q splitting, not a fundamental limitation of neural relays.",
        "fig_nums": list(range(50, 54)),
        "tbl_nums": [24],
    },
    {
        "id": "4.8",
        "title": "End-to-End Joint Optimization",
        "goal": "Compare the modular neural relay approach against a fully joint transmitter-receiver autoencoder.",
        "trials": [
            ("Topology", "SISO"),
            ("Modulation scheme", "Learned latent space (M=16, power-constrained)"),
            ("Channel", "Rayleigh"),
            ("Equalizers", "ZF explicitly at the receiver"),
            ("NN architecture", "MLP Autoencoder (Encoder/Decoder)"),
            ("Special case", "End-to-End (E2E) optimization"),
        ],
        "conclusion": "The E2E autoencoder underperforms both the theoretical limits of classical 16-QAM and the modular two-hop DF relay across all SNR points (67-141% higher BER). The network fails to discover a constellation geometry that surpasses the classical square grid under single-antenna Rayleigh fading, demonstrating that 'black-box' deep learning is inefficient compared to modular designs that leverage classical signal processing algorithms for modulation and equalization.",
        "fig_nums": list(range(46, 50)),
        "tbl_nums": [23],
    },
]

def build_experiment_section(exp):
    lines = []
    lines.append(f"\n### {exp['id']} {exp['title']}")
    lines.append(f"**Goal:** {exp['goal']}")
    lines.append("")
    lines.append("**Trials:**")
    for label, value in exp["trials"]:
        lines.append(f"- **{label}:** {value}")
    lines.append("")
    lines.append(f"**Conclusion:**")
    lines.append(exp["conclusion"])

    # Tables
    tb = tbls_block(exp["tbl_nums"])
    if tb.strip():
        lines.append("")
        lines.append("#### Results Tables")
        lines.append("")
        lines.append(tb)

    # Figures
    fb = figs_block(exp["fig_nums"])
    if fb.strip():
        lines.append("")
        lines.append("#### Results Figures")
        lines.append("")
        lines.append(fb)

    return "\n".join(lines)

print("\nBuilding Experiments chapter ...")
exp_sections = []
for exp in experiments:
    sec = build_experiment_section(exp)
    n_figs = sec.count("![Figure")
    n_tbls = sec.count("|---|")
    print(f"  {exp['id']}: {n_figs} figs, {n_tbls} table rows")
    exp_sections.append(sec)

new_experiments_chapter = "\n## Experiments\n\n" + \
    "The experiments chapter walks through the goals, trials, and conclusions from each experiment, " + \
    "following a systematic evaluation of relay strategies across diverse configurations.\n" + \
    "\n".join(exp_sections) + "\n"

# ─────────────────────────────────────────────────────────────────────────────
# 7. Assemble the new document
# ─────────────────────────────────────────────────────────────────────────────
print("\nAssembling document ...")

# Replace the TOC section
new_doc = (
    front_matter
    + new_toc
    + new_intro_section
    + obj_section
    + new_methods_section
    + new_experiments_chapter
    + discussion_section
    + back_matter
)

print(f"  New document size: {len(new_doc):,} chars")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Write output — NEVER touch thesis.md
# ─────────────────────────────────────────────────────────────────────────────
with open("thesis_restructured.md", "w", encoding="utf-8") as f:
    f.write(new_doc)

print("  Saved to thesis_restructured.md")

# ─────────────────────────────────────────────────────────────────────────────
# 9. Verify
# ─────────────────────────────────────────────────────────────────────────────
print("\nVerification:")
with open("thesis_restructured.md", "r", encoding="utf-8") as f:
    verify = f.read()

headings = re.findall(r"^## .+", verify, re.MULTILINE)
print(f"  ## headings: {headings}")

exp_m = re.search(r"## Experiments(.*?)## Discussion", verify, re.DOTALL)
exp_t = exp_m.group(1) if exp_m else ""
figs_found = re.findall(r"!\[Figure \d+", exp_t)
print(f"  Figures in Experiments: {len(figs_found)}")

for sec_num in ["4.1","4.2","4.3","4.4","4.5","4.6","4.7","4.8"]:
    mm = re.search(r"### " + re.escape(sec_num) + r"(.*?)(?=\n### |\n## )",
                   verify, re.DOTALL)
    if mm:
        c = mm.group(1)
        has_goal       = "**Goal:**" in c
        has_trials     = "**Trials:**" in c
        has_conclusion = "**Conclusion:**" in c
        has_figs       = bool(re.search(r"!\[Figure", c))
        has_tbls       = bool(re.search(r"\|[-|]+\|", c))
        print(f"  {sec_num}: Goal={has_goal} Trials={has_trials} "
              f"Conclusion={has_conclusion} Figs={has_figs} Tables={has_tbls}")
    else:
        print(f"  {sec_num}: NOT FOUND")