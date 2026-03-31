"""
add_table_captions.py
---------------------
Adds captions below all data tables that are currently missing them.
Also adds captions to Tables 15, 16, 24 (found in backup but not labeled).
"""

import re

with open("thesis_restructured.md", "r", encoding="utf-8") as f:
    text = f.read()

print(f"Input size: {len(text):,} chars")

# ─────────────────────────────────────────────────────────────────────────────
# Helper: insert caption below a specific table identified by a unique
# string in its first row
# ─────────────────────────────────────────────────────────────────────────────

def add_caption_after_table(text, first_row_fragment, caption_text, tbl_num):
    """
    Find the markdown table whose first row contains first_row_fragment,
    and add a pandoc-crossref caption below it.
    """
    # Find the table block
    pattern = re.compile(
        r"(\|[^\n]*" + re.escape(first_row_fragment) + r"[^\n]*\|\n"
        r"\|[-| :]+\|\n"
        r"(?:\|[^\n]+\|\n)*)",
        re.MULTILINE
    )
    m = pattern.search(text)
    if not m:
        print(f"  WARNING: Table {tbl_num} not found (fragment: '{first_row_fragment[:50]}')")
        return text, False

    # Check if already has a caption
    after = text[m.end():m.end()+100]
    if re.match(r"\nTable: ", after):
        print(f"  Table {tbl_num}: already has caption, skipping")
        return text, False

    caption_line = f"\nTable: {caption_text} {{#tbl:table{tbl_num}}}\n"
    tbl_content = m.group(0)
    if not tbl_content.endswith("\n"):
        tbl_content += "\n"

    new_block = tbl_content + caption_line
    text = text[:m.start()] + new_block + text[m.end():]
    print(f"  Added caption: Table {tbl_num}: {caption_text[:70]}")
    return text, True

# ─────────────────────────────────────────────────────────────────────────────
# Tables to add captions to (identified by unique first-row fragment)
# ─────────────────────────────────────────────────────────────────────────────

tables_to_add = [
    # (first_row_fragment, caption_text, table_num)

    # Table 15: Activation comparison (from backup, in §4.5)
    (
        "tanh (BPSK) | linear (QAM16) | hardtanh (QAM16)",
        "BER at 16 dB for all relay variants across activation functions and modulations (AWGN and Rayleigh).",
        15
    ),

    # Table 16: Mamba CSI improvements (from backup, in §4.6)
    (
        "AF Baseline | DF Baseline | Mamba S6 (Baseline) | Mamba S6 (+CSI + LN)",
        "16-QAM BER improvements at 20 dB for sequence models with and without CSI injection and layer normalisation (AWGN).",
        16
    ),

    # Table 17: Research positioning (in §1 Literature Review)
    (
        "Aspect | Prior Work | This Thesis",
        "Research positioning: comparison of this thesis against prior work in deep-learning-based relay communication.",
        17
    ),

    # Table 24: 4-class vs 16-class BER (from backup, in §4.7)
    (
        "4-cls BER @ 20 dB | 16-cls BER @ 20 dB | Improvement",
        "BER at 20 dB for all relay variants: 4-class (I/Q split) vs. 16-class (joint 2D) classification for 16-QAM.",
        24
    ),

    # Table 26: Channel SNR comparison (in §1.8 Theoretical Foundations)
    (
        "SNR for BER = $10^{-3}$",
        "Theoretical SNR required to achieve BER = $10^{-3}$ and diversity order for each channel model.",
        26
    ),

    # Table 27: MIMO component roles (in §3.1 System Model)
    (
        "Component | Problem Solved | Location | I",
        "MIMO system components, the problem each solves, and their input/output mapping.",
        27
    ),

    # Table 28: Normalization scaling methods (in §3.4)
    (
        "Model | Parameters | Scaling Method",
        "Normalized 3K-parameter model configurations: parameter count and scaling method for each architecture.",
        28
    ),

    # Table 29: QPSK symbol mapping (in §3.5)
    (
        "Bit pair $(b_0, b_1)$ | Symbol | Quadrant",
        "QPSK Gray-coded symbol mapping: bit pairs to complex symbols.",
        29
    ),

    # Table 30: 16-QAM PAM-4 levels (in §3.5)
    (
        "Bit pair | Level | Bit pair | Level",
        "16-QAM PAM-4 Gray-coded level mapping for in-phase and quadrature components.",
        30
    ),

    # Table 31: Relay complex signal handling (in §3.2)
    (
        "Relay type | Complex signal processing | Rationale",
        "Relay-specific complex signal handling strategies and rationale.",
        31
    ),

    # Table 32: Deployment recommendations (in §5.4)
    (
        "Operating Regime | Recommended Relay | R",
        "Practical deployment recommendations: recommended relay strategy by operating regime.",
        32
    ),

    # Table 33: Hypothesis summary (in §5.1)
    (
        "Hypothesis | Statement | Result",
        "Research hypotheses, their statements, and experimental outcomes.",
        33
    ),
]

added = 0
for frag, cap, num in tables_to_add:
    text, ok = add_caption_after_table(text, frag, cap, num)
    if ok:
        added += 1

print(f"\nAdded {added} new table captions")

# ─────────────────────────────────────────────────────────────────────────────
# Write output
# ─────────────────────────────────────────────────────────────────────────────
with open("thesis_restructured.md", "w", encoding="utf-8") as f:
    f.write(text)

print(f"Output size: {len(text):,} chars")
print("Saved to thesis_restructured.md")

# ─────────────────────────────────────────────────────────────────────────────
# Verify
# ─────────────────────────────────────────────────────────────────────────────
with open("thesis_restructured.md", "r", encoding="utf-8") as f:
    verify = f.read()

captions = re.findall(r"^Table: .+ \{#tbl:table(\d+)\}", verify, re.MULTILINE)
nums = sorted(int(n) for n in captions)
print(f"\nAll {len(captions)} table numbers: {nums}")

# Check for remaining uncaptioned data tables
all_md = list(re.finditer(r"(\|[^\n]+\|\n\|[-| :]+\|\n(?:\|[^\n]+\|\n)*)", verify))
uncaptioned = []
for m in all_md:
    after = verify[m.end():m.end()+100]
    if not re.match(r"\nTable: ", after):
        first_row = m.group(0).split("\n")[0][:70]
        before = verify[max(0,m.start()-80):m.start()].strip().split("\n")[-1][:50]
        uncaptioned.append((before, first_row))

print(f"\nUncaptioned tables remaining: {len(uncaptioned)}")
for b, r in uncaptioned:
    print(f"  before='{b}' | row='{r}'")