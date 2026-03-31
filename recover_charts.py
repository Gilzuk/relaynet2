import re

# Read both files
with open("thesis_backup.md", "r", encoding="utf-8") as f:
    backup = f.read()

with open("thesis_restructured.md", "r", encoding="utf-8") as f:
    restructured = f.read()

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Extract the original Results chapter from the backup
# ─────────────────────────────────────────────────────────────────────────────
results_match = re.search(
    r"## Results(.*?)(?=## Discussion and Conclusions)",
    backup, re.DOTALL
)
results_text = results_match.group(1) if results_match else ""

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Helper: pull a sub-section from the results text
# ─────────────────────────────────────────────────────────────────────────────
def grab_section(text, start_pattern, stop_pattern=None):
    """Return the content between start_pattern and stop_pattern (or end)."""
    if stop_pattern:
        m = re.search(start_pattern + r"(.*?)" + stop_pattern, text, re.DOTALL)
    else:
        m = re.search(start_pattern + r"(.*)", text, re.DOTALL)
    return m.group(1).strip() if m else ""

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Extract figures + tables for each experiment from the backup
# ─────────────────────────────────────────────────────────────────────────────

# Experiment 4.1 – Channel Model Validation
# In the backup this lives in the Methods / Channel Models section AND the
# early part of Results.  The figures 1-7 are already in the restructured
# file (inside the Theoretical Foundations section).  We only need the
# consolidated validation figures that appear in the Results chapter.
exp1_charts = grab_section(
    results_text,
    r"### 6\.1",
    r"### 6\.2"
)
if not exp1_charts:
    # Try alternate heading style
    exp1_charts = grab_section(
        results_text,
        r"### Channel Model Validation",
        r"### (SISO|BPSK|Relay|7\.|6\.2)"
    )

# Experiment 4.2 – SISO BPSK
exp2_charts = grab_section(
    results_text,
    r"### (7\.2|SISO BPSK|AWGN Channel.*Relay|6\.2 SISO)",
    r"### (7\.3|MIMO|Normalized|Higher|7\.4|6\.3)"
)
if not exp2_charts:
    # Grab AWGN + Rayleigh + Rician SISO sections
    awgn_sec  = grab_section(results_text, r"#### (AWGN|7\.2\.1)", r"#### (Rayleigh|7\.2\.2|7\.3)")
    ray_sec   = grab_section(results_text, r"#### (Rayleigh Fading.*Relay|7\.2\.2|7\.3)", r"#### (Rician|7\.2\.3|7\.4)")
    rician_sec= grab_section(results_text, r"#### (Rician.*Relay|7\.2\.3|7\.4)", r"#### (MIMO|7\.5|Normalized)")
    exp2_charts = "\n\n".join(filter(None, [awgn_sec, ray_sec, rician_sec]))

# Experiment 4.3 – MIMO 2x2 BPSK
exp3_charts = grab_section(
    results_text,
    r"### (7\.[5-7]|MIMO 2.2|2.2 MIMO)",
    r"### (7\.8|Normalized|Parameter|Higher|Modulation)"
)
if not exp3_charts:
    zf_sec   = grab_section(results_text, r"#### (ZF|7\.5)", r"#### (MMSE|7\.6)")
    mmse_sec = grab_section(results_text, r"#### (MMSE.*Relay|7\.6)", r"#### (SIC|7\.7)")
    sic_sec  = grab_section(results_text, r"#### (SIC|7\.7)", r"#### (Normalized|7\.8|Parameter)")
    exp3_charts = "\n\n".join(filter(None, [zf_sec, mmse_sec, sic_sec]))

# Experiment 4.4 – Parameter Normalization
exp4_charts = grab_section(
    results_text,
    r"### (7\.8|7\.9|Normalized|Parameter Normalization|Complexity)",
    r"### (7\.10|Higher|Modulation|Constellation)"
)

# Experiment 4.5 – Higher-Order Modulation / Constellation-Aware
exp5_charts = grab_section(
    results_text,
    r"### (7\.10|7\.11|7\.12|Higher.Order|Modulation Scalability|Constellation.Aware)",
    r"### (7\.13|7\.14|7\.15|Input Norm|CSI|LayerNorm)"
)

# Experiment 4.6 – Input Normalization and CSI Injection
exp6_charts = grab_section(
    results_text,
    r"### (7\.13|7\.14|7\.15|Input Norm|LayerNorm|CSI)",
    r"### (7\.16|7\.17|16.Class|End.to.End|E2E)"
)

# Experiment 4.7 – 16-Class 2D Classification
exp7_charts = grab_section(
    results_text,
    r"### (7\.17|16.Class|2D Classif)",
    r"### (7\.18|End.to.End|E2E|Discussion|$)"
)

# Experiment 4.8 – End-to-End
exp8_charts = grab_section(
    results_text,
    r"### (7\.16|End.to.End|E2E)",
    r"### (7\.17|16.Class|Discussion|$)"
)

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Fallback: if section-based extraction failed, use figure-number ranges
# ─────────────────────────────────────────────────────────────────────────────
def extract_figures_by_numbers(text, fig_nums):
    """Extract figure blocks (image + caption) for given figure numbers."""
    blocks = []
    for n in fig_nums:
        # Match the image line
        img = re.search(
            r"(!\[Figure " + str(n) + r"[^]]*\]\([^)]+\))",
            text
        )
        # Match the italic caption line that follows
        cap = re.search(
            r"\*Figure " + str(n) + r"[^*]*\*",
            text
        )
        if img:
            blocks.append(img.group(1))
        if cap:
            blocks.append("\n" + cap.group(0))
    return "\n\n".join(blocks)

def extract_tables_by_numbers(text, tbl_nums):
    """Extract table blocks (header + rows) for given table numbers."""
    blocks = []
    for n in tbl_nums:
        # Look for "Table N" label followed by a markdown table
        m = re.search(
            r"(\*\*Table " + str(n) + r"[^*]*\*\*.*?\n(?:\|.*\n)+)",
            text, re.DOTALL
        )
        if not m:
            m = re.search(
                r"(Table " + str(n) + r"[^\n]*\n(?:\|.*\n)+)",
                text, re.DOTALL
            )
        if m:
            blocks.append(m.group(1).strip())
    return "\n\n".join(blocks)

# Map experiment → figure numbers (from List of Figures in restructured file)
exp_figures = {
    "4.1": list(range(1, 8)),          # Figs 1-7
    "4.2": list(range(9, 12)),         # Figs 9-11
    "4.3": list(range(12, 15)),        # Figs 12-14
    "4.4": list(range(15, 21)),        # Figs 15-20
    "4.5": list(range(21, 37)),        # Figs 21-36
    "4.6": list(range(37, 46)),        # Figs 37-45
    "4.7": list(range(50, 54)),        # Figs 50-53
    "4.8": list(range(46, 50)),        # Figs 46-49
}

exp_tables = {
    "4.1": [],
    "4.2": [1, 2, 3],
    "4.3": [4, 5, 6],
    "4.4": [7, 8, 9, 10, 11, 12, 13],
    "4.5": [14, 15, 16],
    "4.6": [17, 18, 19, 20, 21, 22],
    "4.7": [24],
    "4.8": [23],
}

# Build chart blocks using figure-number extraction as primary method
def build_chart_block(exp_id, section_text):
    """Build the charts block for an experiment section."""
    figs = extract_figures_by_numbers(backup, exp_figures.get(exp_id, []))
    tbls = extract_tables_by_numbers(backup, exp_tables.get(exp_id, []))
    parts = []
    if tbls:
        parts.append("#### Results Tables\n\n" + tbls)
    if figs:
        parts.append("#### Results Figures\n\n" + figs)
    # If figure extraction failed, fall back to section text
    if not parts and section_text.strip():
        parts.append(section_text.strip())
    return "\n\n".join(parts)

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Insert charts into each experiment section in thesis_restructured.md
# ─────────────────────────────────────────────────────────────────────────────
chart_map = {
    "### 4.1": build_chart_block("4.1", exp1_charts),
    "### 4.2": build_chart_block("4.2", exp2_charts),
    "### 4.3": build_chart_block("4.3", exp3_charts),
    "### 4.4": build_chart_block("4.4", exp4_charts),
    "### 4.5": build_chart_block("4.5", exp5_charts),
    "### 4.6": build_chart_block("4.6", exp6_charts),
    "### 4.7": build_chart_block("4.7", exp7_charts),
    "### 4.8": build_chart_block("4.8", exp8_charts),
}

new_text = restructured

for section_header, charts in chart_map.items():
    if not charts.strip():
        print(f"WARNING: No charts found for {section_header}")
        continue

    # Find the section and insert charts right before the next ### heading
    # or before the end of the Experiments chapter
    pattern = (
        r"(" + re.escape(section_header) + r".*?)"
        r"(\*\*Conclusion:\*\*.*?)"
        r"(?=\n### |\n## )"
    )
    replacement = r"\1\2\n\n" + charts.replace("\\", "\\\\")

    new_text_attempt = re.sub(pattern, replacement, new_text, flags=re.DOTALL)
    if new_text_attempt != new_text:
        new_text = new_text_attempt
        print(f"Inserted charts for {section_header}")
    else:
        # Try simpler insertion: append after the Conclusion block
        pattern2 = (
            r"(" + re.escape(section_header) + r".*?)"
            r"(?=\n### |\n## )"
        )
        replacement2 = r"\1\n\n" + charts.replace("\\", "\\\\")
        new_text_attempt2 = re.sub(pattern2, replacement2, new_text, flags=re.DOTALL)
        if new_text_attempt2 != new_text:
            new_text = new_text_attempt2
            print(f"Inserted charts (fallback) for {section_header}")
        else:
            print(f"FAILED to insert charts for {section_header}")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Save — never touch thesis.md
# ─────────────────────────────────────────────────────────────────────────────
with open("thesis_restructured.md", "w", encoding="utf-8") as f:
    f.write(new_text)

print("\nDone. Saved to thesis_restructured.md")