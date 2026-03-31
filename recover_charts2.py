"""
recover_charts2.py
------------------
Rebuilds the Experiments chapter in thesis_restructured.md by:
  1. Keeping the Goal / Trials / Conclusion skeleton already written.
  2. Appending the correct figures and tables from thesis_backup.md
     into each experiment sub-section.

Does NOT touch thesis.md.
"""

import re

# ── load files ────────────────────────────────────────────────────────────────
with open("thesis_backup.md", "r", encoding="utf-8") as f:
    backup = f.read()

with open("thesis_restructured.md", "r", encoding="utf-8") as f:
    restructured = f.read()

# ── helpers ───────────────────────────────────────────────────────────────────

def find_figure(text, n):
    """Return the full ![Figure N: ...](path) line for figure number n."""
    m = re.search(
        r"(!\[Figure " + str(n) + r"[^\]]*\]\([^\)]+\))",
        text
    )
    return m.group(1) if m else None

def find_figure_caption(text, n):
    """Return the *Figure N: ...* italic caption line."""
    m = re.search(
        r"(\*Figure " + str(n) + r"[^\*]+\*)",
        text
    )
    return m.group(1) if m else None

def find_table(text, n):
    """
    Return the markdown table block for Table N.
    Matches 'Table N:' (with colon) or 'Table N ' (with space) to avoid
    matching Table 13 when looking for Table 1.
    """
    # Try to find a label line followed by a markdown table
    patterns = [
        # Bold label: **Table N: ...**
        r"\*\*Table " + str(n) + r"[:\s][^\*]*\*\*[^\n]*\n((?:\|[^\n]*\n)+)",
        # Plain label: Table N: ...
        r"Table " + str(n) + r"[:\s][^\n]*\n((?:\|[^\n]*\n)+)",
        # Label on same line as table header
        r"Table " + str(n) + r"[:\s][^\n]*\n\|[^\n]*\n\|[-|]+\n((?:\|[^\n]*\n)*)",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            # Return label + table
            full = re.search(
                r"(\*?\*?Table " + str(n) + r"[:\s][^\n]*\n(?:\|[^\n]*\n)+)",
                text
            )
            if full:
                return full.group(1).strip()
    return None

def build_figures_block(fig_nums):
    """Build a markdown block with all figures for the given numbers."""
    lines = []
    for n in fig_nums:
        img = find_figure(backup, n)
        cap = find_figure_caption(backup, n)
        if img:
            lines.append(img)
        if cap:
            lines.append("\n" + cap)
        if img or cap:
            lines.append("")  # blank line between figures
    return "\n".join(lines).strip()

def build_tables_block(tbl_nums):
    """Build a markdown block with all tables for the given numbers."""
    blocks = []
    for n in tbl_nums:
        tbl = find_table(backup, n)
        if tbl:
            blocks.append(tbl)
        else:
            print(f"  WARNING: Table {n} not found in backup")
    return "\n\n".join(blocks)

# ── experiment → figure / table mapping ──────────────────────────────────────
exp_figures = {
    "4.1": list(range(1, 8)),      # Figs 1-7  (channel validation)
    "4.2": list(range(9, 12)),     # Figs 9-11 (SISO BPSK)
    "4.3": list(range(12, 15)),    # Figs 12-14 (MIMO BPSK)
    "4.4": list(range(15, 21)),    # Figs 15-20 (normalization + complexity)
    "4.5": list(range(21, 37)),    # Figs 21-36 (higher-order mod + activations)
    "4.6": list(range(37, 46)),    # Figs 37-45 (LayerNorm + CSI)
    "4.7": list(range(50, 54)),    # Figs 50-53 (16-class)
    "4.8": list(range(46, 50)),    # Figs 46-49 (E2E)
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

# ── rebuild the Experiments chapter ──────────────────────────────────────────
# Extract the skeleton Experiments chapter from the restructured file
# (the version WITHOUT charts — we'll rebuild it cleanly)

# The skeleton sections were written by refactor.py and look like:
#   ### 4.N Title
#   **Goal:** ...
#   **Trials:** ...
#   **Conclusion:** ...

# We need to:
# 1. Find the ## Experiments section
# 2. For each ### 4.N sub-section, extract the skeleton (Goal/Trials/Conclusion)
# 3. Append the charts block after the Conclusion

exp_chapter_match = re.search(
    r"(## Experiments\n)(.*?)(?=\n## Discussion)",
    restructured, re.DOTALL
)
if not exp_chapter_match:
    print("ERROR: Could not find ## Experiments chapter in restructured file")
    exit(1)

exp_header = exp_chapter_match.group(1)
exp_body   = exp_chapter_match.group(2)

# Split into sub-sections
# Each sub-section starts with ### 4.N
sub_sections = re.split(r"(?=\n### 4\.)", exp_body)

new_sub_sections = []
for sub in sub_sections:
    # Find which experiment this is
    m = re.match(r"\n### (4\.\d+)", sub)
    if not m:
        new_sub_sections.append(sub)
        continue

    exp_id = m.group(1)

    # Extract just the skeleton (Goal / Trials / Conclusion)
    # Strip any previously inserted chart blocks
    # The skeleton ends at the first "#### Results" or end of sub-section
    skeleton_match = re.match(
        r"(.*?(?:\*\*Conclusion:\*\*.*?))"
        r"(?:\n\n#### Results.*)?$",
        sub, re.DOTALL
    )
    skeleton = skeleton_match.group(1) if skeleton_match else sub

    # Build charts block
    figs_block  = build_figures_block(exp_figures.get(exp_id, []))
    tbls_block  = build_tables_block(exp_tables.get(exp_id, []))

    charts_parts = []
    if tbls_block.strip():
        charts_parts.append("#### Results Tables\n\n" + tbls_block)
    if figs_block.strip():
        charts_parts.append("#### Results Figures\n\n" + figs_block)

    charts_block = "\n\n".join(charts_parts)

    if charts_block.strip():
        new_sub = skeleton + "\n\n" + charts_block
        print(f"  {exp_id}: added {len(exp_figures.get(exp_id,[]))} figs, "
              f"{len(exp_tables.get(exp_id,[]))} tables")
    else:
        new_sub = skeleton
        print(f"  {exp_id}: WARNING — no charts found")

    new_sub_sections.append(new_sub)

new_exp_body = "".join(new_sub_sections)
new_exp_chapter = exp_header + new_exp_body

# Replace the old Experiments chapter in the restructured file
new_restructured = restructured[:exp_chapter_match.start()] + \
                   new_exp_chapter + \
                   restructured[exp_chapter_match.end():]

# ── save ─────────────────────────────────────────────────────────────────────
with open("thesis_restructured.md", "w", encoding="utf-8") as f:
    f.write(new_restructured)

print("\nSaved thesis_restructured.md")

# ── quick verification ────────────────────────────────────────────────────────
with open("thesis_restructured.md", "r", encoding="utf-8") as f:
    verify = f.read()

exp_m = re.search(r"## Experiments(.*?)## Discussion", verify, re.DOTALL)
exp_t = exp_m.group(1) if exp_m else ""
figs_found = re.findall(r"!\[Figure \d+", exp_t)
print(f"\nVerification: {len(figs_found)} figure references in Experiments chapter")

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