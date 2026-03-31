"""
add_missing.py
--------------
Adds to thesis_restructured.md:
  1. Fixes: removes [@eq:...] cross-refs from figure captions and bibliography
  2. Equation citations (paper/textbook, chapter, equation number)
  3. List of Figures
  4. List of Tables
  5. Hebrew abstract
"""

import re

print("Reading files ...")
with open("thesis_restructured.md", "r", encoding="utf-8") as f:
    text = f.read()

with open("thesis_backup.md", "r", encoding="utf-8") as f:
    backup = f.read()

print(f"  Input size: {len(text):,} chars")

# ─────────────────────────────────────────────────────────────────────────────
# 1. FIX BUGS: Remove [@eq:...] from figure captions and bibliography entries
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] Fixing cross-refs in captions and bibliography ...")

# Fix Figure 1 caption: remove ([@eq:confidence-interval]) from caption
text = re.sub(
    r"(\*Figure 1: )\(\[@eq:confidence-interval\]\) ",
    r"\1",
    text
)

# Fix bibliography [1]: remove ([@eq:qpsk-ber]) from reference text
text = re.sub(
    r'("Cooperative diversity in wireless networks: )\(\[@eq:qpsk-ber\]\) (Efficient)',
    r'\1\2',
    text
)

# General cleanup: remove any [@eq:...] that ended up inside *Figure N:...* captions
text = re.sub(
    r"(\*Figure \d+:[^\*]*?)\s*\(\[@eq:[^\]]+\]\)([^\*]*?\*)",
    r"\1\2",
    text
)

# General cleanup: remove any [@eq:...] inside bibliography [N] entries
def clean_bib_line(m):
    line = m.group(0)
    return re.sub(r"\s*\(\[@eq:[^\]]+\]\)", "", line)

text = re.sub(r"^\[\d+\][^\n]+", clean_bib_line, text, flags=re.MULTILINE)

print("  Done.")

# ─────────────────────────────────────────────────────────────────────────────
# 2. EQUATION CITATIONS
#    Add a small source note after each numbered equation block
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Adding equation citations ...")

# Map: eq_label → citation string
EQ_CITATIONS = {
    # Relay / Information Theory
    "relay-capacity-bound": "[3, Thm. 1], [4, Ch. 16]",
    "df-capacity":          "[3, Thm. 1]",
    "df-gaussian-capacity": "[1, Eq. 7], [3]",
    "awgn-hop1":            "[1, Sec. II]",
    "awgn-hop2":            "[1, Sec. II]",
    "af-gain":              "[1, Eq. 1]",
    "af-snr":               "[1, Eq. 2]",
    "df-demod-remod":       "[1, Sec. III]",
    "df-ber-hops":          "[1, Eq. 9], [21, Ch. 14]",
    # ML / Neural Networks
    "optimal-denoiser":     "[23, Ch. 5]",
    "optimal-denoiser-awgn":"[23, Ch. 5]",
    "bias-variance-decomp": "[23, Ch. 5, Eq. 5.18]",
    "mse-loss":             "[23, Ch. 5]",
    # VAE
    "vae-elbo":             "[11, Eq. 2]",
    "vae-loss":             "[11, Eq. 3]",
    # GAN / WGAN-GP
    "gan-minimax":          "[14, Eq. 1]",
    "wasserstein-distance": "[13, Eq. 1]",
    "gradient-penalty":     "[13, Eq. 3]",
    "wgan-gen-loss":        "[13, Eq. 2]",
    "wgan-disc-loss":       "[13, Eq. 2]",
    # Transformer
    "attention":            "[15, Eq. 1]",
    "multihead-attention":  "[15, Eq. 2]",
    "positional-encoding":  "[15, Sec. 3.5]",
    # SSM / Mamba
    "ssm-continuous":       "[17, Eq. 1]",
    "ssm-zoh":              "[17, Eq. 2]",
    "ssm-discrete":         "[17, Eq. 3]",
    "mamba-selective":      "[16, Eq. 4]",
    "mamba-ssd-output":     "[18, Eq. 2]",
    "mamba-ssd-matrix":     "[18, Eq. 3]",
    # MIMO
    "mimo-capacity":        "[26, Eq. 1], [27, Eq. 1]",
    "mimo-received":        "[19, Ch. 7.2]",
    "zf-equalizer":         "[19, Ch. 7.2], [29]",
    "zf-snr":               "[19, Ch. 7.2], [29]",
    "mmse-equalizer":       "[19, Ch. 7.2], [29]",
    "mmse-sinr":            "[19, Ch. 7.2], [29]",
    # Channel Models — Theoretical
    "awgn-channel":         "[21, Ch. 4]",
    "awgn-ber":             "[21, Ch. 4, Eq. 4-3-13]",
    "df-ber-awgn":          "[1, Eq. 9]",
    "af-snr-eff":           "[1, Eq. 2]",
    "rayleigh-channel":     "[21, Ch. 14]",
    "rayleigh-pdf":         "[21, Ch. 14, Eq. 14-4-1]",
    "rayleigh-ber":         "[21, Eq. 14-4-15]",
    "rayleigh-ber-approx":  "[21, Ch. 14]",
    "df-ber-rayleigh":      "[1, Eq. 9]",
    "rician-channel":       "[22, Ch. 2]",
    "rician-pdf":           "[22, Ch. 2, Eq. 2.6]",
    "rician-ber-mgf":       "[22, Ch. 8, Eq. 8.98]",
    "rician-mgf":           "[22, Ch. 8, Eq. 8.99]",
    "mimo-received-2x2":    "[19, Ch. 7.2]",
    "zf-ber-mimo":          "[19, Ch. 7.2], [28]",
    "mmse-ber-mimo":        "[19, Ch. 7.2], [29]",
    # Methods (system model — this work)
    "zf-equalizer-methods": "[19, Ch. 7.2]",
    "mmse-equalizer-methods":"[19, Ch. 7.2]",
    "two-hop-system":       "This work",
    "channel-function":     "This work",
    "power-normalization":  "[1, Eq. 1]",
    "relay-received-siso":  "This work",
    "mimo-interference":    "[19, Ch. 7.2]",
    "mlp-relay":            "[23, Ch. 6]",
    # Simulation
    "ber-estimate":         "[21, Ch. 14]",
    "confidence-interval":  "Standard statistics",
    "wilcoxon-test":        "Standard statistics",
    # Modulation
    "bpsk-mapping":         "[21, Ch. 4, Eq. 4-3-1]",
    "qpsk-mapping":         "[21, Ch. 4, Eq. 4-3-30]",
    "qpsk-ber":             "[21, Ch. 4, Eq. 4-3-31]",
    "qam16-mapping":        "[21, Ch. 5, Eq. 5-2-60]",
    "qam16-ber":            "[21, Ch. 5, Eq. 5-2-79]",
    "psk16-mapping":        "[21, Ch. 5, Eq. 5-2-1]",
    "psk16-decision":       "[21, Ch. 5]",
    "psk16-ber":            "[21, Ch. 5, Eq. 5-2-22]",
    # This work
    "iq-splitting":         "This work",
    "joint-classification": "This work",
    "optimal-denoiser-relay":"[23, Ch. 5]",
    "bias-variance-relay":  "[23, Ch. 5, Eq. 5.18]",
}

def add_eq_citation(m):
    label = m.group(1)
    citation = EQ_CITATIONS.get(label, "")
    if citation and citation != "This work":
        return m.group(0) + f"\n*Source: {citation}*"
    return m.group(0)

text = re.sub(
    r"\$\$ \{#eq:([^\}]+)\}",
    add_eq_citation,
    text
)

citations_added = len(re.findall(r"\n\*Source:", text))
print(f"  Added {citations_added} equation citations")

# ─────────────────────────────────────────────────────────────────────────────
# 3. LIST OF FIGURES
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Building List of Figures ...")

# Extract all unique figure captions (deduplicate by number)
fig_captions = {}
for m in re.finditer(r"\*Figure (\d+): ([^\*]+)\*", text):
    n = int(m.group(1))
    cap = m.group(2).strip()
    if n not in fig_captions:
        fig_captions[n] = cap

# Short summary: first sentence or first 80 chars
def short_cap(cap):
    # Remove LaTeX math for brevity
    cap = re.sub(r"\$[^\$]+\$", "...", cap)
    # First sentence
    m = re.match(r"([^\.]+\.)", cap)
    if m and len(m.group(1)) < 120:
        return m.group(1).strip()
    return cap[:90].rstrip(",;") + "..."

lof_lines = [
    "\n## List of Figures {.unnumbered}\n",
    "| Figure | Short Description | Section |",
    "|--------|-------------------|---------|",
]

# Map figure numbers to sections
fig_section_map = {
    **{n: "§4.1 Channel Validation" for n in range(1, 9)},
    **{n: "§4.2 SISO BPSK" for n in range(9, 12)},
    **{n: "§4.3 MIMO 2×2" for n in range(12, 15)},
    **{n: "§4.4 Normalization" for n in range(15, 21)},
    **{n: "§4.5 Higher-Order Mod." for n in range(21, 37)},
    **{n: "§4.6 CSI Injection" for n in range(37, 46)},
    **{n: "§4.8 E2E" for n in range(46, 50)},
    **{n: "§4.7 16-Class" for n in range(50, 54)},
    8: "§3 Methods",
}

for n in sorted(fig_captions.keys()):
    cap = short_cap(fig_captions[n])
    sec = fig_section_map.get(n, "§4")
    lof_lines.append(f"| {n} | {cap} | {sec} |")

lof_text = "\n".join(lof_lines) + "\n"
print(f"  {len(fig_captions)} unique figures")

# ─────────────────────────────────────────────────────────────────────────────
# 4. LIST OF TABLES
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] Building List of Tables ...")

# Extract all unique table labels
tbl_labels = {}
for m in re.finditer(r"Table (\d+): ([^\n]+)", text):
    n = int(m.group(1))
    label = m.group(2).strip()
    # Skip if it's a cross-ref or very short
    if n not in tbl_labels and len(label) > 10:
        tbl_labels[n] = label

tbl_section_map = {
    1: "§4.2", 2: "§4.2", 3: "§4.2",
    4: "§4.3", 5: "§4.3", 6: "§4.3",
    7: "§4.4", 8: "§4.4", 9: "§4.4",
    10: "§4.4", 11: "§4.4", 12: "§4.4", 13: "§4.4",
    14: "§4.5", 15: "§4.5", 16: "§4.5",
    17: "§4.6", 18: "§4.6", 19: "§4.6",
    20: "§4.6", 21: "§4.6", 22: "§4.6",
    23: "§4.8", 24: "§4.7",
}

lot_lines = [
    "\n## List of Tables {.unnumbered}\n",
    "| Table | Description | Section |",
    "|-------|-------------|---------|",
]

for n in sorted(tbl_labels.keys()):
    label = tbl_labels[n]
    # Truncate
    short = label[:90].rstrip(",;")
    if len(label) > 90:
        short += "..."
    sec = tbl_section_map.get(n, "§4")
    lot_lines.append(f"| {n} | {short} | {sec} |")

lot_text = "\n".join(lot_lines) + "\n"
print(f"  {len(tbl_labels)} unique tables")

# ─────────────────────────────────────────────────────────────────────────────
# 5. HEBREW ABSTRACT
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] Extracting Hebrew abstract ...")

heb_m = re.search(r'(<div dir="rtl">.*?</div>)', backup, re.DOTALL)
if heb_m:
    heb_abstract = heb_m.group(1)
    print(f"  Found Hebrew abstract ({len(heb_abstract)} chars)")
else:
    heb_abstract = ""
    print("  WARNING: Hebrew abstract not found in backup")

heb_section = f"""
## Abstract (Hebrew) {{.unnumbered}}

{heb_abstract}

"""

# ─────────────────────────────────────────────────────────────────────────────
# 6. INSERT ALL ADDITIONS INTO DOCUMENT
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] Inserting additions into document ...")

# Insert List of Figures and List of Tables after the TOC
# and before the Introduction
toc_end = text.find("\n## Introduction and Literature Review")
if toc_end > 0:
    text = (
        text[:toc_end]
        + lof_text
        + lot_text
        + text[toc_end:]
    )
    print("  Inserted List of Figures and List of Tables after TOC")

# Insert Hebrew abstract in the front matter
# It should go after the TOC front matter table (before the Body section)
# Find the front matter section (before Introduction)
# The TOC has "| II | Abstract (Hebrew) |" — replace that with a link
text = text.replace(
    "| II | Abstract (Hebrew) |",
    "| II | [Abstract (Hebrew)](#abstract-hebrew) |"
)

# Insert Hebrew abstract section before the English abstract at the end
# or after the TOC
eng_abs_idx = text.find("\n## Abstract (English)")
if eng_abs_idx > 0:
    text = text[:eng_abs_idx] + heb_section + text[eng_abs_idx:]
    print("  Inserted Hebrew abstract before English abstract")
else:
    # Insert before References
    refs_idx = text.find("\n## References")
    if refs_idx > 0:
        text = text[:refs_idx] + heb_section + text[refs_idx:]
        print("  Inserted Hebrew abstract before References")

# ─────────────────────────────────────────────────────────────────────────────
# 7. WRITE OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
with open("thesis_restructured.md", "w", encoding="utf-8") as f:
    f.write(text)

print(f"\nOutput size: {len(text):,} chars")
print("Saved to thesis_restructured.md")

# ─────────────────────────────────────────────────────────────────────────────
# 8. VERIFY
# ─────────────────────────────────────────────────────────────────────────────
print("\nVerification:")
with open("thesis_restructured.md", "r", encoding="utf-8") as f:
    verify = f.read()

print(f"  List of Figures: {'## List of Figures' in verify}")
print(f"  List of Tables: {'## List of Tables' in verify}")
print(f"  Hebrew abstract: {'Abstract (Hebrew)' in verify}")
print(f"  Hebrew text: {'תקציר' in verify or 'ממסר' in verify}")
source_count = len(re.findall("\n\\*Source:", verify))
print(f"  Equation citations (Source:): {source_count}")
cap_bugs = len(re.findall(r"\*Figure \d+[^\*]*\[@eq:", verify))
bib_bugs = len(re.findall(r"\[\d+\][^\n]*\[@eq:", verify))
print(f"  Remaining [@eq:] in captions: {cap_bugs}")
print(f"  Remaining [@eq:] in bibliography: {bib_bugs}")

# Check headings
headings = re.findall(r"^## .+", verify, re.MULTILINE)
print(f"\n  ## headings: {headings}")