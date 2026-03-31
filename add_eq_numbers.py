"""
add_eq_numbers.py
-----------------
Adds equation numbering and cross-references to thesis_restructured.md.

For each display equation ($$...$$):
  - Adds \tag{N} inside the LaTeX for visual numbering
  - Adds {#eq:eqN} after the closing $$ for pandoc-crossref

For cross-references in the text:
  - Replaces bare "Eq. N" / "Equation N" / "(N)" references with [@eq:eqN]
  - Adds ([@eq:eqN]) after "is:" / "yields:" / "as:" lines that precede equations
"""

import re

# ─────────────────────────────────────────────────────────────────────────────
# Equation labels (meaningful names for pandoc-crossref)
# ─────────────────────────────────────────────────────────────────────────────
EQ_LABELS = {
    1:  "relay-capacity-bound",
    2:  "df-capacity",
    3:  "df-gaussian-capacity",
    4:  "awgn-hop1",
    5:  "awgn-hop2",
    6:  "af-gain",
    7:  "af-snr",
    8:  "df-demod-remod",
    9:  "df-ber-hops",
    10: "optimal-denoiser",
    11: "optimal-denoiser-awgn",
    12: "bias-variance-decomp",
    13: "mse-loss",
    14: "vae-elbo",
    15: "vae-loss",
    16: "gan-minimax",
    17: "wasserstein-distance",
    18: "gradient-penalty",
    19: "wgan-gen-loss",
    20: "wgan-disc-loss",
    21: "attention",
    22: "multihead-attention",
    23: "positional-encoding",
    24: "ssm-continuous",
    25: "ssm-zoh",
    26: "ssm-discrete",
    27: "mamba-selective",
    28: "mamba-ssd-output",
    29: "mamba-ssd-matrix",
    30: "mimo-capacity",
    31: "mimo-received",
    32: "zf-equalizer",
    33: "zf-snr",
    34: "mmse-equalizer",
    35: "mmse-sinr",
    36: "awgn-channel",
    37: "awgn-ber",
    38: "df-ber-awgn",
    39: "af-snr-eff",
    40: "rayleigh-channel",
    41: "rayleigh-pdf",
    42: "rayleigh-ber",
    43: "rayleigh-ber-approx",
    44: "df-ber-rayleigh",
    45: "rician-channel",
    46: "rician-pdf",
    47: "rician-ber-mgf",
    48: "rician-mgf",
    49: "mimo-received-2x2",
    50: "zf-ber-mimo",
    51: "mmse-ber-mimo",
    52: "zf-equalizer-methods",
    53: "mmse-equalizer-methods",
    54: "two-hop-system",
    55: "channel-function",
    56: "power-normalization",
    57: "relay-received-siso",
    58: "mimo-interference",
    59: "mlp-relay",
    60: "ber-estimate",
    61: "confidence-interval",
    62: "wilcoxon-test",
    63: "bpsk-mapping",
    64: "qpsk-mapping",
    65: "qpsk-ber",
    66: "qam16-mapping",
    67: "qam16-ber",
    68: "psk16-mapping",
    69: "psk16-decision",
    70: "psk16-ber",
    71: "iq-splitting",
    72: "joint-classification",
    73: "optimal-denoiser-relay",
    74: "bias-variance-relay",
}

print("Reading thesis_restructured.md ...")
with open("thesis_restructured.md", "r", encoding="utf-8") as f:
    text = f.read()

print(f"  Input size: {len(text):,} chars")

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Find all display equation positions
# ─────────────────────────────────────────────────────────────────────────────
eq_pattern = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
matches = list(eq_pattern.finditer(text))
print(f"  Found {len(matches)} display equations")

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Build replacement map (process in reverse to preserve offsets)
# ─────────────────────────────────────────────────────────────────────────────
# We'll rebuild the text by processing matches in order
result_parts = []
prev_end = 0
eq_counter = 0

for m in matches:
    eq_counter += 1
    n = eq_counter
    label = EQ_LABELS.get(n, f"eq{n}")
    inner = m.group(1)

    # Add \tag{N} to the equation if not already tagged
    # Insert before the closing $$ — add \tag at end of equation content
    inner_stripped = inner.rstrip()

    # Don't double-tag
    if r"\tag{" not in inner_stripped:
        # For aligned/cases environments, add \tag before the closing \end
        if r"\end{aligned}" in inner_stripped:
            inner_new = inner_stripped.replace(
                r"\end{aligned}",
                r"\tag{" + str(n) + r"}" + "\n" + r"\end{aligned}",
                1
            )
        elif r"\end{cases}" in inner_stripped:
            inner_new = inner_stripped + r" \tag{" + str(n) + r"}"
        else:
            inner_new = inner_stripped + r" \tag{" + str(n) + r"}"
    else:
        inner_new = inner_stripped

    # Build the new equation block with pandoc-crossref label
    new_eq = "$$" + inner_new + "\n$$ {#eq:" + label + "}"

    # Append text before this match, then the new equation
    result_parts.append(text[prev_end:m.start()])
    result_parts.append(new_eq)
    prev_end = m.end()

# Append remaining text
result_parts.append(text[prev_end:])
text = "".join(result_parts)

print(f"  After equation numbering: {len(text):,} chars")

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Add cross-references where equations are cited in the text
# ─────────────────────────────────────────────────────────────────────────────
# Build a reverse map: label → number
label_to_num = {v: k for k, v in EQ_LABELS.items()}

# Pattern: "Eq. N" or "Equation N" or "eq. (N)" in prose
# Replace with pandoc-crossref [@eq:label]
def eq_ref(n):
    label = EQ_LABELS.get(n, f"eq{n}")
    return f"[@eq:{label}]"

# Replace explicit "Eq. N" / "Equation N" references
def replace_eq_ref(m):
    n = int(m.group(2))
    if n in EQ_LABELS:
        return m.group(1) + eq_ref(n)
    return m.group(0)

text = re.sub(
    r"(Eq(?:uation)?s?\.?\s+)\((\d+)\)",
    replace_eq_ref,
    text
)
text = re.sub(
    r"(Eq(?:uation)?s?\.?\s+)(\d+)(?!\d)",
    replace_eq_ref,
    text
)

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Add a "Key Equations" reference list at the end of each major section
#         that uses equations (Literature Review, Methods)
# ─────────────────────────────────────────────────────────────────────────────
# Build a compact equation reference table for the appendix
eq_ref_table = "\n\n## Equation Reference {.unnumbered}\n\n"
eq_ref_table += "| No. | Label | Description |\n"
eq_ref_table += "|-----|-------|-------------|\n"

eq_descriptions = {
    1:  "Relay channel capacity upper bound (Cut-Set theorem)",
    2:  "Decode-and-Forward capacity",
    3:  "DF capacity for Gaussian two-hop channel",
    4:  "Received signal at relay (AWGN, Hop 1)",
    5:  "Received signal at destination (Hop 2)",
    6:  "AF amplification gain factor",
    7:  "AF effective end-to-end SNR",
    8:  "DF demodulation and re-modulation",
    9:  "DF end-to-end BER (two independent hops)",
    10: "Optimal Bayesian denoiser (posterior mean)",
    11: "Optimal denoiser for AWGN (tanh form)",
    12: "Bias-variance-irreducible decomposition",
    13: "MSE training loss for neural relay",
    14: "VAE ELBO (evidence lower bound)",
    15: "VAE loss (reconstruction + KL regularization)",
    16: "GAN minimax objective",
    17: "Wasserstein-1 (Earth Mover) distance",
    18: "WGAN gradient penalty",
    19: "WGAN-GP generator loss",
    20: "WGAN-GP discriminator loss",
    21: "Scaled dot-product attention",
    22: "Multi-head attention",
    23: "Sinusoidal positional encoding",
    24: "Continuous-time SSM (state space model)",
    25: "ZOH discretization of SSM",
    26: "Discrete SSM recurrence",
    27: "Mamba S6 selective (input-dependent) parameters",
    28: "Mamba-2 SSD output as structured matrix-vector product",
    29: "Mamba-2 SSD semiseparable matrix",
    30: "MIMO ergodic capacity (Foschini-Telatar)",
    31: "2×2 MIMO received signal model",
    32: "Zero-Forcing (ZF) equalizer",
    33: "ZF post-equalization SNR per stream",
    34: "MMSE equalizer (Wiener filter)",
    35: "MMSE post-equalization SINR per stream",
    36: "AWGN channel model",
    37: "Theoretical BER for BPSK over AWGN",
    38: "DF BER for BPSK over AWGN (two-hop)",
    39: "AF effective SNR (equal-SNR hops)",
    40: "Rayleigh fading channel model",
    41: "Rayleigh fading amplitude PDF",
    42: "Theoretical BER for BPSK over Rayleigh fading",
    43: "High-SNR approximation: Rayleigh BER ≈ 1/(4γ̄)",
    44: "DF BER for BPSK over Rayleigh fading",
    45: "Rician fading channel model (LOS + scatter)",
    46: "Rician fading amplitude PDF",
    47: "Rician BER via MGF approach",
    48: "MGF of instantaneous SNR under Rician fading",
    49: "2×2 MIMO received signal (Rayleigh)",
    50: "ZF BER approximation for 2×2 MIMO",
    51: "MMSE BER approximation for 2×2 MIMO",
    52: "ZF equalizer (Methods chapter)",
    53: "MMSE equalizer (Methods chapter)",
    54: "Two-hop relay system diagram",
    55: "Generic channel function",
    56: "Output power normalization",
    57: "SISO relay received signal",
    58: "MIMO inter-stream interference model",
    59: "MLP relay architecture",
    60: "Monte Carlo BER estimator",
    61: "95% confidence interval for BER",
    62: "Wilcoxon signed-rank test hypothesis",
    63: "BPSK symbol mapping",
    64: "QPSK symbol mapping",
    65: "QPSK BER equals BPSK BER",
    66: "16-QAM I/Q mapping (Gray-coded PAM-4)",
    67: "Approximate BER for 16-QAM over AWGN",
    68: "16-PSK constellation mapping",
    69: "16-PSK hard decision rule",
    70: "Approximate BER for 16-PSK over AWGN",
    71: "I/Q splitting for complex relay processing",
    72: "Joint 2D classification relay output",
    73: "Optimal relay denoiser (AWGN, tanh form)",
    74: "Bias-variance decomposition for relay MSE",
}

for n in range(1, 75):
    label = EQ_LABELS.get(n, f"eq{n}")
    desc = eq_descriptions.get(n, "")
    eq_ref_table += f"| ({n}) | `{label}` | {desc} |\n"

# Insert the equation reference table before the References section
text = text.replace(
    "\n## References\n",
    eq_ref_table + "\n## References\n",
    1
)

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Write output
# ─────────────────────────────────────────────────────────────────────────────
with open("thesis_restructured.md", "w", encoding="utf-8") as f:
    f.write(text)

print(f"  Output size: {len(text):,} chars")
print("  Saved to thesis_restructured.md")

# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Verify
# ─────────────────────────────────────────────────────────────────────────────
print("\nVerification:")
with open("thesis_restructured.md", "r", encoding="utf-8") as f:
    verify = f.read()

tagged = len(re.findall(r"\\tag\{", verify))
labeled = len(re.findall(r"\{#eq:", verify))
crossrefs = len(re.findall(r"\[@eq:", verify))
print(f"  \\tag{{N}} occurrences: {tagged}")
print(f"  {{#eq:...}} labels: {labeled}")
print(f"  [@eq:...] cross-references: {crossrefs}")
print(f"  Equation Reference table: {'## Equation Reference' in verify}")

# Show first 3 equations to confirm format
eq_samples = list(re.finditer(r"\$\$(.+?)\$\$ \{#eq:([^\}]+)\}", verify, re.DOTALL))
print(f"\n  Sample equations (first 3):")
for m in eq_samples[:3]:
    inner = m.group(1).strip().replace("\n", " ")[:70]
    label = m.group(2)
    print(f"    [{label}]: {inner}...")