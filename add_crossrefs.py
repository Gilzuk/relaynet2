"""
add_crossrefs.py
----------------
Adds inline cross-references ([@eq:label]) to thesis_restructured.md
at places where equations are cited by concept in the prose.

Strategy: search for key phrases that introduce or reference specific
equations, and append the cross-reference inline.
"""

import re

with open("thesis_restructured.md", "r", encoding="utf-8") as f:
    text = f.read()

print(f"Input size: {len(text):,} chars")

# ─────────────────────────────────────────────────────────────────────────────
# Cross-reference insertion rules
# Each rule: (search_pattern, replacement_with_crossref)
# We use re.sub with a lambda to append the ref after the matched phrase.
# ─────────────────────────────────────────────────────────────────────────────

# Format: (regex_pattern, eq_label, description)
# The pattern should match the sentence/phrase that introduces or cites the eq.
# We append ([@eq:label]) right after the matched text.

CROSSREF_RULES = [
    # ── Relay capacity ──────────────────────────────────────────────────────
    (r"(capacity of any relay channel is bounded by:)",
     "relay-capacity-bound", "relay capacity bound"),
    (r"(DF capacity.*?is:)",
     "df-capacity", "DF capacity"),
    (r"(two-hop DF capacity is:)",
     "df-gaussian-capacity", "DF Gaussian capacity"),

    # ── AF / DF relay ────────────────────────────────────────────────────────
    (r"(end-to-end SNR for AF relay is:)",
     "af-snr", "AF SNR"),
    (r"(end-to-end BER for DF is:)",
     "df-ber-hops", "DF BER"),
    (r"(AF.*?amplif.*?gain.*?factor.*?:)",
     "af-gain", "AF gain"),

    # ── Optimal denoiser ─────────────────────────────────────────────────────
    (r"(posterior mean.*?reduces to:)",
     "optimal-denoiser-awgn", "optimal denoiser AWGN"),
    (r"(optimal.*?denoiser.*?maps.*?:)",
     "optimal-denoiser", "optimal denoiser"),

    # ── Bias-variance ────────────────────────────────────────────────────────
    (r"(bias.variance.irreducible.*?decomposition.*?:)",
     "bias-variance-decomp", "bias-variance decomp"),
    (r"(MSE.*?training.*?loss.*?:)",
     "mse-loss", "MSE loss"),

    # ── VAE ──────────────────────────────────────────────────────────────────
    (r"(ELBO.*?decomposition.*?:)",
     "vae-elbo", "VAE ELBO"),
    (r"(VAE.*?loss.*?reconstruction.*?regularization.*?:)",
     "vae-loss", "VAE loss"),

    # ── GAN ──────────────────────────────────────────────────────────────────
    (r"(minimax game.*?:)",
     "gan-minimax", "GAN minimax"),
    (r"(Wasserstein-1.*?distance.*?:)",
     "wasserstein-distance", "Wasserstein distance"),
    (r"(gradient.*?penalty.*?:)",
     "gradient-penalty", "gradient penalty"),

    # ── Attention / Transformer ───────────────────────────────────────────────
    (r"(scaled dot-product attention.*?:)",
     "attention", "attention"),
    (r"(multi-head attention.*?:)",
     "multihead-attention", "multi-head attention"),
    (r"(sinusoidal positional encoding.*?:)",
     "positional-encoding", "positional encoding"),

    # ── SSM / Mamba ───────────────────────────────────────────────────────────
    (r"(continuous-time.*?state space.*?:)",
     "ssm-continuous", "SSM continuous"),
    (r"(zero-order hold.*?ZOH.*?discretization.*?:)",
     "ssm-zoh", "SSM ZOH"),
    (r"(discrete.*?recurrence.*?:)",
     "ssm-discrete", "SSM discrete"),
    (r"(selective.*?parameters.*?:)",
     "mamba-selective", "Mamba selective"),

    # ── MIMO ─────────────────────────────────────────────────────────────────
    (r"(MIMO.*?ergodic capacity.*?:)",
     "mimo-capacity", "MIMO capacity"),
    (r"(2\s*[×x]\s*2 MIMO.*?received signal is:)",
     "mimo-received", "MIMO received"),
    (r"(ZF equalizer applies.*?pseudo-inverse.*?:)",
     "zf-equalizer", "ZF equalizer"),
    (r"(post-equalization SNR for stream.*?:)",
     "zf-snr", "ZF SNR"),
    (r"(Wiener filter.*?:)",
     "mmse-equalizer", "MMSE equalizer"),
    (r"(post-equalization SINR for stream.*?:)",
     "mmse-sinr", "MMSE SINR"),

    # ── Channel models ────────────────────────────────────────────────────────
    (r"(AWGN channel.*?adds.*?Gaussian noise.*?:)",
     "awgn-channel", "AWGN channel"),
    (r"(theoretical BER.*?BPSK.*?AWGN.*?is:)",
     "awgn-ber", "AWGN BER"),
    (r"(Rayleigh.*?fading.*?amplitude.*?follows.*?:)",
     "rayleigh-pdf", "Rayleigh PDF"),
    (r"(closed-form.*?Rayleigh.*?BER.*?:)",
     "rayleigh-ber", "Rayleigh BER"),
    (r"(high.SNR.*?Rayleigh.*?:)",
     "rayleigh-ber-approx", "Rayleigh BER approx"),
    (r"(Rician.*?fading.*?amplitude.*?follows.*?:)",
     "rician-pdf", "Rician PDF"),
    (r"(MGF.*?approach.*?:)",
     "rician-ber-mgf", "Rician BER MGF"),
    (r"(moment-generating function.*?MGF.*?:)",
     "rician-mgf", "Rician MGF"),

    # ── Methods ───────────────────────────────────────────────────────────────
    (r"(Monte Carlo BER.*?estimat.*?:)",
     "ber-estimate", "BER estimate"),
    (r"(95%.*?confidence interval.*?:)",
     "confidence-interval", "CI"),
    (r"(Wilcoxon.*?signed-rank.*?:)",
     "wilcoxon-test", "Wilcoxon test"),
    (r"(power.*?normali[sz].*?:)",
     "power-normalization", "power normalization"),

    # ── Modulation ────────────────────────────────────────────────────────────
    (r"(BPSK.*?maps.*?single bit.*?:)",
     "bpsk-mapping", "BPSK mapping"),
    (r"(QPSK.*?maps.*?pairs of bits.*?:)",
     "qpsk-mapping", "QPSK mapping"),
    (r"(QPSK BER equals BPSK.*?:)",
     "qpsk-ber", "QPSK BER"),
    (r"(16-QAM.*?I/Q.*?mapping.*?:)",
     "qam16-mapping", "16-QAM mapping"),
    (r"(approximate BER for 16-QAM.*?:)",
     "qam16-ber", "16-QAM BER"),
    (r"(16-PSK.*?constellation.*?mapping.*?:)",
     "psk16-mapping", "16-PSK mapping"),
    (r"(16-PSK.*?hard decision.*?:)",
     "psk16-decision", "16-PSK decision"),
    (r"(approximate BER for.*?16-PSK.*?:)",
     "psk16-ber", "16-PSK BER"),
    (r"(I/Q splitting.*?:)",
     "iq-splitting", "I/Q splitting"),
    (r"(joint.*?2D.*?classif.*?:)",
     "joint-classification", "joint classification"),
]

changes = 0
for pattern, label, desc in CROSSREF_RULES:
    ref = f" ([@eq:{label}])"
    # Only add if the ref isn't already there
    new_text, n = re.subn(
        pattern + r"(?!\s*\(\[@eq:)",  # don't double-add
        lambda m, r=ref: m.group(0) + r,
        text,
        count=1,
        flags=re.IGNORECASE | re.DOTALL
    )
    if n > 0:
        text = new_text
        changes += 1
        print(f"  + Added [@eq:{label}] ({desc})")

print(f"\nTotal cross-references added: {changes}")

# ─────────────────────────────────────────────────────────────────────────────
# Also add back-references in the Experiments chapter
# where key equations are used in conclusions
# ─────────────────────────────────────────────────────────────────────────────
# In 4.1 conclusion: reference the theoretical BER equations
exp_crossrefs = [
    # (phrase_in_experiments, label)
    (r"(AWGN follows the expected exponential decay)",
     "awgn-ber"),
    (r"(Rayleigh validates the.*?high-SNR slope)",
     "rayleigh-ber-approx"),
    (r"(ZF < MMSE < SIC performance hierarchy)",
     "zf-snr"),
    (r"(1/\(4\\bar\{\\gamma\}\) high-SNR slope)",
     "rayleigh-ber-approx"),
    (r"(per-axis error accumulation)",
     "iq-splitting"),
    (r"(I/Q splitting.*?not a fundamental limitation)",
     "iq-splitting"),
]

for pattern, label in exp_crossrefs:
    ref = f" ([@eq:{label}])"
    new_text, n = re.subn(
        pattern + r"(?!\s*\(\[@eq:)",
        lambda m, r=ref: m.group(0) + r,
        text,
        count=1,
        flags=re.IGNORECASE | re.DOTALL
    )
    if n > 0:
        text = new_text
        print(f"  + Experiments ref: [@eq:{label}]")

# ─────────────────────────────────────────────────────────────────────────────
# Write output
# ─────────────────────────────────────────────────────────────────────────────
with open("thesis_restructured.md", "w", encoding="utf-8") as f:
    f.write(text)

print(f"\nOutput size: {len(text):,} chars")
print("Saved to thesis_restructured.md")

# Verify
total_crossrefs = len(re.findall(r"\[@eq:", text))
print(f"Total [@eq:...] cross-references in document: {total_crossrefs}")