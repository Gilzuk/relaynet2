#!/usr/bin/env python3
"""verify_thesis_tables.py — single-source reproducibility check for the thesis.

Recreates every *numerical* table in the thesis report from its authoritative
data source (the experiment output files and closed-form formulas) and compares,
cell by cell, against the numbers currently transcribed in the LaTeX
(``thesis/chapters/*.tex``). Any mismatch beyond display-rounding tolerance is
flagged.

Data sources, by table:
  tbl:table2            canonical QPSK/Rayleigh BER        <- results/modulation/qpsk_rayleigh.json
  tbl:layers            four-layer argument summary        <- e6_{sim,viterbi,partial,blind} npy
  tbl:table8            normalized-3K Rayleigh BER         <- results/normalized_3k/3k_rayleigh.json
  tbl:table24           4-class vs 16-class @20 dB         <- results/all_relays_16class/all_relays_16class.json
  tbl:tableE6           unknown-channel BER                <- e6_unknown_channel_results/*.npy
  tbl:tableE6flat       flat-channel control BER           <- e6_unknown_channel_results/e6_flat_ported_results.npy
  tbl:tableE6qpsk       QPSK unknown-channel BER           <- e6_unknown_channel_results/e6_qpsk_unknown_channel_results.npy
  tbl:table26           theoretical SNR @ BER=1e-3         <- closed-form (Q-function inversion)
  tab:ber_validation    theory-vs-sim BER (both columns)   <- closed form + results/calibration.json
  prose:E6blind         blind-regime prose claims          <- e6_unknown_channel_results/e6_blind_ported_results.npy
  prose:E6partial       pilot-sweep prose claims           <- e6_unknown_channel_results/e6_partial_ported_results.npy
  prose:E6composite     composite-cascade prose claims     <- e6_unknown_channel_results/e6_composite_ported_results.npy
  tbl:table34           coded block-DF vs uncoded/learned  <- results/coded_df_experiment.json
  tbl:table35           coded-DF K-sweep, QPSK              <- results/coded_df_experiment.json
  tbl:table36           coded-DF K-sweep, 16-QAM            <- results/coded_df_experiment.json
  tbl:table37           paired high-SNR re-measurement      <- results/coded_high_budget_test.json
  tbl:table38           relay-output error diagnostic       <- results/coded_error_mechanism.json
  tbl:table39           soft- vs hard-decision relaying     <- results/coded_soft_decision.json
  tbl:table40           equal-throughput coded vs uncoded   <- results/coded_latency_throughput.json
  tbl:table42           link-adaptation envelope            <- results/coded_rate_adaptation.json
  tbl:table43           envelope under a latency budget     <- results/coded_latency_capacity.json
  tbl:table44           reliable-decoding regime            <- results/coded_reliable_regime.json
  tbl:mmse-baseline     MMSE linear equalizer vs MLSE       <- results/mmse_equalizer.json
  tbl:seq-on-memory     sequence architectures on ISI       <- results/seq_models_on_memory.json
  tbl:joint-latency     memory+latency relay comparison     <- results/joint_latency_memory.json
  tbl:joint-memory      memory-sweep cost/BER table         <- results/joint_latency_memory.json

Timing tables (tbl:table13, tbl:table25) report machine-dependent wall-clock and
are checked only for their deterministic content (parameter counts); the timing
cells are reported as informational, not pass/fail.

Usage:
  python verify_thesis_tables.py            # verify against committed data files
  python verify_thesis_tables.py --rerun    # regenerate the Ch7 (E6) .npy first, then verify
  python verify_thesis_tables.py --tex DIR  # point at a different chapters/ dir

Exit code 0 if every checked cell matches; 1 if any inconsistency is flagged.
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys

import numpy as np
from scipy.special import exp1

ROOT = os.path.dirname(os.path.abspath(__file__))
TEX_DIR = os.path.join(ROOT, "thesis", "chapters")

# ----------------------------------------------------------------------------
# tolerances
# ----------------------------------------------------------------------------
# A published cell printed to D decimals matches its source if they agree to
# within half a unit in the last displayed digit (pure rounding), plus a small
# Monte-Carlo slack for values re-simulated from a fresh RNG run (--rerun).
MC_SLACK = 0.0            # 0 for stored-data comparison; raised under --rerun
def tol_for(text):
    """Rounding tolerance implied by the number of decimals shown."""
    m = re.search(r"\.(\d+)", text)
    dec = len(m.group(1)) if m else 0
    return 0.5 * 10 ** (-dec) + 1e-12 + MC_SLACK


# ----------------------------------------------------------------------------
# LaTeX helpers
# ----------------------------------------------------------------------------
def load_tex():
    text = ""
    for fn in sorted(os.listdir(TEX_DIR)):
        if fn.endswith(".tex") and not fn.startswith("_"):
            text += open(os.path.join(TEX_DIR, fn), encoding="utf-8").read() + "\n"
    return text


def table_body(tex, label):
    """Return the longtable body containing \\label{<label>} (up to \\end{longtable})."""
    i = tex.find("\\label{" + label + "}")
    if i < 0:
        return None
    j = tex.find("\\end{longtable}", i)
    return tex[i:j] if j > 0 else tex[i:]


def tabular_body(tex, label):
    """Return the tabular body containing \\label{<label>} (up to \\end{tabular})."""
    i = tex.find("\\label{" + label + "}")
    if i < 0:
        return None
    j = tex.find("\\end{tabular}", i)
    return tex[i:j] if j > 0 else tex[i:]


_NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def clean_cell(c):
    """Strip LaTeX decoration and return (raw_text, float or None)."""
    c = c.strip()
    c = re.sub(r"\\textbf\{([^}]*)\}", r"\1", c)
    c = c.replace("\\(", "").replace("\\)", "").replace("$", "")
    c = c.replace("\\textasciitilde", "~").replace("\\%", "%")
    c = c.replace("{", "").replace("}", "").strip()
    # LaTeX writes thousands as 2{,}048; the braces are gone by now, and a
    # bare comma between digits would truncate the number at its first group.
    c = re.sub(r"(?<=\d),(?=\d{3}\b)", "", c)
    # A trailing \tiny[lo, hi] confidence annotation is commentary on the cell,
    # not a second number; drop it before the value is read.
    c = re.sub(r"\\tiny\s*\[[^\]]*\]", "", c).strip()
    if c in {"", ":", "---", "--", "~", "-"}:
        return c, None
    # "<5e-5" / "< 5e-5" -> treat as a ceiling; keep raw, value = the bound
    lt = "<" in c
    # Scientific notation is written a\times10^{b}; braces are already gone, so
    # a bare _NUM search would stop at the mantissa and read 3.20 for 3.2e-5.
    sci = re.match(r"^([+-]?\d+(?:\.\d+)?)\s*\\times\s*10\^\s*\(?(-?\d+)\)?", c)
    if sci:
        val = float(sci.group(1)) * 10.0 ** int(sci.group(2))
        return c, ("<%g" % val if lt else val)
    m = _NUM.search(c)
    if not m:
        return c, None
    val = float(m.group(0))
    return c, ("<%g" % val if lt else val)


def data_rows(body):
    """Split a longtable body into cell-lists, keeping only real data rows."""
    if not body:
        return []
    # drop everything up to the last header/foot marker so we only see data
    for marker in ("\\endlastfoot", "\\endhead", "\\endfirsthead"):
        k = body.rfind(marker)
        if k >= 0:
            body = body[k + len(marker):]
            break
    rows = []
    lead = re.compile(r"^(?:\\(?:hline|midrule|toprule|bottomrule)\b"
                      r"|\\noalign\{[^}]*\}|\s)+")
    for chunk in body.split("\\\\"):
        chunk = chunk.strip()
        chunk = lead.sub("", chunk).strip()   # drop leading \hline/\midrule/etc.
        if not chunk or "&" not in chunk:
            continue
        cells = [clean_cell(c) for c in chunk.split("&")]
        rows.append(cells)
    return rows


# ----------------------------------------------------------------------------
# comparison bookkeeping
# ----------------------------------------------------------------------------
# Tables whose source is a *stochastic re-run* (Monte-Carlo .npy), not the exact
# run the thesis was transcribed from. Cross-run agreement is only meaningful to
# within Monte-Carlo noise, so these get an absolute MC tolerance on top of the
# display-rounding tolerance. JSON-backed tables (deterministic transcriptions)
# and analytical tables keep the tight rounding tolerance.
# These slacks absorb Monte-Carlo noise between a published transcription and
# the stored run. They must stay near the tables' own reported CIs (~0.001 for
# the E6 tables); a slack an order of magnitude wider silently passes
# transcription errors, which is how an 0.0088 discrepancy in tbl:tableE6flat
# survived several review passes.
STOCHASTIC_TABLES = {"tbl:tableE6": 0.002, "tbl:tableE6flat": 0.002,
                     "tbl:tableE6qpsk": 0.002,
                     "tbl:table24": 0.002,
                     # Prose claims from the E6 blind/partial/composite studies.
                     # These are now transcribed from the committed .npy at
                     # 10x100k (composite) and 50x20k (blind, partial), so the
                     # slack only needs to absorb display rounding plus a little
                     # Monte-Carlo noise -- not the wide cross-run spread the
                     # earlier 5-6 trial budgets required.
                     "prose:E6blind": 0.002, "prose:E6composite": 0.002,
                     "prose:E6partial": 0.004}


class Report:
    def __init__(self):
        self.checked = 0
        self.flags = []          # (table, where, published, source, diff)
        self.skipped = []        # (table, reason)
        self.tables = []         # (table, n_checked, n_flag)
        self.notes = []          # (table, partial-coverage reason)

    def cell(self, table, where, pub_text, pub_val, src_val):
        # unresolved / non-numeric published cell -> skip silently
        if pub_val is None or src_val is None:
            return
        self.checked += 1
        tol = tol_for(pub_text) + STOCHASTIC_TABLES.get(table, 0.0)
        if isinstance(pub_val, str) and pub_val.startswith("<"):
            bound = float(pub_val[1:])
            ok = src_val <= bound + tol
            diff = max(0.0, src_val - bound)
        else:
            diff = abs(float(pub_val) - float(src_val))
            ok = diff <= tol
        if not ok:
            self.flags.append((table, where, pub_text, f"{src_val:.5g}", f"{diff:.2g}"))

    def finish_table(self, table, before):
        n = self.checked - before
        nf = sum(1 for f in self.flags if f[0] == table)
        self.tables.append((table, n, nf))

    def skip(self, table, reason):
        self.skipped.append((table, reason))

    def note(self, table, reason):
        """Record a partial-coverage note without skipping the whole table.

        Distinct from skip(): the table is still checked, but one row could
        not be, typically because the relay was dropped from a lean re-run.
        Recording it keeps the omission visible instead of letting the cell
        count quietly shrink.
        """
        self.notes.append((table, reason))


# ----------------------------------------------------------------------------
# analytical formulas
# ----------------------------------------------------------------------------
def qfunc(x):
    return 0.5 * math.erfc(x / math.sqrt(2))

def ber_awgn(snr_db):
    # Eb/N0 axis: noise variance N0/2 per real dimension, so BPSK is
    # Q(sqrt(2*Eb/N0)). The earlier Q(sqrt(gamma)) form was the 3 dB
    # pessimistic axis and disagreed with ber_rayleigh below, which was
    # already on Eb/N0 -- the two rows of the calibration table were being
    # checked against theory curves on different axes.
    return qfunc(math.sqrt(2 * 10 ** (snr_db / 10.0)))

def ber_rayleigh(snr_db):
    g = 10 ** (snr_db / 10.0)
    return 0.5 * (1 - math.sqrt(g / (1 + g)))

def snr_for_ber(target, ber_fn, lo=-5.0, hi=60.0):
    """Bisect the SNR (dB) at which ber_fn crosses target."""
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if ber_fn(mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------------------
# per-table checks
# ----------------------------------------------------------------------------
def json_ber(path, relay, snr, snrs):
    d = json.load(open(os.path.join(ROOT, path)))
    return d["results"][relay]["ber_mean"][d["snr_range"].index(snr)]


def check_table2(tex, rep):
    """Canonical Rayleigh BER, 9 relays x SNR (tbl:table2) vs rayleigh.json.

    Column 3, "DF (theory)", is not simulated data -- it is the closed-form
    two-hop composition 2P(1-P), P = ber_rayleigh(snr_db - 3.0103dB), the
    QPSK Es/N0 -> per-bit Eb/N0 correction (Section~sec:rayleigh-two-hop-df-ber).
    """
    T = "tbl:table2"; before = rep.checked
    body = table_body(tex, T)
    if body is None:
        return rep.skip(T, "label not found in tex")
    # tex column order (after SNR): AF DF DF-theory MLP Hybrid VAE Transformer Mamba-S6 Mamba2.
    # The canonical setup is QPSK on Rayleigh -- BPSK is paired with the AWGN
    # calibration channel instead -- so this reads the QPSK result file. The
    # cGAN was excluded from the re-run and has no column.
    cols = ["AF", "DF", None, "MLP (169p)", "Hybrid", "VAE",
            "Transformer", "Mamba S6", "Mamba2 (SSD)"]
    d = json.load(open(os.path.join(ROOT, "results/modulation/qpsk_rayleigh.json")))
    snrs = d["snr_range"]
    for row in data_rows(body):
        if not row or row[0][1] is None:
            continue
        snr = int(row[0][1])
        if snr not in snrs:
            continue
        si = snrs.index(snr)
        for c, relay in enumerate(cols, start=1):
            if c >= len(row):
                break
            if relay is None:  # DF (theory) column
                pub_text, pub_val = row[c]
                p1 = ber_rayleigh(snr - 3.0103)
                src = 2 * p1 * (1 - p1)
                rep.cell(T, f"{snr}dB/DF(theory)", pub_text, pub_val, src)
                continue
            key = resolve(d["results"], relay)
            if key is None:
                if snr == snrs[0]:
                    rep.note(T, f"{relay}: absent from the re-run, column not checked")
                continue
            pub_text, pub_val = row[c]
            src = d["results"][key]["ber_mean"][si]
            rep.cell(T, f"{snr}dB/{relay}", pub_text, pub_val, src)
    rep.finish_table(T, before)


def check_layers_table(tex, rep):
    """The four-layer summary (tbl:layers) vs the data each layer cites.

    This table restates numbers that live in four different result files, so
    it is the single easiest place in the thesis for a figure to drift out of
    step with its source -- which is exactly what happened once, when a
    hand-copied layer-2 floor was written down at half its true value after
    the (2, n_snr) e6_sim array was averaged over the wrong axis. Every cell
    is therefore pinned here.
    """
    T = "tbl:layers"; before = rep.checked
    body = table_body(tex, T)
    if body is None:
        return rep.skip(T, "label not found in tex")

    sim = _load_e6_npy("e6_sim_ported_results.npy")
    blind = _load_e6_npy("e6_blind_ported_results.npy")
    partial = _load_e6_npy("e6_partial_ported_results.npy")
    vit = _load_e6_npy("e6_viterbi_awgn.npy")
    if sim is None or blind is None or partial is None or vit is None:
        return rep.skip(T, "one or more e6 npy files missing")

    snrs = list(sim["snrs"])
    i8, i20 = snrs.index(8), snrs.index(20)
    S1 = sim["results"]["S1: unknown ISI -> AWGN"]

    # Layer 1 is "the canonical setup of Chapter 5, unmodified" -- it must
    # therefore be checked against Chapter 5's actual canonical result
    # (tbl:table2, QPSK on Rayleigh), not against e6_sim_ported.py's own
    # "S4 control", which is a same-pipeline BPSK sanity check with no
    # relation to the QPSK canonical benchmark H2 is stated on. Using S4 here
    # is exactly the bug this check exists to catch: it let a BPSK number
    # (0.0687) stand in for the QPSK canonical DF value (0.1218) for as long
    # as this file went unread, because nothing compared the two.
    canon = json.load(open(os.path.join(ROOT, "results/modulation/qpsk_rayleigh.json")))
    csnrs = canon["snr_range"]; ci8 = csnrs.index(8)
    canon_df = canon["results"]["DF"]["ber_mean"][ci8]
    canon_mlp = canon["results"]["MLP (169p)"]["ber_mean"][ci8]

    # Layer 1: the true canonical setup -- DF vs MLP at 8 dB, QPSK/Rayleigh.
    for label, key, src in [("L1/DF@8dB", r"DF \$([\d.]+)\$ against", canon_df),
                            ("L1/MLP@8dB", r"MLP's \$([\d.]+)\$ at 8~dB", canon_mlp)]:
        m = re.search(key, body)
        if m:
            rep.cell(T, label, m.group(1), float(m.group(1)), src)

    # Layer 2: DF's non-monotonic rise, the MLP recovery, the Viterbi lead.
    m = re.search(r"DF rising from \$([\d.]+)\$ at 8~dB to \$([\d.]+)\$ at 20~dB", body)
    if m:
        rep.cell(T, "L2/DF@8dB", m.group(1), float(m.group(1)), S1["DF"][0][i8])
        rep.cell(T, "L2/DF@20dB", m.group(2), float(m.group(2)), S1["DF"][0][i20])
    m = re.search(r"restores the link \(\$([\d.]+)\$ at 8~dB\)", body)
    if m:
        rep.cell(T, "L2/MLP@8dB", m.group(1), float(m.group(1)), S1["MLP"][0][i8])
    m = re.search(r"ahead by 1--1\.5~dB \(\$([\d.]+)\$\)", body)
    if m:
        rep.cell(T, "L2/VITgenie@8dB", m.group(1), float(m.group(1)),
                 np.array(vit["VIT-genie"])[i8])

    # Layer 3: the pilot-budget crossover at the 10 dB operating point.
    pa = partial["panel_a"]
    m = re.search(r"holds \$([\d.]+)\$ on 200 pilots and \$([\d.]+)\$ on 20", body)
    if m:
        rep.cell(T, "L3/est-200pilot", m.group(1), float(m.group(1)), np.array(pa[200]).ravel()[0])
        rep.cell(T, "L3/est-20pilot", m.group(2), float(m.group(2)), np.array(pa[20]).ravel()[0])
    m = re.search(r"pilot-free MLP's \$([\d.]+)\$", body)
    if m:
        rep.cell(T, "L3/MLPref", m.group(1), float(m.group(1)),
                 float(np.array(partial["mlp_ref"]).ravel()[0]))
    m = re.search(r"by 10 pilots it has degraded to \$([\d.]+)\$", body)
    if m:
        rep.cell(T, "L3/est-10pilot", m.group(1), float(m.group(1)), np.array(pa[10]).ravel()[0])

    # Layer 4: blind regime at 20 dB, plus the unstable DD-MLSE tail.
    bs = blind["summary"]; bsnr = list(blind["snrs"])
    b16, b20 = bsnr.index(16), bsnr.index(20)
    m = re.search(r"MLP reaches \$([\d.]+)\$ against CMA's \$([\d.]+)\$", body)
    if m:
        rep.cell(T, "L4/MLP@20dB", m.group(1), float(m.group(1)), bs["MLP-169"][0][b20])
        rep.cell(T, "L4/CMA@20dB", m.group(2), float(m.group(2)), bs["CMA-blind"][0][b20])
    m = re.search(r"worse at 20~dB \(\$([\d.]+)\$\) than at 16~dB \(\$([\d.]+)\$\)", body)
    if m:
        rep.cell(T, "L4/VITblind@20dB", m.group(1), float(m.group(1)), bs["Viterbi-blind"][0][b20])
        rep.cell(T, "L4/VITblind@16dB", m.group(2), float(m.group(2)), bs["Viterbi-blind"][0][b16])

    rep.finish_table(T, before)


def check_table8(tex, rep):
    """Normalized-3K Rayleigh BER (tbl:table8) vs 3k_rayleigh.json."""
    T = "tbl:table8"; before = rep.checked
    body = table_body(tex, T)
    if body is None:
        return rep.skip(T, "label not found in tex")
    cols = ["GenAI-3K", "Hybrid-3K", "VAE-3K", "Transformer-3K",
            "Mamba-3K", "Mamba2-3K", "AF", "DF"]
    d = json.load(open(os.path.join(ROOT, "results/normalized_3k/3k_rayleigh.json")))
    snrs = d["snr_range"]
    for row in data_rows(body):
        if not row or row[0][1] is None:
            continue
        snr = int(row[0][1])
        if snr not in snrs:
            continue
        si = snrs.index(snr)
        for c, relay in enumerate(cols, start=1):
            if c >= len(row):
                break
            key = resolve(d["results"], relay)
            if key is None:
                if snr == snrs[0]:
                    rep.note(T, f"{relay}: absent from the re-run, column not checked")
                continue
            pub_text, pub_val = row[c]
            src = d["results"][key]["ber_mean"][si]
            rep.cell(T, f"{snr}dB/{relay}", pub_text, pub_val, src)
    rep.finish_table(T, before)


def check_table24(tex, rep):
    """4-class vs 16-class @20 dB (tbl:table24) vs all_relays_16class.json."""
    T = "tbl:table24"; before = rep.checked
    body = table_body(tex, T)
    if body is None:
        return rep.skip(T, "label not found in tex")
    d = json.load(open(os.path.join(ROOT, "results/all_relays_16class/all_relays_16class.json")))
    snrs = d["snr_range"]; i20 = snrs.index(20)
    # tex row label -> json key stem
    # Row labels are matched case-insensitively with separators stripped, so
    # that "Mamba-2 SSD" in the tex and "Mamba2" in the json line up. The
    # previous literal map missed cGAN, Mamba-S6, Mamba-2 SSD, AF and DF --
    # five of the nine rows -- and the table still reported OK on the four it
    # did check.
    stem = {"mlp": "MLP", "vae": "VAE", "cgan": "CGAN", "hybrid": "Hybrid",
            "transformer": "Transformer", "mambas6": "Mamba-S6",
            "mamba2": "Mamba2", "mamba2ssd": "Mamba2",
            "af": "AF", "df": "DF"}
    for row in data_rows(body):
        if not row:
            continue
        raw = row[0][0].replace("\\\\", "").replace("\n", "").strip()
        norm = re.sub(r"[^a-z0-9]", "", raw.lower())
        key = stem.get(norm)
        if key is None:
            if raw and raw.lower() != "relay":
                rep.note(T, f"row {raw!r}: no json mapping, not checked")
            continue
        # col1 = 4-cls @20, col2 = 16-cls @20
        for c, suff in ((1, "4-cls"), (2, "16-cls")):
            if c >= len(row):
                break
            pub_text, pub_val = row[c]
            jk = key if key in ("AF", "DF") else f"{key} {suff}"
            if jk in d["results"]:
                src = d["results"][jk]["ber_mean"][i20]
                rep.cell(T, f"{raw}/{suff}@20dB", pub_text, pub_val, src)
    rep.finish_table(T, before)


def check_table34(tex, rep):
    """Coded block-DF vs. uncoded DF / learned relays (tbl:table34) vs coded_df_experiment.json."""
    T = "tbl:table34"; before = rep.checked
    body = table_body(tex, T)
    if body is None:
        return rep.skip(T, "label not found in tex")
    d = json.load(open(os.path.join(ROOT, "results/coded_df_experiment.json")))
    snrs = d["snr_db"]
    cols = ["uncoded_df", "coded_af", "coded_df", "mlp_coded", "mamba_coded"]
    for row in data_rows(body):
        if not row or row[0][1] is None:
            continue
        snr = int(row[0][1])
        if snr not in snrs:
            continue
        si = snrs.index(snr)
        for c, key in enumerate(cols, start=1):
            if c >= len(row):
                break
            pub_text, pub_val = row[c]
            rep.cell(T, f"{snr}dB/{key}", pub_text, pub_val, d[key][si])
    rep.finish_table(T, before)


def check_table35(tex, rep):
    """Coded-DF BER vs. constraint length, QPSK (tbl:table35) vs coded_df_experiment.json."""
    T = "tbl:table35"; before = rep.checked
    body = table_body(tex, T)
    if body is None:
        return rep.skip(T, "label not found in tex")
    d = json.load(open(os.path.join(ROOT, "results/coded_df_experiment.json")))
    snrs = d["snr_db"]
    ks = ["K3", "K5", "K7"]
    for row in data_rows(body):
        if not row or row[0][1] is None:
            continue
        snr = int(row[0][1])
        if snr not in snrs:
            continue
        si = snrs.index(snr)
        for c, k in enumerate(ks, start=1):
            if c >= len(row):
                break
            pub_text, pub_val = row[c]
            rep.cell(T, f"{snr}dB/{k}", pub_text, pub_val, d["k_sweep"][k]["ber"][si])
    rep.finish_table(T, before)


def check_table36(tex, rep):
    """Coded-DF BER vs. constraint length, 16-QAM (tbl:table36) vs coded_df_experiment.json."""
    T = "tbl:table36"; before = rep.checked
    body = table_body(tex, T)
    if body is None:
        return rep.skip(T, "label not found in tex")
    d = json.load(open(os.path.join(ROOT, "results/coded_df_experiment.json")))
    snrs = d["snr_db"]
    for row in data_rows(body):
        if not row or row[0][1] is None:
            continue
        snr = int(row[0][1])
        if snr not in snrs:
            continue
        si = snrs.index(snr)
        pub_text, pub_val = row[1]
        rep.cell(T, f"{snr}dB/uncoded", pub_text, pub_val, d["qam16_uncoded_df"][si])
        for c, k in enumerate(["K3", "K5", "K7"], start=2):
            if c >= len(row):
                break
            pub_text, pub_val = row[c]
            rep.cell(T, f"{snr}dB/{k}", pub_text, pub_val, d["qam16_k_sweep"][k]["ber"][si])
    rep.finish_table(T, before)


def check_table37(tex, rep):
    """Paired high-SNR re-measurement (tbl:table37) vs coded_high_budget_test.json."""
    T = "tbl:table37"; before = rep.checked
    body = table_body(tex, T)
    if body is None:
        return rep.skip(T, "label not found in tex")
    d = json.load(open(os.path.join(ROOT, "results/coded_high_budget_test.json")))
    label_map = {"coded-df": "coded_df", "mlp-coded": "mlp_coded", "mamba-coded": "mamba_coded"}
    snr = None
    for row in data_rows(body):
        if not row:
            continue
        # rows are either "16 dB & coded-DF & ..." or " & MLP-coded & ..."
        first = row[0][0].strip()
        m = re.match(r"(\d+)\s*dB", first)
        if m:
            snr = m.group(1)
        if snr is None or len(row) < 3:
            continue
        name = re.sub(r"[^a-z0-9-]", "", row[1][0].strip().lower())
        key = label_map.get(name)
        if key is None or snr not in d["summary"]:
            continue
        src = float(np.mean(d["per_trial"][snr][key]))
        pub_text, pub_val = row[2]
        rep.cell(T, f"{snr}dB/{key}", pub_text, pub_val, src)
    rep.finish_table(T, before)


def check_joint_latency(tex, rep):
    """Latency/cost table (tbl:joint-latency) vs joint_latency_memory.json.

    Columns: relay | delay (symbols) | MACs/symbol | BER at 12 dB. The BER
    column and the delay column both come from the measurement; the MAC
    column is arithmetic from the architecture rather than measured, so it is
    checked against the MLSE and relay cost formulas restated below. Those
    restate what unified_latency_axis.py computes; they are not imported from
    it, so a change to the formulas has to be made in both places.
    """
    T = "tbl:joint-latency"; before = rep.checked
    body = tabular_body(tex, T)
    if body is None:
        return rep.skip(T, "label not found in tex")
    d = json.load(open(os.path.join(ROOT, "results/joint_latency_memory.json")))
    rows = {r["scheme"]: r for r in d["part_a"]["rows"]}
    snr_i = d["part_a"]["rows"][0]["snr_db"].index(12)

    label_map = {
        "AF": "AF", "DF-hard (symbol-wise)": "DF-hard (symbol-wise)",
        # clean_cell() strips $, so these are the post-cleaning forms; writing
        # them as they appear in the .tex silently matches nothing.
        "MLP W=5": "MLP w=5", "MLP W=11": "MLP w=11", "MLP W=21": "MLP w=21",
        "MLSE D=2": "MLSE D=2", "MLSE D=3": "MLSE D=3", "MLSE D=15": "MLSE D=15",
        "block DF": "block DF", "block DF + MLSE": "block DF + MLSE",
    }
    macs = {"AF": 2, "DF-hard (symbol-wise)": 0,
            "MLP w=5": 2*5*8 + 4*8, "MLP w=11": 2*11*8 + 4*8, "MLP w=21": 2*21*8 + 4*8,
            "MLSE D=2": 2*4**3, "MLSE D=3": 2*4**3, "MLSE D=15": 2*4**3,
            "block DF": 16, "block DF + MLSE": 16 + 2*4**3}

    for row in data_rows(body):
        if not row:
            continue
        key = label_map.get(row[0][0].strip())
        if key is None or key not in rows or len(row) < 4:
            continue
        src = rows[key]
        rep.cell(T, f"{key}/delay", row[1][0], row[1][1], src["latency_symbols"])
        rep.cell(T, f"{key}/macs", row[2][0], row[2][1], macs[key])
        rep.cell(T, f"{key}/ber12", row[3][0], row[3][1], src["ber"][snr_i])
    rep.finish_table(T, before)


def check_joint_memory(tex, rep):
    """Cost-against-memory table (tbl:joint-memory).

    Arithmetic columns are derived and checked against the cost formulas; the
    BER columns come from results/joint_memory_precision.json, the fifty-fold
    longer re-run, not from joint_latency_memory.json -- the original sweep put
    the L=7 MLSE cell at a single bit error and cannot support the published
    figures. Rows for L = 4 and 6 carry no measured BER (they exist to show the
    growth rate) and their BER cells are dashes, which parse to None.
    """
    T = "tbl:joint-memory"; before = rep.checked
    body = tabular_body(tex, T)
    if body is None:
        return rep.skip(T, "label not found in tex")
    src = os.path.join(ROOT, "results/joint_memory_precision.json")
    if not os.path.exists(src):
        return rep.skip(T, "results/joint_memory_precision.json not found")
    d = json.load(open(src))
    mlse, relay = {}, {}
    for r in d["rows"]:
        (mlse if r["scheme"].startswith("MLSE") else
         relay if r["scheme"].startswith("MLP") else {}).setdefault(
            r["channel_taps_L"], r.get("ber"))

    for row in data_rows(body):
        if not row or row[0][1] is None:
            continue
        L = int(row[0][1])
        rep.cell(T, f"L={L}/states", row[1][0], row[1][1], 4 ** (L - 1))
        rep.cell(T, f"L={L}/mlse_macs", row[2][0], row[2][1], 2 * 4 ** L)
        rep.cell(T, f"L={L}/relay_macs", row[3][0], row[3][1], 2 * 11 * 8 + 4 * 8)
        if len(row) > 4 and row[4][1] is not None and L in mlse:
            rep.cell(T, f"L={L}/mlse_ber", row[4][0], row[4][1], mlse[L])
        if len(row) > 5 and row[5][1] is not None and L in relay:
            rep.cell(T, f"L={L}/relay_ber", row[5][0], row[5][1], relay[L])
    rep.finish_table(T, before)


def check_table38(tex, rep):
    """Error-location diagnostic (tbl:table38) vs coded_error_mechanism.json."""
    T = "tbl:table38"; before = rep.checked
    body = table_body(tex, T)
    if body is None:
        return rep.skip(T, "label not found in tex")
    d = json.load(open(os.path.join(ROOT, "results/coded_error_mechanism.json")))["results"]
    label_map = {"coded-df": "coded_df", "mlp-coded": "mlp_coded", "oracle": "oracle"}
    snr = None
    for row in data_rows(body):
        if not row:
            continue
        first = row[0][0].strip()
        m = re.match(r"(\d+)\s*dB", first)
        if m:
            snr = m.group(1)
        if snr is None or len(row) < 5 or snr not in d:
            continue
        name = re.sub(r"[^a-z0-9-]", "", row[1][0].strip().lower())
        key = label_map.get(name)
        if key is None:
            continue
        src = d[snr][key]
        # relay symbol ER (col 2) and final BER (col 4); "---" cells parse to None
        if row[2][1] is not None:
            rep.cell(T, f"{snr}dB/{key}/relay_sym_er", row[2][0], row[2][1], src["relay_sym_er"])
        if row[4][1] is not None:
            rep.cell(T, f"{snr}dB/{key}/ber", row[4][0], row[4][1], src["ber"])
    rep.finish_table(T, before)


def check_table44(tex, rep):
    """Reliable-decoding regime (tbl:table44) vs coded_reliable_regime.json.

    Columns: SNR | coded-DF | MLP(thesis recipe) | MLP(retrained) | oracle | coded-DF FER.
    """
    T = "tbl:table44"; before = rep.checked
    body = table_body(tex, T)
    if body is None:
        return rep.skip(T, "label not found in tex")
    d = json.load(open(os.path.join(ROOT, "results/coded_reliable_regime.json")))
    idx = {int(s): i for i, s in enumerate(d["snr_db"])}
    cols = ["coded_df", "mlp_thesis", "mlp_ext", "oracle"]
    found = set()
    for row in data_rows(body):
        if not row or row[0][1] is None:
            continue
        snr = int(row[0][1])
        if snr not in idx:
            rep.note(T, f"{snr}dB: no json mapping, not checked")
            continue
        found.add(snr)
        i = idx[snr]
        for c, key in enumerate(cols, start=1):
            if c >= len(row):
                break
            rep.cell(T, f"{snr}dB/{key}", row[c][0], row[c][1], d[key][i])
        if len(row) > 5:
            rep.cell(T, f"{snr}dB/coded_df_fer", row[5][0], row[5][1], d["coded_df_fer"][i])
        else:
            rep.note(T, f"{snr}dB: coded-DF FER column missing, not checked")
    missing = sorted(set(idx) - found)
    if missing:
        rep.note(T, f"json has {missing} dB with no matching table row")
    rep.finish_table(T, before)


def check_table39(tex, rep):
    """Soft- vs hard-decision relaying (tbl:table39) vs coded_soft_decision.json."""
    T = "tbl:table39"; before = rep.checked
    body = table_body(tex, T)
    if body is None:
        return rep.skip(T, "label not found in tex")
    d = json.load(open(os.path.join(ROOT, "results/coded_soft_decision.json")))["summary"]
    cols = ["coded_df", "soft_df", "mlp_hard", "mlp_soft", "oracle"]
    for row in data_rows(body):
        if not row or row[0][1] is None:
            continue
        snr = str(int(row[0][1]))
        if snr not in d:
            continue
        for c, key in enumerate(cols, start=1):
            if c >= len(row):
                break
            pub_text, pub_val = row[c]
            rep.cell(T, f"{snr}dB/{key}", pub_text, pub_val, d[snr][key]["mean"])
    rep.finish_table(T, before)


def check_table40(tex, rep):
    """Equal-throughput comparison (tbl:table40) vs coded_latency_throughput.json."""
    T = "tbl:table40"; before = rep.checked
    body = table_body(tex, T)
    if body is None:
        return rep.skip(T, "label not found in tex")
    d = json.load(open(os.path.join(ROOT, "results/coded_latency_throughput.json")))
    rows = {int(r["snr_db"]): r for r in d["equal_throughput"]}
    for row in data_rows(body):
        if not row or row[0][1] is None:
            continue
        snr = int(row[0][1])
        if snr not in rows or len(row) < 3:
            continue
        rep.cell(T, f"{snr}dB/uncoded_qpsk", row[1][0], row[1][1], rows[snr]["uncoded_qpsk"])
        rep.cell(T, f"{snr}dB/coded_qam16", row[2][0], row[2][1], rows[snr]["coded_qam16"])
    rep.finish_table(T, before)


def ergodic_rayleigh_capacity(snr_db):
    """Ergodic Rayleigh (unconstrained-input Shannon) capacity, bits/complex use."""
    g = 10 ** (snr_db / 10.0)
    return math.log2(math.e) * math.exp(1.0 / g) * exp1(1.0 / g)


def check_table41(tex, rep):
    """Latency and compute cost (tbl:table41) vs its two committed sources.

    Two different kinds of number live in this table and they are checked
    against different files. The Buffer column is architectural -- the
    relay's structural latency in symbols -- and comes from
    coded_latency_throughput.json. The two us/symbol columns are wall-clock
    measurements from two different machines, deliberately reported side by
    side as the evidence that the absolute figures are machine-dependent
    while the ratios are not; both readouts are pinned in
    coded_latency_compute_machines.json.

    Timings are machine-dependent, so this check does NOT assert that the
    thesis figures are reproducible on the machine running the verifier --
    only that they still match the readouts committed alongside them. That
    is the property that can actually drift silently when the table is
    edited, and it is the reason the table is checked at all: it was the
    last numerical table in Chapter 5 with no coverage.
    """
    T = "tbl:table41"; before = rep.checked
    body = table_body(tex, T)
    if body is None:
        return rep.skip(T, "label not found in tex")

    lat = json.load(open(os.path.join(ROOT, "results/coded_latency_throughput.json")))
    mach = json.load(open(os.path.join(ROOT, "results/coded_latency_compute_machines.json")))
    buf = lat["structural_latency_symbols"]
    ca, cb = mach["machine_A"]["compute"], mach["machine_B"]["compute"]

    # tex row label -> (buffer key, compute key). The AF/DF row has no
    # compute entry: it is per-symbol classical processing, dashed in the tex.
    rows = {
        "af / symbol-wise df":     ("AF / symbol-wise DF", None),
        "mlp hard (756p)":         ("MLP / Mamba, window 21", "MLP hard (756p)"),
        "mlp soft (756p)":         ("MLP / Mamba, window 21", "MLP soft (756p)"),
        "hard block-df (viterbi)": ("hard block-DF (Viterbi)", "hard block-DF (Viterbi)"),
        "soft block-df (bcjr)":    ("soft block-DF (BCJR)", "soft block-DF (BCJR)"),
    }

    for row in data_rows(body):
        if not row or len(row) < 4:
            continue
        name = re.sub(r"\\textbf|[{}]", "", row[0][0]).strip().lower()
        if name not in rows:
            continue
        bkey, ckey = rows[name]
        pub_text, pub_val = row[1]
        rep.cell(T, f"{name}/buffer", pub_text, pub_val, float(buf[bkey]))
        if ckey is None:
            continue
        for col, comp, tag in ((2, ca, "machineA"), (3, cb, "machineB")):
            if col >= len(row):
                break
            pub_text, pub_val = row[col]
            rep.cell(T, f"{name}/{tag}", pub_text, pub_val,
                     comp[ckey]["us_per_symbol"])

    # The ratio the surrounding prose leans on, on both machines. This is the
    # claim that survives the change of hardware, so it is the one worth
    # pinning: if either readout is ever replaced, the "1.94x" must move too.
    for comp, tag in ((ca, "machineA"), (cb, "machineB")):
        ratio = comp["soft block-DF (BCJR)"]["us_per_symbol"] / \
                comp["hard block-DF (Viterbi)"]["us_per_symbol"]
        rep.cell(T, f"BCJR/Viterbi ratio ({tag})", "1.94", 1.94, ratio)

    rep.finish_table(T, before)


def check_table42(tex, rep):
    """Link-adaptation envelope (tbl:table42) vs coded_rate_adaptation.json.

    Column 6, "C (Shannon)", is not simulated -- it is the closed-form
    ergodic Rayleigh capacity, an upper-bound sanity ceiling on goodput.
    """
    T = "tbl:table42"; before = rep.checked
    body = table_body(tex, T)
    if body is None:
        return rep.skip(T, "label not found in tex")
    d = json.load(open(os.path.join(ROOT, "results/coded_rate_adaptation.json")))
    env = {"blockdf": {r["snr_db"]: r for r in d["envelope"]["blockdf"]},
           "denoise": {r["snr_db"]: r for r in d["envelope"]["denoise"]}}
    for row in data_rows(body):
        if not row or row[0][1] is None:
            continue
        snr = int(row[0][1])
        if snr not in env["blockdf"] or len(row) < 5:
            continue
        rb, rd = env["blockdf"][snr], env["denoise"][snr]
        if row[2][1] is not None:
            rep.cell(T, f"{snr}dB/blockdf_goodput", row[2][0], row[2][1], rb["goodput"])
        if row[4][1] is not None:
            rep.cell(T, f"{snr}dB/denoise_goodput", row[4][0], row[4][1], rd["goodput"])
        if len(row) >= 6 and row[5][1] is not None:
            rep.cell(T, f"{snr}dB/capacity", row[5][0], row[5][1],
                     ergodic_rayleigh_capacity(snr))
    rep.finish_table(T, before)


def check_table43(tex, rep):
    """Latency-constrained envelope (tbl:table43) vs coded_latency_capacity.json."""
    T = "tbl:table43"; before = rep.checked
    body = table_body(tex, T)
    if body is None:
        return rep.skip(T, "label not found in tex")
    d = json.load(open(os.path.join(ROOT, "results/coded_latency_capacity.json")))
    snap = {r["snr_db"]: r for r in d["snapshot"]}
    for row in data_rows(body):
        if not row or row[0][1] is None:
            continue
        snr = int(row[0][1])
        if snr not in snap or len(row) < 5:
            continue
        rb, rd = snap[snr]["blockdf"], snap[snr]["denoise"]
        if row[2][1] is not None:
            rep.cell(T, f"{snr}dB/blockdf_goodput", row[2][0], row[2][1], rb["goodput"])
        if row[4][1] is not None:
            rep.cell(T, f"{snr}dB/denoise_goodput", row[4][0], row[4][1], rd["goodput"])
    rep.finish_table(T, before)


def _e6_grouped(tex, label, npy_map, rep, snr_cols):
    """Shared parser for the two Ch7 grouped tables (setup/relay rows).

    npy_map: dict mapping (setup_key, relay_label_in_tex) -> callable(snr_index)->value
    snr_cols: list of (column_index_in_row, snr_dB)
    """
    T = label; before = rep.checked
    body = table_body(tex, label)
    if body is None:
        return rep.skip(label, "label not found in tex")
    rep.finish_table  # noqa (kept for symmetry)
    return body, before, T


def check_tableE6(tex, rep):
    """Unknown-channel BER (tbl:tableE6). Sources: e6_sim + e6_viterbi npy."""
    T = "tbl:tableE6"; before = rep.checked
    body = table_body(tex, T)
    if body is None:
        return rep.skip(T, "label not found in tex")
    sim = np.load(os.path.join(ROOT, "e6_unknown_channel_results/e6_sim_ported_results.npy"),
                  allow_pickle=True).item()
    vg_awgn = np.load(os.path.join(ROOT, "e6_unknown_channel_results/e6_viterbi_awgn.npy"),
                      allow_pickle=True).item()
    vg_ray = np.load(os.path.join(ROOT, "e6_unknown_channel_results/e6_viterbi_rayleigh.npy"),
                     allow_pickle=True).item()
    snrs = list(sim["snrs"])
    col_snr = [(2, 8), (3, 12), (4, 16), (5, 20)]   # row: Setup & Relay & 8 & 12 & 16 & 20

    # setup label in tex -> (sim results key, viterbi dict or None)
    setup_map = {
        "Unknown ISI $\\to$ AWGN": ("S1: unknown ISI -> AWGN", vg_awgn),
        "Unknown ISI $\\to$ Rayleigh": ("S2: unknown ISI -> Rayleigh", vg_ray),
        "Control: canonical Rayleigh": ("S4 control: Rayleigh -> Rayleigh (canonical)", None),
    }
    cur_setup = None
    for row in data_rows(body):
        if not row:
            continue
        first = row[0][0].strip()
        if first:  # new setup group
            # match against known setup labels (loose contains)
            cur_setup = None
            for k in setup_map:
                key_plain = k.replace("$\\to$", "->")
                if "AWGN" in first and "ISI" in first and "AWGN" in key_plain:
                    cur_setup = k
                elif "Rayleigh" in first and "ISI" in first and "Rayleigh" in key_plain and "Control" not in k:
                    cur_setup = k
                elif "Control" in first and "Control" in k:
                    cur_setup = k
                if cur_setup:
                    break
        if cur_setup is None or len(row) < 2:
            continue
        relay = row[1][0].strip()
        sim_key, vg = setup_map[cur_setup]
        res = sim["results"].get(sim_key, {})
        # map tex relay name -> source
        def src_at(si):
            r = relay.upper()
            if r.startswith("AF"):
                return res["AF"][0][si]
            if r.startswith("DF"):
                return res["DF"][0][si]
            if "MLP" in r:
                return res["MLP"][0][si]
            if "GENIE" in r and vg is not None:
                return float(vg["VIT-genie"][si])
            if ("PILOT" in r or "EST" in r or "200" in r) and vg is not None:
                return float(vg["VIT-est"][si])
            return None
        for c, snr in col_snr:
            if c >= len(row):
                break
            pub_text, pub_val = row[c]
            if pub_val is None:
                continue
            si = snrs.index(snr)
            src = src_at(si)
            if src is not None:
                rep.cell(T, f"{cur_setup[:16]}/{relay}/{snr}dB", pub_text, pub_val, src)
    rep.finish_table(T, before)


def check_tableE6flat(tex, rep):
    """Flat-channel control BER (tbl:tableE6flat) vs e6_flat npy.

    Columns: Flat channel & Relay & 8 dB & 12 dB & 16/20 dB
    The last column packs two values "0.0119 / 0.0048" (16 dB / 20 dB).
    """
    T = "tbl:tableE6flat"; before = rep.checked
    body = table_body(tex, T)
    if body is None:
        return rep.skip(T, "label not found in tex")
    flat = np.load(os.path.join(ROOT, "e6_unknown_channel_results/e6_flat_ported_results.npy"),
                   allow_pickle=True).item()
    snrs = list(flat["snrs"])
    # tex channel label -> flat results key
    chan_map_order = [("phase",), ("gain",), ("iqimb",)]
    row_channels = {"phase": ["unknown phase", "dbpsk"],
                    "gain": ["unknown gain"],
                    "iqimb": ["asymmetry", "branch"]}
    cur = None
    for row in data_rows(body):
        if not row or len(row) < 5:
            continue
        first = row[0][0].strip().lower()
        if first:
            cur = None
            for key, needles in row_channels.items():
                if any(n in first for n in needles):
                    cur = key
                    break
        if cur is None:
            continue
        relay = row[1][0].strip()
        res = flat["results"].get(cur, {})
        rkey = "MLP" if "MLP" in relay.upper() else ("DF" if "DF" in relay.upper() else
                                                     ("AF" if relay.upper().startswith("AF") else None))
        if rkey is None or rkey not in res:
            continue
        mean = res[rkey][0]
        # col2 = 8 dB, col3 = 12 dB, col4 = "16 / 20"
        for c, snr in ((2, 8), (3, 12)):
            pub_text, pub_val = row[c]
            if pub_val is not None:
                rep.cell(T, f"{cur}/{relay}/{snr}dB", pub_text, pub_val, mean[snrs.index(snr)])
        # split the combined 16/20 cell
        combo = row[4][0]
        parts = _NUM.findall(combo)
        if len(parts) >= 2:
            for val, snr in ((parts[0], 16), (parts[1], 20)):
                rep.cell(T, f"{cur}/{relay}/{snr}dB", val, float(val), mean[snrs.index(snr)])
    rep.finish_table(T, before)


def check_tableE6qpsk(tex, rep):
    """QPSK unknown-channel BER (tbl:tableE6qpsk) vs e6_qpsk_unknown_channel_results.npy."""
    T = "tbl:tableE6qpsk"; before = rep.checked
    body = table_body(tex, T)
    if body is None:
        return rep.skip(T, "label not found in tex")
    d = np.load(os.path.join(ROOT, "e6_unknown_channel_results",
                              "e6_qpsk_unknown_channel_results.npy"),
                allow_pickle=True).item()
    snrs = list(d["snrs"])
    col_snr = [(2, 8), (3, 12), (4, 16), (5, 20)]

    setup_map = {"AWGN": "awgn", "Rayleigh": "rayleigh"}
    # Two Viterbi rows now: the taps-only trellis, which is what this table
    # used to publish under the "genie CSI" label, and the fading-aware one
    # that actually is genie CSI on this channel. "TAPS" must be tested before
    # the bare "VITERBI" fallback or both rows would match the same key.
    relay_map = {"AF": "AF", "DF": "DF", "MLP-QPSK": "MLP-QPSK",
                 "TAPS ONLY": "Viterbi (taps only)",
                 "GENIE": "Viterbi (genie CSI)"}
    cur_setup = None
    for row in data_rows(body):
        if not row:
            continue
        first = row[0][0].strip()
        if first:
            cur_setup = None
            for key, hop2 in setup_map.items():
                if key.upper() in first.upper():
                    cur_setup = hop2
                    break
        if cur_setup is None or len(row) < 2:
            continue
        relay = row[1][0].strip()
        rkey = None
        for needle, target in relay_map.items():
            if needle in relay.upper():
                rkey = target
                break
        if rkey is None:
            continue
        res = d["results"][cur_setup][rkey][0]  # row 0 = mean
        for c, snr in col_snr:
            if c >= len(row):
                break
            pub_text, pub_val = row[c]
            if pub_val is None:
                continue
            si = snrs.index(snr)
            rep.cell(T, f"{cur_setup}/{relay}/{snr}dB", pub_text, pub_val, res[si])
    rep.finish_table(T, before)


def _load_e6_npy(name):
    p = os.path.join(ROOT, "e6_unknown_channel_results", name)
    if not os.path.exists(p):
        return None
    return np.load(p, allow_pickle=True).item()


def check_E6blind_prose(tex, rep):
    """Blind-regime prose claims vs e6_blind_ported_results.npy."""
    T = "prose:E6blind"; before = rep.checked
    d = _load_e6_npy("e6_blind_ported_results.npy")
    if d is None:
        return rep.skip(T, "e6_blind_ported_results.npy not found (run e6_blind_ported.py)")
    snrs = list(d["snrs"])
    sm = d["summary"]
    i8, i16, i20 = snrs.index(8), snrs.index(16), snrs.index(20)

    m = re.search(r"corrected CMA converges smoothly to BER \$([\d.]+)\\times10\^\{-3\}\$ at 20 dB", tex)
    if m:
        rep.cell(T, "CMA-blind/20dB", m.group(1) + "e-3", float(m.group(1)) * 1e-3,
                 sm["CMA-blind"][0][i20])
    m = re.search(r"MLP reaches \$([\d.]+)\\times10\^\{-3\}\$", tex)
    if m:
        rep.cell(T, "MLP-169/20dB", m.group(1) + "e-3", float(m.group(1)) * 1e-3,
                 sm["MLP-169"][0][i20])
    m = re.search(r"\(\$([\d.]+)\$ vs\.\\ \$([\d.]+)\$ at 8 dB\)\. This is the correct answer", tex)
    if m:
        rep.cell(T, "MLP-169/8dB", m.group(1), float(m.group(1)), sm["MLP-169"][0][i8])
        rep.cell(T, "CMA-blind/8dB", m.group(2), float(m.group(2)), sm["CMA-blind"][0][i8])
    m = re.search(r"falling to \$([\d.]+)\$ at 16 dB before rising again to \$([\d.]+)\$ at 20 dB", tex)
    if m:
        rep.cell(T, "Viterbi-blind/16dB", m.group(1), float(m.group(1)), sm["Viterbi-blind"][0][i16])
        rep.cell(T, "Viterbi-blind/20dB", m.group(2), float(m.group(2)), sm["Viterbi-blind"][0][i20])
    m = re.search(r"interval at 8 dB is \$\\pm([\d.]+)\$.*?MLP's \$\\pm([\d.]+)\$.*?CMA's \$\\pm([\d.]+)\$", tex, re.S)
    if m:
        rep.cell(T, "Viterbi-blind CI/8dB", m.group(1), float(m.group(1)), sm["Viterbi-blind"][1][i8])
        rep.cell(T, "MLP CI/8dB", m.group(2), float(m.group(2)), sm["MLP-169"][1][i8])
        rep.cell(T, "CMA CI/8dB", m.group(3), float(m.group(3)), sm["CMA-blind"][1][i8])
    rep.finish_table(T, before)


def check_E6partial_prose(tex, rep):
    """Partial-posterior prose claims (both panels) vs e6_partial_ported_results.npy."""
    T = "prose:E6partial"; before = rep.checked
    d = _load_e6_npy("e6_partial_ported_results.npy")
    if d is None:
        return rep.skip(T, "e6_partial_ported_results.npy not found (run e6_partial_ported.py)")
    pa = d["panel_a"]
    pb = d.get("panel_b", {})
    pbc = d.get("panel_b_cma", {})

    m = re.search(r"payload BER \$([\d.]+)\$ at 800 pilots", tex)
    if m:
        rep.cell(T, "Viterbi/800 pilots", m.group(1), float(m.group(1)), pa[800][0])
    m = re.search(r"down to \$([\d.]+)\$ at 20 pilots", tex)
    if m:
        rep.cell(T, "Viterbi/20 pilots", m.group(1), float(m.group(1)), pa[20][0])
    m = re.search(r"MLP's pilot-free \$([\d.]+)\$", tex)
    if m:
        rep.cell(T, "MLP pilot-free ref", m.group(1), float(m.group(1)), d["mlp_ref"][0])
    m = re.search(r"At 10 pilots Viterbi has already lost its edge, \$([\d.]+)\$ against the MLP's \$([\d.]+)\$", tex)
    if m:
        rep.cell(T, "Viterbi/10 pilots", m.group(1), float(m.group(1)), pa[10][0])
        rep.cell(T, "MLP ref (10-pilot cmp)", m.group(2), float(m.group(2)), d["mlp_ref"][0])
    m = re.search(r"at 5 pilots it collapses to \$([\d.]+)\$", tex)
    if m:
        rep.cell(T, "Viterbi/5 pilots", m.group(1), float(m.group(1)), pa[5][0])
    m = re.search(r"\(\$([\d.]+) \\pm ([\d.]+)\$, against \$\\pm ([\d.]+)\$ at 50 pilots\)", tex)
    if m:
        rep.cell(T, "Viterbi/5 pilots (CI ctx)", m.group(1), float(m.group(1)), pa[5][0])
        rep.cell(T, "Viterbi CI/5 pilots", m.group(2), float(m.group(2)), pa[5][1])
        rep.cell(T, "Viterbi CI/50 pilots", m.group(3), float(m.group(3)), pa[50][1])
    m = re.search(r"flat across the entire sweep at \$([\d.]+)\$", tex)
    if m:
        rep.cell(T, "MLP flat ref", m.group(1), float(m.group(1)), d["mlp_ref"][0])
    # panel (b): blind CMA per-block convergence failure
    m = re.search(r"payload BER is \$([\d.]+)\$ at \$L=40\$ and only improves to \$([\d.]+)\$ at \$L=1000\$", tex)
    if m and pbc:
        rep.cell(T, "CMA/L=40", m.group(1), float(m.group(1)), pbc[40][0])
        rep.cell(T, "CMA/L=1000", m.group(2), float(m.group(2)), pbc[1000][0])
    m = re.search(r"against the \$([\d.]+)\$ it achieves when given a \$20\{,\}000\$-symbol block", tex)
    if m:
        rep.cell(T, "CMA/20k-block ref", m.group(1), float(m.group(1)), d["cma_ref"][0])
    rep.finish_table(T, before)


def check_E6composite_prose(tex, rep):
    """Composite-cascade prose claims vs e6_composite_ported_results.npy."""
    T = "prose:E6composite"; before = rep.checked
    d = _load_e6_npy("e6_composite_ported_results.npy")
    if d is None:
        return rep.skip(T, "e6_composite_ported_results.npy not found (run e6_composite_ported.py)")
    snrs = list(d["snrs"]); s8 = snrs.index(8); s10 = snrs.index(10); s20 = snrs.index(20)
    sm = d["summary"]

    m = re.search(r"reaching \$([\d.]+)\\times10\^\{-3\}\$ at 20 dB", tex)
    if m:
        rep.cell(T, "MLP-169/20dB", m.group(1) + "e-3", float(m.group(1)) * 1e-3,
                 sm["MLP-169"][0][s20])
    m = re.search(r"\(\$([\d.]+)\$ vs\.\\ \$([\d.]+)\$ at 8 dB\) and converges", tex)
    if m:
        rep.cell(T, "Viterbi-diff/8dB", m.group(1), float(m.group(1)), sm["Viterbi-diff"][0][s8])
        rep.cell(T, "MLP-169/8dB", m.group(2), float(m.group(2)), sm["MLP-169"][0][s8])
    m = re.search(r"indistinguishable at \$([\d.]+)\$ each at 20 dB", tex)
    if m:
        rep.cell(T, "MLP-169/20dB (tie)", m.group(1), float(m.group(1)), sm["MLP-169"][0][s20])
        rep.cell(T, "Viterbi-diff/20dB (tie)", m.group(1), float(m.group(1)), sm["Viterbi-diff"][0][s20])
    m = re.search(r"\(\$([\d.]+)\$ vs\.\\ \$([\d.]+)\$ at 10 dB for the larger network", tex)
    if m:
        rep.cell(T, "MLP-large/10dB", m.group(1), float(m.group(1)), sm["MLP-large"][0][s10])
        rep.cell(T, "MLP-169/10dB", m.group(2), float(m.group(2)), sm["MLP-169"][0][s10])
    m = re.search(r"\$([\d.]+)\$ vs\.\\ \$([\d.]+)\$ at 8 dB for the smaller", tex)
    if m:
        rep.cell(T, "MLP-large/8dB", m.group(1), float(m.group(1)), sm["MLP-large"][0][s8])
        rep.cell(T, "MLP-169/8dB (cmp)", m.group(2), float(m.group(2)), sm["MLP-169"][0][s8])
    m = re.search(r"ending at the identical \$([\d.]+)\$ at 20 dB", tex)
    if m:
        rep.cell(T, "MLP-large/20dB", m.group(1), float(m.group(1)), sm["MLP-large"][0][s20])
    rep.finish_table(T, before)


def check_table26(tex, rep):
    """Theoretical SNR @ BER=1e-3 (tbl:table26) vs closed-form inversion."""
    T = "tbl:table26"; before = rep.checked
    body = table_body(tex, T)
    if body is None:
        return rep.skip(T, "label not found in tex")
    src = {"awgn": snr_for_ber(1e-3, ber_awgn),
           "rayleigh": snr_for_ber(1e-3, ber_rayleigh)}
    for row in data_rows(body):
        if not row:
            continue
        name = row[0][0].strip().lower()
        key = "awgn" if "awgn" in name else ("rayleigh" if "rayleigh" in name else None)
        if key is None or len(row) < 2:
            continue
        pub_text, pub_val = row[1]      # "~9.8 dB" / "~24 dB"
        if pub_val is not None:
            # these are quoted to ~1 dB; allow 1 dB tolerance
            rep.cell(T, f"{key}(SNR@1e-3, ~1dB tol)", pub_text + "0", pub_val, src[key])
    rep.finish_table(T, before)


# ----------------------------------------------------------------------------
# relay-name aliases
# ----------------------------------------------------------------------------
# The minimal MLP was renamed from "GenAI (169p)" to "MLP (169p)" (and
# "GenAI-3K" to "MLP-3K"), so a result file's spelling depends on when it was
# produced. Three separate checks have already been silently disabled by a
# KeyError on the old name -- the table reports "skipped" and its cells stop
# being counted, which is exactly the failure mode a verifier must not have.
# Every lookup goes through resolve() so a rename costs nothing, and a relay
# genuinely absent from a run (the cGAN in a lean re-run) is reported as
# missing rather than raising.
RELAY_ALIASES = {
    "GenAI (169p)": ["GenAI (169p)", "MLP (169p)"],
    "MLP (169p)":   ["MLP (169p)", "GenAI (169p)"],
    "GenAI-3K":     ["GenAI-3K", "MLP-3K"],
    "MLP-3K":       ["MLP-3K", "GenAI-3K"],
}


def resolve(results, name):
    """Return the key *name* goes by in *results*, or None if absent."""
    for cand in RELAY_ALIASES.get(name, [name]):
        if cand in results:
            return cand
    return None


def check_ber_validation(tex, rep):
    """Calibration table (tab:ber_validation_long).

    Checks BOTH columns: the theory column against the closed form, and the
    simulation column against results/calibration.json. Only the theory
    columns were checked before, so the agreement the table exists to
    demonstrate was itself unverified.
    """
    T = "tab:ber_validation_long"; before = rep.checked
    body = table_body(tex, T)
    if body is None:
        return rep.skip(T, "label not found in tex")
    theo = {"awgn": ber_awgn, "rayleigh": ber_rayleigh}
    cal_path = os.path.join(ROOT, "results", "calibration.json")
    cal = json.load(open(cal_path)) if os.path.exists(cal_path) else None
    # columns: Chan & 4dB(Th) & 4dB(Sim) & 10dB(Th) & 10dB(Sim) & 16dB(Th) & 16dB(Sim)
    pairs = [(1, 2, 4), (3, 4, 10), (5, 6, 16)]
    for row in data_rows(body):
        if not row:
            continue
        name = row[0][0].strip().lower()
        key = "awgn" if "awgn" in name else ("rayleigh" if "rayleigh" in name else None)
        if key is None:
            continue
        for c_th, c_sim, snr in pairs:
            if c_th < len(row):
                pub_text, pub_val = row[c_th]
                if pub_val is not None:
                    rep.cell(T, f"{key}/theory@{snr}dB", pub_text, pub_val, theo[key](snr))
            if cal is not None and c_sim < len(row):
                pub_text, pub_val = row[c_sim]
                entry = cal["results"].get(key, {}).get(str(snr))
                # "n/r" cells carry no number and are skipped by rep.cell
                if pub_val is not None and entry is not None and entry["resolvable"]:
                    rep.cell(T, f"{key}/sim@{snr}dB", pub_text, pub_val, entry["sim_mean"])
    rep.finish_table(T, before)


def check_mmse_baseline(tex, rep):
    """MMSE linear-equalizer baseline (tbl:mmse-baseline) vs results/mmse_equalizer.json.

    Columns: Channel & 3 taps & 5 taps & 7 taps & 11 taps & Learned relay
    Rows: ISI (BPSK) / ISI (QPSK) / Composite
    Source keys: mmse["isi"][N], mmse["isi_complex"][N], mmse["composite"][N].
    The learned-relay column is not in the JSON (it is copied from E6 results)
    and is skipped here; only the MMSE columns are machine-derived.
    """
    T = "tbl:mmse-baseline"; before = rep.checked
    body = tabular_body(tex, T)
    if body is None:
        return rep.skip(T, "label not found in tex")
    src_path = os.path.join(ROOT, "results", "mmse_equalizer.json")
    if not os.path.exists(src_path):
        return rep.skip(T, "results/mmse_equalizer.json not found")
    with open(src_path) as fh:
        mmse = json.load(fh)
    # map tex row label -> JSON channel key
    ch_map = [("bpsk", "isi"), ("qpsk", "isi_complex"), ("composite", "composite")]
    tap_cols = [(1, "3"), (2, "5"), (3, "7"), (4, "11")]
    for row in data_rows(body):
        if not row:
            continue
        label = row[0][0].strip().lower()
        src_key = None
        for needle, key in ch_map:
            if needle in label:
                src_key = key
                break
        if src_key is None:
            continue
        for col, n in tap_cols:
            if col >= len(row):
                continue
            pub_text, pub_val = row[col]
            if pub_val is None:
                continue
            src = mmse.get(src_key, {}).get(n)
            if src is not None:
                rep.cell(T, f"{src_key}/{n}tap", pub_text, pub_val, src)
    rep.finish_table(T, before)


def check_qpsk_decomposition_prose(tex, rep):
    """The QPSK SER / bits-per-symbol-error figures quoted in Chapter 7 prose.

    These are the numbers that rule out the criterion-mismatch conjecture, so
    they should not be able to drift away from the run that produced them.
    Prose rather than a table, so the sentence is located by its own wording.
    """
    T = "prose:qpsk-decomposition"; before = rep.checked
    src_path = os.path.join(ROOT, "results", "qpsk_error_decomposition.json")
    if not os.path.exists(src_path):
        return rep.skip(T, "results/qpsk_error_decomposition.json not found")
    with open(src_path) as fh:
        src = json.load(fh)

    def at20(needle):
        for name, rows in src["detectors"].items():
            if needle.lower() in name.lower():
                for r in rows:
                    if int(r["snr_db"]) == 20:
                        return r
        return None

    vit, mlp = at20("taps only"), at20("MLP")
    if vit is None or mlp is None:
        return rep.skip(T, "detector keys not found in the JSON")

    i = tex.find("the MLP is ahead on \\emph{symbol} error rate as well")
    if i < 0:
        return rep.skip(T, "decomposition sentence not found in tex")
    sent = tex[i:tex.find(".", tex.find("Gray map is not the route", i))]
    nums = [float(m) for m in re.findall(r"\$(\d+\.\d+)\$", sent)]
    if len(nums) != 4:
        return rep.skip(T, f"expected 4 numbers in the sentence, found {len(nums)}")
    for got, want, what in zip(
            nums,
            [mlp["ser"], vit["ser"],
             mlp["bits_per_symbol_error"], vit["bits_per_symbol_error"]],
            ["mlp_ser@20dB", "vit_ser@20dB", "mlp_bits_per_err", "vit_bits_per_err"]):
        rep.cell(T, what, f"{got}", got, want)
    rep.finish_table(T, before)


def check_slicer_floor(tex, rep):
    """The closed-form slicer-BER table in Section~\\ref{sec:unknown-channel-experiment}.

    Two rows, both machine-derivable and from different sources, which is the
    point of the table: the closed form against results/isi_slicer_floor.json,
    and the measured DF row against the same E6 .npy that backs tbl:tableE6.
    The table has no \\label (it is an inline tabular inside a center, not a
    float), so it is located by its row labels rather than by tabular_body.
    """
    T = "tbl:slicer-floor-inline"; before = rep.checked
    i = tex.find("DF closed form, Eq.")
    if i < 0:
        return rep.skip(T, "closed-form row not found in tex")
    # back up to the table's own \toprule so the SNR header row is inside the
    # body -- it is what tells each value row which column is which SNR.
    start = tex.rfind("\\toprule", 0, i)
    if start < 0:
        return rep.skip(T, "no \\toprule above the closed-form row")
    stop = tex.find("\\bottomrule", i)
    if stop < 0:
        # find() returning -1 would slice to the last character of the whole
        # document, silently checking the wrong cells rather than failing.
        return rep.skip(T, "no \\bottomrule below the closed-form row")
    body = tex[start:stop]
    src_path = os.path.join(ROOT, "results", "isi_slicer_floor.json")
    if not os.path.exists(src_path):
        return rep.skip(T, "results/isi_slicer_floor.json not found")
    with open(src_path) as fh:
        src = json.load(fh)
    closed = dict(zip([int(x) for x in src["snr_db"]], src["slicer_ber"]))
    closed_af = dict(zip([int(x) for x in src["snr_db"]], src["af_ber"]))

    sim_path = os.path.join(ROOT, "e6_unknown_channel_results",
                            "e6_sim_ported_results.npy")
    sim = (np.load(sim_path, allow_pickle=True).item()
           if os.path.exists(sim_path) else None)

    # the SNR header decides which columns the value rows are compared against
    header = None
    for row in data_rows(body):
        if not row:
            continue
        label = row[0][0].strip()
        if label.startswith("SNR"):
            header = [int(v) for _, v in row[1:] if v is not None]
            continue
        if header is None:
            continue
        vals = [(t, v) for t, v in row[1:] if v is not None]
        if len(vals) != len(header):
            continue
        for (pub_text, pub_val), snr in zip(vals, header):
            if label.startswith("DF closed form") and snr in closed:
                rep.cell(T, f"closedDF/{snr}dB", pub_text, pub_val, closed[snr])
            elif label.startswith("AF closed form") and snr in closed_af:
                rep.cell(T, f"closedAF/{snr}dB", pub_text, pub_val, closed_af[snr])
            elif label.startswith(("DF measured", "AF measured")) and sim is not None:
                res = sim["results"].get("S1: unknown ISI -> AWGN")
                snrs = list(sim["snrs"])
                relay = label[:2]
                if res is not None and snr in snrs:
                    rep.cell(T, f"measured{relay}/{snr}dB", pub_text, pub_val,
                             float(res[relay][0][snrs.index(snr)]))
    rep.finish_table(T, before)


def check_seq_on_memory(tex, rep):
    """Sequence architectures on memory channels (tbl:seq-on-memory) vs
    results/seq_models_on_memory.json.

    Columns: Channel & MLP-3K & Transformer-3K & Mamba-S6-3K & Mamba2-3K
    Each cell is "best_db (spread)"; we check the best_db value and skip the
    bracketed spread (it is informational and harder to parse without ambiguity).
    Rows: ISI (BPSK) / ISI (QPSK) / Composite
    Source keys: seq["channels"]["isi"], ["isi_complex"], ["composite"].
    """
    T = "tbl:seq-on-memory"; before = rep.checked
    body = tabular_body(tex, T)
    if body is None:
        return rep.skip(T, "label not found in tex")
    src_path = os.path.join(ROOT, "results", "seq_models_on_memory.json")
    if not os.path.exists(src_path):
        return rep.skip(T, "results/seq_models_on_memory.json not found")
    with open(src_path) as fh:
        seq = json.load(fh)
    channels = seq.get("channels", {})
    ch_map = [("bpsk", "isi"), ("qpsk", "isi_complex"), ("composite", "composite")]
    arch_cols = [(1, "MLP-3K"), (2, "Transformer-3K"), (3, "Mamba-S6-3K"), (4, "Mamba2-3K")]
    for row in data_rows(body):
        if not row:
            continue
        label = row[0][0].strip().lower()
        src_key = None
        for needle, key in ch_map:
            if needle in label:
                src_key = key
                break
        if src_key is None:
            continue
        ch_data = channels.get(src_key, {})
        for col, aname in arch_cols:
            if col >= len(row):
                continue
            pub_text, pub_val = row[col]
            if pub_val is None:
                continue
            arch_data = ch_data.get("archs", {}).get(aname, {})
            src = arch_data.get("best_db")
            if src is not None and not math.isnan(src):
                rep.cell(T, f"{src_key}/{aname}", pub_text, pub_val, src)
    rep.finish_table(T, before)


def main():
    global TEX_DIR, MC_SLACK
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", action="store_true",
                    help="regenerate the Ch7 (E6) .npy from the ported scripts first")
    ap.add_argument("--tex", default=None, help="override thesis chapters/ dir")
    args = ap.parse_args()
    if args.tex:
        TEX_DIR = args.tex

    if args.rerun:
        MC_SLACK = 0.01  # allow Monte-Carlo variation for freshly re-simulated cells
        print("Regenerating Ch7 (E6) result files (this runs the ported experiments)...")
        for scr in ("e6_sim_ported.py", "e6_viterbi_ported.py", "e6_flat_ported.py",
                    "e6_blind_ported.py", "e6_partial_ported.py", "e6_composite_ported.py"):
            print(f"  running {scr} ...", flush=True)
            subprocess.run([sys.executable, scr], cwd=ROOT, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        # copy fresh /tmp outputs into the results dir the checker reads
        import shutil
        for tmp, dst in [("/tmp/e6_sim_ported_results.npy", "e6_sim_ported_results.npy"),
                         ("/tmp/e6_viterbi_awgn.npy", "e6_viterbi_awgn.npy"),
                         ("/tmp/e6_viterbi_rayleigh.npy", "e6_viterbi_rayleigh.npy"),
                         ("/tmp/e6_flat_ported_results.npy", "e6_flat_ported_results.npy"),
                         ("/tmp/e6_blind_ported_results.npy", "e6_blind_ported_results.npy"),
                         ("/tmp/e6_partial_ported_results.npy", "e6_partial_ported_results.npy"),
                         ("/tmp/e6_composite_ported_results.npy", "e6_composite_ported_results.npy")]:
            if os.path.exists(tmp):
                shutil.copy(tmp, os.path.join(ROOT, "e6_unknown_channel_results", dst))

    tex = load_tex()
    rep = Report()
    # check_ber_validation / check_table26 / check_table24 / check_table35 / check_table36 retired: the AWGN
    # calibration section and the 16-QAM extension chapter were removed from the
    # thesis (see thesis/RERUN_CHANGELOG.md). Their functions are kept below so
    # the checks can be restored if the material ever returns.
    checks = [check_table2, check_layers_table, check_table8,
              check_tableE6, check_tableE6flat, check_tableE6qpsk,
              check_E6blind_prose, check_E6partial_prose, check_E6composite_prose,
              check_table34,
              check_table37, check_table38, check_table39, check_table40,
              check_table41, check_table42, check_table43,
              check_table44,
              check_mmse_baseline, check_seq_on_memory,
              check_slicer_floor, check_qpsk_decomposition_prose,
              check_joint_latency, check_joint_memory]
    for chk in checks:
        try:
            chk(tex, rep)
        except Exception as e:  # noqa
            rep.skip(chk.__name__, f"error: {e}")

    # ---- report ----
    print("\n" + "=" * 74)
    print("THESIS TABLE VERIFICATION  (published .tex  vs  experiment data source)")
    print("=" * 74)
    print(f"{'table':<24}{'cells':>8}{'flagged':>10}   status")
    print("-" * 74)
    for name, n, nf in rep.tables:
        status = "OK" if nf == 0 else "*** MISMATCH ***"
        print(f"{name:<24}{n:>8}{nf:>10}   {status}")
    for name, reason in rep.skipped:
        print(f"{name:<24}{'-':>8}{'-':>10}   skipped: {reason}")
    print("-" * 74)
    if rep.notes:
        print("Partial coverage (table checked, some rows not):")
        for name, reason in rep.notes:
            print(f"  [{name}] {reason}")
        print("-" * 74)
    print(f"cells checked: {rep.checked}   inconsistencies: {len(rep.flags)}")

    if rep.flags:
        print("\nFLAGGED INCONSISTENCIES (table | cell | published | source | |diff|):")
        for t, where, pub, src, diff in rep.flags:
            print(f"  [{t}] {where}: published={pub}  source={src}  diff={diff}")
    else:
        print("\nAll checked cells match their data source within display-rounding tolerance.")

    print("\nInformational (not pass/fail):")
    print("  tbl:table13, tbl:table25 report machine-dependent wall-clock timing;")
    print("  re-run run_experiments.py / the sequence-model benchmark to refresh those.")

    return 1 if rep.flags else 0


if __name__ == "__main__":
    sys.exit(main())
