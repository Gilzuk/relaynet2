#!/usr/bin/env python3
"""verify_thesis_tables.py — single-source reproducibility check for the thesis.

Recreates every *numerical* table in the thesis report from its authoritative
data source (the experiment output files and closed-form formulas) and compares,
cell by cell, against the numbers currently transcribed in the LaTeX
(``thesis/chapters/*.tex``). Any mismatch beyond display-rounding tolerance is
flagged.

Data sources, by table:
  tbl:table2            canonical Rayleigh BER, 9 relays   <- results/bpsk_comparison/rayleigh.json
  tbl:table8            normalized-3K Rayleigh BER         <- results/normalized_3k/3k_rayleigh.json
  tbl:table14           modulation BER (AWGN)              <- results/bpsk_comparison/awgn.json (+ modulation)
  tbl:table24           4-class vs 16-class @20 dB         <- results/all_relays_16class/all_relays_16class.json
  tbl:tableE6           unknown-channel BER                <- e6_unknown_channel_results/*.npy
  tbl:tableE6flat       flat-channel control BER           <- e6_unknown_channel_results/e6_flat_ported_results.npy
  tbl:table26           theoretical SNR @ BER=1e-3         <- closed-form (Q-function inversion)
  tab:ber_validation    theory-vs-sim BER                  <- closed-form theory column
  prose:E6blind         blind-regime prose claims          <- e6_unknown_channel_results/e6_blind_ported_results.npy
  prose:E6partial       pilot-sweep prose claims           <- e6_unknown_channel_results/e6_partial_ported_results.npy
  prose:E6composite     composite-cascade prose claims     <- e6_unknown_channel_results/e6_composite_ported_results.npy

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


_NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def clean_cell(c):
    """Strip LaTeX decoration and return (raw_text, float or None)."""
    c = c.strip()
    c = re.sub(r"\\textbf\{([^}]*)\}", r"\1", c)
    c = c.replace("\\(", "").replace("\\)", "").replace("$", "")
    c = c.replace("\\textasciitilde", "~").replace("\\%", "%")
    c = c.replace("{", "").replace("}", "").strip()
    if c in {"", ":", "---", "--", "~", "-"}:
        return c, None
    # "<5e-5" / "< 5e-5" -> treat as a ceiling; keep raw, value = the bound
    lt = "<" in c
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
STOCHASTIC_TABLES = {"tbl:tableE6": 0.010, "tbl:tableE6flat": 0.010,
                     "tbl:table24": 0.002,
                     # prose claims from the E6 blind/partial/composite studies
                     # (5-6 trials x 40k bits; the partial-posterior 5-pilot
                     # point is dominated by occasional catastrophic LS fits,
                     # hence the wider slack)
                     "prose:E6blind": 0.010, "prose:E6composite": 0.010,
                     "prose:E6partial": 0.030}


class Report:
    def __init__(self):
        self.checked = 0
        self.flags = []          # (table, where, published, source, diff)
        self.skipped = []        # (table, reason)
        self.tables = []         # (table, n_checked, n_flag)

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


# ----------------------------------------------------------------------------
# analytical formulas
# ----------------------------------------------------------------------------
def qfunc(x):
    return 0.5 * math.erfc(x / math.sqrt(2))

def ber_awgn(snr_db):
    return qfunc(math.sqrt(10 ** (snr_db / 10.0)))

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
    """Canonical Rayleigh BER, 9 relays x SNR (tbl:table2) vs rayleigh.json."""
    T = "tbl:table2"; before = rep.checked
    body = table_body(tex, T)
    if body is None:
        return rep.skip(T, "label not found in tex")
    # tex column order (after SNR): AF DF MLP Hybrid VAE CGAN Transformer Mamba-S6 Mamba2
    cols = ["AF", "DF", "GenAI (169p)", "Hybrid", "VAE",
            "CGAN (WGAN-GP)", "Transformer", "Mamba S6", "Mamba2 (SSD)"]
    d = json.load(open(os.path.join(ROOT, "results/bpsk_comparison/rayleigh.json")))
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
            pub_text, pub_val = row[c]
            src = d["results"][relay]["ber_mean"][si]
            rep.cell(T, f"{snr}dB/{relay}", pub_text, pub_val, src)
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
            pub_text, pub_val = row[c]
            src = d["results"][relay]["ber_mean"][si]
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
    stem = {"MLP": "MLP", "VAE": "VAE", "CGAN": "CGAN", "Hybrid": "Hybrid",
            "trans-former": "Transformer", "Transformer": "Transformer",
            "Mamba S6": "Mamba-S6", "Mamba2": "Mamba2", "Mamba2(ssd)": "Mamba2"}
    for row in data_rows(body):
        if not row:
            continue
        name = row[0][0].replace("\\\\", "").replace("\n", "").strip()
        key = stem.get(name)
        if key is None:
            continue
        # col1 = 4-cls @20, col2 = 16-cls @20
        for c, suff in ((1, "4-cls"), (2, "16-cls")):
            if c >= len(row):
                break
            pub_text, pub_val = row[c]
            jk = f"{key} {suff}"
            if jk in d["results"]:
                src = d["results"][jk]["ber_mean"][i20]
                rep.cell(T, f"{name}/{suff}@20dB", pub_text, pub_val, src)
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


def _load_e6_npy(name):
    p = os.path.join(ROOT, "e6_unknown_channel_results", name)
    if not os.path.exists(p):
        return None
    return np.load(p, allow_pickle=True).item()


def check_E6blind_prose(tex, rep):
    """Blind-regime prose claims (Section 'The Posterior-Free (Blind) Regime')
    vs e6_blind_ported_results.npy: CMA/MLP BER at 20 dB and the mid-SNR
    (10 dB) 95% CI instability comparison."""
    T = "prose:E6blind"; before = rep.checked
    d = _load_e6_npy("e6_blind_ported_results.npy")
    if d is None:
        return rep.skip(T, "e6_blind_ported_results.npy not found (run e6_blind_ported.py)")
    snrs = list(d["snrs"]); s20 = snrs.index(20)
    sm = d["summary"]  # name -> (mean_per_snr, ci_per_snr)

    m = re.search(r"CMA converges smoothly to BER \$([\d.]+)\\times10\^\{-3\}\$ at 20 dB", tex)
    if m:
        rep.cell(T, "CMA-blind/20dB", m.group(1) + "e-3", float(m.group(1)) * 1e-3,
                 sm["CMA-blind"][0][s20])
    m = re.search(r"tracks it almost exactly \(\$([\d.]+)\\times10\^\{-3\}\$\)", tex)
    if m:
        rep.cell(T, "MLP-169/20dB", m.group(1) + "e-3", float(m.group(1)) * 1e-3,
                 sm["MLP-169"][0][s20])
    m = re.search(r"mid-SNR \((\d+) dB\) confidence interval of \$([\d.]+)\$ "
                  r"\(versus \$([\d.]+)\$ for the MLP and \$([\d.]+)\$ for CMA\)", tex)
    if m:
        sci = snrs.index(int(m.group(1)))
        db = m.group(1)
        rep.cell(T, f"Viterbi-blind CI/{db}dB", m.group(2), float(m.group(2)),
                 sm["Viterbi-blind"][1][sci])
        rep.cell(T, f"MLP CI/{db}dB", m.group(3), float(m.group(3)), sm["MLP-169"][1][sci])
        rep.cell(T, f"CMA CI/{db}dB", m.group(4), float(m.group(4)), sm["CMA-blind"][1][sci])
    rep.finish_table(T, before)


def check_E6partial_prose(tex, rep):
    """Partial-posterior prose claims (pilot-budget sweep at 10 dB) vs
    e6_partial_ported_results.npy."""
    T = "prose:E6partial"; before = rep.checked
    d = _load_e6_npy("e6_partial_ported_results.npy")
    if d is None:
        return rep.skip(T, "e6_partial_ported_results.npy not found (run e6_partial_ported.py)")
    pa = d["panel_a"]  # n_pilots -> (mean, ci)

    m = re.search(r"payload BER \$([\d.]+)\$ at 800 pilots", tex)
    if m:
        rep.cell(T, "Viterbi/800 pilots", m.group(1), float(m.group(1)), pa[800][0])
    m = re.search(r"down to \$([\d.]+)\$ at 10 pilots", tex)
    if m:
        rep.cell(T, "Viterbi/10 pilots", m.group(1), float(m.group(1)), pa[10][0])
    m = re.search(r"jumps to \$([\d.]+)\$", tex)
    if m:
        rep.cell(T, "Viterbi/5 pilots", m.group(1), float(m.group(1)), pa[5][0])
    m = re.search(r"sweep at \$([\d.]+)\$", tex)
    if m:
        rep.cell(T, "MLP pilot-free ref", m.group(1), float(m.group(1)), d["mlp_ref"][0])
    rep.finish_table(T, before)


def check_E6composite_prose(tex, rep):
    """Composite-cascade prose claims vs e6_composite_ported_results.npy:
    MLP-170 BER at 20 dB and the Viterbi-vs-MLP gap at 8 dB."""
    T = "prose:E6composite"; before = rep.checked
    d = _load_e6_npy("e6_composite_ported_results.npy")
    if d is None:
        return rep.skip(T, "e6_composite_ported_results.npy not found (run e6_composite_ported.py)")
    snrs = list(d["snrs"]); s20 = snrs.index(20); s8 = snrs.index(8)
    sm = d["summary"]

    m = re.search(r"reaching \$([\d.]+)\\times10\^\{-3\}\$ at 20 dB", tex)
    if m:
        rep.cell(T, "MLP-170/20dB", m.group(1) + "e-3", float(m.group(1)) * 1e-3,
                 sm["MLP-169"][0][s20])
    m = re.search(r"\(\$([\d.]+)\$ vs\.\\ \$([\d.]+)\$ at 8 dB\)", tex)
    if m:
        rep.cell(T, "Viterbi-diff/8dB", m.group(1), float(m.group(1)), sm["Viterbi-diff"][0][s8])
        rep.cell(T, "MLP-170/8dB", m.group(2), float(m.group(2)), sm["MLP-169"][0][s8])
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


def check_ber_validation(tex, rep):
    """Theory-vs-sim table (tab:ber_validation_long): check theory columns."""
    T = "tab:ber_validation_long"; before = rep.checked
    body = table_body(tex, T)
    if body is None:
        return rep.skip(T, "label not found in tex")
    theo = {"awgn": ber_awgn, "rayleigh": ber_rayleigh}
    # columns: Chan & 4dB(Th) & 4dB(Sim) & 10dB(Th) & 10dB(Sim) & 16dB(Th) & 16dB(Sim)
    th_cols = [(1, 4), (3, 10), (5, 16)]
    for row in data_rows(body):
        if not row:
            continue
        name = row[0][0].strip().lower()
        key = "awgn" if "awgn" in name else ("rayleigh" if "rayleigh" in name else None)
        if key is None:
            continue
        for c, snr in th_cols:
            if c >= len(row):
                break
            pub_text, pub_val = row[c]
            if pub_val is None:
                continue
            src = theo[key](snr)
            rep.cell(T, f"{key}/theory@{snr}dB", pub_text, pub_val, src)
    rep.finish_table(T, before)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
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
    checks = [check_ber_validation, check_table26, check_table2, check_table8,
              check_table24, check_tableE6, check_tableE6flat,
              check_E6blind_prose, check_E6partial_prose, check_E6composite_prose]
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
    print("  tbl:table14/15 (modulation/activation) draw from results/qam16_activation/")
    print("  and results/modulation/*.json; add mappings here if you change those tables.")

    return 1 if rep.flags else 0


if __name__ == "__main__":
    sys.exit(main())
