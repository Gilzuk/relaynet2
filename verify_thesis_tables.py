#!/usr/bin/env python3
"""verify_thesis_tables.py — single-source reproducibility check for the thesis.

Checks the explicitly listed numerical tables and selected prose against their
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
                        (learned-relay column               <- results/mlp_min_size_all_channels.json)
  tbl:seq-on-memory     sequence architectures on ISI       <- results/seq_models_on_memory.json
  tbl:joint-latency     memory+latency relay comparison     <- results/joint_latency_memory.json
  tbl:joint-memory      memory-sweep cost/BER table         <- results/joint_latency_memory.json
  prose:nlbias          nonlinear-bias (S3) numbers, Ch. 6  <- e6_unknown_channel_results/e6_sim_ported_results.npy
  prose:coded-fer       20-dB frame-error rates, Ch. 5      <- results/coded_reliable_regime.json
  prose:mamba-lookahead causal-readout energy shares       <- tap vectors (size-sweep JSON and script)
  prose:timing-mmse-median Fig. 6.7(b) speedup; Table 6.6 median lead <- e6_complexity_ported_results.npy, size sweep
  prose:seq-rerun       Sec. 6.4.2 rerun statements         <- results/seq_models_on_memory_{readout,bidirectional,
                                                               rerun_controls,repeat}.json, _repeat_probe.jsonl

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

# Matches the exponent of a cell written in scientific notation, in either the
# LaTeX form (3.20\times10^-5, braces already stripped by clean_cell) or the
# plain form (5e-5). Group 1 is the mantissa, group 2 the exponent.
_SCI_PARTS = re.compile(
    r"([+-]?\d+(?:\.\d+)?)\s*(?:\\times\s*10\^\s*[{(]?\s*(-?\d+)\s*[})]?|[eE]([+-]?\d+))"
)


def tol_for(text):
    """Half a displayed unit, including a scientific-notation exponent.

    A fixed absolute epsilon would swamp rare-event BERs. Use relative
    floating-point slack instead. Bounds are tested strictly by Report.cell.
    """
    plain = text.replace("{", "").replace("}", "")
    number = re.search(
        r"([+-]?\d+(?:\.\d*)?)(?:[eE]([+-]?\d+)|\s*\\times\s*10\^\s*([+-]?\d+))?",
        plain,
    )
    if number is None:
        raise ValueError(f"No displayed number in {text!r}")
    mantissa = number.group(1)
    decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
    exponent = int(number.group(2) or number.group(3) or 0)
    return 0.5 * 10.0 ** (exponent - decimals) + MC_SLACK


# ----------------------------------------------------------------------------
# LaTeX helpers
# ----------------------------------------------------------------------------
def load_tex():
    """Read only compiled chapters; ignore comments and no-op review macros.

    The --tex scratch override supplies chapter contents, while main.tex is
    still the authority for which chapters are active in the manuscript.
    """
    from scripts.overleaf_project import no_comments
    from scripts.strip_rev import strip_rev
    with open(os.path.join(ROOT, "thesis", "main.tex"), encoding="utf-8") as handle:
        main = no_comments(handle.read())
    names = re.findall(r'\\(?:include|input)\{chapters/([^}]+)\}', main)
    parts = []
    for name in names:
        with open(os.path.join(TEX_DIR, name + ".tex"), encoding="utf-8") as handle:
            source = no_comments(handle.read())
        source = source.replace(r'\AK{', r'\REV{').replace(r'\GZ{', r'\REV{')
        parts.append(strip_rev(source))
    return "\n".join(parts)


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
    # re.search, not re.match: a cell may carry a leading word ("exploratory
    # <3.0\times10^-10"), and an anchored pattern then fell through to _NUM,
    # which reads the mantissa alone and returned the bound 3 for 3e-10.
    sci = re.search(r"([+-]?\d+(?:\.\d+)?)\s*\\times\s*10\^\s*[{(]?\s*(-?\d+)\s*[})]?", c)
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


# ----------------------------------------------------------------------------
# Coverage floors
# ----------------------------------------------------------------------------
# The number of cells each check is expected to examine. A check that examines
# fewer has stopped seeing part of what it covers -- a renamed row label, a
# moved anchor, a table it can no longer parse -- and until now that failed
# silently, because only *mismatches* were reported and a check that examined
# nothing reported "OK". Three real defects hid here: a parameter-count check
# that validated constants against themselves, two unguarded find() calls that
# could slice the wrong region, and an earlier joint-latency check that read 3
# of 10 rows while printing OK.
#
# Raise a floor when a check legitimately gains cells; never lower one to make
# a failure go away -- a drop means the check lost sight of something.
MIN_CELLS = {
    "tbl:table2": 54, "tbl:layers": 15, "tbl:table8": 48,
    # The unvalidated 20-dB MLP bound is intentionally no longer a numeric cell.
    "tbl:tableE6": 19, "tbl:tableE6flat": 24, "tbl:tableE6qpsk": 20,
    # E6partial 13 -> 12 (2026-10-03): the 10-pilot sentence no longer restates the MLP reference.
    # E6partial 12 -> 13 (2026-10-13): the 10-pilot 95% interval is now stated and checked.
    "prose:E6blind": 12, "prose:E6partial": 13, "prose:E6composite": 6,
    "tbl:table34": 30, "tbl:table37": 6, "tbl:table38": 10,
    "tbl:table39": 30, "tbl:table40": 12, "tbl:table41": 15,
    "tbl:table42": 27, "tbl:table43": 18, "tbl:table44": 20,
    # mmse-baseline 12 -> 15 (2026-10-13): the learned-relay column is recomputed from the size sweep.
    "tbl:mmse-baseline": 15, "tbl:seq-on-memory": 12,
    # arch:relay-param-counts was 7. Two of those cells came from a shape claim
    # inside a \REV{} annotation, which renders as nothing and was removed with
    # the rest of the revision history; the rendered appendix claim remains.
    "arch:relay-param-counts": 5,
    "consistency:proof-copies": 7, "tbl:slicer-floor-inline": 24,
    "prose:qpsk-decomposition": 4, "prose:qpsk-controls": 3, "prose:scalar-map": 7,
    "prose:af-conventional": 21, "prose:isi-penalty": 18, "prose:hybrid-branch": 21, "tbl:coded-matched": 46, "prose:flat-conclusion": 2, "coded-matched-secondary": 40,
    "prose:seed-spread-native": 6,
    "tbl:bcjr-benchmark": 12, "prose:soft-df-calibration": 2,
    "prose:mmse-monotonicity": 12,
    "tbl:joint-latency": 30, "tbl:joint-memory": 21,
    "prose:nlbias": 3, "prose:coded-fer": 2, "prose:mamba-lookahead": 2,
    # seq-rerun (2026-10-13): Sec. 6.4.2 statements resting on the sequence-on-memory rerun.
    "prose:seq-rerun": 19,
    # red-team fixes (2026-10-14): corrected blind intervals, fixed-gain AF, Fig. 6.7(b), Table 6.6 median.
    "prose:timing-mmse-median": 6,
    "prose:khina-round2": 60,
}

# Checks allowed to skip, with the reason. A skip means the check could not run
# at all; anything not listed here is a failure rather than a quiet line in the
# report.
ALLOWED_SKIPS = {}


class Report:
    def __init__(self):
        self.checked = 0
        self.flags = []          # (table, where, published, source, diff)
        self.skipped = []        # (table, reason)
        self.tables = []         # (table, n_checked, n_flag)
        self.notes = []          # (table, partial-coverage reason)
        self.coverage = []       # (table, n_checked, floor) - examined too few
        self.unexpected_skips = []  # (table, reason) - could not run at all

    def cell(self, table, where, pub_text, pub_val, src_val):
        # unresolved / non-numeric published cell -> skip silently
        if pub_val is None or src_val is None:
            return
        self.checked += 1
        tol = tol_for(pub_text) + STOCHASTIC_TABLES.get(table, 0.0)
        if isinstance(pub_val, str) and pub_val.startswith("<"):
            bound = float(pub_val[1:])
            # A published strict upper bound is not a rounded point estimate.
            ok = src_val < bound
            diff = max(0.0, src_val - bound)
        else:
            diff = abs(float(pub_val) - float(src_val))
            # Allow one ulp-scale relative cushion for decimal ties produced
            # by binary floating-point arithmetic; this does not change the
            # displayed-unit tolerance or the strict upper-bound rule above.
            ok = diff <= tol * (1.0 + 1e-12)
        if not ok:
            self.flags.append((table, where, pub_text, f"{src_val:.5g}", f"{diff:.2g}"))

    def finish_table(self, table, before):
        n = self.checked - before
        nf = sum(1 for f in self.flags if f[0] == table)
        self.tables.append((table, n, nf))
        floor = MIN_CELLS.get(table)
        if floor is not None and n < floor:
            self.coverage.append((table, n, floor))

    def zero_error_bound(self, table, where, pub_text, pub_val, bits):
        """Validate the printed nominal rule-of-three bound, not 0 < anything.

        This checks transcription and exposure only. It does not establish
        independent-bit coverage for a correlated-error experiment.
        """
        if bits <= 0:
            raise ValueError("A zero-error bound requires a positive bit budget")
        self.checked += 1
        expected = 3.0 / bits
        is_bound = isinstance(pub_val, str) and pub_val.startswith("<")
        value = float(pub_val[1:]) if is_bound else float(pub_val or 0)
        diff = abs(value - expected)
        # No stochastic absolute tolerance: these are exact exposure records.
        ok = is_bound and diff <= tol_for(pub_text) * (1 + 1e-12)
        if not ok:
            self.flags.append((table, where, pub_text, f"nominal <{expected:g}", f"{diff:g}"))

    def skip(self, table, reason):
        self.skipped.append((table, reason))
        if table not in ALLOWED_SKIPS:
            self.unexpected_skips.append((table, reason))

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


def matched_viterbi(name, snr):
    """genie / pilot-LS Viterbi on the current Hop-2 noise convention.

    e6_viterbi_awgn.npy is NOT the source for these any more. Those vectors
    were generated when the real AWGN branch put the whole of N0 in its single
    dimension, which is 3.01 dB pessimistic, so they were not comparable with
    the relay rows printed beside them -- the stored 0.0072 at 8 dB is 0.001354
    on the current convention. e6_matched_protocol.py re-measured them, and
    e6_matched_highsnr.py gives 16 and 20 dB a predeclared 1e8 bits so those
    cells resolve to 3e-8 bounds rather than the 3e-6 a 1e6-bit budget allows.

    Returns the empirical rate. Zero-error transcription checks must also use
    matched_viterbi_counts and Report.zero_error_bound to validate the exposure.
    """
    if snr in (16, 20):
        r = _MATCHED_HI["results"][name][str(snr)]
        return r["ber"] if r["errors"] else 0.0
    i = _MATCHED["snrs"].index(snr)
    e = _MATCHED["results"][name]["errors"][i]
    b = _MATCHED["results"][name]["bits"][i]
    return (e / b) if e else 0.0


def matched_viterbi_counts(name, snr):
    if snr in (16, 20):
        record = _MATCHED_HI["results"][name][str(snr)]
        return record["errors"], record["bits"]
    i = _MATCHED["snrs"].index(snr)
    record = _MATCHED["results"][name]
    return record["errors"][i], record["bits"][i]


_MATCHED = json.load(open(os.path.join(ROOT, "results/e6_matched_protocol.json")))
_MATCHED_HI = json.load(open(os.path.join(ROOT, "results/e6_matched_highsnr.json")))


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
    awgn_viterbi_name = ("codex_viterbi_consistency_awgn.npy"
                         if os.path.exists(os.path.join(
                             ROOT, "e6_unknown_channel_results/codex_viterbi_consistency_awgn.npy"))
                         else "e6_viterbi_awgn.npy")
    vit = _load_e6_npy(awgn_viterbi_name)
    if sim is None or blind is None or partial is None or vit is None:
        return rep.skip(T, "one or more e6 npy files missing")

    snrs = list(sim["snrs"])
    i8, i20 = snrs.index(8), snrs.index(20)
    S1 = sim["results"]["S1: unknown ISI -> AWGN"]
    fixed = json.load(open(os.path.join(
        ROOT, "e6_unknown_channel_results/codex_isi_fixed_budget_validation.json")))

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
    m = re.search(r"restores the link \(\$([\d.]+)\$ at 8~dB", body)
    if m:
        rep.cell(T, "L2/MLP@8dB", m.group(1), float(m.group(1)), S1["MLP"][0][i8])
    # This cell was previously read off e6_viterbi_awgn.npy and printed as
    # 0.0072, which made it look as though the MLP led at 8 dB. On the current
    # noise convention it is 0.001354 and the ordering is the other way round.
    m = re.search(r"\$([\d.]+)\$ against that \$[\d.]+\$ at 8~dB", body)
    if m:
        rep.cell(T, "L2/VITgenie@8dB", m.group(1), float(m.group(1)),
                 matched_viterbi("VIT-genie", 8))
    m = re.search(r"below \$3\.0\\times10\^\{-8\}\$ at 16~dB against the MLP's "
                  r"\$([\d.]+)\\times10\^\{-7\}\$", body)
    if m:
        rep.cell(T, "L2/MLP@16dB", m.group(1) + "e-7", float(m.group(1)) * 1e-7,
                 fixed["points"]["16"]["MLP"]["mean"])

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
    m = re.search(r"At 8~dB the MLP reaches \$([\d.]+)\$ against CMA's \$([\d.]+)\$", body)
    if m:
        b8 = bsnr.index(8)
        rep.cell(T, "L4/MLP@8dB", m.group(1), float(m.group(1)), bs["MLP-169"][0][b8])
        rep.cell(T, "L4/CMA@8dB", m.group(2), float(m.group(2)), bs["CMA-blind"][0][b8])
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
            "MLSE D=2": 2**4, "MLSE D=3": 2**4, "MLSE D=15": 2**4,
            "block DF": 16, "block DF + MLSE": 16 + 2**4}
    # MLSE rows give the exactly factorized count 2^(L+1) first and the
    # implemented joint trellis 2*M^L in brackets (Khina round 2).
    joint = {"MLSE D=2": 2*4**3, "MLSE D=3": 2*4**3, "MLSE D=15": 2*4**3,
             "block DF + MLSE": 16 + 2*4**3}

    for row in data_rows(body):
        if not row:
            continue
        key = label_map.get(row[0][0].strip())
        if key is None or key not in rows or len(row) < 4:
            continue
        src = rows[key]
        rep.cell(T, f"{key}/delay", row[1][0], row[1][1], src["latency_symbols"])
        rep.cell(T, f"{key}/macs", row[2][0], row[2][1], macs[key])
        if key in joint:
            m = re.search(r"\((\d+)\)", row[2][0])
            if m:
                rep.cell(T, f"{key}/macs joint", m.group(1), float(m.group(1)), joint[key])
            else:
                rep.cell(T, f"{key}/macs joint", "missing", float("nan"), joint[key])
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
        rep.cell(T, f"L={L}/mlse_macs_factorized", row[1][0], row[1][1], 2 ** (L + 1))
        rep.cell(T, f"L={L}/mlse_macs_joint", row[2][0], row[2][1], 2 * 4 ** L)
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
            src = d[snr][key]["mean"]
            rep.cell(T, f"{snr}dB/{key}", pub_text, pub_val, src)
    rep.finish_table(T, before)


def check_coded_matched(tex, rep):
    """Matched-metric coded comparison (tbl:coded-matched) and its prose.

    Source: results/coded_matched_metrics.json (coded_matched_metrics.py). The
    learned-relay cells hold two values, "plain / weighted" destination metric.
    """
    T = "tbl:coded-matched"; before = rep.checked
    p = os.path.join(ROOT, "results", "coded_matched_metrics.json")
    if not os.path.exists(p):
        return rep.skip(T, "results/coded_matched_metrics.json not found")
    body = table_body(tex, T)
    if body is None:
        return rep.skip(T, "label not found in tex")
    d = json.load(open(p)); S = d["summary"]
    cols = {1: ["coded_df|plain"], 2: ["coded_df_m|matched"], 3: ["soft_df_m|matched"],
            4: ["mlp_hard|plain", "mlp_hard|matched"],
            5: ["mlp_soft|plain", "mlp_soft|matched"], 6: ["oracle|matched"]}
    for row in data_rows(body):
        if not row or row[0][1] is None:
            continue
        snr = str(int(row[0][1]))
        if snr not in S:
            continue
        for c, keys in cols.items():
            if c >= len(row):
                break
            parts = row[c][0].split("/") if len(keys) > 1 else [row[c][0]]
            for part, key in zip(parts, keys):
                txt, val = clean_cell(part)
                rep.cell(T, f"{snr}dB/{key}", txt, val, S[snr][key]["mean"])
    # error counts quoted in the caption and prose
    bits = d["n_frames"] * d["frame_info_bits"]
    count = lambda snr, k: round(sum(d["per_trial"][snr][k]) * bits)
    for pat, src in [
        (r"matched block DF makes (\d+) and (\d+) errors", [count("16", "coded_df_m|matched"), count("20", "coded_df_m|matched")]),
        (r"and the oracle (\d+) and (\d+)\}", [count("16", "oracle|matched"), count("20", "oracle|matched")]),
        (r"it makes (\d+) errors in \$3\\times10\^\{6\}\$ bits \(\$1\.0\\times10\^\{-6\}\$\), against \$([\d{},]+)\$",
         [count("20", "coded_df_m|matched"), min(count("20", k) for k in ("mlp_hard|plain", "mlp_hard|matched", "mlp_soft|plain", "mlp_soft|matched"))]),
    ]:
        m = re.search(pat, tex)
        if m:
            for g, v in zip(m.groups(), src):
                txt, val = clean_cell(g)
                rep.cell(T, "error count", txt, val, v)
    rep.finish_table(T, before)

def check_flat_conclusion_prose(tex, rep):
    """MLP-vs-DF differences quoted in the flat-channel conclusion (Ch. 6).

    Source: e6_unknown_channel_results/e6_flat_ported_results.npy (C7 rerun).
    """
    T = "prose:flat-conclusion"; before = rep.checked
    p = os.path.join(ROOT, "e6_unknown_channel_results/e6_flat_ported_results.npy")
    if not os.path.exists(p):
        return rep.skip(T, "e6_flat_ported_results.npy not found")
    d = np.load(p, allow_pickle=True).item()
    snrs = list(d["snrs"])

    def mean(x):
        x = np.asarray(x[0] if isinstance(x, (tuple, list)) else x)
        return x.mean(1) if x.ndim > 1 else x

    diff = {k: mean(v["MLP"]) - mean(v["DF"]) for k, v in d["results"].items()}
    tab = max(np.abs(v[[snrs.index(x) for x in (8, 12, 16, 20)]]).max() for v in diff.values())
    m = re.search(r"the largest difference is \$([\d.]+)\$ across the tabulated points", tex)
    if m:
        rep.cell(T, "max |MLP-DF| at 8/12/16/20 dB", m.group(1), float(m.group(1)), tab)
    m = re.search(r"by up to \$([\d.]+)\$ at 0~dB", tex)
    if m:
        rep.cell(T, "phase MLP-DF at 0 dB", m.group(1), float(m.group(1)), diff["phase"][snrs.index(0)])
    rep.finish_table(T, before)

def check_coded_matched_secondary(tex, rep):
    """Matched-metric values added to Tables 38, 40, 42, 43 and the coded prose (C4b).

    Sources: results/coded_*_matched.json and results/coded_matched_envelopes.json.
    """
    T = "coded-matched-secondary"; before = rep.checked
    R = lambda f: json.load(open(os.path.join(ROOT, "results", f)))
    try:
        em = R("coded_error_mechanism_matched.json")["results"]
        kq = R("coded_df_experiment_matched.json")
        lt = {r["snr_db"]: r for r in R("coded_latency_throughput_matched.json")["equal_throughput"]}
        env = R("coded_matched_envelopes.json")["envelopes"]
        ms = R("coded_min_size_matched.json")
    except FileNotFoundError as e:
        return rep.skip(T, f"missing {e.filename}")
    # Table 38 matched rows
    body = table_body(tex, "tbl:table38")
    snr = None
    names = {"coded-dfmatched": "coded_df", "mlp-codedweighteddest": "mlp_coded_wdest", "oraclematched": "oracle"}
    for row in data_rows(body or ""):
        m = re.match(r"(\d+)\s*dB", row[0][0].strip())
        if m:
            snr = m.group(1)
        key = names.get(re.sub(r"[^a-z0-9-]", "", row[1][0].strip().lower())) if len(row) > 4 else None
        if key is None or snr not in em:
            continue
        src = em[snr][key]
        if row[2][1] is not None:
            rep.cell(T, f"t38/{snr}/{key}/relay_sym_er", row[2][0], row[2][1], src["relay_sym_er"])
        if row[4][1] is not None:
            rep.cell(T, f"t38/{snr}/{key}/ber", row[4][0], row[4][1], src["ber"])
    # Table 40 matched columns
    for row in data_rows(table_body(tex, "tbl:table40") or ""):
        if len(row) >= 6 and row[0][1] is not None and int(row[0][1]) in lt:
            r = lt[int(row[0][1])]
            rep.cell(T, f"t40/{int(row[0][1])}/coded_matched", row[4][0], row[4][1], r["coded_qam16"])
            rep.cell(T, f"t40/{int(row[0][1])}/gain_matched", row[5][0], row[5][1],
                     r["uncoded_qpsk"] / r["coded_qam16"])
    # Tables 42/43 matched block-DF column ("MCS, G")
    for lab, key, col in (("tbl:table42", "unconstrained", 6), ("tbl:table43", "150", 5)):
        rows = {r["snr_db"]: r["blockdf"] for r in env[key]}
        for row in data_rows(table_body(tex, lab) or ""):
            if len(row) > col and row[0][1] is not None and int(row[0][1]) in rows:
                g = re.search(r",\s*([\d.]+)\s*$", row[col][0])
                if g:
                    rep.cell(T, f"{lab}/{int(row[0][1])}/blockdf_matched", g.group(1), float(g.group(1)),
                             rows[int(row[0][1])]["goodput"])
    # prose: K sweep at 8 dB and the coded size-sweep penalty range
    m = re.search(r"at 8~dB the BER is \$([\d.]+)\$, \$([\d.]+)\$ and \$([\d.]+)\\times10\^\{(-\d+)\}\$ for \$K=3,5,7\$", tex)
    if m:
        i8 = kq["snr_db"].index(8)
        vals = [float(m.group(1)), float(m.group(2)), float(m.group(3)) * 10 ** int(m.group(4))]
        for v, txt, K in zip(vals, [m.group(1), m.group(2), m.group(3)], ("K3", "K5", "K7")):
            rep.cell(T, f"ksweep/{K}@8dB", txt, v, kq["k_sweep"][K]["ber"][i8])
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
    """Unknown-channel BER (tbl:tableE6). Sources: historical npy + fixed JSON."""
    T = "tbl:tableE6"; before = rep.checked
    body = table_body(tex, T)
    if body is None:
        return rep.skip(T, "label not found in tex")
    sim = np.load(os.path.join(ROOT, "e6_unknown_channel_results/e6_sim_ported_results.npy"),
                  allow_pickle=True).item()
    # Prefer the explicitly prefixed rerun when it exists.  The historical
    # array is retained as an audit artifact, but was generated with the old
    # real-AWGN convention and must not continue to drive the published AWGN
    # reference rows after the channel implementation was corrected.
    awgn_viterbi_name = ("codex_viterbi_consistency_awgn.npy"
                         if os.path.exists(os.path.join(
                             ROOT, "e6_unknown_channel_results/codex_viterbi_consistency_awgn.npy"))
                         else "e6_viterbi_awgn.npy")
    vg_awgn = np.load(os.path.join(ROOT, "e6_unknown_channel_results", awgn_viterbi_name),
                      allow_pickle=True).item()
    vg_ray = np.load(os.path.join(ROOT, "e6_unknown_channel_results/e6_viterbi_rayleigh.npy"),
                     allow_pickle=True).item()
    fixed = json.load(open(os.path.join(
        ROOT, "e6_unknown_channel_results/codex_isi_fixed_budget_validation.json")))
    snrs = list(sim["snrs"])
    compact = "Setup & Relay" not in body
    col_snr = [(1, 8), (2, 12), (3, 16), (4, 20)] if compact else [(2, 8), (3, 12), (4, 16), (5, 20)]

    # setup label in tex -> (sim results key, viterbi dict or None)
    setup_map = {
        "Unknown ISI $\\to$ AWGN": ("S1: unknown ISI -> AWGN", vg_awgn),
        "Unknown ISI $\\to$ Rayleigh": ("S2: unknown ISI -> Rayleigh", vg_ray),
        "Control: canonical Rayleigh": ("S4 control: Rayleigh -> Rayleigh (canonical)", None),
    }
    cur_setup = "Unknown ISI $\\to$ AWGN" if compact else None
    for row in data_rows(body):
        if not row:
            continue
        first = row[0][0].strip()
        if first and not compact:  # new setup group
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
        relay = row[0 if compact else 1][0].strip()
        sim_key, vg = setup_map[cur_setup]
        res = sim["results"].get(sim_key, {})
        # map tex relay name -> source
        def src_at(si):
            r = relay.upper()
            snr = str(int(snrs[si]))
            replacement = (
                cur_setup == "Unknown ISI $\\to$ AWGN"
                and snr in fixed["points"]
                and ("AF" if r.startswith("AF") else
                     "DF" if r.startswith("DF") else
                     "MLP" if "MLP" in r else None)
            )
            if replacement:
                return fixed["points"][snr][replacement]["mean"]
            if r.startswith("AF"):
                return res["AF"][0][si]
            if r.startswith("DF"):
                return res["DF"][0][si]
            if "MLP" in r:
                return res["MLP"][0][si]
            # AWGN setup: the Viterbi rows come from the matched-protocol
            # rerun, not from the superseded-convention npy (see
            # matched_viterbi). The Rayleigh setup still uses its own npy.
            awgn = cur_setup == "Unknown ISI $\\to$ AWGN"
            if "GENIE" in r:
                if awgn:
                    return matched_viterbi("VIT-genie", int(snr))
                return float(vg["VIT-genie"][si]) if vg is not None else None
            if "PILOT" in r or "EST" in r or "200" in r:
                if awgn:
                    return matched_viterbi("VIT-est", int(snr))
                return float(vg["VIT-est"][si]) if vg is not None else None
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
                if cur_setup == "Unknown ISI $\\to$ AWGN" and "VITERBI" in relay.upper():
                    name = "VIT-genie" if "GENIE" in relay.upper() else "VIT-est"
                    errors, bits = matched_viterbi_counts(name, snr)
                    if errors == 0:
                        rep.zero_error_bound(T, f"{relay}/{snr}dB", pub_text, pub_val, bits)
                        continue
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

    m = re.search(r"corrected CMA (?:converges smoothly to|reaches) BER \$([\d.]+)\\times10\^\{-3\}\$ at 20 dB", tex)
    if m:
        rep.cell(T, "CMA-blind/20dB", m.group(1) + "e-3", float(m.group(1)) * 1e-3,
                 sm["CMA-blind"][0][i20])
    m = re.search(r"MLP reaches \$([\d.]+)\\times10\^\{-3\}\$", tex)
    if m:
        rep.cell(T, "MLP-169/20dB", m.group(1) + "e-3", float(m.group(1)) * 1e-3,
                 sm["MLP-169"][0][i20])
    m = re.search(r"(?:\(\$([\d.]+)\$ vs\.\\ \$([\d.]+)\$ at 8 dB\)\. This is the correct answer"
                  r"|MLP at \$([\d.]+)\$ against CMA's \$([\d.]+)\$ at 8 dB)", tex)
    if m:
        mlp8, cma8 = (m.group(1) or m.group(3)), (m.group(2) or m.group(4))
        rep.cell(T, "MLP-169/8dB", mlp8, float(mlp8), sm["MLP-169"][0][i8])
        rep.cell(T, "CMA-blind/8dB", cma8, float(cma8), sm["CMA-blind"][0][i8])
    # Two phrasings of the same pair of numbers; the second branch fills groups
    # 3/4, so read whichever branch matched rather than assuming groups 1/2.
    m = re.search(r"(?:falling to \$([\d.]+)\$ at 16 dB before rising again to \$([\d.]+)\$ at 20 dB"
                  r"|\$([\d.]+)\$ at 16 dB and \$([\d.]+)\$ at 20 dB)", tex)
    if m:
        v16, v20 = (m.group(1) or m.group(3)), (m.group(2) or m.group(4))
        rep.cell(T, "Viterbi-blind/16dB", v16, float(v16), sm["Viterbi-blind"][0][i16])
        rep.cell(T, "Viterbi-blind/20dB", v20, float(v20), sm["Viterbi-blind"][0][i20])
    # Intervals and resolution claims come from the corrected-interval rerun
    # (e6_blind_intervals.py): classical intervals on the 50 unique draws, the
    # MLP's hierarchical over its three trained instances. The stored npy
    # intervals pool triplicated classical draws and are not used for these.
    ip = os.path.join(ROOT, "results", "e6_blind_intervals.json")
    iv = json.load(open(ip))["rows"] if os.path.exists(ip) else None
    m = re.search(r"interval at 8 dB is \$\\pm([\d.]+)\$.*?MLP's \$\\pm([\d.]+)\$.*?CMA's \$\\pm([\d.]+)\$", tex, re.S)
    if m and iv:
        r8 = iv["8"]
        rep.cell(T, "Viterbi-blind CI/8dB (unique draws)", m.group(1), float(m.group(1)), r8["Viterbi-blind"]["ci_unique_draws"])
        rep.cell(T, "MLP CI/8dB (hierarchical)", m.group(2), float(m.group(2)), r8["MLP-169"]["ci_hierarchical"])
        rep.cell(T, "CMA CI/8dB (unique draws)", m.group(3), float(m.group(3)), r8["CMA-blind"]["ci_unique_draws"])
    m = re.search(r"resolved from 0 to 12 dB \(Wilcoxon \$p \\le ([\d.]+)\$\) but not from 14 to 20 dB \(\$p \\ge ([\d.]+)\$\)", tex)
    if m and iv:
        lo = [iv[str(k)]["MLP_minus_CMA"]["wilcoxon_p_draws"] for k in (0, 2, 4, 6, 8, 10, 12)]
        hi = [iv[str(k)]["MLP_minus_CMA"]["wilcoxon_p_draws"] for k in (14, 16, 18, 20)]
        rep.cell(T, "max p, 0-12 dB (bound)", m.group(1), "<%g" % (float(m.group(1)) + 1e-12), max(lo))
        rep.cell(T, "min p, 14-20 dB", m.group(2), float(m.group(2)), min(hi))
    m = re.search(r"at 20 dB the MLP is better on only (\d+) of the 50 draws", tex)
    if m and iv:
        rep.cell(T, "draws MLP better at 20 dB", m.group(1), float(m.group(1)), iv["20"]["MLP_minus_CMA"]["draws_mlp_better"])
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

    m = re.search(r"payload BER \$([\d.]+)\$ at 200 pilots", tex)
    if m:
        rep.cell(T, "Viterbi/200 pilots", m.group(1), float(m.group(1)), pa[200][0])
    m = re.search(r"(?:down to|and) \$([\d.]+)\$ at 20 pilots", tex)
    if m:
        rep.cell(T, "Viterbi/20 pilots", m.group(1), float(m.group(1)), pa[20][0])
    m = re.search(r"MLP's pilot-free \$([\d.]+)\$", tex)
    if m:
        rep.cell(T, "MLP pilot-free ref", m.group(1), float(m.group(1)), d["mlp_ref"][0])
    m = re.search(r"At 10 pilots Viterbi has (?:already )?lost its edge, \$([\d.]+)\$ against the MLP's \$([\d.]+)\$", tex)
    if m:
        rep.cell(T, "Viterbi/10 pilots", m.group(1), float(m.group(1)), pa[10][0])
        rep.cell(T, "MLP ref (10-pilot cmp)", m.group(2), float(m.group(2)), d["mlp_ref"][0])
    m = re.search(r"At 10 pilots its BER is \$([\d.]+)\$", tex)
    if m:
        rep.cell(T, "Viterbi/10 pilots", m.group(1), float(m.group(1)), pa[10][0])
    m = re.search(r"with a 95\\% interval of \$\\pm ([\d.]+)\$ that contains the MLP's value", tex)
    if m:
        rep.cell(T, "Viterbi CI/10 pilots", m.group(1), float(m.group(1)), pa[10][1])
    m = re.search(r"at 5 pilots it (?:collapses to|reaches) \$([\d.]+)\$", tex)
    if m:
        rep.cell(T, "Viterbi/5 pilots", m.group(1), float(m.group(1)), pa[5][0])
    m = re.search(r"\$([\d.]+) \\pm ([\d.]+)\$, against \$\\pm ([\d.]+)\$ at 50 pilots", tex)
    if m:
        rep.cell(T, "Viterbi/5 pilots (CI ctx)", m.group(1), float(m.group(1)), pa[5][0])
        rep.cell(T, "Viterbi CI/5 pilots", m.group(2), float(m.group(2)), pa[5][1])
        rep.cell(T, "Viterbi CI/50 pilots", m.group(3), float(m.group(3)), pa[50][1])
    m = re.search(r"(?:flat across the entire sweep at|MLP (?:reference )?remains at) \$([\d.]+)\$", tex)
    if m:
        rep.cell(T, "MLP flat ref", m.group(1), float(m.group(1)), d["mlp_ref"][0])
    # panel (b): blind CMA per-block convergence failure
    m = re.search(r"CMA records \$([\d.]+)\$ at \$L=40\$ and \$([\d.]+)\$ at \$L=1000\$", tex)
    if m and pbc:
        rep.cell(T, "CMA/L=40", m.group(1), float(m.group(1)), pbc[40][0])
        rep.cell(T, "CMA/L=1000", m.group(2), float(m.group(2)), pbc[1000][0])
    m = re.search(r"versus its \$([\d.]+)\$ long-block reference", tex)
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
    m = re.search(r"\(\$([\d.]+)\$ vs\.\\ \$([\d.]+)\$(?: at 8 dB)?\) and both round", tex)
    if m:
        rep.cell(T, "Viterbi-diff/8dB", m.group(1), float(m.group(1)), sm["Viterbi-diff"][0][s8])
        rep.cell(T, "MLP-169/8dB", m.group(2), float(m.group(2)), sm["MLP-169"][0][s8])
    m = re.search(r"both round to \$([\d.]+)\$ at 20 dB", tex)
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
    m = re.search(r"ending at a rounded \$([\d.]+)\$ at 20 dB", tex)
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
    The learned-relay column is recomputed from results/mlp_min_size_all_channels.json:
    the lowest best-of-three-seeds penalty against the sweep's MLSE baseline over
    every size-sweep configuration on the channel (worst penalty over the reachable
    targets, the Appendix 9.6 definition), as the table caption states.
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
        if len(row) > 5 and row[5][1] is not None:
            best = _best_sweep_penalty(src_key)
            if best is not None:
                rep.cell(T, f"{src_key}/learned relay", row[5][0], row[5][1], best)
    rep.finish_table(T, before)


def _best_sweep_penalty(channel):
    """Lowest best-of-seeds required-SNR penalty over the size sweep (Table 6.6)."""
    from ber_metrics import penalty_table
    path = os.path.join(ROOT, "results", "mlp_min_size_all_channels.json")
    if not os.path.exists(path):
        return None
    ms = json.load(open(path))
    c = ms["channels"].get(channel)
    if c is None:
        return None
    snr, base = ms["snr_db"], np.array(c["baseline_ber"])
    vals = []
    for cfg in c["sweep"]:
        per_seed = [penalty_table(snr, np.array(r["ber"]), base)["worst_db_penalty"]
                    for r in cfg["seed_runs"]]
        per_seed = [v for v in per_seed if v == v]
        if per_seed:
            vals.append(min(per_seed))
    return min(vals) if vals else None



def check_scalar_map_prose(tex, rep):
    """Scalar MAP relay figures quoted in Ch. 6 prose (results/e6_scalar_map.json).

    The paragraph answers the reviewer's question whether the MLP's gain over
    zero-threshold DF on the ISI channel is a decision-rule effect or a
    temporal-context effect, so its seven numbers are pinned to the run.
    """
    T = "prose:scalar-map"; before = rep.checked
    src_path = os.path.join(ROOT, "results", "e6_scalar_map.json")
    if not os.path.exists(src_path):
        return rep.skip(T, "results/e6_scalar_map.json not found")
    d = json.load(open(src_path))
    R, snrs = d["results"], d["snrs"]
    ber = lambda k, snr: R[k]["ber"][snrs.index(snr)]
    i = tex.find("it was measured under the matched protocol (\\texttt{e6\\_scalar\\_map.py})")
    if i < 0:
        return rep.skip(T, "scalar-MAP sentence not found in tex")
    stop = tex.find("temporal context its window spans", i)
    if stop < 0:
        return rep.skip(T, "scalar-MAP paragraph does not reach its closing clause")
    seg = tex[i:stop]
    nums = re.findall(r"\$(\d+\.\d+)(?:\\times10\^\{(-?\d+)\})?\$", seg)
    if len(nums) != 7:
        return rep.skip(T, f"expected 7 numbers in the paragraph, found {len(nums)}")
    want = [("smap@8dB", ber("scalar-MAP", 8)), ("smap@12dB", ber("scalar-MAP", 12)),
            ("smap@16dB", ber("scalar-MAP", 16)), ("smap@20dB", ber("scalar-MAP", 20)),
            ("DF@20dB", ber("DF", 20)), ("smap@12dB(again)", ber("scalar-MAP", 12)),
            ("MLP@12dB", ber("MLP", 12))]
    for (mant, exp), (what, src) in zip(nums, want):
        text = mant + (f"\\times10^{{{exp}}}" if exp else "")
        val = float(mant) * (10.0 ** int(exp) if exp else 1.0)
        rep.cell(T, what, text, val, src)
    rep.finish_table(T, before)

def check_af_conventional_prose(tex, rep):
    """AF block-length and conventional-AF figures in the Ch. 8 H1 discussion.

    Source: results/af_block_and_conventional.json, plus the committed canonical
    MLP and AF means (results/modulation/qpsk_rayleigh.json) for the two margins.
    Each number is located by the words around it rather than by position.
    """
    T = "prose:af-conventional"; before = rep.checked
    p = os.path.join(ROOT, "results", "af_block_and_conventional.json")
    if not os.path.exists(p):
        return rep.skip(T, "results/af_block_and_conventional.json not found")
    d = json.load(open(p))
    c = json.load(open(os.path.join(ROOT, "results", "modulation", "qpsk_rayleigh.json")))["results"]
    i = lambda snr: d["snrs"].index(snr)
    imp, vg = d["implemented_af_by_block"], d["conventional_af_variable_gain"]
    fg = d["conventional_af_fixed_gain"]
    mlp0, af0 = c["MLP (169p)"]["ber_mean"][0], c["AF"]["ber_mean"][0]
    checks = [
        (r"\$([\d.]+)\$ for 50-symbol blocks", ["impl@0dB,N=50"], [imp["50"][i(0)]]),
        (r"\$([\d.]+)\$ at the canonical 5\{,\}000", ["impl@0dB,N=5000"], [imp["5000"][i(0)]]),
        (r"\$([\d.]+)\$ at 50\{,\}000 \(", ["impl@0dB,N=50000"], [imp["50000"][i(0)]]),
        (r"reaches \$([\d.]+)\$ at 0~dB, \$([\d.]+)\$ at 4~dB and \$([\d.]+)\$ at 8~dB",
         ["convAF@0dB", "convAF@4dB", "convAF@8dB"], [vg[i(0)], vg[i(4)], vg[i(8)]]),
        (r"margin at 0~dB falls from \$([\d.]+)\$ to \$([\d.]+)\$",
         ["margin vs implemented AF", "margin vs conventional AF"], [af0 - mlp0, vg[i(0)] - mlp0]),
        (r"does better at 0 and 2~dB \(\$([\d.]+)\$ and \$([\d.]+)\$\)",
         ["fixedAF@0dB", "fixedAF@2dB"], [fg[i(0)], fg[i(2)]]),
        (r"against the fixed-gain one it is \$([\d.]+)\$", ["margin vs fixed-gain AF"], [fg[i(0)] - mlp0]),
        (r"a fixed-gain AF does better at 0~dB \(\$([\d.]+)\$\), leaving \$([\d.]+)\$",
         ["fixedAF@0dB (Ch. 5)", "margin vs fixed-gain AF (Ch. 5)"], [fg[i(0)], fg[i(0)] - mlp0]),
    ]
    # The margin range quoted in both abstracts, Ch. 8 finding 1 and Table 8.1:
    # the smaller end against the fixed-gain AF, the larger against variable gain.
    for m in re.finditer(r"margin is (?:only )?\$([\d.]+)\$--\$([\d.]+)\$"
                         r"|\\textenglish\{0 dB\} הוא \\textenglish\{([\d.]+)--([\d.]+)\}", tex):
        lo_, hi_ = (m.group(1) or m.group(3)), (m.group(2) or m.group(4))
        rep.cell(T, "margin range low (fixed gain)", lo_, float(lo_), fg[i(0)] - mlp0)
        rep.cell(T, "margin range high (variable gain)", hi_, float(hi_), vg[i(0)] - mlp0)
    for pat, labels, srcs in checks:
        m = re.search(pat, tex)
        if not m:
            continue
        for g, lab, src in zip(m.groups(), labels, srcs):
            rep.cell(T, lab, g, float(g), src)
    rep.finish_table(T, before)

def check_isi_penalty_prose(tex, rep):
    """MLP-vs-genie-Viterbi required-SNR penalties on the fixed 3-tap ISI chain.

    Quoted in Ch. 6 (Layer table and H5 conclusion), Ch. 8, Ch. 9, the abstract
    and the Hebrew abstract. Source: results/e6_isi_penalty.json, written by
    e6_isi_penalty.py from results/e6_matched_protocol.json.
    """
    T = "prose:isi-penalty"; before = rep.checked
    p = os.path.join(ROOT, "results", "e6_isi_penalty.json")
    if not os.path.exists(p):
        return rep.skip(T, "results/e6_isi_penalty.json not found")
    pen = json.load(open(p))["penalties"]
    v = [pen[k]["penalty_db"] for k in ("1e-01", "1e-02", "1e-03", "1e-04")]
    lab = ["1e-1", "1e-2", "1e-3", "1e-4"]
    patterns = [
        (r"\$\+([\d.]+)\$~dB at a \$10\^\{-1\}\$ target, \$\+([\d.]+)\$ at \$10\^\{-2\}\$, "
         r"\$\+([\d.]+)\$ at \$10\^\{-3\}\$ and \$\+([\d.]+)\$ at \$10\^\{-4\}\$", [0, 1, 2, 3]),
        (r"\$([\d.]+)\$, \$([\d.]+)\$, \$([\d.]+)\$, and \$([\d.]+)\$~dB at BER "
         r"\$10\^\{-1\}\$ through \$10\^\{-4\}\$", [0, 1, 2, 3]),
        (r"vary from \$([\d.]+)\$ to \$([\d.]+)\$~dB with the target", [0, 3]),
        (r"penalties of \$([\d.]+)\$--\$([\d.]+)\$~dB", [0, 3]),
        (r"\\textenglish\{([\d.]+)--([\d.]+) dB\}", [0, 3]),
    ]
    for pat, idx in patterns:
        for m in re.finditer(pat, tex):
            for g, i in zip(m.groups(), idx):
                rep.cell(T, f"MLP-vs-genie@{lab[i]}", g, float(g), v[i])
    rep.finish_table(T, before)

def check_hybrid_branch_prose(tex, rep):
    """Which branch the canonical Hybrid took (Sec. 4.2 rationale, Ch. 7, Ch. 8).

    The Hybrid switches on a received-power SNR estimate, not on the operating
    SNR. Source: results/hybrid_branch_selection.json, written by
    hybrid_branch_selection.py, which replays the canonical run's Hop-1 draws and
    checks the replay against the published per-trial BERs.
    """
    T = "prose:hybrid-branch"; before = rep.checked
    p = os.path.join(ROOT, "results", "hybrid_branch_selection.json")
    if not os.path.exists(p):
        return rep.skip(T, "results/hybrid_branch_selection.json not found")
    d = json.load(open(p))
    by = {int(round(r["snr_db"])): r["mlp_axis_blocks"] for r in d["rows"]}
    n = d["rows"][0]["axis_blocks"]
    # The prose says the replay reproduces every all-DF block's published BER.
    rep.cell(T, "replay BER mismatches", "0", 0.0, d["replay_check"]["ber_mismatches"])
    m = re.search(r"The threshold is a fixed \$(\d+)\$~dB", tex)
    if m:
        rep.cell(T, "threshold", m.group(1), float(m.group(1)), d["threshold_db"])
    m = re.search(r"MLP branch on all (\d+) axis-blocks at each SNR from 0 to 6~dB, on (\d+) of (\d+) "
                  r"at 8~dB, (\d+) at 10~dB and (\d+) at 12~dB, and on none from 14~dB upward", tex)
    if m:
        rep.cell(T, "axis-blocks", m.group(1), float(m.group(1)), n)
        for snr in (0, 2, 4, 6):
            rep.cell(T, f"MLP@{snr}dB", m.group(1), float(m.group(1)), by[snr])
        rep.cell(T, "MLP@8dB", m.group(2), float(m.group(2)), by[8])
        rep.cell(T, "axis-blocks@8dB", m.group(3), float(m.group(3)), n)
        rep.cell(T, "MLP@10dB", m.group(4), float(m.group(4)), by[10])
        rep.cell(T, "MLP@12dB", m.group(5), float(m.group(5)), by[12])
        for snr in (14, 16, 18, 20):
            rep.cell(T, f"MLP@{snr}dB", "0", 0.0, by[snr])  # "none"
    # "between 6 and 8 dB": the last SNR with the MLP branch on every block, and
    # the first with it on fewer than half.
    snrs = sorted(by)
    lower = max(s_ for s_ in snrs if all(by[t] == n for t in snrs if t <= s_))
    upper = min(s_ for s_ in snrs if by[s_] < n / 2)
    for k, m in enumerate(re.finditer(r"between (\d+) and (\d+)~dB", tex)):
        ctx = tex[max(0, m.start() - 400):m.start()]
        if "switch" not in ctx:
            continue
        rep.cell(T, f"switch lower #{k}", m.group(1), float(m.group(1)), lower)
        rep.cell(T, f"switch upper #{k}", m.group(2), float(m.group(2)), upper)
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
    # Both lookups guarded: on -1 the inner find would make the outer one
    # search from the end of the document and the slice would cover an
    # unrelated region, checking the wrong numbers instead of failing.
    tail = tex.find("Gray map is not the route", i)
    if tail < 0:
        return rep.skip(T, "sentence does not reach the expected closing clause")
    stop = tex.find(".", tail)
    if stop < 0:
        return rep.skip(T, "no sentence terminator after the closing clause")
    sent = tex[i:stop]
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


def check_qpsk_controls_prose(tex, rep):
    """The three 20 dB trellis-control SERs quoted in Chapter 7 prose.

    These are the numbers that pin the withdrawn reversal on the benchmark's
    channel model rather than on the trellis, so they should not be able to
    drift away from the run that produced them (qpsk_trellis_controls.py).
    """
    T = "prose:qpsk-controls"; before = rep.checked
    src_path = os.path.join(ROOT, "results", "qpsk_trellis_controls.json")
    if not os.path.exists(src_path):
        return rep.skip(T, "results/qpsk_trellis_controls.json not found")
    with open(src_path) as fh:
        src = json.load(fh)["controls"]

    i = tex.find("separate the two explanations at 20~dB")
    if i < 0:
        return rep.skip(T, "controls sentence not found in tex")
    # ". The trellis" rather than the first "." -- the last quoted control
    # value has decimal places, so a bare period search would stop inside it.
    stop = tex.find(". The trellis", tex.find("given the fading gains as well", i))
    if stop < 0:
        return rep.skip(T, "sentence does not reach the expected closing clause")
    sent = tex[i:stop]
    nums = [float(m) for m in re.findall(r"\$(\d+\.\d+)\$", sent)]
    if len(nums) != 3:
        return rep.skip(T, f"expected 3 numbers in the sentence, found {len(nums)}")
    for got, want, what in zip(
            nums,
            [src["taps_only_faded"]["ser"],
             src["taps_only_fading_removed"]["ser"],
             src["genie_csi_faded"]["ser"]],
            ["taps_only_faded_ser", "fading_removed_ser", "genie_csi_ser"]):
        rep.cell(T, what, f"{got}", got, want)
    rep.finish_table(T, before)


def check_bcjr_benchmark(tex, rep):
    """tbl:bcjr-benchmark against results/qpsk_bcjr_benchmark.json.

    These land in the thesis as the answer to Future Work item 4, so they get
    the same binding as every other published number: the table cannot drift
    from the run that produced it.
    """
    T = "tbl:bcjr-benchmark"; before = rep.checked
    src_path = os.path.join(ROOT, "results", "qpsk_bcjr_benchmark.json")
    if not os.path.exists(src_path):
        return rep.skip(T, "results/qpsk_bcjr_benchmark.json not found")
    comp = json.load(open(src_path))["comparison"]
    body = tabular_body(tex, T)
    if body is None:
        return rep.skip(T, "label not found in tex")
    for row in data_rows(body):
        if not row or row[0][1] is None:
            continue
        snr = str(int(row[0][1]))
        if snr not in comp or len(row) < 3:
            continue
        rep.cell(T, f"{snr}dB/viterbi", row[1][0], row[1][1], comp[snr]["viterbi_ber"])
        rep.cell(T, f"{snr}dB/bcjr", row[2][0], row[2][1], comp[snr]["bcjr_ber"])
    rep.finish_table(T, before)


def check_soft_df_calibration_prose(tex, rep):
    """The two variance-doubled figures in tbl:table39's footnote.

    They came from coded_soft_df_calibration.py, a different run from the table
    they annotate, and until now neither the verifier nor the provenance audit
    knew they existed -- two published numbers with nothing binding them to
    their source. The table's own 20 dB cell is the nominal run, the one
    Figure fig57 plots; an earlier revision showed this sweep's value there and
    mislabelled it as the plotted run.
    """
    T = "prose:soft-df-calibration"; before = rep.checked
    src_path = os.path.join(ROOT, "results", "coded_soft_df_calibration.json")
    if not os.path.exists(src_path):
        return rep.skip(T, "results/coded_soft_df_calibration.json not found")
    src = json.load(open(src_path))["results"]["20"]
    i = tex.find("With the assumed variance doubled, a separate sensitivity sweep gives")
    if i < 0:
        return rep.skip(T, "footnote sentence not found in tex")
    sent = tex[i:i + 260]
    nums = [float(m) for m in re.findall(r"\$(\d+\.\d+)\$", sent)]
    if len(nums) != 2:
        return rep.skip(T, f"expected 2 numbers in the footnote, found {len(nums)}")
    # the footnote quotes the sweep's doubled-variance run; every factor from 2
    # upward lands on the same value to the published digits
    doubled = src["soft_by_factor"]["2.0"]
    rep.cell(T, "soft_df_variance_doubled", f"{nums[0]}", nums[0], doubled)
    rep.cell(T, "hard_block_df_20dB", f"{nums[1]}", nums[1], src["hard"])
    rep.finish_table(T, before)


def check_seed_spread_native_prose(tex, rep):
    """The six across-seed spreads quoted in Limitations item 8.

    These entered the thesis as the answer to reviewer item M2, so they get the
    same binding as every other published number.
    """
    T = "prose:seed-spread-native"; before = rep.checked
    src_path = os.path.join(ROOT, "results", "seed_spread_native_architectures.json")
    if not os.path.exists(src_path):
        return rep.skip(T, "results/seed_spread_native_architectures.json not found")
    src = json.load(open(src_path))["architectures"]
    i = tex.find("The across-seed spread in SNR")
    if i < 0:
        return rep.skip(T, "spread sentence not found in tex")
    sent = tex[i:i + 460]
    names = {"Mamba-S6": "Mamba S6", "the Hybrid": "Hybrid", "Mamba-2": "Mamba2 (SSD)",
             "the MLP": "MLP (169p)", "the Transformer": "Transformer", "the VAE": "VAE"}
    found = re.findall(r"\$(\d+\.\d+)\$(?:~dB)? for ([A-Za-z0-9 -]+?)(?=,|\s+and|\.|\s*\()", sent)
    if len(found) != 6:
        return rep.skip(T, f"expected 6 spreads in the sentence, found {len(found)}")
    # Comparable targets (Khina round 2): a seed that misses a target DF reaches
    # makes the spread a lower bound, which the sentence must say ("at least").
    from ber_metrics import penalty_common, spread_of_seeds
    d = json.load(open(src_path))
    bounded = sent[sent.find("at least"):] if "at least" in sent else ""
    for val, label in found:
        key = names.get(label.strip())
        if key is None:
            return rep.skip(T, f"unrecognised architecture label {label!r}")
        spr, flag = spread_of_seeds([penalty_common(d["snr_db"], r["ber"], d["df_ber"])
                                     for r in src[key]["runs"]])
        rep.cell(T, f"spread/{key}", val, float(val), spr)
        rep.cell(T, f"spread/{key} marked as a bound", "1" if label.strip() in bounded else "0",
                 float(label.strip() in bounded), float(flag == ">"))
    rep.finish_table(T, before)


def check_mmse_monotonicity_prose(tex, rep):
    """The MMSE-vs-taps figures Chapter 6 uses to explain the non-monotonicity.

    Twelve numbers quoted in prose -- four attained MMSE values at 16 dB and the
    per-target penalties at 1e-1 and 1e-2 -- against
    results/mmse_equalizer_detail.json. They are the evidence that the published
    non-monotonicity belongs to the worst-target metric rather than to the
    equalizer, so they should not be able to drift away from the run.
    """
    T = "prose:mmse-monotonicity"; before = rep.checked
    src_path = os.path.join(ROOT, "results", "mmse_equalizer_detail.json")
    if not os.path.exists(src_path):
        return rep.skip(T, "results/mmse_equalizer_detail.json not found")
    with open(src_path) as fh:
        det = json.load(fh).get("isi_complex", {})
    if not det:
        return rep.skip(T, "isi_complex not in the detail JSON")

    i = tex.find("For the QPSK linear-MMSE detail run")
    if i < 0:
        return rep.skip(T, "monotonicity sentence not found in tex")
    stop = tex.find("These measurements do not imply", i)
    if stop < 0:
        return rep.skip(T, "no closing clause after the monotonicity sentence")
    nums = [float(x) for x in re.findall(r"[-+]?\d+\.\d+", tex[i:stop])]
    taps = ("3", "5", "7", "11")
    want = ([det[n]["attained_mmse"]["16"] for n in taps]
            + [det[n]["per_target_db"]["0.1"] for n in taps]
            + [det[n]["per_target_db"]["0.01"] for n in taps])
    if len(nums) != len(want):
        return rep.skip(T, f"expected {len(want)} numbers, found {len(nums)}")
    names = ([f"mmse16/{n}tap" for n in taps]
             + [f"pen1e-1/{n}tap" for n in taps]
             + [f"pen1e-2/{n}tap" for n in taps])
    for got, exp, nm in zip(nums, want, names):
        rep.cell(T, nm, f"{got}", got, exp)
    rep.finish_table(T, before)


# Relay architectures the thesis names by parameter count, and the script that
# defines each. Three networks of the same class land on two counts, two of them
# on 169 by coincidence, which is exactly how the unknown-ISI relay came to be
# published as 169 when it is 170. Checked against the source rather than a
# stored number so the labels cannot drift again.
RELAY_ARCHITECTURES = [
    ("canonical (run_experiments.py)", 5, 24, 169),
    ("unknown-ISI (e6_sim_ported.py)", 11, 13, 170),
    ("composite/blind/pilot (2W I/Q)", 22, 7, 169),
    ("composite large", 22, 48, 1153),
]


def check_relay_param_counts(tex, rep):
    """Every "i -> h -> 1, N parameters" claim in the thesis, checked two ways.

    A two-layer perceptron with `i` inputs, `h` hidden units and one output has
    i*h + h + h + 1 parameters. This reads the shape-and-count claims out of the
    LaTeX rather than trusting a stored number, checks each one's arithmetic,
    and checks that the shape is one the code actually builds. The earlier
    version validated only the hardcoded constants against themselves, which
    could not fail and could not have caught the defect it was written for: the
    thesis called an 11 -> 13 -> 1 relay "169 parameters" when it is 170.
    """
    T = "arch:relay-param-counts"; before = rep.checked
    known = {(i, h): n for _, i, h, n in RELAY_ARCHITECTURES}

    # "$11 \to 13 \to 1$, which is $170$ parameters" and its variants: take the
    # first number quoted within a short window after the shape.
    pat = re.compile(r"\$(\d+)\s*\\to\s*(\d+)\s*\\to\s*1\$(.{0,120}?)\$(\d+)\$", re.S)
    found = 0
    for m in pat.finditer(tex):
        i, h, claimed = int(m.group(1)), int(m.group(2)), int(m.group(4))
        found += 1
        rep.cell(T, f"tex {i}->{h}->1", str(claimed), float(claimed),
                 float(i * h + h + h + 1))
        if (i, h) not in known:
            rep.cell(T, f"tex {i}->{h}->1 is an architecture the code builds",
                     "yes", 0.0, 1.0)
    if not found:
        return rep.skip(T, "no 'i -> h -> 1' shape claims found in the tex")

    # and the arithmetic of every architecture the code does build
    for name, i, h, claimed in RELAY_ARCHITECTURES:
        rep.cell(T, f"code {name}", str(claimed), float(claimed),
                 float(i * h + h + h + 1))
    rep.finish_table(T, before)


# The six headline-claim proofs exist twice: thesis Appendix F (section 9.6) and
# the README's "Appendix - Proof of Claims". Nothing linked them, and a
# correction has now landed in one copy and not the other twice running -- the
# MAP/MLSE conflation and the unsupported windowed-MAP bound were both fixed in
# the thesis first and left standing in the README. These are the load-bearing
# quantities the two share; they must agree.
PROOF_COPIES = {
    "cursor h0": (r"0\.758", 2),
    "interferer sum": (r"0\.910", 2),
    "flipped amplitude": (r"0\.152", 2),
    "unknown-ISI relay params": (r"\b170\b", 2),
    # No fixed-channel zero is used to explain the random nonlinear cascade.
    "local BCJR endpoint": (r"relay-output", 2),
}


def check_proof_copies_agree(tex, rep):
    """Thesis Appendix F and the README proof-of-claims must not drift apart.

    Both state the same six proofs. This checks the quantities they share appear
    in both, so a correction to one copy that misses the other is caught here
    rather than by a reader finding the two documents disagreeing.
    """
    T = "consistency:proof-copies"; before = rep.checked
    readme = os.path.join(ROOT, "README.md")
    if not os.path.exists(readme):
        return rep.skip(T, "README.md not found")
    rm = open(readme, encoding="utf-8").read()
    i = rm.find("Proof of Claims")
    if i < 0:
        return rep.skip(T, "README has no proof-of-claims section")
    rm = rm[i:]
    j = tex.find("Verification of Headline Claims")
    if j < 0:
        return rep.skip(T, "thesis has no Appendix F")
    ap = tex[j:]

    for name, (pat, _) in PROOF_COPIES.items():
        in_tex = 1.0 if re.search(pat, ap) else 0.0
        in_rm = 1.0 if re.search(pat, rm) else 0.0
        # both copies must carry it: compare the README against the thesis
        rep.cell(T, f"{name} in both copies", str(in_rm), in_rm, in_tex)

    # and neither copy may still call MLSE BER-optimal
    for label, body in (("thesis", ap), ("README", rm)):
        bad = 1.0 if re.search(r"MAP/MLSE detection is optimal", body) else 0.0
        rep.cell(T, f"{label}: no MAP/MLSE conflation", str(bad), bad, 0.0)
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
    from ber_metrics import penalty_common, best_of_seeds, spread_of_seeds
    snr = seq["snr_db"]

    def comparable(ch, aname):
        """Best and spread over seeds on the targets MLSE reaches (Khina round 2)."""
        c = channels[ch]
        res = [penalty_common(snr, r["ber"], c["baseline_ber"]) for r in c["archs"][aname]["runs"]]
        return best_of_seeds(res), spread_of_seeds(res)
    raw_rows = {}
    for line in body.split("\\\\"):
        cells = [x.strip() for x in line.split("&")]
        if len(cells) == 5:
            raw_rows[cells[0].lower()] = cells
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
            if aname not in ch_data.get("archs", {}):
                continue
            (best, bflag), (spr, sflag) = comparable(src_key, aname)
            rep.cell(T, f"{src_key}/{aname}", pub_text, pub_val, best)
            raw = next((v for k, v in raw_rows.items() if label in k or k in label), None)
            if raw is not None:
                m = re.search(r"\((\$\{>\}|>)?\s*([\d.]+)", raw[col])
                if m:
                    shown = (">" if m.group(1) else "") + m.group(2)
                    rep.cell(T, f"{src_key}/{aname} spread", m.group(2), float(m.group(2)), spr)
                    rep.cell(T, f"{src_key}/{aname} spread bound marked", "1" if shown[:1] == ">" else "0",
                             float(shown[:1] == ">"), float(sflag == ">"))
    rep.finish_table(T, before)


def check_nlbias_prose(tex, rep):
    """Nonlinear-bias channel (setup S3) numbers in Sec. 6.2 vs e6_sim_ported_results.npy."""
    T = "prose:nlbias"; before = rep.checked
    d = _load_e6_npy("e6_sim_ported_results.npy")
    if d is None:
        return rep.skip(T, "e6_sim_ported_results.npy not found")
    m = re.search(r"at 8~dB AF, zero-threshold DF and the 170-parameter MLP reach "
                  r"\$([\d.]+)\$, \$([\d.]+)\$ and \$([\d.]+)\$", tex)
    if not m:
        return rep.skip(T, "nonlinear-bias sentence not found in tex")
    r = d["results"]["S3: nonlinear+bias -> AWGN"]
    i = list(d["snrs"]).index(8)
    for k, g in zip(("AF", "DF", "MLP"), m.groups()):
        rep.cell(T, f"{k}@8dB", g, float(g), float(np.array(r[k][0])[i]))
    rep.finish_table(T, before)


def check_coded_fer_prose(tex, rep):
    """20-dB frame-error rates quoted in Sec. 5.5 vs results/coded_reliable_regime.json."""
    T = "prose:coded-fer"; before = rep.checked
    p = os.path.join(ROOT, "results", "coded_reliable_regime.json")
    if not os.path.exists(p):
        return rep.skip(T, "results/coded_reliable_regime.json not found")
    d = json.load(open(p))
    m = re.search(r"frame errors occur in \$([\d.]+)\\%\$ of coded-DF frames and "
                  r"\$([\d.]+)\\%\$ even with an oracle relay", tex)
    if not m:
        return rep.skip(T, "frame-error sentence not found in tex")
    i = d["snr_db"].index(20)
    rep.cell(T, "coded-DF FER@20dB (%)", m.group(1), float(m.group(1)), 100 * d["coded_df_fer"][i])
    rep.cell(T, "oracle FER@20dB (%)", m.group(2), float(m.group(2)), 100 * d["oracle_fer"][i])
    rep.finish_table(T, before)


def check_mamba_lookahead_prose(tex, rep):
    """Share of x[n]'s energy after the causal Mamba readout point (Sec. 6.4.2), from the taps."""
    T = "prose:mamba-lookahead"; before = rep.checked
    m = re.search(r"which carry \$([\d.]+)\\%\$ of the energy of \$x\[n\]\$ on the ISI taps "
                  r"and \$([\d.]+)\\%\$ on the composite", tex)
    if not m:
        return rep.skip(T, "Mamba look-ahead sentence not found in tex")
    ms = json.load(open(os.path.join(ROOT, "results", "mlp_min_size_all_channels.json")))
    src = open(os.path.join(ROOT, "mlp_min_size_all_channels.py"), encoding="utf-8").read()
    comp = re.search(r"H_COMPOSITE = np\.array\(\[([^\]]+)\]\)", src)
    for g, taps in ((m.group(1), ms["isi_taps"]),
                    (m.group(2), [float(x) for x in comp.group(1).split(",")] if comp else None)):
        if taps is None:
            continue
        e = np.array(taps, dtype=float) ** 2
        rep.cell(T, "future-tap energy share (%)", g, float(g), 100 * e[1:].sum() / e.sum())
    rep.finish_table(T, before)


def check_seq_rerun_prose(tex, rep):
    """Sec. 6.4.2 statements that rest on the sequence-on-memory rerun.

    Table 6.7's dagger note and margin come from the published run; the
    reproducibility paragraph and the look-ahead sentence come from the rerun
    (readout, bidirectional, controls) and the repeat/probe runs. Every number
    is recomputed from the committed result files, and the three claims made in
    words (ten best values within the stated margin, the bidirectional readout
    trailing the centre readout everywhere, bit-for-bit repeats on one machine)
    are checked as counts.
    """
    from ber_metrics import penalty_table
    T = "prose:seq-rerun"; before = rep.checked
    R = os.path.join(ROOT, "results")
    names = ("seq_models_on_memory.json", "seq_models_on_memory_readout.json",
             "seq_models_on_memory_bidirectional.json",
             "seq_models_on_memory_rerun_controls.json",
             "seq_models_on_memory_repeat.json", "seq_models_on_memory_repeat_probe.jsonl")
    if not all(os.path.exists(os.path.join(R, n)) for n in names):
        return rep.skip(T, "a sequence-on-memory rerun result file is missing")
    pub, ro, bi, ctl, rpt = (json.load(open(os.path.join(R, n))) for n in names[:5])
    probes = [json.loads(l) for l in open(os.path.join(R, names[5])) if l.strip()]
    snr = pub["snr_db"]

    def per_target(rec, run):
        pt = penalty_table(snr, run["ber"], rec["baseline_ber"])["per_target"]
        return {t: v["db_penalty"] for t, v in pt.items() if v["snr_base"] == v["snr_base"]}

    # Table 6.7 dagger note (comparable targets, Khina round 2): the Mamba-2 QPSK
    # seed that misses 1e-3 is a lower bound, and the earlier rule scored it lower.
    from ber_metrics import penalty_common
    q = pub["channels"]["isi_complex"]
    m = re.search(r"penalty is only bounded below \(\$>\+([\d.]+)\$\).*?scored it at \$\+([\d.]+)\$", tex, re.S)
    if m:
        res = [penalty_common(snr, r["ber"], q["baseline_ber"]) for r in q["archs"]["Mamba2-3K"]["runs"]]
        lb = [r["worst_db"] for r in res if r["lower_bound"]]
        rep.cell(T, "Mamba-2 QPSK missing seed, lower bound", m.group(1), float(m.group(1)), lb[0] if lb else float("nan"))
        rep.cell(T, "Mamba-2 QPSK missing seed, earlier rule", m.group(2), float(m.group(2)),
                 min(r["db"] for r in q["archs"]["Mamba2-3K"]["runs"]))
    # MLP lead over the best sequence architecture on the two ISI channels.
    m = re.search(r"by \$([\d.]+)\$ and \$([\d.]+)\$~dB on the two ISI cases", tex)
    if m:
        from ber_metrics import best_of_seeds
        for g, ch in zip(m.groups(), ("isi", "isi_complex")):
            c = pub["channels"][ch]
            b = {a: best_of_seeds([penalty_common(snr, r["ber"], c["baseline_ber"]) for r in c["archs"][a]["runs"]])[0]
                 for a in c["archs"]}
            rep.cell(T, f"MLP lead, {ch}", g, float(g),
                     min(b[a] for a in ("Transformer-3K", "Mamba-S6-3K", "Mamba2-3K")) - b["MLP-3K"])
    # Reproducibility paragraph.
    rerun = {"MLP-3K": (ctl, "MLP-3K"), "Transformer-3K": (ctl, "Transformer-3K"),
             "Mamba-S6-3K": (ro, "Mamba-S6-3K|center"), "Mamba2-3K": (ro, "Mamba2-3K|center")}
    m = re.search(r"reproduced eleven of the twelve best values within \$([\d.]+)\$~dB "
                  r"\(not the Transformer's at QPSK\)", tex)
    if m:
        from ber_metrics import best_of_seeds
        lim = float(m.group(1))
        def cbest(d, c, a):
            rec = d["channels"][c]
            return best_of_seeds([penalty_common(snr, r["ber"], rec["baseline_ber"])
                                  for r in rec["archs"][a]["runs"]])[0]
        diffs = {(c, a): abs(cbest(pub, c, a) - cbest(src, c, key))
                 for c in ("isi", "isi_complex", "composite")
                 for a, (src, key) in rerun.items()}
        named = {("isi_complex", "Transformer-3K")}
        rep.cell(T, "largest best-value difference, other eleven cells", m.group(1), lim,
                 max(v for k, v in diffs.items() if k not in named))
        within = sum(1 for k, v in diffs.items() if v <= lim + tol_for(m.group(1)))
        rep.cell(T, "best values within the margin (count)", "11", 11, within)
    m = re.search(r"Mamba-S6 spread \$([\d.]+)\$ and \$([\d.]+)\$~dB instead of \$([\d.]+)\$ "
                  r"and \$([\d.]+)\$, and the Transformer \$([\d.]+)\$ and \$([\d.]+)\$ instead "
                  r"of \$([\d.]+)\$ and \$([\d.]+)\$", tex)
    if m:
        g = m.groups()
        src = [ro["channels"][c]["archs"]["Mamba-S6-3K|center"]["spread"] for c in ("isi", "isi_complex")]
        src += [pub["channels"][c]["archs"]["Mamba-S6-3K"]["spread"] for c in ("isi", "isi_complex")]
        src += [ctl["channels"][c]["archs"]["Transformer-3K"]["spread"] for c in ("isi", "isi_complex")]
        src += [pub["channels"][c]["archs"]["Transformer-3K"]["spread"] for c in ("isi", "isi_complex")]
        for k, (t, v) in enumerate(zip(g, src)):
            rep.cell(T, f"ISI spread {k}", t, float(t), v)
    m = re.search(r"the same Transformer seed moved by \$([\d.]+)\$~dB", tex)
    if m:
        c = next(c for c in rpt["cases"] if c["arch"] == "Transformer-3K")
        rep.cell(T, "Transformer cross-machine shift", m.group(1), float(m.group(1)),
                 abs(c["db"] - c["committed_rerun_db"]))
    if "Seeded training repeats bit for bit on one machine" in tex:
        # every repeated configuration must give one weight hash and the repeat's BER curve
        bad = 0
        for c in rpt["cases"]:
            same = [p for p in probes if (p["arch"], p["channel"], p["seed"])
                    == (c["arch"], c["channel"], c["seed"])]
            bad += (len({p["weights_sha256"] for p in same}) > 1
                    or any(p["ber"] != c["ber"] for p in same))
        rep.cell(T, "configurations not repeating bit for bit", "0", 0, bad)
    if "only the feedforward relay retrained identically on both machines" in tex:
        same = [c["arch"] for c in rpt["cases"] if c["identical_to_committed_rerun"]]
        rep.cell(T, "architectures identical across machines (count)", "1", 1,
                 len(same) if same == ["MLP-3K"] else len(same) + 10)
    # Look-ahead sentence in the limitation paragraph.
    m = re.search(r"the first failed to train \(BER about \$([\d.]+)\$ at 20~dB\)", tex)
    if m:
        b20 = [r["ber"][-1] for ch in ro["channels"].values()
               for k, e in ch["archs"].items() if k.endswith("|last") for r in e["runs"]]
        rep.cell(T, "full-window readout BER at 20 dB (min)", m.group(1), float(m.group(1)), min(b20))
        rep.cell(T, "full-window readout BER at 20 dB (max)", m.group(1), float(m.group(1)), max(b20))
    if "the second trailed the centre readout at every target on every channel" in tex:
        worse = 0
        for c, rec in ro["channels"].items():
            brec = bi["channels"][c]
            for a in ("Mamba-S6-3K", "Mamba2-3K"):
                cen = [per_target(rec, r) for r in rec["archs"][f"{a}|center"]["runs"]]
                bid = [per_target(brec, r) for r in brec["archs"][f"{a}|bidirectional"]["runs"]]
                for t in cen[0]:
                    cmax = max(d[t] for d in cen)
                    bmin = min((d[t] if d[t] == d[t] else float("inf")) for d in bid)
                    worse += not (bmin > cmax)
        rep.cell(T, "targets where bidirectional does not trail centre", "0", 0, worse)
    rep.finish_table(T, before)


def check_timing_and_mmse_median_prose(tex, rep):
    """Fig. 6.7(b) speedup and Table 6.6's median-initialization lead.

    The speedup range is read from e6_complexity_ported_results.npy (median of
    seven single-threaded runs per block length). The median-initialization lead
    takes, on each channel, the size-sweep configuration with the best
    best-of-seeds penalty (the one Table 6.6 reports), uses its median seed
    instead, and subtracts it from the best MMSE penalty.
    """
    from ber_metrics import penalty_table
    T = "prose:timing-mmse-median"; before = rep.checked
    tp = os.path.join(ROOT, "e6_unknown_channel_results", "e6_complexity_ported_results.npy")
    if os.path.exists(tp):
        pb = np.load(tp, allow_pickle=True).item()["panel_b"]
        ratios = [v[2] for v in pb.values()]
        for m in re.finditer(r"MLP is \$([\d.]+)\$--\$([\d.]+)\\times\$ faster", tex):
            rep.cell(T, "speedup low", m.group(1), float(m.group(1)), min(ratios))
            rep.cell(T, "speedup high", m.group(2), float(m.group(2)), max(ratios))
    m = re.search(r"\(\$([\d.]+)\$ to \$([\d.]+)\$~dB at the median initialization\)", tex)
    mp = os.path.join(ROOT, "results", "mlp_min_size_all_channels.json")
    ep = os.path.join(ROOT, "results", "mmse_equalizer.json")
    if m and os.path.exists(mp) and os.path.exists(ep):
        ms, mm = json.load(open(mp)), json.load(open(ep))
        leads = []
        for ch in ("isi", "isi_complex", "composite"):
            c = ms["channels"][ch]
            snr, base = ms["snr_db"], np.array(c["baseline_ber"])
            best = None
            for cfg in c["sweep"]:
                ps = [penalty_table(snr, np.array(r["ber"]), base)["worst_db_penalty"] for r in cfg["seed_runs"]]
                ps = [v for v in ps if v == v]
                if ps and (best is None or min(ps) < best[0]):
                    best = (min(ps), float(np.median(ps)))
            leads.append(min(mm[ch].values()) - best[1])
        rep.cell(T, "median-init lead low", m.group(1), float(m.group(1)), min(leads))
        rep.cell(T, "median-init lead high", m.group(2), float(m.group(2)), max(leads))
    rep.finish_table(T, before)


def check_khina_round2(tex, rep):
    """Numbers introduced in answer to Khina's second round.

    * Table 4.4 (tbl:canonical-runs) and the Sec. 4.3.3 run identification,
      against results/canonical_run_provenance.json;
    * the comparable-target penalties of Tables 9.4 (coded row), 9.5 and 9.6
      and of the Sec. 7.2.1, 4.3.4 and appendix prose, recomputed here from the
      stored per-seed curves with ber_metrics.penalty_common rather than read
      from penalty_common_targets.json, so the check is independent of that
      script.
    """
    from ber_metrics import penalty_common, spread_of_seeds
    T = "prose:khina-round2"; before = rep.checked
    R = os.path.join(ROOT, "results")
    prov = json.load(open(os.path.join(R, "canonical_run_provenance.json")))
    # Table 4.4: parameters, windows and gradient windows per row.
    body = tabular_body(tex, "tbl:canonical-runs")
    rows = {"MLP": ("table_5_2", ["MLP (169p)"]), "Hybrid (MLP part)": ("table_5_2", ["Hybrid"]),
            "VAE": ("table_5_2", ["VAE"]), "Transformer": ("table_5_2", ["Transformer"]),
            "Mamba-S6 / Mamba-2": ("table_5_2", ["Mamba S6", "Mamba2 (SSD)"]),
            "MLP-3K, Hybrid-3K": ("table_5_3", ["MLP-3K", "Hybrid-3K"]),
            "VAE-3K": ("table_5_3", ["VAE-3K"]),
            "Sequence models, 3K": ("table_5_3", ["Transformer-3K", "Mamba-3K", "Mamba2-3K"])}
    if body is not None:
        for line in body.split("\\\\"):
            c = [x.strip() for x in line.split("&")]
            if len(c) != 7 or c[0] not in rows:
                continue
            tab, names = rows[c[0]]
            recs = [prov["relays"][tab][n] for n in names]
            params = [int(x) for x in re.findall(r"\d+", c[1].replace("{,}", ""))]
            want = sorted({r["params"] for r in recs})
            rep.cell(T, f"{c[0]}/params", c[1], float(params[0]), float(want[0]))
            if len(params) > 1:
                rep.cell(T, f"{c[0]}/params (2)", c[1], float(params[-1]), float(want[-1]))
            rep.cell(T, f"{c[0]}/window", c[2], float(c[2]), float(recs[0]["window"]))
            snrs = [int(x) for x in re.findall(r"\d+", c[3])]
            rep.cell(T, f"{c[0]}/training SNRs", c[3], float(snrs == recs[0]["training_snrs_db"]), 1.0)
            w = [int(x) for x in re.findall(r"\d+", c[4].replace("{,}", ""))]
            rep.cell(T, f"{c[0]}/windows", c[4], float(w[0]), float(recs[0]["windows"]))
            if len(w) > 1:
                rep.cell(T, f"{c[0]}/gradient windows", c[4], float(w[1]), float(recs[0]["gradient_windows"]))
    # Run identification in Sec. 4.3.3.
    for key, run in prov["runs"].items():
        for field in ("code_commit", "data_commit"):
            rep.cell(T, f"{key}/{field}", run[field], float(run[field] in tex), 1.0)
        rep.cell(T, f"{key}/AF and DF reproduce", "1", 1.0,
                 float(all(run["af_df_reproduce_bit_for_bit"].values())))
    m = re.search(r"in \$2\{,\}892\$~s for training", tex)
    rep.cell(T, "Table 5.2 stage seconds", "2892", float(bool(m)) * 2892, float(prov["runs"]["table_5_2"]["stage_seconds"]))
    m = re.search(r"\$746\.6\$~s", tex)
    rep.cell(T, "Table 5.3 stage seconds", "746.6", float(bool(m)) * 746.6, float(prov["runs"]["table_5_3"]["stage_seconds"]))
    # Size-sweep seed spreads (Table 9.5 pooled row; Sec. 4.3.4 and 7.2.1 prose).
    ms = json.load(open(os.path.join(R, "mlp_min_size_all_channels.json")))
    sp, flags = [], []
    for c in ms["channels"].values():
        base = c.get("baseline_ber", c["df_ber"])
        for cfg in c["sweep"]:
            v, f = spread_of_seeds([penalty_common(ms["snr_db"], r["ber"], base) for r in cfg["seed_runs"]])
            if v == v:
                sp.append(v); flags.append(f)
    m = re.search(r"All\s*& (\d+) & ([\d.]+) & \$\{\\geq\}([\d.]+)\$ & \$\{\\geq\}([\d.]+)\$", tex)
    if m:
        rep.cell(T, "Table 9.5 pooled configurations", m.group(1), float(m.group(1)), float(len(sp)))
        rep.cell(T, "Table 9.5 pooled median", m.group(2), float(m.group(2)), float(np.median(sp)))
        rep.cell(T, "Table 9.5 pooled 90th percentile", m.group(3), float(m.group(3)), float(np.percentile(sp, 90)))
        rep.cell(T, "Table 9.5 pooled maximum", m.group(4), float(m.group(4)), float(max(sp)))
    for m in re.finditer(r"median (?:across-initialization spread is )?\$([\d.]+)\$~dB(?: over (\d+) configurations| is not a universal)", tex):
        rep.cell(T, "pooled median spread (prose)", m.group(1), float(m.group(1)), float(np.median(sp)))
        if m.group(2):
            rep.cell(T, "pooled configurations (prose)", m.group(2), float(m.group(2)), float(len(sp)))
    # Table 9.6: per-seed penalties, bounds and spreads.
    ss = json.load(open(os.path.join(R, "seed_spread_architectures.json")))
    ss_body = tabular_body(tex, "tbl:seed-spread-3k") or ""
    for a, v in ss["architectures"].items():
        res = [penalty_common(ss["snr_db"], r["ber"], ss["df_ber"]) for r in v["runs"]]
        m = re.search(re.escape(a) + r"\s*& [\d{},]+ & (.*?) & (.*?) \\\\", ss_body)
        if not m:
            continue
        cells = [x.strip() for x in m.group(1).split(",")]
        for k, (cell, r) in enumerate(zip(cells, res)):
            num = float(re.search(r"([\d.]+)", cell).group(1))
            rep.cell(T, f"Table 9.6 {a} seed {k}", cell, num, r["worst_db"])
            rep.cell(T, f"Table 9.6 {a} seed {k} bound marked", cell, float("geq" in cell), float(r["lower_bound"]))
        spr, f = spread_of_seeds(res)
        rep.cell(T, f"Table 9.6 {a} spread", m.group(2), float(re.search(r"([\d.]+)", m.group(2)).group(1)), spr)
        rep.cell(T, f"Table 9.6 {a} spread bound marked", m.group(2), float("geq" in m.group(2)), float(f == ">"))
        if a == "Transformer-3K":
            m2 = re.search(r"whereas the Transformer's is at least \$([\d.]+)\$~dB", tex)
            if m2:
                rep.cell(T, "Transformer spread (Sec. 7.6)", m2.group(1), float(m2.group(1)), spr)
    # Coded size sweep (Sec. 7.2.1): worst-of-seeds penalties against block DF.
    rng = {}
    for f in ("coded_min_size.json", "coded_min_size_matched.json"):
        c = json.load(open(os.path.join(R, f)))
        vals = [penalty_common(c["snr_db"], np.max([r["ber"] for r in x["seed_runs"]], axis=0), c["df_ber"])
                for x in c["sweep"]]
        rng[f] = vals
    m = re.search(r"penalties of \$\+([\d.]+)\$--\$\+([\d.]+)\$~dB under its hard read-out against unweighted block DF, and at least \$\+([\d.]+)\$~dB against matched block DF", tex)
    if m:
        u = [r["worst_db"] for r in rng["coded_min_size.json"]]
        rep.cell(T, "coded sweep, unweighted low", m.group(1), float(m.group(1)), min(u))
        rep.cell(T, "coded sweep, unweighted high", m.group(2), float(m.group(2)), max(u))
        rep.cell(T, "coded sweep, matched lowest", m.group(3), float(m.group(3)),
                 min(r["worst_db"] for r in rng["coded_min_size_matched.json"]))
    # Window 1 -> 7 gain: "more than 8 dB" must be a true lower bound on every three-tap channel.
    gains = []
    for c in ms["channels"].values():
        if c["memory"] <= 1:
            continue
        base = c.get("baseline_ber", c["df_ber"])
        best = {}
        for w in (1, 7):
            res = [penalty_common(ms["snr_db"], r["ber"], base) for x in c["sweep"] if x["window"] == w for r in x["seed_runs"]]
            best[w] = min(r["worst_db"] for r in res)
        gains.append(best[1] - best[7])
    n8 = len(re.findall(r"more than 8~dB", tex))
    rep.cell(T, "window-gain statements found", str(n8), float(n8 >= 3), 1.0)
    rep.cell(T, "smallest window 1->7 gain exceeds 8 dB", "8", 8.0, 8.0 if min(gains) > 8.0 else min(gains))
    # Table 9.4 coded row: no smaller configuration meets the strict budgets.
    m = re.search(r"coded & code & (.*?) \\\\", tex)
    if m:
        cells = [x.strip() for x in m.group(1).split("&")]
        rep.cell(T, "Table 9.4 coded row, strict budgets dashed", m.group(1), float(all(x == "---" for x in cells[:3])), 1.0)
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
              check_relay_param_counts, check_proof_copies_agree,
              check_slicer_floor, check_qpsk_decomposition_prose, check_scalar_map_prose,
              check_af_conventional_prose, check_isi_penalty_prose, check_hybrid_branch_prose,
              check_coded_matched, check_flat_conclusion_prose, check_coded_matched_secondary,
              check_qpsk_controls_prose, check_bcjr_benchmark,
              check_seed_spread_native_prose, check_khina_round2,
              check_soft_df_calibration_prose,
              check_mmse_monotonicity_prose,
              check_joint_latency, check_joint_memory,
              check_nlbias_prose, check_coded_fer_prose, check_mamba_lookahead_prose,
              check_seq_rerun_prose, check_timing_and_mmse_median_prose]
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

    if rep.coverage:
        print("\nCOVERAGE SHORTFALL (a check stopped seeing part of what it covers):")
        for t, n, floor in rep.coverage:
            print(f"  [{t}] examined {n} cells, expected at least {floor}")
        print("  A drop means the check lost sight of something -- a renamed row,")
        print("  a moved anchor, a table it can no longer parse. Fix the check, or")
        print("  raise the floor in MIN_CELLS only if the coverage genuinely shrank.")

    if rep.unexpected_skips:
        print("\nCHECK COULD NOT RUN (not in ALLOWED_SKIPS):")
        for t, reason in rep.unexpected_skips:
            print(f"  [{t}] {reason}")

    print("\nInformational (not pass/fail):")
    print("  tbl:table13, tbl:table25 report machine-dependent wall-clock timing;")
    print("  re-run run_experiments.py / the sequence-model benchmark to refresh those.")

    # A check that examined nothing, or could not run, is a failure. Previously
    # only mismatches set the exit code, so "I checked nothing" and "everything
    # matches" were the same green.
    return 1 if (rep.flags or rep.coverage or rep.unexpected_skips) else 0


if __name__ == "__main__":
    sys.exit(main())
