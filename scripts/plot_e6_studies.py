#!/usr/bin/env python3
"""Regenerate the Chapter 7 composite / blind / partial-posterior figures.

Reads the committed .npy outputs of e6_composite_ported.py, e6_blind_ported.py
and e6_partial_ported.py and rebuilds the four figures the thesis includes:

    results/e6_composite.png
    results/e6_blind.png
    results/e6_partial_pilot_budget_sweep.png
    results/e6_partial_short_blocks_overhead.png

Figures are written to BOTH results/ and thesis/results/ so the repository copy
and the copy main.tex compiles against never drift apart.

Until this script existed the four figures had no regeneration path in the
repository -- only their cached .npy inputs survived -- so a rerun of the
experiments could not be reflected in the document. Run it after any rerun:

    python3 scripts/plot_e6_studies.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPY_DIR = os.path.join(ROOT, "e6_unknown_channel_results")
OUT_DIRS = [os.path.join(ROOT, "results"), os.path.join(ROOT, "thesis", "results")]

# Shared per-method styling, so the composite and blind panels stay legible
# side by side in the compiled document.
STYLE = {
    "AF":                  dict(color="tab:orange", marker="s", ls="--"),
    "DF-diff":             dict(color="tab:red",    marker="o", ls="-"),
    "Viterbi-diff":        dict(color="tab:purple", marker="v", ls="-."),
    "Viterbi-blind":       dict(color="tab:purple", marker="v", ls="-."),
    "CMA-blind":           dict(color="tab:orange", marker="s", ls="--"),
    "MLP-169":             dict(color="tab:blue",   marker="^", ls="-"),
    "MLP-large":           dict(color="tab:green",  marker="D", ls=":"),
}
LABELS = {
    "AF": "AF",
    "DF-diff": "DF (differential)",
    "Viterbi-diff": r"Pilot-LS Viterbi + diff. (matched to ISI+phase)",
    "Viterbi-blind": "Decision-directed blind MLSE (no pilots)",
    "CMA-blind": "CMA blind equalizer (no pilots)",
    "MLP-169": "MLP-169 (no impairment knowledge)",
    "MLP-large": "MLP-1153",
}


def _save(fig, name):
    for d in OUT_DIRS:
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def _load(name):
    return np.load(os.path.join(NPY_DIR, name), allow_pickle=True).item()


def plot_ber_panel(d, order, title, name, floor=None, mlp_label=None):
    """BER-vs-SNR semilog panel with 95% CI bands."""
    snrs = np.asarray(d["snrs"], dtype=float)
    fig, ax = plt.subplots(figsize=(10, 6))
    for key in order:
        if key not in d["summary"]:
            continue
        mu, ci = (np.asarray(v, dtype=float) for v in d["summary"][key])
        st = STYLE[key]
        label = mlp_label if (key == "MLP-169" and mlp_label) else LABELS[key]
        ax.semilogy(snrs, np.maximum(mu, 1e-5), label=label, markersize=6, **st)
        ax.fill_between(snrs, np.maximum(mu - ci, 1e-6), np.maximum(mu + ci, 1e-6),
                        color=st["color"], alpha=0.18, lw=0)
    if floor is not None:
        ax.axhline(floor, color="0.4", ls=":", lw=1.2)
        ax.text(0.3, floor * 1.06, "memoryless-relay floor", color="0.4", fontsize=9)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    n = d.get("n_trials", "?")
    b = d.get("n_bits", "?")
    ax.annotate(f"{n} trials $\\times$ {b:,} bits/SNR point",
                xy=(0.99, 0.98), xycoords="axes fraction",
                ha="right", va="top", fontsize=8, color="0.35")
    _save(fig, name)


def plot_composite():
    d = _load("e6_composite_ported_results.npy")
    plot_ber_panel(
        d,
        ["AF", "DF-diff", "Viterbi-diff", "MLP-169", "MLP-large"],
        r"Composite channel: ISI $\times$ PA-nonlinearity $\times$ unknown phase (DBPSK)",
        "e6_composite.png",
        floor=0.25,
    )


def plot_blind():
    d = _load("e6_blind_ported_results.npy")
    plot_ber_panel(
        d,
        ["DF-diff", "Viterbi-blind", "CMA-blind", "MLP-169"],
        "Posterior-free (blind) composite channel: no pilots, no channel prior",
        "e6_blind.png",
        mlp_label="MLP-169 (family-trained, blind at test)",
    )


def plot_partial_pilots():
    d = _load("e6_partial_ported_results.npy")
    pilots = list(d["pilots"])
    mus = np.array([d["panel_a"][p][0] for p in pilots])
    cis = np.array([d["panel_a"][p][1] for p in pilots])
    mlp_mu = d["mlp_ref"][0]
    cma_mu = d["cma_ref"][0]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axhline(mlp_mu, color="tab:blue", lw=2, label="MLP-169 (0 pilots)")
    ax.axhline(cma_mu, color="tab:orange", ls="--", lw=2, label="CMA blind (0 pilots)")
    ax.errorbar(pilots, mus, yerr=cis, color="tab:purple", ls="-.", marker="v",
                capsize=3, label="Pilot-aided Viterbi")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Number of pilots (partial posterior)")
    ax.set_ylabel(f"Payload BER @ {d['op_snr']:.0f} dB")
    ax.set_title("(a) Partial posterior: pilot-budget sweep")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    worst = pilots[int(np.argmax(mus))]
    ax.annotate("estimation variance\ndominates", xy=(worst, mus.max()),
                xytext=(0.62, 0.62), textcoords="axes fraction",
                fontsize=9, color="0.35",
                arrowprops=dict(arrowstyle="->", color="0.55", lw=1))
    _save(fig, "e6_partial_pilot_budget_sweep.png")


def plot_partial_blocks():
    d = _load("e6_partial_ported_results.npy")
    Ls = list(d["block_lengths"])
    vit = np.array([d["panel_b"][L][0] for L in Ls])
    vit_ci = np.array([d["panel_b"][L][1] for L in Ls])
    frac = np.array([1.0 - d["panel_b"][L][2] for L in Ls])
    mlp_mu = d["mlp_ref"][0]
    cma = d.get("panel_b_cma")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.errorbar(Ls, vit, yerr=vit_ci, color="tab:purple", ls="-.", marker="v",
                capsize=3, label="Viterbi BER (10 pilots/block)")
    ax.axhline(mlp_mu, color="tab:blue", lw=2, label="MLP BER (0 pilots)")
    if cma is not None:
        cmu = np.array([cma[L][0] for L in Ls])
        cci = np.array([cma[L][1] for L in Ls])
        ax.errorbar(Ls, cmu, yerr=cci, color="tab:orange", ls="--", marker="s",
                    capsize=3, label="CMA blind BER (0 pilots)")
    ax.set_xscale("log")
    ax.set_xlabel("Block length $L$ (symbols)")
    ax.set_ylabel(f"Payload BER @ {d['op_snr']:.0f} dB")
    ax.set_title("(b) Short blocks: the pilot-overhead cost")
    ax.grid(True, which="both", alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(Ls, frac, color="0.45", ls=":", marker="s",
             label="Classical data rate (1-overhead)")
    ax2.set_ylabel("Fraction of block carrying data")
    ax2.set_ylim(0.6, 1.02)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=9)
    _save(fig, "e6_partial_short_blocks_overhead.png")


def main():
    print("Regenerating Chapter 7 study figures from committed .npy data...")
    plot_composite()
    plot_blind()
    plot_partial_pilots()
    plot_partial_blocks()
    print("Done.")


if __name__ == "__main__":
    main()
