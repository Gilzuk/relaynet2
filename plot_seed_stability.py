"""Charts for the equal-budget seed-stability study and the Transformer failure.

Follows CHART_GUIDELINES.md and reuses the palette validated for the
minimum-size charts (six-check validator: lightness band, chroma floor, CVD
separation dE 11.0 against a target of 8, normal-vision floor, contrast --
all pass, no warnings). Every trial is plotted individually; nothing is
averaged away, because the spread between trials is the subject.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PALETTE = ["#0072B2", "#D55E00", "#009E73", "#7B3294", "#8C6D00"]
MARKERS = ["o", "s", "^", "D", "v"]
GREY, BLACK = "#8A8A8A", "#222222"
LW = 1.3
H4_BAR = 0.18          # the effect H4's 4.1% high-SNR spread corresponds to

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 16, "axes.labelsize": 14,
    "legend.fontsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})


def chart_spread():
    d = json.load(open("results/seed_spread_architectures.json"))
    items = sorted(d["architectures"].items(), key=lambda kv: kv[1]["spread_db"])
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for i, (name, v) in enumerate(items):
        vals = [r["db_penalty"] for r in v["runs"]]
        ax.plot([min(vals), max(vals)], [i, i], color=GREY, linewidth=2.0,
                solid_capstyle="round", zorder=2)
        ax.plot(vals, np.full(len(vals), i, dtype=float), linestyle="none",
                marker=MARKERS[i % len(MARKERS)], markersize=9,
                color=PALETTE[i % len(PALETTE)], markeredgecolor="white",
                markeredgewidth=1.2, zorder=3,
                label=f"{name}  (spread {v['spread_db']:.3f} dB)")
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([n for n, _ in items])
    ax.set_xlabel("SNR penalty vs symbol-wise DF (dB) — one marker per initialization")
    ax.set_title("Every architecture is reproducible except the Transformer")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.annotate("this initialization\nfailed to converge", xy=(2.039, 4),
                xytext=(1.30, 3.05), fontsize=11, color=BLACK,
                arrowprops=dict(arrowstyle="->", color=BLACK, lw=0.8))
    ax.text(0.02, -0.20,
            f"H4's high-SNR architecture spread (4.1% BER) corresponds to about "
            f"{H4_BAR:.2f} dB;\nonly the Transformer's spread exceeds it, so only it "
            f"cannot be resolved from a single run.",
            transform=ax.transAxes, fontsize=11, color=GREY, va="top")
    fig.tight_layout()
    fig.savefig("results/arch_seed_spread.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def chart_loss_vs_penalty():
    d = json.load(open("results/transformer_instability.json"))
    t, m = d["runs"]["Transformer-3K"], d["runs"]["Mamba-S6-3K"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for runs, name, c, mk in ((t, "Transformer-3K (8 seeds)", PALETTE[1], MARKERS[1]),
                              (m, "Mamba-S6-3K control (4 seeds)", PALETTE[2], MARKERS[2])):
        ax.plot([r["final_loss"] for r in runs], [r["db_penalty"] for r in runs],
                linestyle="none", marker=mk, markersize=9, color=c,
                markeredgecolor="white", markeredgewidth=1.2, label=name)
    r0 = [r for r in t if r["seed"] == 0][0]
    ax.annotate("seed 0: loss 11x the median,\nthe only run that fails",
                xy=(r0["final_loss"], r0["db_penalty"]), xytext=(0.0026, 1.62),
                fontsize=11, color=BLACK,
                arrowprops=dict(arrowstyle="->", color=BLACK, lw=0.8))
    lo = np.array([r["final_loss"] for r in t])
    db = np.array([r["db_penalty"] for r in t])
    ax.set_xscale("log")
    ax.set_xlabel("Final training loss")
    ax.set_ylabel("SNR penalty vs DF (dB)")
    ax.set_title("Loss detects the failure, but not fine-grained quality")
    keep = lo <= 5 * np.median(lo)
    ax.text(0.03, 0.95,
            f"r = {np.corrcoef(lo, db)[0,1]:+.3f} over all 8 seeds, but that is the\n"
            f"outlier's leverage: r = {np.corrcoef(lo[keep], db[keep])[0,1]:+.2f} "
            f"(p = 0.41) among the 7\nthat converge, where loss carries no BER signal.",
            transform=ax.transAxes, fontsize=11, color=BLACK, va="top")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig("results/transformer_loss_vs_penalty.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def chart_seed_curves():
    d = json.load(open("results/transformer_instability.json"))
    snrs = d["snr_db"]; df = np.asarray(d["df_ber"], float)
    t = d["runs"]["Transformer-3K"]
    med = np.median([r["final_loss"] for r in t])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(snrs, df, color=BLACK, linewidth=LW, linestyle="-.", marker="+",
                markersize=7, label="DF (0 params)")
    first = True
    for r in t:
        bad = r["final_loss"] > 5 * med
        ax.semilogy(snrs, r["ber"], linewidth=LW if bad else 0.9,
                    color=PALETTE[1] if bad else GREY,
                    marker="s" if bad else None, markersize=6,
                    zorder=3 if bad else 2,
                    label=("seed 0 (failed to converge)" if bad
                           else ("seeds 1-7 (converged)" if first else None)))
        if not bad:
            first = False
    allb = [v for r in t for v in r["ber"]] + list(df)
    ax.set_ylim(min(v for v in allb if v > 0) / 10.0, 1.0)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER")
    ax.set_title("Transformer-3K: all eight initializations")
    ax.annotate("the failed run tracks the others at 0 dB\n"
                "and separates as SNR rises", xy=(12, 0.0812),
                xytext=(1.2, 0.0075), fontsize=11, color=BLACK,
                arrowprops=dict(arrowstyle="->", color=BLACK, lw=0.8))
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig("results/transformer_seed_ber_curves.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    chart_spread(); chart_loss_vs_penalty(); chart_seed_curves()
    print("wrote 3 charts to results/")
