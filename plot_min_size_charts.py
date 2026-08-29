"""Charts for the minimum-relay-size study.

Follows CHART_GUIDELINES.md: unique colour+marker per curve, thin lines,
legend outside the plot area, grey/black baselines, 8x5 at 1.5:1, title 16 /
axes 14 / legend 12, light gridlines, descriptive filenames, and a BER
summary table alongside the plots.

Palette is the Okabe-Ito-derived set below, validated with the dataviz
six-check validator (lightness band, chroma floor, CVD separation, normal
vision floor, contrast vs surface) -- all five pass, worst adjacent CVD
separation dE 11.0 against a target of 8. Markers carry identity as well as
colour, so the series stay separable in greyscale and under any CVD.

Everything is read from the stored result JSONs; no experiment is re-run.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ber_metrics import penalty_table, DEFAULT_TARGETS

PALETTE = ["#0072B2", "#D55E00", "#009E73", "#7B3294", "#8C6D00"]
MARKERS = ["o", "s", "^", "D", "v"]
GREY, BLACK = "#8A8A8A", "#222222"
LW = 1.3
FIGSIZE = (8, 5)

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 16, "axes.labelsize": 14,
    "legend.fontsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

WINDOWS = [1, 3, 5, 7]


def load(p):
    return json.load(open(p))


def best_db_by_window(r, snrs):
    """Best (lowest) dB penalty achieved at each window width."""
    base = np.asarray(r.get("baseline_ber", r["df_ber"]), dtype=float)
    out = {}
    for w in WINDOWS:
        vals = []
        for x in r["sweep"]:
            if x["window"] != w:
                continue
            for s in x["seed_runs"]:
                p = penalty_table(snrs, s["ber"], base)
                if p["targets_reached"]:
                    vals.append(p["worst_db_penalty"])
        out[w] = min(vals) if vals else np.nan
    return out


# ── 1. the central finding: window helps only where the channel has memory ──
def chart_window_crossover(d):
    snrs = d["snr_db"]
    groups = [("Memoryless channels (1 tap)", 1),
              ("Channels with memory (3 taps)", 3)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

    for ax, (title, mem) in zip(axes, groups):
        chans = [(n, r) for n, r in d["channels"].items() if r["memory"] == mem]
        for i, (n, r) in enumerate(chans):
            by_w = best_db_by_window(r, snrs)
            xs = [w for w in WINDOWS if by_w[w] == by_w[w]]
            ys = [by_w[w] for w in xs]
            ax.plot(xs, ys, marker=MARKERS[i % len(MARKERS)],
                    color=PALETTE[i % len(PALETTE)], linewidth=LW,
                    markersize=6, label=f"{n}  ({r['baseline']})")
        ax.axhline(0, color=BLACK, linewidth=1.0, linestyle="--", zorder=1)
        ax.set_title(title)
        ax.set_xlabel("Relay window (symbols)")
        ax.set_xticks(WINDOWS)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16),
                  frameon=False, ncol=1)
    axes[0].set_ylabel("SNR penalty vs classical baseline (dB)")
    # the "matches baseline" note lives on the right panel, where there is
    # room; on the left it collided with the inset
    axes[1].annotate("0 dB = matches baseline", xy=(6.0, 0), xytext=(4.4, 5.6),
                     fontsize=11, color=BLACK,
                     arrowprops=dict(arrowstyle="->", color=BLACK, lw=0.8))

    # The memoryless panel is flat on the shared scale -- which is the point,
    # so the scale stays shared -- but that hides whether the window costs
    # anything at all. Inset zooms the four near-zero channels; nlbias sits at
    # -2.9 dB and is deliberately outside it.
    ins = axes[0].inset_axes([0.30, 0.46, 0.56, 0.34])
    for i, (n, r) in enumerate([(n, r) for n, r in d["channels"].items()
                                if r["memory"] == 1 and n != "nlbias"]):
        by_w = best_db_by_window(r, snrs)
        xs = [w for w in WINDOWS if by_w[w] == by_w[w]]
        ins.plot(xs, [by_w[w] for w in xs], marker=MARKERS[i % len(MARKERS)],
                 color=PALETTE[i % len(PALETTE)], linewidth=1.0, markersize=4)
    ins.axhline(0, color=BLACK, linewidth=0.8, linestyle="--")
    ins.set_xticks(WINDOWS)
    ins.tick_params(labelsize=9)
    ins.set_title("zoom, dB (nlbias excluded)", fontsize=9, color=GREY, pad=2)
    ins.grid(alpha=0.2)
    fig.suptitle("A wider window pays off only when the channel has memory",
                 fontsize=16, y=0.99)
    fig.tight_layout()
    fig.savefig("results/minsize_window_crossover.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


# ── 2. headline: how far the best small relay is from its baseline ──────────
def chart_db_penalty_by_channel(d):
    snrs = d["snr_db"]
    rows = []
    for n, r in d["channels"].items():
        base = np.asarray(r.get("baseline_ber", r["df_ber"]), dtype=float)
        best = None
        for x in r["sweep"]:
            for s in x["seed_runs"]:
                p = penalty_table(snrs, s["ber"], base)
                if p["targets_reached"] and (best is None
                                             or p["worst_db_penalty"] < best[0]):
                    best = (p["worst_db_penalty"], x)
        if best:
            rows.append((n, best[0], best[1], r))
    rows.sort(key=lambda z: z[1])

    fig, ax = plt.subplots(figsize=(9, 5))
    names = [z[0] for z in rows]
    vals = [z[1] for z in rows]
    # diverging: one hue each side of a neutral zero, never a hue at the middle
    cols = [PALETTE[0] if v <= 0 else PALETTE[1] for v in vals]
    y = np.arange(len(rows))
    ax.barh(y, vals, color=cols, height=0.62, zorder=3)
    ax.axvline(0, color=BLACK, linewidth=1.0, zorder=4)
    ax.set_yticks(y)
    # config goes in the tick label, only the dB value sits at the bar end --
    # putting both at the bar end made the negative-bar labels run left into
    # the tick labels and overprint them
    ax.set_yticklabels(
        [f"{n} [{z[3]['baseline']}]\nw={z[2]['window']} h1-{z[2]['hidden']}p"
         f" ({z[2]['params']}p)" for n, z in zip(names, rows)], fontsize=10)
    ax.set_xlabel("SNR penalty vs classical baseline (dB)")
    for yi, v in zip(y, vals):
        off = 0.10 if v >= 0 else -0.10
        ax.text(v + off, yi, f"{v:+.2f} dB", va="center",
                ha="left" if v >= 0 else "right", fontsize=11, color=BLACK)
    ax.set_xlim(min(vals) - 1.2, max(vals) + 1.2)
    ax.set_title("Best small relay against each channel's own baseline\n"
                 "left of zero = better than the classical baseline",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig("results/minsize_db_penalty_by_channel.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


# ── 3. how small can it be and still match MLP-169, on the 3-tap channels ───
def chart_vs_mlp169(d):
    """Small multiples, one line per window.

    An earlier version plotted a single line per channel against parameter
    count, sorted by parameter count. That produced a sawtooth that was pure
    artifact: ordering by size interleaves windows (4p is w=1 h1-1p, 6p is
    w=3 h1-1p, 7p is w=1 h1-2p, 8p is w=5 h1-1p), so the line kept jumping
    between window-1 configurations that fail badly and wider ones that do
    not, implying a trajectory no experiment traced. Parameter count alone
    does not order these configurations; window does, within it.
    """
    chans = [(n, r) for n, r in d["channels"].items() if r["memory"] == 3]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, sharey=True)

    for ax, (n, r) in zip(axes.ravel(), chans):
        ref = [x for x in r["sweep"] if x["window"] == 5 and x["hidden"] == 24]
        if not ref:
            continue
        refb = np.mean([s["ber"] for s in ref[0]["seed_runs"]], axis=0)
        usable = [k for k in range(len(refb)) if refb[k] > 1e-5]
        for i, w in enumerate(WINDOWS):
            pts = []
            for x in sorted([z for z in r["sweep"] if z["window"] == w],
                            key=lambda z: z["params"]):
                worst = np.max([s["ber"] for s in x["seed_runs"]], axis=0)
                if not usable:
                    continue
                pts.append((x["params"],
                            100 * max((worst[k] - refb[k]) / refb[k]
                                      for k in usable)))
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        marker=MARKERS[i % len(MARKERS)],
                        color=PALETTE[i % len(PALETTE)], linewidth=LW,
                        markersize=6, label=f"window {w}")
        ax.axhline(2.0, color=BLACK, linewidth=1.0, linestyle="--")
        ax.axvline(169, color=GREY, linewidth=1.0, linestyle=":")
        ax.set_xscale("log")
        # symlog, not log: configurations that BEAT MLP-169 have negative
        # penalties and a log axis cannot show them. An earlier version
        # clamped them to a 1e-2 floor, which drew "much better than MLP-169"
        # and "0.01% worse" as the same flat line.
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_ylim(-8, 3e6)
        ax.set_title(n, fontsize=14)

    axes[0][0].text(190, 4e4, "MLP-169", fontsize=10, color=GREY, rotation=90)
    axes[0][0].text(4.5, 2.8, "2% match bar", fontsize=10, color=BLACK)
    axes[1][0].text(4.5, -6.0, "below 0 = beats MLP-169", fontsize=10,
                    color=GREY)
    for ax in axes[1]:
        ax.set_xlabel("Relay parameters")
    for ax in axes[:, 0]:
        ax.set_ylabel("BER penalty vs MLP-169 (%)")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="center left", bbox_to_anchor=(1.0, 0.5),
               frameon=False)
    fig.suptitle("Window, not parameter count, decides how close a relay gets"
                 " to MLP-169", fontsize=16, y=1.0)
    fig.tight_layout()
    fig.savefig("results/minsize_vs_mlp169_3tap.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


# ── 4. coded: a threshold in width, and a wall past it ──────────────────────
def chart_coded_threshold():
    c = load("results/coded_min_size.json")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, h in enumerate(sorted({r["hidden"] for r in c["sweep"]})):
        g = sorted([r for r in c["sweep"] if r["hidden"] == h],
                   key=lambda z: z["params"])
        ax.plot([r["params"] for r in g],
                [100 * r["worst_rel_operational"] for r in g],
                marker=MARKERS[i % len(MARKERS)], color=PALETTE[i % len(PALETTE)],
                linewidth=LW, markersize=6, label=f"hidden {h}")
    ax.axhline(0, color=BLACK, linewidth=1.0, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("Relay parameters")
    ax.set_ylabel("Worst-case BER penalty vs block DF (%)")
    ax.set_title("Coded relay: width sets a threshold, size does not help")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
              title="neurons per layer")
    ax.annotate("width 2 and 8 pinned at ~+50%\nfrom 18 to 540 parameters",
                xy=(520, 51.7), xytext=(22, 40), fontsize=11, color=BLACK,
                arrowprops=dict(arrowstyle="->", color=BLACK, lw=0.8,
                                connectionstyle="arc3,rad=-0.15"))
    ax.annotate("width 16 steps down,\nthen loses it again",
                xy=(244, 35.3), xytext=(430, 20), fontsize=11, color=BLACK,
                arrowprops=dict(arrowstyle="->", color=BLACK, lw=0.8,
                                connectionstyle="arc3,rad=0.15"))
    fig.tight_layout()
    fig.savefig("results/coded_minsize_width_threshold.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


# ── 5. the underlying BER curves, for one channel with memory ───────────────
def chart_ber_curves(d, channel="composite"):
    r = d["channels"][channel]
    snrs = d["snr_db"]
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # baselines in grey/black, per guideline 8
    ax.semilogy(snrs, r["af_ber"], color=GREY, linewidth=LW, linestyle="--",
                marker="x", markersize=5, label="AF (0p)")
    ax.semilogy(snrs, r["df_ber"], color="#5A5A5A", linewidth=LW,
                linestyle="-.", marker="+", markersize=7, label="DF (0p)")
    ax.semilogy(snrs, r["baseline_ber"], color=BLACK, linewidth=LW,
                marker="*", markersize=7, label=f"{r['baseline']} (0p)")

    picks, i = [], 0
    for w, h in ((1, 1), (3, 1), (7, 16)):
        m = [x for x in r["sweep"] if x["window"] == w and x["hidden"] == h]
        if m:
            picks.append(m[0])
    ref = [x for x in r["sweep"] if x["window"] == 5 and x["hidden"] == 24]
    picks += ref
    for x in picks:
        b = np.mean([s["ber"] for s in x["seed_runs"]], axis=0)
        lab = f"w={x['window']} h1-{x['hidden']}p ({x['params']}p)"
        if (x["window"], x["hidden"]) == (5, 24):
            lab += "  = MLP-169"
        ax.semilogy(snrs, b, marker=MARKERS[i % len(MARKERS)],
                    color=PALETTE[i % len(PALETTE)], linewidth=LW,
                    markersize=6, label=lab)
        i += 1

    allb = [v for c in (r["af_ber"], r["df_ber"], r["baseline_ber"]) for v in c]
    allb += [v for x in picks for s in x["seed_runs"] for v in s["ber"]]
    lo = min(v for v in allb if v > 0)
    ax.set_ylim(lo / 10.0, 1.0)          # exactly one decade below the min
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER")
    ax.set_title(f"BER on {channel}: relay size against the classical baselines")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(f"results/minsize_ber_curves_{channel}.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


# ── BER summary table at key SNR points, per guideline 18 ───────────────────
def summary_table(d, out="results/minsize_summary_table.md"):
    snrs = d["snr_db"]
    key = [4, 16]
    idx = [snrs.index(k) for k in key if k in snrs]
    lines = ["| channel | baseline | validity | best config | dB penalty |"
             + "".join(f" BER@{snrs[i]}dB base | BER@{snrs[i]}dB relay |" for i in idx),
             "|---|---|---|---|---|" + "---|" * (2 * len(idx))]
    for n, r in d["channels"].items():
        base = np.asarray(r.get("baseline_ber", r["df_ber"]), dtype=float)
        best = None
        for x in r["sweep"]:
            for s in x["seed_runs"]:
                p = penalty_table(snrs, s["ber"], base)
                if p["targets_reached"] and (best is None
                                             or p["worst_db_penalty"] < best[0]):
                    best = (p["worst_db_penalty"], x, s)
        if not best:
            lines.append(f"| {n} | {r['baseline']} | "
                         f"{r['baseline_diagnostics']['verdict']} | -- | -- |"
                         + "".join(f" {base[i]:.5f} | -- |" for i in idx))
            continue
        v, x, s = best
        cfg = f"w={x['window']} h1-{x['hidden']}p ({x['params']}p)"
        lines.append(f"| {n} | {r['baseline']} | "
                     f"{r['baseline_diagnostics']['verdict']} | {cfg} | {v:+.2f} |"
                     + "".join(f" {base[i]:.5f} | {s['ber'][i]:.5f} |" for i in idx))
    open(out, "w").write("\n".join(lines) + "\n")
    return "\n".join(lines)


if __name__ == "__main__":
    d = load("results/mlp_min_size_all_channels.json")
    chart_window_crossover(d)
    chart_db_penalty_by_channel(d)
    chart_vs_mlp169(d)
    chart_coded_threshold()
    chart_ber_curves(d, "composite")
    print(summary_table(d))
    print("\nwrote 5 charts + summary table to results/")
