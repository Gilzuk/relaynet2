"""
fix_qam16_legends.py
====================
Post-processes every QAM16 PNG referenced in thesis.md:
  - Moves legend BELOW the axes (horizontal, ncols = ceil(n/2), max 6/row)
  - Adds leader lines from each curve to a label at a representative SNR
  - Saves back to the same path (originals backed up as *_orig.png)

JSON structure used by all experiments:
  d['snr_range']          -> list of SNR values
  d['results'][label]     -> {'ber_mean': [...], 'ci_lower': [...], 'ci_upper': [...]}
"""
import os, sys, io, json, math, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Colorblind-safe palette + markers ────────────────────────────────────────
PALETTE = [
    "#0072B2","#D55E00","#009E73","#E69F00",
    "#56B4E9","#CC79A7","#F0E442","#882255",
    "#4B0082","#8B4513","#2ca02c","#17becf",
    "#888888","#333333",
]
MARKERS = ["o","s","^","D","v","p","h","X","*","P","<",">","1","2"]
LSTYLES = ["-","--","-.",":","-","--","-.",":","-","--","-.",":","-","--"]

# Fixed styles for AF/DF baselines
BASELINE_STYLE = {
    "AF": {"color": "#888888", "marker": "<", "ls": ":"},
    "DF": {"color": "#333333", "marker": ">", "ls": ":"},
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def parse_results(data):
    """Return (snr, ber_dict, ci_dict) from the standard JSON format."""
    snr = data["snr_range"]
    ber_dict = {}
    ci_dict  = {}
    for label, v in data["results"].items():
        ber = np.asarray(v["ber_mean"])
        ber_dict[label] = ber
        if "ci_lower" in v and "ci_upper" in v:
            ci_dict[label] = (
                np.maximum(np.asarray(v["ci_lower"]), 1e-8),
                np.asarray(v["ci_upper"]),
            )
    return snr, ber_dict, ci_dict


def assign_styles(labels):
    """Assign color/marker/linestyle to each label."""
    colors, markers, lstyles = {}, {}, {}
    ai_idx = 0
    for label in labels:
        up = label.upper()
        if up in BASELINE_STYLE:
            s = BASELINE_STYLE[up]
            colors[label]  = s["color"]
            markers[label] = s["marker"]
            lstyles[label] = s["ls"]
        else:
            colors[label]  = PALETTE[ai_idx % len(PALETTE)]
            markers[label] = MARKERS[ai_idx % len(MARKERS)]
            lstyles[label] = LSTYLES[ai_idx % len(LSTYLES)]
            ai_idx += 1
    return colors, markers, lstyles


# ── Legend below axes ─────────────────────────────────────────────────────────
def legend_below(ax, fig, n_items, max_per_row=6, fontsize=9):
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    if ax.get_legend():
        ax.get_legend().remove()
    ncols = min(n_items, max_per_row)
    nrows = math.ceil(n_items / ncols)
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=ncols,
        fontsize=fontsize,
        framealpha=0.92,
        borderaxespad=0.4,
        handlelength=2.2,
        columnspacing=0.9,
        handletextpad=0.5,
    )
    # Push axes up to make room for legend rows
    fig.subplots_adjust(bottom=0.10 + 0.055 * nrows)


# ── Leader lines ──────────────────────────────────────────────────────────────
def add_leader_lines(ax, snr, ber_dict, colors, markers, ann_snr_idx=-4):
    """
    For each curve draw a short arrow from the curve at ann_snr_idx
    to a text label placed to the right, staggered vertically in log space
    to avoid overlap.
    """
    snr = list(snr)
    idx = ann_snr_idx % len(snr)
    x_curve = snr[idx]
    x_range = snr[-1] - snr[0]
    x_text  = x_curve + x_range * 0.03   # small offset to the right

    # Collect (log_y, label) sorted top-to-bottom
    entries = []
    for label, ber in ber_dict.items():
        ber = np.asarray(ber)
        y = ber[idx]
        if y > 0:
            entries.append((math.log10(y), label, y))
    entries.sort(key=lambda e: e[0], reverse=True)  # high BER first

    # Stagger: ensure min 0.22 decades between text labels
    placed = []   # list of log_y_text already placed
    for log_y, label, y_curve in entries:
        log_y_text = log_y
        # Push down if too close to an already-placed label
        for py in placed:
            if abs(py - log_y_text) < 0.22:
                log_y_text = py - 0.24
        placed.append(log_y_text)
        y_text = 10 ** log_y_text

        color = colors.get(label, "black")
        ax.annotate(
            label,
            xy=(x_curve, y_curve),
            xytext=(x_text, y_text),
            fontsize=6.5,
            color=color,
            fontweight="bold",
            arrowprops=dict(
                arrowstyle="-",
                color=color,
                lw=0.9,
                connectionstyle="arc3,rad=0.0",
            ),
            va="center",
            ha="left",
            clip_on=False,
        )


# ── Axes styling ──────────────────────────────────────────────────────────────
def style_ber_axes(ax, min_ber, title):
    ax.set_ylim(bottom=max(min_ber / 10, 1e-6), top=0.65)
    ax.yaxis.set_minor_locator(LogLocator(subs="all", numticks=20))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", linewidth=0.5, alpha=0.45)
    ax.grid(True, which="minor", linewidth=0.3, alpha=0.15)
    ax.set_xlabel("SNR (dB)", fontsize=13)
    ax.set_ylabel("Bit Error Rate (BER)", fontsize=13)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.tick_params(labelsize=11)


# ── Core redraw function ──────────────────────────────────────────────────────
def redraw(json_path, out_png, title,
           max_per_row=6, leader_idx=-4, figsize=(11, 6),
           skip_leaders_if_gt=20):
    """
    Load JSON, redraw BER figure with:
      - legend below (horizontal)
      - leader lines (skipped if >skip_leaders_if_gt curves to avoid clutter)
    """
    if not os.path.exists(json_path):
        print(f"  SKIP (no JSON): {json_path}")
        return
    if not os.path.exists(out_png):
        print(f"  SKIP (no PNG):  {out_png}")
        return

    data = load_json(json_path)
    snr, ber_dict, ci_dict = parse_results(data)
    colors, markers, lstyles = assign_styles(list(ber_dict.keys()))

    all_bers = np.concatenate([b for b in ber_dict.values()])
    min_ber  = all_bers[all_bers > 0].min() if np.any(all_bers > 0) else 1e-4

    fig, ax = plt.subplots(figsize=figsize)

    for label, ber in ber_dict.items():
        ax.semilogy(
            snr, ber,
            color=colors[label], marker=markers[label],
            markersize=4, linewidth=1.4, linestyle=lstyles[label],
            label=label,
        )
        if label in ci_dict:
            lo, hi = ci_dict[label]
            ax.fill_between(snr, lo, hi, color=colors[label], alpha=0.10)

    style_ber_axes(ax, min_ber, title)

    n = len(ber_dict)
    if n <= skip_leaders_if_gt:
        add_leader_lines(ax, snr, ber_dict, colors, markers, ann_snr_idx=leader_idx)

    legend_below(ax, fig, n_items=n, max_per_row=max_per_row)

    # Backup original
    orig = out_png.replace(".png", "_orig.png")
    if not os.path.exists(orig):
        shutil.copy2(out_png, orig)

    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fixed ({n} curves): {out_png}")


# ── Top-3 helper ──────────────────────────────────────────────────────────────
def redraw_top3(json_path, out_png, title, n_top=3):
    if not os.path.exists(json_path):
        print(f"  SKIP (no JSON): {json_path}")
        return
    if not os.path.exists(out_png):
        print(f"  SKIP (no PNG):  {out_png}")
        return

    data = load_json(json_path)
    snr, ber_dict, ci_dict = parse_results(data)

    # Separate baselines from AI variants
    baselines = {k: v for k, v in ber_dict.items() if k.upper() in BASELINE_STYLE}
    ai_vars   = {k: v for k, v in ber_dict.items() if k.upper() not in BASELINE_STYLE}

    # Rank AI variants by mean BER in upper half of SNR range
    half = len(snr) // 2
    ranking = sorted(ai_vars, key=lambda k: np.mean(ai_vars[k][half:]))
    top_keys = ranking[:n_top]

    TOP_CLR = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7"]
    TOP_MRK = ["o", "s", "^", "D", "v"]

    plot_dict = {}
    colors, markers, lstyles = {}, {}, {}

    for k, v in baselines.items():
        plot_dict[k] = v
        s = BASELINE_STYLE[k.upper()]
        colors[k]  = s["color"]
        markers[k] = s["marker"]
        lstyles[k] = s["ls"]

    for i, k in enumerate(top_keys):
        plot_dict[k] = ai_vars[k]
        colors[k]  = TOP_CLR[i % len(TOP_CLR)]
        markers[k] = TOP_MRK[i % len(TOP_MRK)]
        lstyles[k] = "-"

    all_bers = np.concatenate(list(plot_dict.values()))
    min_ber  = all_bers[all_bers > 0].min() if np.any(all_bers > 0) else 1e-4

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, ber in plot_dict.items():
        ax.semilogy(
            snr, ber,
            color=colors[label], marker=markers[label],
            markersize=6, linewidth=1.6, linestyle=lstyles[label],
            label=label,
        )
        if label in ci_dict:
            lo, hi = ci_dict[label]
            ax.fill_between(snr, lo, hi, color=colors[label], alpha=0.13)

    style_ber_axes(ax, min_ber, title)
    add_leader_lines(ax, snr, plot_dict, colors, markers)
    legend_below(ax, fig, n_items=len(plot_dict), max_per_row=len(plot_dict))

    orig = out_png.replace(".png", "_orig.png")
    if not os.path.exists(orig):
        shutil.copy2(out_png, orig)

    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fixed (top-{n_top}): {out_png}")


# ═══════════════════════════════════════════════════════════════════════════════
#  JOBS — every QAM16 PNG referenced in thesis.md
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 62)
print("  Fixing QAM16 figure legends")
print("=" * 62)

# Fig 25 — 16-QAM relay comparison AWGN
redraw(
    "results/modulation/qam16_awgn.json",
    "results/modulation/qam16__awgn_ci.png",
    "16-QAM Relay Comparison — AWGN Channel",
    max_per_row=5,
)

# Fig 26 — 16-QAM relay comparison Rayleigh
redraw(
    "results/modulation/qam16_rayleigh.json",
    "results/modulation/qam16__rayleigh_ci.png",
    "16-QAM Relay Comparison — Rayleigh Fading",
    max_per_row=5,
)

# Fig 28 — 16-QAM activation experiment AWGN
redraw(
    "results/qam16_activation/qam16_activation_awgn.json",
    "results/qam16_activation/qam16_activation_awgn.png",
    "16-QAM Activation Experiment — AWGN Channel",
    max_per_row=5,
)

# Fig 29 — 16-QAM activation experiment Rayleigh
redraw(
    "results/qam16_activation/qam16_activation_rayleigh.json",
    "results/qam16_activation/qam16_activation_rayleigh.png",
    "16-QAM Activation Experiment — Rayleigh Fading",
    max_per_row=5,
)

# Fig 34 — QAM16 activation comparison AWGN
redraw(
    "results/activation_comparison/activation_qam16_awgn.json",
    "results/activation_comparison/qam16_activation_awgn.png",
    "QAM16 Activation Comparison — AWGN Channel",
    max_per_row=5,
)

# Fig 35 — QAM16 activation comparison Rayleigh
redraw(
    "results/activation_comparison/activation_qam16_rayleigh.json",
    "results/activation_comparison/qam16_activation_rayleigh.png",
    "QAM16 Activation Comparison — Rayleigh Fading",
    max_per_row=5,
)

# Fig 39 — Full CSI experiment (50 curves) — leaders skipped, legend only
redraw(
    "results/csi/csi_experiment_qam16_rayleigh.json",
    "results/csi/csi_experiment_qam16_rayleigh.png",
    "16-QAM CSI Experiment — All 48 Neural Variants (Rayleigh Fading)",
    max_per_row=6,
    figsize=(14, 8),
    skip_leaders_if_gt=20,   # 50 curves -> skip leaders
)

# Fig 40 — Top-3 CSI
redraw_top3(
    "results/csi/csi_experiment_qam16_rayleigh.json",
    "results/csi/top3_qam16_rayleigh.png",
    "Top-3 Neural Relays vs Classical — 16-QAM Rayleigh Fading",
    n_top=3,
)

print("\nAll done.")

import os, sys, io, json, math, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Shared style helpers ──────────────────────────────────────────────────────

def legend_below(ax, fig, ncols=None, fontsize=9):
    """Move legend below axes, horizontal layout."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    n = len(handles)
    if ncols is None:
        ncols = min(n, max(3, math.ceil(n / 2)))
    # Remove existing legend
    if ax.get_legend():
        ax.get_legend().remove()
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=ncols,
        fontsize=fontsize,
        framealpha=0.9,
        borderaxespad=0.5,
        handlelength=2.0,
        columnspacing=1.0,
    )
    fig.subplots_adjust(bottom=0.22 + 0.045 * math.ceil(n / ncols))


def add_leader_lines(ax, snr_values, ber_dict, colors, markers, label_snr_idx=-3):
    """
    Draw a short annotating arrow from the end of each curve to a
    text label placed just outside the curve, avoiding overlap.
    label_snr_idx: which SNR index to annotate (default: 3rd from right)
    """
    snr = np.asarray(snr_values)
    idx = label_snr_idx % len(snr)
    x_ann = snr[idx]

    # Collect (y_value, label) pairs and sort by y to stagger offsets
    entries = []
    for label, ber in ber_dict.items():
        ber = np.asarray(ber)
        if ber[idx] > 0:
            entries.append((ber[idx], label))
    entries.sort(key=lambda e: e[0], reverse=True)

    # Assign vertical offsets in log space to avoid overlap
    used_y = []
    for y_val, label in entries:
        color = colors.get(label, "black")
        marker = markers.get(label, "o")

        # Find a y_text that doesn't collide with already-placed labels
        y_text = y_val
        log_y = math.log10(max(y_val, 1e-9))
        for uy in used_y:
            if abs(math.log10(max(uy, 1e-9)) - math.log10(max(y_text, 1e-9))) < 0.25:
                y_text = 10 ** (math.log10(max(y_text, 1e-9)) + 0.28)
        used_y.append(y_text)

        # x offset: place label to the right of the annotation point
        x_text = x_ann + (snr[-1] - snr[0]) * 0.04

        ax.annotate(
            label,
            xy=(x_ann, y_val),
            xytext=(x_text, y_text),
            fontsize=7,
            color=color,
            arrowprops=dict(
                arrowstyle="-",
                color=color,
                lw=0.8,
                connectionstyle="arc3,rad=0.0",
            ),
            va="center",
            ha="left",
            clip_on=False,
        )


def ber_axes_style(ax, min_ber, title, xlabel="SNR (dB)", ylabel="Bit Error Rate"):
    ax.set_ylim(bottom=max(min_ber / 10, 1e-6), top=0.6)
    ax.yaxis.set_minor_locator(LogLocator(subs="all", numticks=20))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", linewidth=0.5, alpha=0.4)
    ax.grid(True, which="minor", linewidth=0.3, alpha=0.15)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=11)


# ═══════════════════════════════════════════════════════════════════════════════
#  1.  results/modulation/qam16__awgn_ci.png
#      results/modulation/qam16__rayleigh_ci.png
# ═══════════════════════════════════════════════════════════════════════════════

def fix_modulation_qam16():
    """Figures 25 & 26 — basic QAM16 relay comparison."""
    for channel in ("awgn", "rayleigh"):
        src = f"results/modulation/qam16__{channel}_ci.png"
        if not os.path.exists(src):
            print(f"  SKIP (not found): {src}")
            continue

        # Load JSON if available, else skip
        json_path = f"results/modulation/qam16_{channel}_results.json"
        if not os.path.exists(json_path):
            # Try alternate names
            for alt in [
                f"results/modulation/qam16_{channel}.json",
                f"results/modulation/{channel}_qam16.json",
            ]:
                if os.path.exists(alt):
                    json_path = alt
                    break
            else:
                print(f"  SKIP (no JSON): {src}")
                continue

        with open(json_path) as f:
            data = json.load(f)

        snr = data["snr"]
        variants = data.get("variants", {})
        baselines = data.get("baselines", {})

        # Build ber_dict preserving order: baselines first, then variants
        ber_dict = {}
        colors = {}
        markers = {}
        lstyles = {}

        PALETTE = [
            "#0072B2","#D55E00","#009E73","#E69F00",
            "#56B4E9","#CC79A7","#F0E442","#882255",
            "#888888","#333333",
        ]
        MLIST = ["o","s","^","D","v","p","h","X","<",">"]

        for i, (k, v) in enumerate(baselines.items()):
            label = k.upper()
            ber_dict[label] = v["ber_mean"]
            colors[label] = PALETTE[-(i+1)]
            markers[label] = MLIST[-(i+1)]
            lstyles[label] = ":"

        for i, (k, v) in enumerate(variants.items()):
            ber_dict[k] = v["ber_mean"]
            colors[k] = PALETTE[i % len(PALETTE)]
            markers[k] = MLIST[i % len(MLIST)]
            lstyles[k] = "-" if i % 3 == 0 else ("--" if i % 3 == 1 else "-.")

        all_bers = np.concatenate([np.asarray(b) for b in ber_dict.values()])
        min_ber = all_bers[all_bers > 0].min() if np.any(all_bers > 0) else 1e-4

        fig, ax = plt.subplots(figsize=(10, 6))
        for label, ber in ber_dict.items():
            ber = np.asarray(ber)
            ci_lo = ci_hi = None
            if label in variants and "ber_ci95_lo" in variants[label]:
                ci_lo = np.asarray(variants[label]["ber_ci95_lo"])
                ci_hi = np.asarray(variants[label]["ber_ci95_hi"])
            ax.semilogy(snr, ber, color=colors[label], marker=markers[label],
                        markersize=5, linewidth=1.4, linestyle=lstyles[label], label=label)
            if ci_lo is not None:
                ax.fill_between(snr, np.maximum(ci_lo, 1e-8), ci_hi,
                                color=colors[label], alpha=0.10)

        ch_title = "AWGN" if channel == "awgn" else "Rayleigh Fading"
        ber_axes_style(ax, min_ber, f"16-QAM Relay Comparison — {ch_title}")
        add_leader_lines(ax, snr, {k: np.asarray(v) for k, v in ber_dict.items()},
                         colors, markers)
        legend_below(ax, fig)
        shutil.copy2(src, src.replace(".png", "_orig.png"))
        fig.savefig(src, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Fixed: {src}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Generic fixer for any experiment JSON → PNG
# ═══════════════════════════════════════════════════════════════════════════════

PALETTE = [
    "#0072B2","#D55E00","#009E73","#E69F00",
    "#56B4E9","#CC79A7","#F0E442","#882255",
    "#4B0082","#8B4513","#888888","#333333",
]
MLIST = ["o","s","^","D","v","p","h","X","*","P","<",">"]
LLIST = ["-","--","-.",":","-","--","-.",":","-","--","-."]


def build_style_maps(variants_keys, baselines_keys):
    colors, markers, lstyles = {}, {}, {}
    for i, k in enumerate(baselines_keys):
        label = k.upper()
        colors[label]  = PALETTE[-(i+1) % len(PALETTE)]
        markers[label] = MLIST[-(i+1) % len(MLIST)]
        lstyles[label] = ":"
    for i, k in enumerate(variants_keys):
        colors[k]  = PALETTE[i % len(PALETTE)]
        markers[k] = MLIST[i % len(MLIST)]
        lstyles[k] = LLIST[i % len(LLIST)]
    return colors, markers, lstyles


def fix_from_json(json_path, out_png, title, ncols=None):
    """Generic: load JSON, redraw BER figure with fixed legend."""
    if not os.path.exists(json_path):
        print(f"  SKIP (no JSON): {json_path}")
        return
    if not os.path.exists(out_png):
        print(f"  SKIP (no PNG): {out_png}")
        return

    with open(json_path) as f:
        data = json.load(f)

    snr = data["snr"]
    variants  = data.get("variants", {})
    baselines = data.get("baselines", {})

    colors, markers, lstyles = build_style_maps(
        list(variants.keys()), list(baselines.keys())
    )

    ber_dict = {}
    for k, v in baselines.items():
        label = k.upper()
        ber_dict[label] = np.asarray(v["ber_mean"])
    for k, v in variants.items():
        ber_dict[k] = np.asarray(v["ber_mean"])

    all_bers = np.concatenate(list(ber_dict.values()))
    min_ber = all_bers[all_bers > 0].min() if np.any(all_bers > 0) else 1e-4

    fig, ax = plt.subplots(figsize=(11, 6))

    for label, ber in ber_dict.items():
        src_key = label.lower()
        ci_lo = ci_hi = None
        if label in variants:
            vd = variants[label]
            if "ber_ci95_lo" in vd:
                ci_lo = np.maximum(np.asarray(vd["ber_ci95_lo"]), 1e-8)
                ci_hi = np.asarray(vd["ber_ci95_hi"])
        ax.semilogy(snr, ber, color=colors[label], marker=markers[label],
                    markersize=5, linewidth=1.4, linestyle=lstyles[label], label=label)
        if ci_lo is not None:
            ax.fill_between(snr, ci_lo, ci_hi, color=colors[label], alpha=0.10)

    ber_axes_style(ax, min_ber, title)
    add_leader_lines(ax, snr, ber_dict, colors, markers)
    legend_below(ax, fig, ncols=ncols)

    shutil.copy2(out_png, out_png.replace(".png", "_orig.png"))
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fixed: {out_png}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CSI experiment — 48 variants (very dense legend)
# ═══════════════════════════════════════════════════════════════════════════════

def fix_csi_qam16():
    json_path = "results/csi/csi_experiment_qam16_rayleigh.json"
    out_png   = "results/csi/csi_experiment_qam16_rayleigh.png"
    top3_png  = "results/csi/top3_qam16_rayleigh.png"

    if not os.path.exists(json_path):
        print(f"  SKIP (no JSON): {json_path}")
        return

    with open(json_path) as f:
        data = json.load(f)

    snr       = data["snr"]
    variants  = data.get("variants", {})
    baselines = data.get("baselines", {})

    colors, markers, lstyles = build_style_maps(
        list(variants.keys()), list(baselines.keys())
    )

    ber_dict = {}
    for k, v in baselines.items():
        label = k.upper()
        ber_dict[label] = np.asarray(v["ber_mean"])
    for k, v in variants.items():
        ber_dict[k] = np.asarray(v["ber_mean"])

    all_bers = np.concatenate(list(ber_dict.values()))
    min_ber = all_bers[all_bers > 0].min() if np.any(all_bers > 0) else 1e-4

    # ── Main figure: all 48+ variants ────────────────────────────────────────
    if os.path.exists(out_png):
        n_curves = len(ber_dict)
        ncols = min(n_curves, 6)   # max 6 per row in legend
        fig, ax = plt.subplots(figsize=(13, 7))

        for label, ber in ber_dict.items():
            ax.semilogy(snr, ber, color=colors[label], marker=markers[label],
                        markersize=4, linewidth=1.2, linestyle=lstyles[label],
                        label=label, alpha=0.85)

        ber_axes_style(ax, min_ber,
                       "16-QAM CSI Experiment — All Variants (Rayleigh Fading)")
        # For 48 curves, leader lines would be too cluttered — skip them,
        # use the horizontal legend only
        legend_below(ax, fig, ncols=ncols, fontsize=7)

        shutil.copy2(out_png, out_png.replace(".png", "_orig.png"))
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Fixed (CSI all): {out_png}")

    # ── Top-3 figure ─────────────────────────────────────────────────────────
    if os.path.exists(top3_png):
        upper_half = len(snr) // 2
        ranking = sorted(
            [k for k in variants],
            key=lambda v: np.mean(np.asarray(variants[v]["ber_mean"])[upper_half:])
        )
        top3 = ranking[:3]
        TOP3_CLR = ["#0072B2", "#D55E00", "#009E73"]
        TOP3_MRK = ["o", "s", "^"]

        t3_ber = {}
        t3_col = {}
        t3_mrk = {}

        for k, v in baselines.items():
            label = k.upper()
            t3_ber[label] = np.asarray(v["ber_mean"])
            t3_col[label] = "#888888" if label == "AF" else "#333333"
            t3_mrk[label] = "<" if label == "AF" else ">"

        for i, vn in enumerate(top3):
            t3_ber[vn] = np.asarray(variants[vn]["ber_mean"])
            t3_col[vn] = TOP3_CLR[i]
            t3_mrk[vn] = TOP3_MRK[i]

        all_t3 = np.concatenate(list(t3_ber.values()))
        min_t3 = all_t3[all_t3 > 0].min() if np.any(all_t3 > 0) else 1e-4

        fig, ax = plt.subplots(figsize=(10, 6))
        for label, ber in t3_ber.items():
            ls = ":" if label in ("AF", "DF") else "-"
            ax.semilogy(snr, ber, color=t3_col[label], marker=t3_mrk[label],
                        markersize=6, linewidth=1.5, linestyle=ls, label=label)
            if label in variants and "ber_ci95_lo" in variants[label]:
                ci_lo = np.maximum(np.asarray(variants[label]["ber_ci95_lo"]), 1e-8)
                ci_hi = np.asarray(variants[label]["ber_ci95_hi"])
                ax.fill_between(snr, ci_lo, ci_hi, color=t3_col[label], alpha=0.12)

        ber_axes_style(ax, min_t3,
                       "Top-3 Neural Relays vs Classical — 16-QAM Rayleigh")
        add_leader_lines(ax, snr, t3_ber, t3_col, t3_mrk)
        legend_below(ax, fig, ncols=len(t3_ber))

        shutil.copy2(top3_png, top3_png.replace(".png", "_orig.png"))
        fig.savefig(top3_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Fixed (CSI top3): {top3_png}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN — run all fixers
# ═══════════════════════════════════════════════════════════════════════════════

JOBS = [
    # (json_path, out_png, title, ncols)
    (
        "results/qam16_activation/qam16_activation_qam16.json",
        "results/qam16_activation/qam16_activation_awgn.png",
        "16-QAM Activation Experiment — AWGN", None,
    ),
    (
        "results/qam16_activation/qam16_activation_qam16.json",
        "results/qam16_activation/qam16_activation_rayleigh.png",
        "16-QAM Activation Experiment — Rayleigh Fading", None,
    ),
    (
        "results/activation_comparison/qam16_activation_qam16.json",
        "results/activation_comparison/qam16_activation_awgn.png",
        "QAM16 Activation Comparison — AWGN", None,
    ),
    (
        "results/activation_comparison/qam16_activation_qam16.json",
        "results/activation_comparison/qam16_activation_rayleigh.png",
        "QAM16 Activation Comparison — Rayleigh Fading", None,
    ),
]

# Find actual JSON files
def find_json(directory):
    found = []
    for f in os.listdir(directory):
        if f.endswith(".json"):
            found.append(os.path.join(directory, f))
    return found


print("=" * 60)
print("  Fixing QAM16 figure legends")
print("=" * 60)

# Fix modulation comparison figures
fix_modulation_qam16()

# Fix activation experiment figures
for json_dir, png_path, title, ncols in [
    ("results/qam16_activation",
     "results/qam16_activation/qam16_activation_awgn.png",
     "16-QAM Activation Experiment — AWGN", None),
    ("results/qam16_activation",
     "results/qam16_activation/qam16_activation_rayleigh.png",
     "16-QAM Activation Experiment — Rayleigh Fading", None),
    ("results/activation_comparison",
     "results/activation_comparison/qam16_activation_awgn.png",
     "QAM16 Activation Comparison — AWGN", None),
    ("results/activation_comparison",
     "results/activation_comparison/qam16_activation_rayleigh.png",
     "QAM16 Activation Comparison — Rayleigh Fading", None),
]:
    jsons = find_json(json_dir)
    if not jsons:
        print(f"  SKIP (no JSON in {json_dir})")
        continue
    # Use the first JSON found (most experiments save one per directory)
    fix_from_json(jsons[0], png_path, title, ncols)

# Fix CSI figures
fix_csi_qam16()

print("\nDone.")
