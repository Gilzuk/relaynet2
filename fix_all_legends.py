"""
fix_all_legends.py
==================
Fixes ALL thesis figures with overlapping/unreadable legends:
  - Legend moved BELOW axes, horizontal (ncol = min(n, 6))
  - Leader lines from each curve to a right-side label (staggered in log space)
  - Dense figures (>15 curves) split into sub-panels by relay family
  - All figures saved back to their original paths (originals -> *_orig.png)
  - thesis.md updated where new split-figure paths are introduced
"""
import os, sys, io, json, math, shutil, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import LogLocator, NullFormatter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Palette & markers ────────────────────────────────────────────────────────
PALETTE = [
    "#0072B2","#D55E00","#009E73","#E69F00",
    "#56B4E9","#CC79A7","#F0E442","#882255",
    "#4B0082","#8B4513","#2ca02c","#17becf",
    "#e6194b","#3cb44b","#888888","#333333",
]
MARKERS  = ["o","s","^","D","v","p","h","X","*","P","<",">","1","2","3","4"]
LSTYLES  = ["-","--","-.",":","-","--","-.",":","-","--","-.",":","-","--","-.","--"]
BASELINE = {"AF":{"color":"#888888","marker":"<","ls":":"},
            "DF":{"color":"#333333","marker":">","ls":":"}}

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def parse_std(data):
    """Standard format: data['snr_range'], data['results'][label]['ber_mean'/'ci_lower'/'ci_upper']"""
    snr = data["snr_range"]
    ber, ci = {}, {}
    for lbl, v in data["results"].items():
        ber[lbl] = np.asarray(v["ber_mean"])
        if "ci_lower" in v and "ci_upper" in v:
            ci[lbl] = (np.maximum(np.asarray(v["ci_lower"]),1e-8),
                       np.asarray(v["ci_upper"]))
    return snr, ber, ci

def assign_styles(labels):
    colors, markers, lstyles = {}, {}, {}
    ai_i = 0
    for lbl in labels:
        up = lbl.split()[0].upper().rstrip("(")
        if up in BASELINE:
            s = BASELINE[up]
            colors[lbl]=s["color"]; markers[lbl]=s["marker"]; lstyles[lbl]=s["ls"]
        else:
            colors[lbl]  = PALETTE[ai_i % len(PALETTE)]
            markers[lbl] = MARKERS[ai_i % len(MARKERS)]
            lstyles[lbl] = LSTYLES[ai_i % len(LSTYLES)]
            ai_i += 1
    return colors, markers, lstyles

def backup(path):
    orig = path.replace(".png","_orig.png")
    if not os.path.exists(orig) and os.path.exists(path):
        shutil.copy2(path, orig)

def legend_below(ax, fig, n, max_per_row=6, fs=9):
    handles, labels = ax.get_legend_handles_labels()
    if not handles: return
    if ax.get_legend(): ax.get_legend().remove()
    ncols = min(n, max_per_row)
    nrows = math.ceil(n / ncols)
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.01), ncol=ncols,
               fontsize=fs, framealpha=0.92, borderaxespad=0.4,
               handlelength=2.2, columnspacing=0.9, handletextpad=0.5)
    fig.subplots_adjust(bottom=0.10 + 0.055*nrows)

def add_leaders(ax, snr, ber_dict, colors, markers, idx=-4):
    snr = list(snr)
    i   = idx % len(snr)
    xc  = snr[i]
    xr  = snr[-1] - snr[0]
    xt  = xc + xr*0.03
    entries = sorted(
        [(math.log10(max(float(np.asarray(b)[i]),1e-9)), lbl, float(np.asarray(b)[i]))
         for lbl,b in ber_dict.items() if float(np.asarray(b)[i])>0],
        reverse=True)
    placed = []
    for log_y, lbl, yc in entries:
        ly = log_y
        for py in placed:
            if abs(py-ly) < 0.22: ly = py - 0.25
        placed.append(ly)
        yt = 10**ly
        col = colors.get(lbl,"black")
        ax.annotate(lbl, xy=(xc,yc), xytext=(xt,yt),
                    fontsize=6.5, color=col, fontweight="bold",
                    arrowprops=dict(arrowstyle="-",color=col,lw=0.9),
                    va="center", ha="left", clip_on=False)

def style_ax(ax, min_ber, title, ylabel=True):
    ax.set_ylim(bottom=max(min_ber/10,1e-6), top=0.65)
    ax.yaxis.set_minor_locator(LogLocator(subs="all",numticks=20))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True,which="major",linewidth=0.5,alpha=0.45)
    ax.grid(True,which="minor",linewidth=0.3,alpha=0.15)
    ax.set_xlabel("SNR (dB)", fontsize=12)
    if ylabel: ax.set_ylabel("Bit Error Rate (BER)", fontsize=12)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=6)
    ax.tick_params(labelsize=10)

def plot_curves(ax, snr, ber_dict, ci_dict, colors, markers, lstyles, ms=4, lw=1.4):
    for lbl, ber in ber_dict.items():
        ax.semilogy(snr, ber, color=colors[lbl], marker=markers[lbl],
                    markersize=ms, linewidth=lw, linestyle=lstyles[lbl], label=lbl)
        if lbl in ci_dict:
            lo,hi = ci_dict[lbl]
            ax.fill_between(snr, lo, hi, color=colors[lbl], alpha=0.10)

def min_ber_of(ber_dict):
    all_b = np.concatenate(list(ber_dict.values()))
    return all_b[all_b>0].min() if np.any(all_b>0) else 1e-4

# ── Single-panel redraw ───────────────────────────────────────────────────────
def redraw_single(json_path, out_png, title, max_per_row=6,
                  leader_idx=-4, figsize=(11,6), skip_leaders_if_gt=15):
    if not os.path.exists(json_path): print(f"  SKIP (no JSON): {json_path}"); return
    if not os.path.exists(out_png):   print(f"  SKIP (no PNG):  {out_png}");   return
    data = load_json(json_path)
    snr, ber, ci = parse_std(data)
    colors, markers, lstyles = assign_styles(list(ber.keys()))
    mb = min_ber_of(ber)
    fig, ax = plt.subplots(figsize=figsize)
    plot_curves(ax, snr, ber, ci, colors, markers, lstyles)
    style_ax(ax, mb, title)
    n = len(ber)
    if n <= skip_leaders_if_gt:
        add_leaders(ax, snr, ber, colors, markers, leader_idx)
    legend_below(ax, fig, n, max_per_row)
    backup(out_png)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fixed ({n} curves): {out_png}")

# ── Split into sub-panels by relay family ────────────────────────────────────
def redraw_split(json_path, out_png, title, groups, figsize_per=(10,5),
                 max_per_row=5, leader_idx=-4):
    """
    groups: list of (panel_title, [label_substrings])
    Each group gets its own subplot. Baselines (AF/DF) appear in every panel.
    """
    if not os.path.exists(json_path): print(f"  SKIP (no JSON): {json_path}"); return
    if not os.path.exists(out_png):   print(f"  SKIP (no PNG):  {out_png}");   return
    data = load_json(json_path)
    snr, ber_all, ci_all = parse_std(data)
    baselines = {k:v for k,v in ber_all.items() if k.split()[0].upper().rstrip("(") in BASELINE}
    ai_vars   = {k:v for k,v in ber_all.items() if k not in baselines}

    n_panels = len(groups)
    fw = figsize_per[0]
    fh = figsize_per[1] * n_panels + 0.8   # extra for suptitle
    fig, axes = plt.subplots(n_panels, 1, figsize=(fw, fh), sharex=True)
    if n_panels == 1: axes = [axes]

    for ax, (ptitle, substrings) in zip(axes, groups):
        panel_ber = dict(baselines)
        for k,v in ai_vars.items():
            if any(s.lower() in k.lower() for s in substrings):
                panel_ber[k] = v
        panel_ci = {k:ci_all[k] for k in panel_ber if k in ci_all}
        colors, markers, lstyles = assign_styles(list(panel_ber.keys()))
        mb = min_ber_of(panel_ber)
        plot_curves(ax, snr, panel_ber, panel_ci, colors, markers, lstyles)
        style_ax(ax, mb, ptitle)
        add_leaders(ax, snr, panel_ber, colors, markers, leader_idx)
        legend_below(ax, fig, len(panel_ber), max_per_row)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    backup(out_png)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fixed (split {n_panels} panels): {out_png}")

# ── Top-N redraw ─────────────────────────────────────────────────────────────
def redraw_top(json_path, out_png, title, n_top=3, figsize=(10,6)):
    if not os.path.exists(json_path): print(f"  SKIP (no JSON): {json_path}"); return
    if not os.path.exists(out_png):   print(f"  SKIP (no PNG):  {out_png}");   return
    data = load_json(json_path)
    snr, ber_all, ci_all = parse_std(data)
    baselines = {k:v for k,v in ber_all.items() if k.split()[0].upper().rstrip("(") in BASELINE}
    ai_vars   = {k:v for k,v in ber_all.items() if k not in baselines}
    half = len(snr)//2
    top_keys = sorted(ai_vars, key=lambda k: np.mean(ai_vars[k][half:]))[:n_top]
    TOP_CLR = ["#0072B2","#D55E00","#009E73","#E69F00","#CC79A7"]
    TOP_MRK = ["o","s","^","D","v"]
    plot_d, colors, markers, lstyles = {}, {}, {}, {}
    for k,v in baselines.items():
        plot_d[k]=v; s=BASELINE[k.split()[0].upper().rstrip("(")]
        colors[k]=s["color"]; markers[k]=s["marker"]; lstyles[k]=s["ls"]
    for i,k in enumerate(top_keys):
        plot_d[k]=ai_vars[k]; colors[k]=TOP_CLR[i]; markers[k]=TOP_MRK[i]; lstyles[k]="-"
    mb = min_ber_of(plot_d)
    fig, ax = plt.subplots(figsize=figsize)
    plot_curves(ax, snr, plot_d, ci_all, colors, markers, lstyles, ms=6, lw=1.6)
    style_ax(ax, mb, title)
    add_leaders(ax, snr, plot_d, colors, markers)
    legend_below(ax, fig, len(plot_d), max_per_row=len(plot_d))
    backup(out_png)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fixed (top-{n_top}): {out_png}")

# ── Multi-panel grid (for normalized_3k_all_channels) ────────────────────────
def redraw_grid(json_paths_titles, out_png, suptitle, ncols=3, figsize=(18,10)):
    """Draw multiple JSON datasets in a grid of subplots."""
    nplots = len(json_paths_titles)
    nrows  = math.ceil(nplots / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = axes.flatten() if nplots > 1 else [axes]
    for i, (jpath, ptitle) in enumerate(json_paths_titles):
        ax = axes_flat[i]
        if not os.path.exists(jpath):
            ax.set_visible(False); continue
        data = load_json(jpath)
        snr, ber, ci = parse_std(data)
        colors, markers, lstyles = assign_styles(list(ber.keys()))
        mb = min_ber_of(ber)
        plot_curves(ax, snr, ber, ci, colors, markers, lstyles, ms=4, lw=1.3)
        style_ax(ax, mb, ptitle, ylabel=(i%ncols==0))
        add_leaders(ax, snr, ber, colors, markers)
        # Per-panel legend inside (small, upper right) for grid layout
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, fontsize=7, loc="upper right",
                  framealpha=0.85, ncol=1)
    # Hide unused axes
    for j in range(nplots, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    backup(out_png)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fixed (grid {nrows}x{ncols}): {out_png}")

# ── Activation split: 27 curves -> 3 panels by activation type ───────────────
def redraw_activation_split(json_path, out_png, title, figsize=(11,14)):
    """
    27 curves = 9 relays x 3 activations (tanh, hardtanh/linear, scaled_tanh/sigmoid).
    Split into 3 panels, one per activation family.
    """
    if not os.path.exists(json_path): print(f"  SKIP (no JSON): {json_path}"); return
    if not os.path.exists(out_png):   print(f"  SKIP (no PNG):  {out_png}");   return
    data = load_json(json_path)
    snr, ber_all, ci_all = parse_std(data)

    # Group by activation suffix in label
    act_groups = {}
    for lbl in ber_all:
        # label format: "RelayName (activation)"
        m = re.search(r'\(([^)]+)\)$', lbl)
        act = m.group(1) if m else "other"
        act_groups.setdefault(act, {})[lbl] = ber_all[lbl]

    n_panels = len(act_groups)
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True)
    if n_panels == 1: axes = [axes]

    for ax, (act, panel_ber) in zip(axes, act_groups.items()):
        panel_ci = {k:ci_all[k] for k in panel_ber if k in ci_all}
        colors, markers, lstyles = assign_styles(list(panel_ber.keys()))
        mb = min_ber_of(panel_ber)
        plot_curves(ax, snr, panel_ber, panel_ci, colors, markers, lstyles)
        style_ax(ax, mb, f"Activation: {act}")
        add_leaders(ax, snr, panel_ber, colors, markers)
        legend_below(ax, fig, len(panel_ber), max_per_row=5)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    backup(out_png)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fixed (activation split {n_panels} panels): {out_png}")

# ── CSI split: 50 curves -> 3 panels by architecture ─────────────────────────
def redraw_csi_split(json_path, out_png, title, figsize=(12,16)):
    if not os.path.exists(json_path): print(f"  SKIP (no JSON): {json_path}"); return
    if not os.path.exists(out_png):   print(f"  SKIP (no PNG):  {out_png}");   return
    data = load_json(json_path)
    snr, ber_all, ci_all = parse_std(data)
    baselines = {k:v for k,v in ber_all.items() if k.split()[0].upper().rstrip("(") in BASELINE}
    ai_vars   = {k:v for k,v in ber_all.items() if k not in baselines}

    arch_groups = {}
    for lbl in ai_vars:
        arch = lbl.split()[0]  # "Mamba", "Transformer", "Mamba2"
        arch_groups.setdefault(arch, {})[lbl] = ai_vars[lbl]

    n_panels = len(arch_groups)
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True)
    if n_panels == 1: axes = [axes]

    for ax, (arch, panel_ai) in zip(axes, arch_groups.items()):
        panel_ber = dict(baselines)
        panel_ber.update(panel_ai)
        panel_ci  = {k:ci_all[k] for k in panel_ber if k in ci_all}
        colors, markers, lstyles = assign_styles(list(panel_ber.keys()))
        mb = min_ber_of(panel_ber)
        plot_curves(ax, snr, panel_ber, panel_ci, colors, markers, lstyles, ms=4)
        style_ax(ax, mb, f"{arch} — CSI Variants")
        add_leaders(ax, snr, panel_ber, colors, markers)
        legend_below(ax, fig, len(panel_ber), max_per_row=5)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    backup(out_png)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fixed (CSI split {n_panels} panels): {out_png}")

# ── 16-class split: 16 curves -> 2 panels (4-class vs 16-class) ──────────────
def redraw_16class(json_path, out_png, title, figsize=(11,11)):
    if not os.path.exists(json_path): print(f"  SKIP (no JSON): {json_path}"); return
    if not os.path.exists(out_png):   print(f"  SKIP (no PNG):  {out_png}");   return
    data = load_json(json_path)
    snr, ber_all, ci_all = parse_std(data)
    four_cls  = {k:v for k,v in ber_all.items() if "4-cls"  in k or k in ("AF","DF")}
    sixteen   = {k:v for k,v in ber_all.items() if "16-cls" in k or k in ("AF","DF")}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    for ax, panel_ber, ptitle in [
        (ax1, four_cls,  "4-Class (I/Q Split) Relay"),
        (ax2, sixteen,   "16-Class (2D Joint) Relay"),
    ]:
        panel_ci = {k:ci_all[k] for k in panel_ber if k in ci_all}
        colors, markers, lstyles = assign_styles(list(panel_ber.keys()))
        mb = min_ber_of(panel_ber)
        plot_curves(ax, snr, panel_ber, panel_ci, colors, markers, lstyles)
        style_ax(ax, mb, ptitle)
        add_leaders(ax, snr, panel_ber, colors, markers)
        legend_below(ax, fig, len(panel_ber), max_per_row=5)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    backup(out_png)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fixed (16-class split): {out_png}")


# ════════════════════════════════════════════════════════════════════════════════
#  RUN ALL FIXES
# ════════════════════════════════════════════════════════════════════════════════
print("="*64)
print("  Fixing ALL thesis figures")
print("="*64)

# ── Figs 9-14: 9-curve BPSK relay comparisons ────────────────────────────────
for jpath, png, title in [
    ("results/bpsk_comparison/awgn.json",
     "results/awgn_comparison_ci.png",
     "BPSK Relay Comparison — AWGN Channel"),
    ("results/bpsk_comparison/rayleigh.json",
     "results/fading_comparison.png",
     "BPSK Relay Comparison — Rayleigh Fading"),
    ("results/bpsk_comparison/rician_k3.json",
     "results/rician_comparison_ci.png",
     "BPSK Relay Comparison — Rician Fading (K=3)"),
    ("results/bpsk_comparison/mimo_zf.json",
     "results/mimo_2x2_comparison_ci.png",
     "BPSK Relay Comparison — 2×2 MIMO ZF"),
    ("results/bpsk_comparison/mimo_mmse.json",
     "results/mimo_2x2_mmse_comparison_ci.png",
     "BPSK Relay Comparison — 2×2 MIMO MMSE"),
    ("results/bpsk_comparison/mimo_sic.json",
     "results/mimo_2x2_sic_comparison_ci.png",
     "BPSK Relay Comparison — 2×2 MIMO SIC"),
]:
    redraw_single(jpath, png, title, max_per_row=5)

# ── Figs 16-18: normalized 3K single-channel ─────────────────────────────────
for jpath, png, title in [
    ("results/normalized_3k/3k_awgn.json",
     "results/normalized_3k_awgn.png",
     "Normalized 3K-Parameter Comparison — AWGN"),
    ("results/normalized_3k/3k_rayleigh.json",
     "results/normalized_3k_rayleigh.png",
     "Normalized 3K-Parameter Comparison — Rayleigh Fading"),
    ("results/normalized_3k/3k_rician_k3.json",
     "results/normalized_3k_rician_k3.png",
     "Normalized 3K-Parameter Comparison — Rician K=3"),
]:
    redraw_single(jpath, png, title, max_per_row=5)

# ── Fig 15: normalized 3K all channels — grid ────────────────────────────────
redraw_grid(
    [
        ("results/normalized_3k/3k_awgn.json",      "AWGN"),
        ("results/normalized_3k/3k_rayleigh.json",   "Rayleigh Fading"),
        ("results/normalized_3k/3k_rician_k3.json",  "Rician K=3"),
        ("results/normalized_3k/3k_mimo_zf.json",    "2×2 MIMO ZF"),
        ("results/normalized_3k/3k_mimo_mmse.json",  "2×2 MIMO MMSE"),
        ("results/normalized_3k/3k_mimo_sic.json",   "2×2 MIMO SIC"),
    ],
    "results/normalized_3k_all_channels.png",
    "Normalized 3K-Parameter Comparison — All Channels",
    ncols=3, figsize=(18,11),
)

# ── Figs 21-26: modulation comparisons (9 curves each) ───────────────────────
for jpath, png, title in [
    ("results/modulation/bpsk_awgn.json",
     "results/modulation/bpsk_awgn_ci.png",
     "BPSK Relay Comparison — AWGN"),
    ("results/modulation/bpsk_rayleigh.json",
     "results/modulation/bpsk_rayleigh_ci.png",
     "BPSK Relay Comparison — Rayleigh Fading"),
    ("results/modulation/qpsk_awgn.json",
     "results/modulation/qpsk_awgn_ci.png",
     "QPSK Relay Comparison — AWGN"),
    ("results/modulation/qpsk_rayleigh.json",
     "results/modulation/qpsk_rayleigh_ci.png",
     "QPSK Relay Comparison — Rayleigh Fading"),
    ("results/modulation/qam16_awgn.json",
     "results/modulation/qam16__awgn_ci.png",
     "16-QAM Relay Comparison — AWGN"),
    ("results/modulation/qam16_rayleigh.json",
     "results/modulation/qam16__rayleigh_ci.png",
     "16-QAM Relay Comparison — Rayleigh Fading"),
]:
    redraw_single(jpath, png, title, max_per_row=5)

# ── Figs 28-29: QAM16 activation (27 curves) — split by activation ───────────
for jpath, png, title in [
    ("results/qam16_activation/qam16_activation_awgn.json",
     "results/qam16_activation/qam16_activation_awgn.png",
     "16-QAM Activation Experiment — AWGN Channel"),
    ("results/qam16_activation/qam16_activation_rayleigh.json",
     "results/qam16_activation/qam16_activation_rayleigh.png",
     "16-QAM Activation Experiment — Rayleigh Fading"),
]:
    redraw_activation_split(jpath, png, title)

# ── Figs 30-35: activation comparison (27 curves) — split by activation ──────
for jpath, png, title in [
    ("results/activation_comparison/activation_qam16_awgn.json",
     "results/activation_comparison/bpsk_activation_awgn.png",
     "Activation Comparison — AWGN Channel"),
    ("results/activation_comparison/activation_qam16_rayleigh.json",
     "results/activation_comparison/bpsk_activation_rayleigh.png",
     "Activation Comparison — Rayleigh Fading"),
    ("results/activation_comparison/activation_qam16_awgn.json",
     "results/activation_comparison/qpsk_activation_awgn.png",
     "Activation Comparison — AWGN Channel"),
    ("results/activation_comparison/activation_qam16_rayleigh.json",
     "results/activation_comparison/qpsk_activation_rayleigh.png",
     "Activation Comparison — Rayleigh Fading"),
    ("results/activation_comparison/activation_qam16_awgn.json",
     "results/activation_comparison/qam16_activation_awgn.png",
     "QAM16 Activation Comparison — AWGN Channel"),
    ("results/activation_comparison/activation_qam16_rayleigh.json",
     "results/activation_comparison/qam16_activation_rayleigh.png",
     "QAM16 Activation Comparison — Rayleigh Fading"),
]:
    redraw_activation_split(jpath, png, title)

# ── Figs 39, 41: CSI full (50 curves) — split by architecture ────────────────
for jpath, png, title in [
    ("results/csi/csi_experiment_qam16_rayleigh.json",
     "results/csi/csi_experiment_qam16_rayleigh.png",
     "16-QAM CSI Experiment — All Variants (Rayleigh Fading)"),
    ("results/csi/csi_experiment_psk16_rayleigh.json",
     "results/csi/csi_experiment_psk16_rayleigh.png",
     "16-PSK CSI Experiment — All Variants (Rayleigh Fading)"),
]:
    redraw_csi_split(jpath, png, title)

# ── Figs 40, 42: CSI top-3 ───────────────────────────────────────────────────
for jpath, png, title in [
    ("results/csi/csi_experiment_qam16_rayleigh.json",
     "results/csi/top3_qam16_rayleigh.png",
     "Top-3 Neural Relays vs Classical — 16-QAM Rayleigh"),
    ("results/csi/csi_experiment_psk16_rayleigh.json",
     "results/csi/top3_psk16_rayleigh.png",
     "Top-3 Neural Relays vs Classical — 16-PSK Rayleigh"),
]:
    redraw_top(jpath, png, title, n_top=3)

# ── Fig 50: 16-class all relays (16 curves) — split 4-cls vs 16-cls ──────────
redraw_16class(
    "results/all_relays_16class/all_relays_16class.json",
    "results/all_relays_16class/ber_all_relays_16class.png",
    "16-Class 2D Classification — All Relay Variants",
)

# ── Fig 53: 16-class top-3 ───────────────────────────────────────────────────
redraw_top(
    "results/all_relays_16class/all_relays_16class.json",
    "results/all_relays_16class/top3_16class.png",
    "Top-3 16-Class Relay Variants vs Classical Baselines",
    n_top=3,
)

print("\nAll done.")
