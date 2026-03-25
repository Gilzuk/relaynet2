"""
16-Class QAM16 — All Relay Architectures Comparison
=====================================================
Tests classify_2d=True (16-class 2D) on every trainable relay:
  1. MLP (MinimalGenAIRelay)
  2. VAE (VAERelay)
  3. CGAN (CGANRelay)
  4. Hybrid (HybridRelay)
  5. Transformer (TransformerRelayWrapper)
  6. Mamba S6 (MambaRelayWrapper)
  7. Mamba2 SSD (Mamba2RelayWrapper)

Each with:
  - 4-class per-axis baseline (classify=True)
  - 16-class 2D (classify_2d=True)

Plus AF & DF baselines.
"""

import sys, os, json, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from relaynet.relays.genai import MinimalGenAIRelay
from relaynet.relays.vae import VAERelay
from relaynet.relays.cgan import CGANRelay
from relaynet.relays.hybrid import HybridRelay
from relaynet.relays.af import AmplifyAndForwardRelay
from relaynet.relays.df import DecodeAndForwardRelay
from relaynet.simulation.runner import run_monte_carlo

from checkpoints.checkpoint_18_transformer_relay import TransformerRelayWrapper
from checkpoints.checkpoint_20_mamba_s6_relay import MambaRelayWrapper
from checkpoints.checkpoint_23_mamba2_relay import Mamba2RelayWrapper

OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "all_relays_16class")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Config ────────────────────────────────────────────────────────
MODULATION     = "qam16"
TRAINING_SNRS  = [5, 10, 15, 20, 25]
SEED           = 42
EVAL_SNRS      = list(range(0, 21, 2))
MC_BITS        = 10_000
MC_TRIALS      = 10


# ═══════════════════════════════════════════════════════════════════
#  VARIANT DEFINITIONS
# ═══════════════════════════════════════════════════════════════════
#  (name, relay_class, kwargs_override, train_kwargs_override)
VARIANTS = [
    # ── MLP ──
    ("MLP 4-cls", MinimalGenAIRelay, dict(
        window_size=5, hidden_size=24, classify=True, training_modulation=MODULATION,
        prefer_gpu=False,
    ), dict(num_samples=50000, epochs=100, training_modulation=MODULATION)),
    ("MLP 16-cls", MinimalGenAIRelay, dict(
        window_size=1, hidden_size=24, classify_2d=True, training_modulation=MODULATION,
        prefer_gpu=False,
    ), dict(num_samples=50000, epochs=200, training_modulation=MODULATION)),

    # ── VAE ──
    ("VAE 4-cls", VAERelay, dict(
        window_size=7, latent_size=8, beta=0.1, hidden_sizes=(32, 16),
        classify=True, training_modulation=MODULATION, prefer_gpu=False,
    ), dict(num_samples=50000, epochs=100, training_modulation=MODULATION)),
    ("VAE 16-cls", VAERelay, dict(
        window_size=1, latent_size=8, beta=0.1, hidden_sizes=(32, 16),
        classify_2d=True, training_modulation=MODULATION, prefer_gpu=False,
    ), dict(num_samples=50000, epochs=200, training_modulation=MODULATION)),

    # ── CGAN ──
    ("CGAN 4-cls", CGANRelay, dict(
        window_size=7, noise_size=8, lambda_gp=10, lambda_l1=20, n_critic=3,
        g_hidden_sizes=(32, 16), c_hidden_sizes=(32, 16),
        classify=True, training_modulation=MODULATION, prefer_gpu=False,
    ), dict(num_samples=20000, epochs=80, training_modulation=MODULATION)),
    ("CGAN 16-cls", CGANRelay, dict(
        window_size=1, noise_size=8, lambda_gp=10, lambda_l1=20, n_critic=3,
        g_hidden_sizes=(32, 16), c_hidden_sizes=(32, 16),
        classify_2d=True, training_modulation=MODULATION, prefer_gpu=False,
    ), dict(num_samples=20000, epochs=80, training_modulation=MODULATION)),

    # ── Hybrid ──
    ("Hybrid 4-cls", HybridRelay, dict(
        snr_threshold=5.0, mlp_window_size=5, mlp_hidden_size=24,
        classify=True, training_modulation=MODULATION, prefer_gpu=False,
    ), dict(num_samples=30000, epochs=100, training_modulation=MODULATION)),
    ("Hybrid 16-cls", HybridRelay, dict(
        snr_threshold=5.0, mlp_window_size=1, mlp_hidden_size=24,
        classify_2d=True, training_modulation=MODULATION, prefer_gpu=False,
    ), dict(num_samples=30000, epochs=200, training_modulation=MODULATION)),

    # ── Transformer ──
    ("Transformer 4-cls", TransformerRelayWrapper, dict(
        window_size=5, d_model=32, num_layers=2,
        classify=True, training_modulation=MODULATION, prefer_gpu=False,
    ), dict(num_samples=50000, epochs=100, training_modulation=MODULATION)),
    ("Transformer 16-cls", TransformerRelayWrapper, dict(
        window_size=5, d_model=32, num_layers=2,
        classify_2d=True, training_modulation=MODULATION, prefer_gpu=False,
    ), dict(num_samples=50000, epochs=200, training_modulation=MODULATION)),

    # ── Mamba S6 ──
    ("Mamba-S6 4-cls", MambaRelayWrapper, dict(
        window_size=5, d_model=32, d_state=16, num_layers=2,
        classify=True, training_modulation=MODULATION, prefer_gpu=False,
    ), dict(num_samples=50000, epochs=100, training_modulation=MODULATION)),
    ("Mamba-S6 16-cls", MambaRelayWrapper, dict(
        window_size=5, d_model=32, d_state=16, num_layers=2,
        classify_2d=True, training_modulation=MODULATION, prefer_gpu=False,
    ), dict(num_samples=50000, epochs=200, training_modulation=MODULATION)),

    # ── Mamba2 SSD ──
    ("Mamba2 4-cls", Mamba2RelayWrapper, dict(
        window_size=5, d_model=32, d_state=16, num_layers=2,
        classify=True, training_modulation=MODULATION, prefer_gpu=False,
    ), dict(num_samples=50000, epochs=100, training_modulation=MODULATION)),
    ("Mamba2 16-cls", Mamba2RelayWrapper, dict(
        window_size=5, d_model=32, d_state=16, num_layers=2,
        classify_2d=True, training_modulation=MODULATION, prefer_gpu=False,
    ), dict(num_samples=50000, epochs=200, training_modulation=MODULATION)),
]


# ═══════════════════════════════════════════════════════════════════
#  TRAIN + EVALUATE
# ═══════════════════════════════════════════════════════════════════
trained = {}

for vname, relay_cls, rkw, tkw in VARIANTS:
    np.random.seed(SEED)
    print(f"\n{'=' * 60}")
    print(f"  {vname}")
    print(f"{'=' * 60}")

    relay = relay_cls(**rkw)
    print(f"  Params: {relay.num_params}")
    print(f"  Training ...")

    relay.train(training_snrs=TRAINING_SNRS, seed=SEED, **tkw)

    print(f"  Evaluating BER ...")
    snrs, bers, trials = run_monte_carlo(
        relay, EVAL_SNRS, MC_BITS, MC_TRIALS, modulation=MODULATION)

    lo = np.percentile(trials, 2.5, axis=1)
    hi = np.percentile(trials, 97.5, axis=1)

    trained[vname] = {
        "relay": relay,
        "params": relay.num_params,
        "ber_mean": bers,
        "ber_trials": trials,
        "ci95_lo": lo,
        "ci95_hi": hi,
    }
    print(f"  BER@20dB = {bers[-1]:.6f}  avg = {np.mean(bers):.6f}")


# Baselines
print(f"\n  Computing AF & DF baselines ...")
_, bers_af, _ = run_monte_carlo(
    AmplifyAndForwardRelay(), EVAL_SNRS, MC_BITS, MC_TRIALS, modulation=MODULATION)
_, bers_df, _ = run_monte_carlo(
    DecodeAndForwardRelay(), EVAL_SNRS, MC_BITS, MC_TRIALS, modulation=MODULATION)
print(f"  AF BER@20dB = {bers_af[-1]:.6f}")
print(f"  DF BER@20dB = {bers_df[-1]:.6f}")


# ═══════════════════════════════════════════════════════════════════
#  JSON
# ═══════════════════════════════════════════════════════════════════
vnames = [v[0] for v in VARIANTS]
results_json = {
    "timestamp": datetime.datetime.now().isoformat(),
    "modulation": MODULATION,
    "experiment": "all_relays_16class_vs_4class",
    "config": {
        "training_snrs": TRAINING_SNRS, "seed": SEED,
        "mc_bits": MC_BITS, "mc_trials": MC_TRIALS,
    },
    "variants": {},
    "baselines": {
        "af": {"ber_mean": [float(b) for b in bers_af]},
        "df": {"ber_mean": [float(b) for b in bers_df]},
    },
    "snr": EVAL_SNRS,
}
for vn in vnames:
    t = trained[vn]
    results_json["variants"][vn] = {
        "params": t["params"],
        "ber_mean": [float(b) for b in t["ber_mean"]],
        "ber_ci95_lo": [float(b) for b in t["ci95_lo"]],
        "ber_ci95_hi": [float(b) for b in t["ci95_hi"]],
    }
json_path = os.path.join(OUT_DIR, "all_relays_16class.json")
with open(json_path, "w") as f:
    json.dump(results_json, f, indent=2)
print(f"\n  JSON -> {json_path}")


# ═══════════════════════════════════════════════════════════════════
#  CHART STYLE
# ═══════════════════════════════════════════════════════════════════
# Colorblind-friendly palette
COLORS = {
    "MLP 4-cls":           "#D55E00",
    "MLP 16-cls":          "#E69F00",
    "VAE 4-cls":           "#56B4E9",
    "VAE 16-cls":          "#0072B2",
    "CGAN 4-cls":          "#CC79A7",
    "CGAN 16-cls":         "#882255",
    "Hybrid 4-cls":        "#009E73",
    "Hybrid 16-cls":       "#44AA99",
    "Transformer 4-cls":   "#F0E442",
    "Transformer 16-cls":  "#B8860B",
    "Mamba-S6 4-cls":      "#7570B3",
    "Mamba-S6 16-cls":     "#4B0082",
    "Mamba2 4-cls":        "#E7298A",
    "Mamba2 16-cls":       "#A50026",
    "AF":                  "#888888",
    "DF":                  "#333333",
}
MARKERS = {
    "MLP 4-cls":           "o",
    "MLP 16-cls":          "s",
    "VAE 4-cls":           "^",
    "VAE 16-cls":          "v",
    "CGAN 4-cls":          "D",
    "CGAN 16-cls":         "d",
    "Hybrid 4-cls":        "p",
    "Hybrid 16-cls":       "h",
    "Transformer 4-cls":   "P",
    "Transformer 16-cls":  "X",
    "Mamba-S6 4-cls":      "*",
    "Mamba-S6 16-cls":     "H",
    "Mamba2 4-cls":        "8",
    "Mamba2 16-cls":       "1",
    "AF":                  "<",
    "DF":                  ">",
}
LSTYLES = {
    "MLP 4-cls":           "--",
    "MLP 16-cls":          "-",
    "VAE 4-cls":           "--",
    "VAE 16-cls":          "-",
    "CGAN 4-cls":          "--",
    "CGAN 16-cls":         "-",
    "Hybrid 4-cls":        "--",
    "Hybrid 16-cls":       "-",
    "Transformer 4-cls":   "--",
    "Transformer 16-cls":  "-",
    "Mamba-S6 4-cls":      "--",
    "Mamba-S6 16-cls":     "-",
    "Mamba2 4-cls":        "--",
    "Mamba2 16-cls":       "-",
    "AF":                  ":",
    "DF":                  ":",
}


# ═══════════════════════════════════════════════════════════════════
#  CHART 1 — BER vs SNR (all variants)
# ═══════════════════════════════════════════════════════════════════
print("  Generating BER chart ...")
fig, ax = plt.subplots(figsize=(10, 6))

ax.semilogy(EVAL_SNRS, bers_af, color=COLORS["AF"], marker="<",
            markersize=5, linewidth=1.0, linestyle=":", label="AF baseline")
ax.semilogy(EVAL_SNRS, bers_df, color=COLORS["DF"], marker=">",
            markersize=5, linewidth=1.0, linestyle=":", label="DF baseline")

for vn in vnames:
    t = trained[vn]
    ax.semilogy(EVAL_SNRS, t["ber_mean"], color=COLORS[vn], marker=MARKERS[vn],
                markersize=5, linewidth=1.3, linestyle=LSTYLES[vn],
                label=f"{vn} ({t['params']}p)")
    ax.fill_between(EVAL_SNRS, t["ci95_lo"], t["ci95_hi"],
                    color=COLORS[vn], alpha=0.06)

all_bers = np.concatenate([t["ber_mean"] for t in trained.values()] + [bers_af, bers_df])
min_ber = all_bers[all_bers > 0].min() if np.any(all_bers > 0) else 1e-4
ax.set_ylim(bottom=min_ber / 10, top=0.55)
ax.yaxis.set_minor_locator(LogLocator(subs="all", numticks=20))
ax.yaxis.set_minor_formatter(NullFormatter())
ax.grid(True, which="major", linewidth=0.5, alpha=0.4)
ax.grid(True, which="minor", linewidth=0.3, alpha=0.15)
ax.set_xlabel("SNR (dB)", fontsize=14)
ax.set_ylabel("Bit Error Rate", fontsize=14)
ax.set_title(f"All Relays — 16-Class vs 4-Class, {MODULATION.upper()}", fontsize=16)
ax.tick_params(labelsize=12)
ax.legend(fontsize=7, loc="upper right", framealpha=0.9, ncol=3)

# Inset zoom at high SNR
hi_snrs = [s for s in EVAL_SNRS if s >= 12]
if len(hi_snrs) >= 3:
    axins = inset_axes(ax, width="38%", height="32%", loc="center left",
                       bbox_to_anchor=(0.02, 0.02, 1, 1),
                       bbox_transform=ax.transAxes)
    for vn in vnames:
        hi_idx = [EVAL_SNRS.index(s) for s in hi_snrs]
        vals = trained[vn]["ber_mean"][hi_idx[0]:hi_idx[-1] + 1]
        axins.semilogy(hi_snrs, vals, color=COLORS[vn], marker=MARKERS[vn],
                       markersize=3, linewidth=1.0, linestyle=LSTYLES[vn])
    axins.semilogy(hi_snrs, bers_df[[EVAL_SNRS.index(s) for s in hi_snrs]],
                   color=COLORS["DF"], marker=">", markersize=3, linewidth=0.8, linestyle=":")
    axins.grid(True, which="both", linewidth=0.3, alpha=0.3)
    axins.set_title("High-SNR zoom", fontsize=9)
    axins.tick_params(labelsize=8)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", lw=0.6)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "ber_all_relays_16class.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  BER chart saved.")


# ═══════════════════════════════════════════════════════════════════
#  CHART 2 — Grouped bar chart: BER@20dB  (4-cls vs 16-cls per arch)
# ═══════════════════════════════════════════════════════════════════
print("  Generating grouped bar chart ...")
architectures = ["MLP", "VAE", "CGAN", "Hybrid", "Transformer", "Mamba-S6", "Mamba2"]
snr_20_idx = EVAL_SNRS.index(20)

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(architectures))
width = 0.3

ber_4cls = [trained[f"{a} 4-cls"]["ber_mean"][snr_20_idx] for a in architectures]
ber_16cls = [trained[f"{a} 16-cls"]["ber_mean"][snr_20_idx] for a in architectures]

bars1 = ax.bar(x - width / 2, ber_4cls, width,
               label="4-class per-axis", color="#D55E00", edgecolor="black", linewidth=0.5, alpha=0.8)
bars2 = ax.bar(x + width / 2, ber_16cls, width,
               label="16-class 2D", color="#0072B2", edgecolor="black", linewidth=0.5, alpha=0.8)

ax.axhline(y=bers_df[snr_20_idx], color=COLORS["DF"], linewidth=1.5, linestyle="--",
           label=f"DF = {bers_df[snr_20_idx]:.5f}")

for bars, vals in [(bars1, ber_4cls), (bars2, ber_16cls)]:
    for b, v in zip(bars, vals):
        txt = f"{v:.4f}" if v >= 0.0001 else f"{v:.1e}"
        ax.text(b.get_x() + b.get_width() / 2, v + max(max(ber_4cls), max(ber_16cls)) * 0.03,
                txt, ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(architectures, fontsize=12)
ax.set_ylabel("BER @ 20 dB", fontsize=14)
ax.set_title(f"BER @ 20 dB — 4-Class vs 16-Class per Architecture, {MODULATION.upper()}", fontsize=14)
ax.legend(fontsize=11, loc="upper right")
ax.grid(True, axis="y", linewidth=0.4, alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "grouped_bar_16class.png"), dpi=150)
plt.close(fig)
print(f"  Grouped bar chart saved.")


# ═══════════════════════════════════════════════════════════════════
#  CHART 3 — Heatmap: BER for each variant × SNR
# ═══════════════════════════════════════════════════════════════════
print("  Generating heatmap ...")
ber_matrix = np.array([trained[vn]["ber_mean"] for vn in vnames])
fig, ax = plt.subplots(figsize=(12, 5))
im = ax.imshow(ber_matrix, aspect="auto", cmap="RdYlGn_r")
ax.set_xticks(range(len(EVAL_SNRS)))
ax.set_xticklabels([str(s) for s in EVAL_SNRS], fontsize=11)
ax.set_yticks(range(len(vnames)))
ax.set_yticklabels(vnames, fontsize=9)
ax.set_xlabel("SNR (dB)", fontsize=14)
ax.set_title(f"BER Heatmap — All Relays 4-cls vs 16-cls, {MODULATION.upper()}", fontsize=15)
for i in range(len(vnames)):
    for j in range(len(EVAL_SNRS)):
        v = ber_matrix[i, j]
        txt = f"{v:.3f}" if v >= 0.01 else f"{v:.4f}"
        clr = "white" if v > 0.25 else "black"
        ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=clr)
fig.colorbar(im, ax=ax, label="BER", shrink=0.8)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "heatmap_all_relays_16class.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Heatmap saved.")


# ═══════════════════════════════════════════════════════════════════
#  CHART 4 — Top-3 chart (best 16-class variants vs baselines)
# ═══════════════════════════════════════════════════════════════════
print("  Generating top-3 chart ...")
# Rank by average BER in upper-half SNR range
upper_half = [i for i, s in enumerate(EVAL_SNRS) if s >= EVAL_SNRS[-1] / 2]
ranking = []
for vn in vnames:
    avg_hi = np.mean(trained[vn]["ber_mean"][upper_half[0]:upper_half[-1] + 1])
    ranking.append((avg_hi, vn))
ranking.sort()
top3 = [vn for _, vn in ranking[:3]]

fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(EVAL_SNRS, bers_af, color="#AAAAAA", marker="<", markersize=5,
            linewidth=1.0, linestyle=":", label="AF baseline")
ax.semilogy(EVAL_SNRS, bers_df, color="#333333", marker=">", markersize=5,
            linewidth=1.0, linestyle=":", label="DF baseline")
top3_colors = ["#E69F00", "#0072B2", "#009E73"]
top3_markers = ["s", "v", "h"]
for i, vn in enumerate(top3):
    t = trained[vn]
    ax.semilogy(EVAL_SNRS, t["ber_mean"], color=top3_colors[i], marker=top3_markers[i],
                markersize=6, linewidth=1.5, linestyle="-",
                label=f"#{i+1} {vn} ({t['params']}p)")
    ax.fill_between(EVAL_SNRS, t["ci95_lo"], t["ci95_hi"],
                    color=top3_colors[i], alpha=0.10)

ax.set_ylim(bottom=min_ber / 10, top=0.55)
ax.yaxis.set_minor_locator(LogLocator(subs="all", numticks=20))
ax.yaxis.set_minor_formatter(NullFormatter())
ax.grid(True, which="major", linewidth=0.5, alpha=0.4)
ax.grid(True, which="minor", linewidth=0.3, alpha=0.15)
ax.set_xlabel("SNR (dB)", fontsize=14)
ax.set_ylabel("Bit Error Rate", fontsize=14)
ax.set_title(f"Top-3 Performers — 16-Class QAM16", fontsize=16)
ax.tick_params(labelsize=12)
ax.legend(fontsize=10, loc="upper right", framealpha=0.9)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "top3_16class.png"), dpi=150)
plt.close(fig)
print(f"  Top-3 chart saved.")


# ═══════════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  BER SUMMARY TABLE")
print("=" * 70)
key_snrs = [4, 10, 16, 20]
header = f"  {'Variant':20s}" + "".join(f"  {s:6d}dB" for s in key_snrs) + f"  {'AvgBER':>10s}  {'Params':>6s}"
print(header)
print("  " + "-" * (len(header) - 2))
for vn in vnames:
    t = trained[vn]
    row = f"  {vn:20s}"
    for s in key_snrs:
        if s in EVAL_SNRS:
            row += f"  {t['ber_mean'][EVAL_SNRS.index(s)]:8.5f}"
    row += f"  {np.mean(t['ber_mean']):10.6f}  {t['params']:6d}"
    print(row)
for bname, bbers in [("AF", bers_af), ("DF", bers_df)]:
    row = f"  {bname:20s}"
    for s in key_snrs:
        if s in EVAL_SNRS:
            row += f"  {bbers[EVAL_SNRS.index(s)]:8.5f}"
    row += f"  {np.mean(bbers):10.6f}  {'---':>6s}"
    print(row)

print("\n  GAP vs DF @ 20 dB:")
df_20 = bers_df[snr_20_idx]
for vn in vnames:
    mlp_20 = trained[vn]["ber_mean"][snr_20_idx]
    gap = mlp_20 - df_20
    print(f"    {vn:20s}  BER={mlp_20:.5f}  gap={gap:+.5f}")

print(f"\n  Top-3: {', '.join(top3)}")
print(f"\n  All outputs in: {OUT_DIR}")
