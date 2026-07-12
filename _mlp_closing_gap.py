"""
MLP Classification — Closing the DF Gap (QAM16)
=================================================
Systematically applies three improvements:
  1. window_size=1  (eliminate irrelevant neighbour noise)
  2. Training SNR range includes high SNR (5–25 dB)
  3. Deeper/wider network  (64-wide, 2 hidden layers)

Variants (all classification, CrossEntropyLoss):
  A) Baseline:     win=5,  SNR={5,10,15},       24 hidden, 1 layer
  B) Win=1:        win=1,  SNR={5,10,15},       24 hidden, 1 layer
  C) Wide SNR:     win=5,  SNR={5,10,15,20,25}, 24 hidden, 1 layer
  D) Win1+WideSNR: win=1,  SNR={5,10,15,20,25}, 24 hidden, 1 layer
  E) Wider:        win=1,  SNR={5,10,15,20,25}, 64 hidden, 1 layer
  F) Deeper:       win=1,  SNR={5,10,15,20,25}, 64 hidden, 2 layers

• 70/15/15 split, early stopping (patience=15)
• Monte-Carlo BER: 10 trials × 10,000 bits
• Charts + JSON
"""

import sys, os, json, datetime
sys.path.insert(0, ".")
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from relaynet.utils.activations import (
    generate_training_targets, make_torch_activation,
    get_num_classes, get_constellation_levels, symbols_to_class_indices,
    get_clip_range,
)
from relaynet.simulation.runner import run_monte_carlo
from relaynet.relays.genai import MinimalGenAIRelay
from relaynet.relays.af import AmplifyAndForwardRelay
from relaynet.relays.df import DecodeAndForwardRelay

OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "classify_closing_gap")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Shared Config ────────────────────────────────────────────────
MODULATION    = "qam16"
NUM_SAMPLES   = 50_000
MAX_EPOCHS    = 300
PATIENCE      = 15
BATCH_SIZE    = 64
LR            = 1e-3
SEED          = 42
EVAL_SNRS     = list(range(0, 21, 2))
MC_BITS       = 10_000
MC_TRIALS     = 10

CLIP = get_clip_range(MODULATION)
NUM_CLASSES = get_num_classes(MODULATION)
LEVELS_NP = get_constellation_levels(MODULATION)

device = torch.device("cpu")

# ═══════════════════════════════════════════════════════════════════
#  VARIANT DEFINITIONS
# ═══════════════════════════════════════════════════════════════════
#  (name, window_size, training_snrs, hidden_sizes)
#  hidden_sizes is a list: [24] = 1 layer, [64] = 1 wide layer, [64,64] = 2 layers
VARIANTS = [
    ("A: Baseline (w5,snr3,h24)",  5, [5, 10, 15],         [24]),
    ("B: Win=1",                   1, [5, 10, 15],         [24]),
    ("C: Wide SNR",                5, [5, 10, 15, 20, 25], [24]),
    ("D: Win1 + Wide SNR",         1, [5, 10, 15, 20, 25], [24]),
    ("E: Win1 + WideSNR + h64",    1, [5, 10, 15, 20, 25], [64]),
    ("F: Win1 + WideSNR + 2×64",   1, [5, 10, 15, 20, 25], [64, 64]),
]


# ═══════════════════════════════════════════════════════════════════
#  DATA GENERATOR  (per-variant, since window_size and SNR list vary)
# ═══════════════════════════════════════════════════════════════════
def make_dataset(window_size, training_snrs):
    """Generate windowed training data and split 70/15/15."""
    pad = window_size // 2
    samples_per_snr = NUM_SAMPLES // len(training_snrs)
    X_all, y_all = [], []

    for snr in training_snrs:
        clean, noisy = generate_training_targets(
            samples_per_snr, snr, training_modulation=MODULATION,
            seed=42 + int(snr),
        )
        for i in range(pad, len(noisy) - pad):
            X_all.append(noisy[i - pad: i + pad + 1])
            y_all.append(clean[i])

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.float32)
    y_cls = symbols_to_class_indices(y, MODULATION)

    N = len(X)
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(N)
    n_train = int(0.70 * N)
    n_val = int(0.15 * N)

    splits = {}
    for name, sl in [("train", slice(0, n_train)),
                     ("val",   slice(n_train, n_train + n_val)),
                     ("test",  slice(n_train + n_val, None))]:
        X_s = torch.as_tensor(X[idx[sl]], device=device)
        yc_s = torch.as_tensor(y_cls[idx[sl]], dtype=torch.long, device=device)
        splits[name] = (X_s, yc_s)

    print(f"    Samples: {N}  -> Train {n_train}, Val {n_val}, Test {N - n_train - n_val}")
    return splits


# ═══════════════════════════════════════════════════════════════════
#  MODEL BUILDER
# ═══════════════════════════════════════════════════════════════════
def build_model(window_size, hidden_sizes):
    """Build classification MLP with variable depth/width."""
    layers = []
    in_dim = window_size
    for h in hidden_sizes:
        layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
        in_dim = h
    layers.append(nn.Linear(in_dim, NUM_CLASSES))
    model = nn.Sequential(*layers).to(device)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return model


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ═══════════════════════════════════════════════════════════════════
#  ACCURACY HELPER
# ═══════════════════════════════════════════════════════════════════
def acc_classify(model, X_t, y_cls_t):
    with torch.no_grad():
        return (model(X_t).argmax(dim=-1) == y_cls_t).float().mean().item()


# ═══════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════
def train_variant(name, model, splits):
    Xt_tr, yc_tr = splits["train"]
    Xt_vl, yc_vl = splits["val"]
    Xt_te, yc_te = splits["test"]

    print(f"\n{'=' * 60}\n  {name}  ({count_params(model)} params)\n{'=' * 60}")
    opt = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    hist = {"epoch": [], "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [], "test_acc": []}
    best_val = float("inf")
    best_ep = 0
    patience_ctr = 0
    best_state = None

    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(Xt_tr), device=device)
        ep_loss = 0.0
        n_b = 0
        for i in range(0, len(Xt_tr), BATCH_SIZE):
            sl = perm[i:i + BATCH_SIZE]
            loss = criterion(model(Xt_tr[sl]), yc_tr[sl])
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); n_b += 1
        ep_loss /= n_b

        model.eval()
        with torch.no_grad():
            vl = criterion(model(Xt_vl), yc_vl).item()

        tr_a = acc_classify(model, Xt_tr, yc_tr)
        vl_a = acc_classify(model, Xt_vl, yc_vl)
        te_a = acc_classify(model, Xt_te, yc_te)

        hist["epoch"].append(ep)
        hist["train_loss"].append(ep_loss)
        hist["val_loss"].append(vl)
        hist["train_acc"].append(tr_a)
        hist["val_acc"].append(vl_a)
        hist["test_acc"].append(te_a)

        if vl < best_val:
            best_val = vl; best_ep = ep; patience_ctr = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1

        if ep % 20 == 0 or ep == 1 or patience_ctr == PATIENCE:
            print(f"  Ep {ep:3d} | loss={ep_loss:.4f} val={vl:.4f} | "
                  f"tr_acc={tr_a:.4f} vl_acc={vl_a:.4f} te_acc={te_a:.4f}")

        if patience_ctr >= PATIENCE:
            print(f"  Early stop ep {ep} (best {best_ep})")
            break

    model.load_state_dict(best_state)
    model.eval()
    te_acc = acc_classify(model, Xt_te, yc_te)
    print(f"  Best epoch: {best_ep}  Test acc: {te_acc:.4f}")
    return best_state, hist, best_ep, te_acc


# ═══════════════════════════════════════════════════════════════════
#  TRAIN ALL VARIANTS
# ═══════════════════════════════════════════════════════════════════
trained = {}

for vname, win, snrs, hsizes in VARIANTS:
    print(f"\n  Preparing data for {vname}  (win={win}, snrs={snrs})")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    splits = make_dataset(win, snrs)
    model = build_model(win, hsizes)
    state, hist, best_ep, te_acc = train_variant(vname, model, splits)
    model.load_state_dict(state)
    trained[vname] = {
        "model": model, "hist": hist, "best_ep": best_ep,
        "te_acc": te_acc, "params": count_params(model),
        "window_size": win, "hidden_sizes": hsizes,
    }


# ═══════════════════════════════════════════════════════════════════
#  BER EVALUATION
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  BER EVALUATION -- Monte-Carlo")
print("=" * 60)

ber_results = {}

for vname, info in trained.items():
    model = info["model"]
    win = info["window_size"]
    hsizes = info["hidden_sizes"]

    relay = MinimalGenAIRelay(
        window_size=win, hidden_size=hsizes[0],
        prefer_gpu=False,
        classify=True, training_modulation=MODULATION,
    )
    relay._torch_model = model
    relay._use_torch = True
    relay.is_trained = True

    snrs, bers, trials = run_monte_carlo(
        relay, EVAL_SNRS, MC_BITS, MC_TRIALS, modulation=MODULATION,
    )
    lo = np.percentile(trials, 2.5, axis=1)
    hi = np.percentile(trials, 97.5, axis=1)
    ber_results[vname] = {
        "ber_mean": bers, "ber_trials": trials,
        "ci95_lo": lo, "ci95_hi": hi,
    }
    print(f"  {vname:35s} -> BER@20dB={bers[-1]:.6f}  avg={np.mean(bers):.6f}")

# Baselines
print("  Computing AF & DF baselines ...")
_, bers_af, _ = run_monte_carlo(
    AmplifyAndForwardRelay(), EVAL_SNRS, MC_BITS, MC_TRIALS, modulation=MODULATION)
_, bers_df, _ = run_monte_carlo(
    DecodeAndForwardRelay(), EVAL_SNRS, MC_BITS, MC_TRIALS, modulation=MODULATION)
print(f"  {'AF':35s} -> BER@20dB={bers_af[-1]:.6f}  avg={np.mean(bers_af):.6f}")
print(f"  {'DF':35s} -> BER@20dB={bers_df[-1]:.6f}  avg={np.mean(bers_df):.6f}")


# ═══════════════════════════════════════════════════════════════════
#  JSON PERSISTENCE
# ═══════════════════════════════════════════════════════════════════
results_json = {
    "timestamp": datetime.datetime.now().isoformat(),
    "modulation": MODULATION,
    "experiment": "classify_closing_df_gap",
    "config": {
        "num_samples": NUM_SAMPLES, "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE, "batch_size": BATCH_SIZE, "lr": LR,
        "seed": SEED, "mc_bits": MC_BITS, "mc_trials": MC_TRIALS,
    },
    "variants": {},
    "baselines": {
        "af": {"ber_mean": [float(b) for b in bers_af]},
        "df": {"ber_mean": [float(b) for b in bers_df]},
    },
    "snr": EVAL_SNRS,
}
for vname, info in trained.items():
    br = ber_results[vname]
    results_json["variants"][vname] = {
        "params": info["params"],
        "window_size": info["window_size"],
        "hidden_sizes": info["hidden_sizes"],
        "best_epoch": info["best_ep"],
        "test_accuracy": float(info["te_acc"]),
        "ber_mean": [float(b) for b in br["ber_mean"]],
        "ber_ci95_lo": [float(b) for b in br["ci95_lo"]],
        "ber_ci95_hi": [float(b) for b in br["ci95_hi"]],
        "history": info["hist"],
    }

json_path = os.path.join(OUT_DIR, "closing_gap_qam16.json")
with open(json_path, "w") as f:
    json.dump(results_json, f, indent=2)
print(f"\n  JSON -> {json_path}")


# ═══════════════════════════════════════════════════════════════════
#  CHART STYLE
# ═══════════════════════════════════════════════════════════════════
COLORS = {
    "A: Baseline (w5,snr3,h24)": "#D55E00",
    "B: Win=1":                  "#E69F00",
    "C: Wide SNR":               "#CC79A7",
    "D: Win1 + Wide SNR":        "#56B4E9",
    "E: Win1 + WideSNR + h64":   "#009E73",
    "F: Win1 + WideSNR + 2×64":  "#0072B2",
    "AF": "#888888", "DF": "#333333",
}
MARKERS = {
    "A: Baseline (w5,snr3,h24)": "o",
    "B: Win=1":                  "s",
    "C: Wide SNR":               "^",
    "D: Win1 + Wide SNR":        "D",
    "E: Win1 + WideSNR + h64":   "p",
    "F: Win1 + WideSNR + 2×64":  "h",
    "AF": "<", "DF": ">",
}
LSTYLES = {
    "A: Baseline (w5,snr3,h24)": "--",
    "B: Win=1":                  "--",
    "C: Wide SNR":               "-.",
    "D: Win1 + Wide SNR":        "-.",
    "E: Win1 + WideSNR + h64":   "-",
    "F: Win1 + WideSNR + 2×64":  "-",
    "AF": ":", "DF": ":",
}
vnames = [v[0] for v in VARIANTS]


# ═══════════════════════════════════════════════════════════════════
#  CHART 1 -- BER vs SNR (all variants + baselines)
# ═══════════════════════════════════════════════════════════════════
print("  Generating BER chart ...")
fig, ax = plt.subplots(figsize=(10, 6))

ax.semilogy(EVAL_SNRS, bers_af, color=COLORS["AF"], marker=MARKERS["AF"],
            markersize=5, linewidth=1.0, linestyle=":", label="AF baseline")
ax.semilogy(EVAL_SNRS, bers_df, color=COLORS["DF"], marker=MARKERS["DF"],
            markersize=5, linewidth=1.0, linestyle=":", label="DF baseline")

for vn in vnames:
    br = ber_results[vn]
    ax.semilogy(EVAL_SNRS, br["ber_mean"], color=COLORS[vn], marker=MARKERS[vn],
                markersize=5, linewidth=1.3, linestyle=LSTYLES[vn],
                label=f"{vn} ({trained[vn]['params']}p)")
    ax.fill_between(EVAL_SNRS, br["ci95_lo"], br["ci95_hi"],
                    color=COLORS[vn], alpha=0.08)

all_bers = np.concatenate([br["ber_mean"] for br in ber_results.values()] + [bers_af, bers_df])
min_ber = all_bers[all_bers > 0].min() if np.any(all_bers > 0) else 1e-4
ax.set_ylim(bottom=min_ber / 10, top=0.55)
ax.yaxis.set_minor_locator(LogLocator(subs="all", numticks=20))
ax.yaxis.set_minor_formatter(NullFormatter())
ax.grid(True, which="major", linewidth=0.5, alpha=0.4)
ax.grid(True, which="minor", linewidth=0.3, alpha=0.15)
ax.set_xlabel("SNR (dB)", fontsize=14)
ax.set_ylabel("Bit Error Rate", fontsize=14)
ax.set_title(f"Closing the DF Gap -- MLP Classification, {MODULATION.upper()}", fontsize=16)
ax.tick_params(labelsize=12)
ax.legend(fontsize=8, loc="upper right", framealpha=0.9, ncol=1)

# Zoomed inset for high SNR
hi_snrs = [s for s in EVAL_SNRS if s >= 12]
if len(hi_snrs) >= 3:
    axins = inset_axes(ax, width="38%", height="32%", loc="center left",
                       bbox_to_anchor=(0.02, 0.02, 1, 1),
                       bbox_transform=ax.transAxes)
    for vn in vnames:
        br = ber_results[vn]
        hi_idx = [EVAL_SNRS.index(s) for s in hi_snrs]
        vals = br["ber_mean"][hi_idx[0]:hi_idx[-1] + 1]
        axins.semilogy(hi_snrs, vals, color=COLORS[vn], marker=MARKERS[vn],
                       markersize=3, linewidth=1.0, linestyle=LSTYLES[vn])
    axins.semilogy(hi_snrs, bers_df[[EVAL_SNRS.index(s) for s in hi_snrs]],
                   color=COLORS["DF"], marker=">", markersize=3, linewidth=0.8, linestyle=":")
    axins.grid(True, which="both", linewidth=0.3, alpha=0.3)
    axins.set_title("High-SNR zoom", fontsize=9)
    axins.tick_params(labelsize=8)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", lw=0.6)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "ber_closing_gap_qam16.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  BER chart saved.")


# ═══════════════════════════════════════════════════════════════════
#  CHART 2 -- Improvement waterfall (BER@20dB per variant)
# ═══════════════════════════════════════════════════════════════════
print("  Generating waterfall chart ...")
fig, ax = plt.subplots(figsize=(10, 5))

snr_20_idx = EVAL_SNRS.index(20)
ber_at_20 = [ber_results[vn]["ber_mean"][snr_20_idx] for vn in vnames]
df_20 = bers_df[snr_20_idx]

short_names = [vn.split(": ", 1)[1] for vn in vnames]
bars = ax.bar(range(len(vnames)), ber_at_20,
              color=[COLORS[vn] for vn in vnames], edgecolor="black", linewidth=0.5)
ax.axhline(y=df_20, color=COLORS["DF"], linewidth=1.5, linestyle="--",
           label=f"DF = {df_20:.5f}")
ax.axhline(y=bers_af[snr_20_idx], color=COLORS["AF"], linewidth=1.0, linestyle=":",
           label=f"AF = {bers_af[snr_20_idx]:.5f}", alpha=0.6)

for i, (b, v) in enumerate(zip(bars, ber_at_20)):
    ax.text(i, v + max(ber_at_20) * 0.01, f"{v:.4f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xticks(range(len(vnames)))
ax.set_xticklabels(short_names, fontsize=10, rotation=20, ha="right")
ax.set_ylabel("BER @ 20 dB", fontsize=14)
ax.set_title(f"BER @ 20 dB — Progressive Improvements, {MODULATION.upper()}", fontsize=15)
ax.legend(fontsize=11, loc="upper right")
ax.grid(True, axis="y", linewidth=0.4, alpha=0.4)
ax.tick_params(labelsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "waterfall_ber20_qam16.png"), dpi=150)
plt.close(fig)
print(f"  Waterfall chart saved.")


# ═══════════════════════════════════════════════════════════════════
#  CHART 3 -- Top-3 + baselines
# ═══════════════════════════════════════════════════════════════════
print("  Generating Top-3 chart ...")
upper_half_idx = [i for i, s in enumerate(EVAL_SNRS) if s >= EVAL_SNRS[-1] // 2]
ranking = sorted(ber_results.keys(),
                 key=lambda v: np.mean(ber_results[v]["ber_mean"][upper_half_idx[0]:]))
top3 = ranking[:3]
print(f"  Top-3: {top3}")

TOP3_CLR = ["#0072B2", "#D55E00", "#009E73"]
fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(EVAL_SNRS, bers_af, color="#888888", marker="<", markersize=5,
            linewidth=1.0, linestyle=":", label="AF baseline")
ax.semilogy(EVAL_SNRS, bers_df, color="#333333", marker=">", markersize=5,
            linewidth=1.0, linestyle=":", label="DF baseline")
for i, vn in enumerate(top3):
    br = ber_results[vn]
    ax.semilogy(EVAL_SNRS, br["ber_mean"], color=TOP3_CLR[i], marker=MARKERS[vn],
                markersize=6, linewidth=1.4, linestyle="-",
                label=f"#{i + 1} {vn}")
    ax.fill_between(EVAL_SNRS, br["ci95_lo"], br["ci95_hi"], color=TOP3_CLR[i], alpha=0.12)

ax.set_ylim(bottom=min_ber / 10, top=0.55)
ax.yaxis.set_minor_locator(LogLocator(subs="all", numticks=20))
ax.yaxis.set_minor_formatter(NullFormatter())
ax.grid(True, which="major", linewidth=0.5, alpha=0.4)
ax.grid(True, which="minor", linewidth=0.3, alpha=0.15)
ax.set_xlabel("SNR (dB)", fontsize=14)
ax.set_ylabel("Bit Error Rate", fontsize=14)
ax.set_title(f"Top-3 Improved MLP Variants -- {MODULATION.upper()}", fontsize=16)
ax.tick_params(labelsize=12)
ax.legend(fontsize=10, loc="upper right", framealpha=0.9)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "top3_closing_gap_qam16.png"), dpi=150)
plt.close(fig)
print(f"  Top-3 chart saved.")


# ═══════════════════════════════════════════════════════════════════
#  CHART 4 -- Accuracy vs Epoch
# ═══════════════════════════════════════════════════════════════════
print("  Generating accuracy chart ...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
for ax, split, key in [
    (axes[0], "Train", "train_acc"),
    (axes[1], "Validation", "val_acc"),
    (axes[2], "Test", "test_acc"),
]:
    for vn in vnames:
        h = trained[vn]["hist"]
        ax.plot(h["epoch"], h[key], color=COLORS[vn], linewidth=1.1,
                linestyle=LSTYLES[vn], label=vn)
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_title(f"{split} Accuracy", fontsize=14)
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.tick_params(labelsize=11)
axes[0].set_ylabel("Accuracy", fontsize=14)
axes[2].legend(fontsize=7, loc="lower right", framealpha=0.9)
fig.suptitle(f"Training Accuracy -- Closing the DF Gap, {MODULATION.upper()}", fontsize=16, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "accuracy_closing_gap_qam16.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Accuracy chart saved.")


# ═══════════════════════════════════════════════════════════════════
#  CHART 5 -- Loss vs Epoch
# ═══════════════════════════════════════════════════════════════════
print("  Generating loss chart ...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, key, title in [
    (axes[0], "train_loss", "Train Loss"),
    (axes[1], "val_loss", "Validation Loss"),
]:
    for vn in vnames:
        h = trained[vn]["hist"]
        ax.plot(h["epoch"], h[key], color=COLORS[vn], linewidth=1.1,
                linestyle=LSTYLES[vn], label=vn)
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Loss", fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.tick_params(labelsize=11)
axes[1].legend(fontsize=7, loc="upper right", framealpha=0.9)
fig.suptitle(f"Training Loss -- Closing the DF Gap, {MODULATION.upper()}", fontsize=16, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "loss_closing_gap_qam16.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Loss chart saved.")


# ═══════════════════════════════════════════════════════════════════
#  CHART 6 -- Heatmap
# ═══════════════════════════════════════════════════════════════════
print("  Generating heatmap ...")
ber_matrix = np.array([ber_results[vn]["ber_mean"] for vn in vnames])
short_names_hm = [vn.split(": ", 1)[1] for vn in vnames]

fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(ber_matrix, aspect="auto", cmap="RdYlGn_r")
ax.set_xticks(range(len(EVAL_SNRS)))
ax.set_xticklabels([str(s) for s in EVAL_SNRS], fontsize=11)
ax.set_yticks(range(len(vnames)))
ax.set_yticklabels(short_names_hm, fontsize=10)
ax.set_xlabel("SNR (dB)", fontsize=14)
ax.set_title(f"BER Heatmap -- Closing the DF Gap, {MODULATION.upper()}", fontsize=15)
for i in range(len(vnames)):
    for j in range(len(EVAL_SNRS)):
        v = ber_matrix[i, j]
        txt = f"{v:.3f}" if v >= 0.01 else f"{v:.4f}"
        clr = "white" if v > 0.25 else "black"
        ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=clr)
fig.colorbar(im, ax=ax, label="BER", shrink=0.8)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "heatmap_closing_gap_qam16.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Heatmap saved.")


# ═══════════════════════════════════════════════════════════════════
#  SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  BER SUMMARY TABLE")
print("=" * 70)
key_snrs = [4, 10, 16, 20]
header = f"  {'Variant':35s}" + "".join(f"  {s:6d}dB" for s in key_snrs) + f"  {'AvgBER':>10s}"
print(header)
print("  " + "-" * (len(header) - 2))
for vn in vnames:
    br = ber_results[vn]["ber_mean"]
    row = f"  {vn:35s}"
    for s in key_snrs:
        if s in EVAL_SNRS:
            row += f"  {br[EVAL_SNRS.index(s)]:8.5f}"
    row += f"  {np.mean(br):10.6f}"
    print(row)
for bname, bbers in [("AF", bers_af), ("DF", bers_df)]:
    row = f"  {bname:35s}"
    for s in key_snrs:
        if s in EVAL_SNRS:
            row += f"  {bbers[EVAL_SNRS.index(s)]:8.5f}"
    row += f"  {np.mean(bbers):10.6f}"
    print(row)

# ─── Gap analysis ─────────────────────────────────────────────────
print("\n  GAP vs DF @ 20 dB:")
df_20 = bers_df[snr_20_idx]
for vn in vnames:
    mlp_20 = ber_results[vn]["ber_mean"][snr_20_idx]
    gap = mlp_20 - df_20
    pct = (gap / max(df_20, 1e-8)) * 100
    print(f"    {vn:35s}  BER={mlp_20:.5f}  gap={gap:+.5f}  ({pct:+.0f}%)")

print("\n" + "=" * 70)
print("  FINAL SUMMARY")
print("=" * 70)
for vn in vnames:
    info = trained[vn]
    br = ber_results[vn]
    print(f"  {vn:35s} | {info['params']:5d}p | best_ep={info['best_ep']:3d} | "
          f"te_acc={info['te_acc']:.4f} | ber@20={br['ber_mean'][snr_20_idx]:.5f}")
print(f"  {'DF':35s} | {'---':>5s}p | {'---':>8s} | {'---':>10s} | ber@20={df_20:.5f}")
print(f"\n  All outputs in: {OUT_DIR}")
