"""
MLP Classification — Hidden & Output Activation Comparison (QAM16)
===================================================================
Probes how different activations affect classification performance.

Hidden-activation variants (output = raw logits):
  1. ReLU  (baseline)
  2. Tanh
  3. Sigmoid
  4. Scaled Tanh

Output-activation variants (hidden = ReLU, activation applied to logits):
  5. Tanh output
  6. Scaled Tanh output
  7. Sigmoid output
  8. ReLU output

• 70/15/15 train/val/test split, Early stopping (patience=15)
• Monte-Carlo BER: 10 trials × 10,000 bits
• Charts per CHART_GUIDELINES.md + JSON persistence
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

OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "classify_activations")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Config ───────────────────────────────────────────────────────
MODULATION    = "qam16"
WINDOW_SIZE   = 5
HIDDEN_SIZE   = 24
NUM_SAMPLES   = 50_000
TRAINING_SNRS = [5, 10, 15]
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

np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cpu")


# ═══════════════════════════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════════════════════════
print("Generating training data ...")
pad = WINDOW_SIZE // 2
samples_per_snr = NUM_SAMPLES // len(TRAINING_SNRS)
X_all, y_all = [], []

for snr in TRAINING_SNRS:
    clean, noisy = generate_training_targets(
        samples_per_snr, snr, training_modulation=MODULATION,
        seed=42 + int(snr),
    )
    for i in range(pad, len(noisy) - pad):
        X_all.append(noisy[i - pad: i + pad + 1])
        y_all.append(clean[i])

X = np.array(X_all, dtype=np.float32)
y = np.array(y_all, dtype=np.float32).reshape(-1, 1)
y_cls = symbols_to_class_indices(y.ravel(), MODULATION)

N = len(X)
idx = np.random.permutation(N)
n_train = int(0.70 * N)
n_val = int(0.15 * N)
X_train = X[idx[:n_train]]
X_val   = X[idx[n_train:n_train + n_val]]
X_test  = X[idx[n_train + n_val:]]
yc_train = y_cls[idx[:n_train]]
yc_val   = y_cls[idx[n_train:n_train + n_val]]
yc_test  = y_cls[idx[n_train + n_val:]]

Xt_tr  = torch.as_tensor(X_train, device=device)
Xt_val = torch.as_tensor(X_val,   device=device)
Xt_te  = torch.as_tensor(X_test,  device=device)
yc_tr  = torch.as_tensor(yc_train, dtype=torch.long, device=device)
yc_vl  = torch.as_tensor(yc_val,   dtype=torch.long, device=device)
yc_te  = torch.as_tensor(yc_test,  dtype=torch.long, device=device)

print(f"  Samples: {N}  -> Train {n_train}, Val {n_val}, Test {N - n_train - n_val}")
print(f"  Classes: {NUM_CLASSES}  Levels: {LEVELS_NP}")
print()


# ═══════════════════════════════════════════════════════════════════
#  MODEL BUILDER
# ═══════════════════════════════════════════════════════════════════
class ScaledTanh(nn.Module):
    """clip_range * tanh(x)"""
    def __init__(self, clip):
        super().__init__()
        self.clip = clip
    def forward(self, x):
        return self.clip * torch.tanh(x)


def _make_hidden_act(name):
    """Return a hidden-layer activation module."""
    return {
        "relu":        nn.ReLU(),
        "tanh":        nn.Tanh(),
        "sigmoid":     nn.Sigmoid(),
        "scaled_tanh": ScaledTanh(CLIP),
    }[name]


def _make_output_act(name):
    """Return an output-layer activation module (applied to logits)."""
    return {
        "tanh":        nn.Tanh(),
        "sigmoid":     nn.Sigmoid(),
        "relu":        nn.ReLU(),
        "scaled_tanh": ScaledTanh(CLIP),
    }[name]


def build_model(hidden_act="relu", output_act=None):
    """
    Build classification MLP.
      hidden_act: activation after the hidden Linear
      output_act: optional activation applied to logits (after final Linear)
    """
    layers = [
        nn.Linear(WINDOW_SIZE, HIDDEN_SIZE),
        _make_hidden_act(hidden_act),
        nn.Linear(HIDDEN_SIZE, NUM_CLASSES),
    ]
    if output_act is not None:
        layers.append(_make_output_act(output_act))
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
def train_variant(name, model):
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
            vl = criterion(model(Xt_val), yc_vl).item()

        tr_a = acc_classify(model, Xt_tr, yc_tr)
        vl_a = acc_classify(model, Xt_val, yc_vl)
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
#  VARIANTS
# ═══════════════════════════════════════════════════════════════════
VARIANTS = [
    # (name, hidden_act, output_act)
    # ── Hidden activation sweep (output = raw logits) ──
    ("H:ReLU",         "relu",        None),
    ("H:Tanh",         "tanh",        None),
    ("H:Sigmoid",      "sigmoid",     None),
    ("H:Scaled Tanh",  "scaled_tanh", None),
    # ── Output activation sweep (hidden = ReLU) ──
    ("O:Tanh",         "relu",        "tanh"),
    ("O:Scaled Tanh",  "relu",        "scaled_tanh"),
    ("O:Sigmoid",      "relu",        "sigmoid"),
    ("O:ReLU",         "relu",        "relu"),
]


# ═══════════════════════════════════════════════════════════════════
#  TRAIN ALL
# ═══════════════════════════════════════════════════════════════════
trained = {}

for vname, h_act, o_act in VARIANTS:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = build_model(hidden_act=h_act, output_act=o_act)
    state, hist, best_ep, te_acc = train_variant(vname, model)
    model.load_state_dict(state)
    trained[vname] = {
        "model": model, "hist": hist, "best_ep": best_ep,
        "te_acc": te_acc, "params": count_params(model),
    }


# ═══════════════════════════════════════════════════════════════════
#  BER EVALUATION  (Monte-Carlo)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  BER EVALUATION -- Monte-Carlo")
print("=" * 60)

ber_results = {}

for vname, info in trained.items():
    model = info["model"]
    relay = MinimalGenAIRelay(
        window_size=WINDOW_SIZE, hidden_size=HIDDEN_SIZE,
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
    print(f"  {vname:20s} -> avg BER(10-20dB) = {np.mean(bers[5:]):.6f}")

# AF / DF baselines
print("  Computing AF & DF baselines ...")
_, bers_af, trials_af = run_monte_carlo(
    AmplifyAndForwardRelay(), EVAL_SNRS, MC_BITS, MC_TRIALS, modulation=MODULATION)
_, bers_df, trials_df = run_monte_carlo(
    DecodeAndForwardRelay(), EVAL_SNRS, MC_BITS, MC_TRIALS, modulation=MODULATION)
print(f"  {'AF':20s} -> avg BER(10-20dB) = {np.mean(bers_af[5:]):.6f}")
print(f"  {'DF':20s} -> avg BER(10-20dB) = {np.mean(bers_df[5:]):.6f}")


# ═══════════════════════════════════════════════════════════════════
#  JSON PERSISTENCE
# ═══════════════════════════════════════════════════════════════════
results_json = {
    "timestamp": datetime.datetime.now().isoformat(),
    "modulation": MODULATION,
    "experiment": "classify_activation_sweep",
    "config": {
        "window_size": WINDOW_SIZE, "hidden_size": HIDDEN_SIZE,
        "num_samples": NUM_SAMPLES, "training_snrs": TRAINING_SNRS,
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
        "batch_size": BATCH_SIZE, "lr": LR, "seed": SEED,
        "mc_bits": MC_BITS, "mc_trials": MC_TRIALS,
        "clip_range": float(CLIP),
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
        "best_epoch": info["best_ep"],
        "test_accuracy": float(info["te_acc"]),
        "ber_mean": [float(b) for b in br["ber_mean"]],
        "ber_ci95_lo": [float(b) for b in br["ci95_lo"]],
        "ber_ci95_hi": [float(b) for b in br["ci95_hi"]],
        "ber_per_trial": br["ber_trials"].tolist(),
        "history": info["hist"],
    }

json_path = os.path.join(OUT_DIR, "classify_activations_qam16.json")
with open(json_path, "w") as f:
    json.dump(results_json, f, indent=2)
print(f"\n  JSON -> {json_path}")


# ═══════════════════════════════════════════════════════════════════
#  CHART STYLE
# ═══════════════════════════════════════════════════════════════════
COLORS = {
    "H:ReLU":        "#0072B2",
    "H:Tanh":        "#D55E00",
    "H:Sigmoid":     "#009E73",
    "H:Scaled Tanh": "#E69F00",
    "O:Tanh":        "#56B4E9",
    "O:Scaled Tanh": "#CC79A7",
    "O:Sigmoid":     "#F0E442",
    "O:ReLU":        "#882255",
    "AF": "#888888", "DF": "#333333",
}
MARKERS = {
    "H:ReLU": "o", "H:Tanh": "s", "H:Sigmoid": "^", "H:Scaled Tanh": "D",
    "O:Tanh": "p", "O:Scaled Tanh": "h", "O:Sigmoid": "v", "O:ReLU": "X",
    "AF": "<", "DF": ">",
}
LSTYLES = {
    "H:ReLU": "-", "H:Tanh": "-", "H:Sigmoid": "-", "H:Scaled Tanh": "-",
    "O:Tanh": "--", "O:Scaled Tanh": "--", "O:Sigmoid": "--", "O:ReLU": "--",
    "AF": ":", "DF": ":",
}
vnames = [v[0] for v in VARIANTS]


# ═══════════════════════════════════════════════════════════════════
#  CHART 1 -- BER vs SNR  (all variants)
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
ax.set_title(f"Classification Activation Sweep -- {MODULATION.upper()}", fontsize=16)
ax.tick_params(labelsize=12)
ax.legend(fontsize=9, loc="upper right", framealpha=0.9, ncol=1)

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
fig.savefig(os.path.join(OUT_DIR, "ber_classify_activations_qam16.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  BER chart saved.")


# ═══════════════════════════════════════════════════════════════════
#  CHART 2 -- BER split: hidden vs output panels
# ═══════════════════════════════════════════════════════════════════
print("  Generating hidden vs output panel chart ...")
hidden_vnames = [v[0] for v in VARIANTS if v[2] is None]
output_vnames = [v[0] for v in VARIANTS if v[2] is not None]

fig, (ax_h, ax_o) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

for ax, subset, title in [
    (ax_h, hidden_vnames, "Hidden Activation Sweep"),
    (ax_o, output_vnames, "Output Activation Sweep"),
]:
    ax.semilogy(EVAL_SNRS, bers_af, color="#888888", marker="<", markersize=4,
                linewidth=0.8, linestyle=":", label="AF", alpha=0.6)
    ax.semilogy(EVAL_SNRS, bers_df, color="#333333", marker=">", markersize=4,
                linewidth=0.8, linestyle=":", label="DF", alpha=0.6)
    for vn in subset:
        br = ber_results[vn]
        ax.semilogy(EVAL_SNRS, br["ber_mean"], color=COLORS[vn], marker=MARKERS[vn],
                    markersize=5, linewidth=1.3, linestyle=LSTYLES[vn],
                    label=f"{vn} ({trained[vn]['params']}p)")
        ax.fill_between(EVAL_SNRS, br["ci95_lo"], br["ci95_hi"],
                        color=COLORS[vn], alpha=0.10)
    ax.set_ylim(bottom=min_ber / 10, top=0.55)
    ax.yaxis.set_minor_locator(LogLocator(subs="all", numticks=20))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", linewidth=0.5, alpha=0.4)
    ax.grid(True, which="minor", linewidth=0.3, alpha=0.15)
    ax.set_xlabel("SNR (dB)", fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9)
    ax.tick_params(labelsize=12)

ax_h.set_ylabel("Bit Error Rate", fontsize=14)
fig.suptitle(f"Classification Activation Comparison -- {MODULATION.upper()}", fontsize=16, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "ber_hidden_vs_output_qam16.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Panel chart saved.")


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
ax.set_title(f"Top-3 Classification Activations -- {MODULATION.upper()}", fontsize=16)
ax.tick_params(labelsize=12)
ax.legend(fontsize=11, loc="upper right", framealpha=0.9)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "top3_classify_activations_qam16.png"), dpi=150)
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
axes[2].legend(fontsize=8, loc="lower right", framealpha=0.9, ncol=1)
fig.suptitle(f"Training Accuracy -- Classification Activation Sweep, {MODULATION.upper()}",
             fontsize=16, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "accuracy_classify_activations_qam16.png"),
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

axes[1].legend(fontsize=8, loc="upper right", framealpha=0.9, ncol=1)
fig.suptitle(f"Training Loss -- Classification Activation Sweep, {MODULATION.upper()}",
             fontsize=16, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "loss_classify_activations_qam16.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Loss chart saved.")


# ═══════════════════════════════════════════════════════════════════
#  CHART 6 -- Heatmap
# ═══════════════════════════════════════════════════════════════════
print("  Generating heatmap ...")
ber_matrix = np.array([ber_results[vn]["ber_mean"] for vn in vnames])

fig, ax = plt.subplots(figsize=(10, 4.5))
im = ax.imshow(ber_matrix, aspect="auto", cmap="RdYlGn_r")
ax.set_xticks(range(len(EVAL_SNRS)))
ax.set_xticklabels([str(s) for s in EVAL_SNRS], fontsize=11)
ax.set_yticks(range(len(vnames)))
ax.set_yticklabels(vnames, fontsize=10)
ax.set_xlabel("SNR (dB)", fontsize=14)
ax.set_title(f"BER Heatmap -- Classification Activations, {MODULATION.upper()}", fontsize=15)

for i in range(len(vnames)):
    for j in range(len(EVAL_SNRS)):
        v = ber_matrix[i, j]
        txt = f"{v:.3f}" if v >= 0.01 else f"{v:.4f}"
        clr = "white" if v > 0.25 else "black"
        ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=clr)

fig.colorbar(im, ax=ax, label="BER", shrink=0.8)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "heatmap_classify_activations_qam16.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Heatmap saved.")


# ═══════════════════════════════════════════════════════════════════
#  CHART 7 -- Winner per SNR
# ═══════════════════════════════════════════════════════════════════
print("  Generating winner chart ...")
fig, ax = plt.subplots(figsize=(10, 6))

winner_per_snr = []
for j in range(len(EVAL_SNRS)):
    best_v = min(vnames, key=lambda v: ber_results[v]["ber_mean"][j])
    winner_per_snr.append(best_v)

for vn in vnames:
    br = ber_results[vn]
    is_winner = any(w == vn for w in winner_per_snr)
    alpha = 1.0 if is_winner else 0.2
    lw = 1.4 if is_winner else 0.8
    ax.semilogy(EVAL_SNRS, br["ber_mean"], color=COLORS[vn], marker=MARKERS[vn],
                markersize=5, linewidth=lw, linestyle=LSTYLES[vn],
                alpha=alpha, label=vn)
    for j, snr in enumerate(EVAL_SNRS):
        if winner_per_snr[j] == vn:
            ax.plot(snr, br["ber_mean"][j], color=COLORS[vn], marker=MARKERS[vn],
                    markersize=9, markeredgewidth=2, markeredgecolor="black", zorder=10)

ax.semilogy(EVAL_SNRS, bers_af, color="#888888", linewidth=0.8, linestyle=":", alpha=0.4, label="AF")
ax.semilogy(EVAL_SNRS, bers_df, color="#333333", linewidth=0.8, linestyle=":", alpha=0.4, label="DF")
ax.set_ylim(bottom=min_ber / 10, top=0.55)
ax.grid(True, which="major", linewidth=0.4, alpha=0.3)
ax.set_xlabel("SNR (dB)", fontsize=14)
ax.set_ylabel("Bit Error Rate", fontsize=14)
ax.set_title(f"Winner per SNR -- Classification Activations, {MODULATION.upper()}", fontsize=16)
ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
ax.tick_params(labelsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "winner_classify_activations_qam16.png"), dpi=150)
plt.close(fig)
print(f"  Winner chart saved.")


# ═══════════════════════════════════════════════════════════════════
#  SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  BER SUMMARY TABLE")
print("=" * 60)
key_snrs = [4, 10, 16, 20]
header = f"  {'Variant':20s}" + "".join(f"  {s:6d} dB" for s in key_snrs) + f"  {'AvgBER':>10s}"
print(header)
print("  " + "-" * (len(header) - 2))
for vn in vnames:
    br = ber_results[vn]["ber_mean"]
    row = f"  {vn:20s}"
    for s in key_snrs:
        if s in EVAL_SNRS:
            row += f"  {br[EVAL_SNRS.index(s)]:9.5f}"
    row += f"  {np.mean(br):10.6f}"
    print(row)
for bname, bbers in [("AF", bers_af), ("DF", bers_df)]:
    row = f"  {bname:20s}"
    for s in key_snrs:
        if s in EVAL_SNRS:
            row += f"  {bbers[EVAL_SNRS.index(s)]:9.5f}"
    row += f"  {np.mean(bbers):10.6f}"
    print(row)

print("\n" + "=" * 60)
print("  FINAL SUMMARY")
print("=" * 60)
for vn in vnames:
    info = trained[vn]
    br = ber_results[vn]
    print(f"  {vn:20s} | {info['params']:4d}p | best_ep={info['best_ep']:3d} | "
          f"te_acc={info['te_acc']:.4f} | avg_ber={np.mean(br['ber_mean']):.6f}")

print(f"\n  All outputs in: {OUT_DIR}")
