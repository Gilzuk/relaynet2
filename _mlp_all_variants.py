"""
MLP Output-Head Comparison — QAM16
===================================
Variants:
  1. Classify  (4-class CE)
  2. Classify + InputLN
  3. Regress tanh
  4. Regress hardtanh
  5. Regress hardtanh + InputLN
  6. Regress scaled_tanh
  7. Regress scaled_tanh + InputLN

• 70/15/15 train/val/test split
• Early stopping (patience=15)
• Train/Val/Test accuracy tracked per epoch
• Monte-Carlo BER: 10 trials × 10,000 bits
• Charts per CHART_GUIDELINES.md
• JSON persistence
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

OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "classify_vs_regress")
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

CLIP = get_clip_range(MODULATION)  # 3/sqrt(10) ~ 0.9487
NUM_CLASSES = get_num_classes(MODULATION)  # 4
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
X_train, X_val, X_test = X[idx[:n_train]], X[idx[n_train:n_train+n_val]], X[idx[n_train+n_val:]]
y_train, y_val, y_test = y[idx[:n_train]], y[idx[n_train:n_train+n_val]], y[idx[n_train+n_val:]]
yc_train, yc_val, yc_test = y_cls[idx[:n_train]], y_cls[idx[n_train:n_train+n_val]], y_cls[idx[n_train+n_val:]]

Xt_tr  = torch.as_tensor(X_train, device=device)
Xt_val = torch.as_tensor(X_val,   device=device)
Xt_te  = torch.as_tensor(X_test,  device=device)
yr_tr  = torch.as_tensor(y_train, device=device)
yr_vl  = torch.as_tensor(y_val,   device=device)
yr_te  = torch.as_tensor(y_test,  device=device)
yc_tr  = torch.as_tensor(yc_train, dtype=torch.long, device=device)
yc_vl  = torch.as_tensor(yc_val,   dtype=torch.long, device=device)
yc_te  = torch.as_tensor(yc_test,  dtype=torch.long, device=device)
levels_t = torch.as_tensor(LEVELS_NP, dtype=torch.float32, device=device)

print(f"  Samples: {N}  -> Train {n_train}, Val {n_val}, Test {N-n_train-n_val}")
print(f"  Classes: {NUM_CLASSES}  Levels: {LEVELS_NP}")
print()


# ═══════════════════════════════════════════════════════════════════
#  MODEL BUILDER
# ═══════════════════════════════════════════════════════════════════
def build_model(classify, activation="tanh", use_input_norm=False):
    """Build MLP with optional InputLN and activation variant."""
    layers = []
    if use_input_norm:
        layers.append(nn.LayerNorm(WINDOW_SIZE))
    layers.extend([nn.Linear(WINDOW_SIZE, HIDDEN_SIZE), nn.ReLU()])
    if classify:
        layers.append(nn.Linear(HIDDEN_SIZE, NUM_CLASSES))
    else:
        layers.append(nn.Linear(HIDDEN_SIZE, 1))
        layers.append(make_torch_activation(activation, clip_range=CLIP))
    model = nn.Sequential(*layers).to(device)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return model


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ═══════════════════════════════════════════════════════════════════
#  ACCURACY HELPERS
# ═══════════════════════════════════════════════════════════════════
def acc_classify(model, X_t, y_cls_t):
    with torch.no_grad():
        return (model(X_t).argmax(dim=-1) == y_cls_t).float().mean().item()


def acc_regress(model, X_t, y_t):
    with torch.no_grad():
        out = model(X_t).flatten()
        pred_idx = torch.argmin(torch.abs(out.unsqueeze(1) - levels_t.unsqueeze(0)), dim=1)
        true_idx = torch.argmin(torch.abs(y_t.flatten().unsqueeze(1) - levels_t.unsqueeze(0)), dim=1)
        return (pred_idx == true_idx).float().mean().item()


# ═══════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════
def train_variant(name, model, classify):
    """Train one variant. Returns (best_model_state, history, best_epoch, test_acc)."""
    print(f"\n{'='*60}\n  {name}  ({count_params(model)} params)\n{'='*60}")
    opt = optim.Adam(model.parameters(), lr=LR)
    if classify:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

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
            out = model(Xt_tr[sl])
            if classify:
                loss = criterion(out, yc_tr[sl])
            else:
                loss = criterion(out, yr_tr[sl])
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); n_b += 1
        ep_loss /= n_b

        model.eval()
        with torch.no_grad():
            if classify:
                vl = criterion(model(Xt_val), yc_vl).item()
            else:
                vl = criterion(model(Xt_val), yr_vl).item()

        if classify:
            tr_a = acc_classify(model, Xt_tr, yc_tr)
            vl_a = acc_classify(model, Xt_val, yc_vl)
            te_a = acc_classify(model, Xt_te, yc_te)
        else:
            tr_a = acc_regress(model, Xt_tr, yr_tr)
            vl_a = acc_regress(model, Xt_val, yr_vl)
            te_a = acc_regress(model, Xt_te, yr_te)

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
    if classify:
        te_acc = acc_classify(model, Xt_te, yc_te)
    else:
        te_acc = acc_regress(model, Xt_te, yr_te)
    print(f"  Best epoch: {best_ep}  Test acc: {te_acc:.4f}")
    return best_state, hist, best_ep, te_acc


# ═══════════════════════════════════════════════════════════════════
#  DEFINE VARIANTS
# ═══════════════════════════════════════════════════════════════════
VARIANTS = [
    # (name, classify, activation, use_input_norm)
    ("Classify",              True,  None,          False),
    ("Classify + InputLN",    True,  None,          True),
    ("Regress tanh",          False, "tanh",        False),
    ("Regress hardtanh",      False, "hardtanh",    False),
    ("Regress hardtanh + LN", False, "hardtanh",    True),
    ("Regress scaled_tanh",       False, "scaled_tanh", False),
    ("Regress scaled_tanh + LN",  False, "scaled_tanh", True),
]


# ═══════════════════════════════════════════════════════════════════
#  TRAIN ALL
# ═══════════════════════════════════════════════════════════════════
trained = {}  # name -> {model, hist, best_ep, te_acc, classify, params}

for vname, cls, act, ln in VARIANTS:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = build_model(classify=cls, activation=act or "tanh", use_input_norm=ln)
    state, hist, best_ep, te_acc = train_variant(vname, model, cls)
    model.load_state_dict(state)
    trained[vname] = {
        "model": model, "hist": hist, "best_ep": best_ep,
        "te_acc": te_acc, "classify": cls, "params": count_params(model),
    }


# ═══════════════════════════════════════════════════════════════════
#  BER EVALUATION  (Monte-Carlo)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  BER EVALUATION -- Monte-Carlo")
print("=" * 60)

ber_results = {}  # name -> {ber_mean, ber_trials, ci95_lo, ci95_hi}

for vname, info in trained.items():
    cls = info["classify"]
    model = info["model"]

    # Build a relay and inject trained model
    relay = MinimalGenAIRelay(
        window_size=WINDOW_SIZE, hidden_size=HIDDEN_SIZE,
        prefer_gpu=False,
        classify=cls, training_modulation=MODULATION,
        output_activation="hardtanh" if not cls else "tanh",
        clip_range=CLIP if not cls else None,
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
    print(f"  {vname:30s} -> avg BER(10-20dB) = {np.mean(bers[5:]):.6f}")

# AF / DF baselines
print("  Computing AF & DF baselines ...")
_, bers_af, trials_af = run_monte_carlo(
    AmplifyAndForwardRelay(), EVAL_SNRS, MC_BITS, MC_TRIALS, modulation=MODULATION)
_, bers_df, trials_df = run_monte_carlo(
    DecodeAndForwardRelay(), EVAL_SNRS, MC_BITS, MC_TRIALS, modulation=MODULATION)
print(f"  {'AF':30s} -> avg BER(10-20dB) = {np.mean(bers_af[5:]):.6f}")
print(f"  {'DF':30s} -> avg BER(10-20dB) = {np.mean(bers_df[5:]):.6f}")


# ═══════════════════════════════════════════════════════════════════
#  JSON PERSISTENCE  (#21)
# ═══════════════════════════════════════════════════════════════════
results_json = {
    "timestamp": datetime.datetime.now().isoformat(),
    "modulation": MODULATION,
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

json_path = os.path.join(OUT_DIR, "mlp_all_variants_qam16.json")
with open(json_path, "w") as f:
    json.dump(results_json, f, indent=2)
print(f"\n  JSON -> {json_path}")


# ═══════════════════════════════════════════════════════════════════
#  CHART STYLE SETUP
# ═══════════════════════════════════════════════════════════════════
# Colorblind-friendly palette (#9, #13)
COLORS = {
    "Classify":              "#0072B2",  # blue
    "Classify + InputLN":    "#56B4E9",  # sky blue
    "Regress tanh":          "#D55E00",  # vermillion
    "Regress hardtanh":      "#E69F00",  # amber
    "Regress hardtanh + LN": "#F0E442",  # yellow
    "Regress scaled_tanh":       "#009E73",  # teal
    "Regress scaled_tanh + LN":  "#CC79A7",  # rose
    "AF": "#888888", "DF": "#333333",
}
MARKERS = {
    "Classify":              "o",
    "Classify + InputLN":    "p",
    "Regress tanh":          "s",
    "Regress hardtanh":      "^",
    "Regress hardtanh + LN": "v",
    "Regress scaled_tanh":       "D",
    "Regress scaled_tanh + LN":  "h",
    "AF": "<", "DF": ">",
}
LSTYLES = {
    "Classify":              "-",
    "Classify + InputLN":    "-",
    "Regress tanh":          "--",
    "Regress hardtanh":      "--",
    "Regress hardtanh + LN": "--",
    "Regress scaled_tanh":       "-.",
    "Regress scaled_tanh + LN":  "-.",
    "AF": ":", "DF": ":",
}
vnames = [v[0] for v in VARIANTS]


# ═══════════════════════════════════════════════════════════════════
#  CHART 1 -- BER vs SNR  (all variants + baselines)
# ═══════════════════════════════════════════════════════════════════
print("  Generating BER chart ...")
fig, ax = plt.subplots(figsize=(10, 6))  # #10

# Baselines (#8)
ax.semilogy(EVAL_SNRS, bers_af, color=COLORS["AF"], marker=MARKERS["AF"],
            markersize=5, linewidth=1.0, linestyle=LSTYLES["AF"], label="AF baseline")
ax.semilogy(EVAL_SNRS, bers_df, color=COLORS["DF"], marker=MARKERS["DF"],
            markersize=5, linewidth=1.0, linestyle=LSTYLES["DF"], label="DF baseline")

# Variants (#1, #2, #7, #14)
for vn in vnames:
    br = ber_results[vn]
    ax.semilogy(EVAL_SNRS, br["ber_mean"], color=COLORS[vn], marker=MARKERS[vn],
                markersize=5, linewidth=1.3, linestyle=LSTYLES[vn],
                label=f"{vn} ({trained[vn]['params']}p)")
    ax.fill_between(EVAL_SNRS, br["ci95_lo"], br["ci95_hi"],
                    color=COLORS[vn], alpha=0.08)

# Y-axis (#6)
all_bers = np.concatenate([br["ber_mean"] for br in ber_results.values()] + [bers_af, bers_df])
min_ber = all_bers[all_bers > 0].min() if np.any(all_bers > 0) else 1e-4
ax.set_ylim(bottom=min_ber / 10, top=0.55)
ax.yaxis.set_minor_locator(LogLocator(subs="all", numticks=20))
ax.yaxis.set_minor_formatter(NullFormatter())

# Grid (#12)
ax.grid(True, which="major", linewidth=0.5, alpha=0.4)
ax.grid(True, which="minor", linewidth=0.3, alpha=0.15)

# Labels (#11)
ax.set_xlabel("SNR (dB)", fontsize=14)
ax.set_ylabel("Bit Error Rate", fontsize=14)
ax.set_title(f"MLP Output-Head Comparison -- {MODULATION.upper()}", fontsize=16)
ax.tick_params(labelsize=12)

# Legend (#3)
ax.legend(fontsize=9, loc="upper right", framealpha=0.9, ncol=1)

# Zoomed inset for high-SNR (#15)
hi_snrs = [s for s in EVAL_SNRS if s >= 12]
if len(hi_snrs) >= 3:
    axins = inset_axes(ax, width="38%", height="32%", loc="center left",
                       bbox_to_anchor=(0.02, 0.02, 1, 1),
                       bbox_transform=ax.transAxes)
    for vn in vnames:
        br = ber_results[vn]
        hi_idx = [EVAL_SNRS.index(s) for s in hi_snrs]
        vals = br["ber_mean"][hi_idx[0]:hi_idx[-1]+1]
        axins.semilogy(hi_snrs, vals, color=COLORS[vn], marker=MARKERS[vn],
                       markersize=3, linewidth=1.0, linestyle=LSTYLES[vn])
    hi_idx_b = [EVAL_SNRS.index(s) for s in hi_snrs]
    axins.semilogy(hi_snrs, bers_df[hi_idx_b[0]:hi_idx_b[-1]+1], color=COLORS["DF"],
                   marker=MARKERS["DF"], markersize=3, linewidth=0.8, linestyle=":")
    axins.grid(True, which="both", linewidth=0.3, alpha=0.3)
    axins.set_title("High-SNR zoom", fontsize=9)
    axins.tick_params(labelsize=8)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", lw=0.6)

fig.tight_layout()
ber_path = os.path.join(OUT_DIR, "ber_all_variants_qam16.png")
fig.savefig(ber_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  BER chart -> {ber_path}")


# ═══════════════════════════════════════════════════════════════════
#  CHART 2 -- Top-3 + baselines  (#22)
# ═══════════════════════════════════════════════════════════════════
print("  Generating Top-3 chart ...")
upper_half_idx = [i for i, s in enumerate(EVAL_SNRS) if s >= EVAL_SNRS[-1] // 2]
ranking = sorted(ber_results.keys(),
                 key=lambda v: np.mean(ber_results[v]["ber_mean"][upper_half_idx[0]:]))
top3 = ranking[:3]
print(f"  Top-3 (by avg BER upper-half SNR): {top3}")

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
                label=f"#{i+1} {vn}")
    ax.fill_between(EVAL_SNRS, br["ci95_lo"], br["ci95_hi"], color=TOP3_CLR[i], alpha=0.12)

ax.set_ylim(bottom=min_ber / 10, top=0.55)
ax.yaxis.set_minor_locator(LogLocator(subs="all", numticks=20))
ax.yaxis.set_minor_formatter(NullFormatter())
ax.grid(True, which="major", linewidth=0.5, alpha=0.4)
ax.grid(True, which="minor", linewidth=0.3, alpha=0.15)
ax.set_xlabel("SNR (dB)", fontsize=14)
ax.set_ylabel("Bit Error Rate", fontsize=14)
ax.set_title(f"Top-3 MLP Variants -- {MODULATION.upper()}", fontsize=16)
ax.tick_params(labelsize=12)
ax.legend(fontsize=11, loc="upper right", framealpha=0.9)
fig.tight_layout()
top3_path = os.path.join(OUT_DIR, "top3_variants_qam16.png")
fig.savefig(top3_path, dpi=150)
plt.close(fig)
print(f"  Top-3 chart -> {top3_path}")


# ═══════════════════════════════════════════════════════════════════
#  CHART 3 -- Accuracy vs Epoch (all variants, three panels)
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
fig.suptitle(f"Training Accuracy -- MLP Variants on {MODULATION.upper()}", fontsize=16, y=1.01)
fig.tight_layout()
acc_path = os.path.join(OUT_DIR, "accuracy_all_variants_qam16.png")
fig.savefig(acc_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Accuracy chart -> {acc_path}")


# ═══════════════════════════════════════════════════════════════════
#  CHART 4 -- Loss vs Epoch
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
fig.suptitle(f"Training Loss -- MLP Variants on {MODULATION.upper()}", fontsize=16, y=1.01)
fig.tight_layout()
loss_path = os.path.join(OUT_DIR, "loss_all_variants_qam16.png")
fig.savefig(loss_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Loss chart -> {loss_path}")


# ═══════════════════════════════════════════════════════════════════
#  CHART 5 -- Summary heatmap  (#19)
# ═══════════════════════════════════════════════════════════════════
print("  Generating heatmap ...")
ber_matrix = np.array([ber_results[vn]["ber_mean"] for vn in vnames])

fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(ber_matrix, aspect="auto", cmap="RdYlGn_r")
ax.set_xticks(range(len(EVAL_SNRS)))
ax.set_xticklabels([str(s) for s in EVAL_SNRS], fontsize=11)
ax.set_yticks(range(len(vnames)))
ax.set_yticklabels(vnames, fontsize=10)
ax.set_xlabel("SNR (dB)", fontsize=14)
ax.set_title(f"BER Heatmap -- MLP Variants, {MODULATION.upper()}", fontsize=15)

for i in range(len(vnames)):
    for j in range(len(EVAL_SNRS)):
        v = ber_matrix[i, j]
        txt = f"{v:.3f}" if v >= 0.01 else f"{v:.4f}"
        clr = "white" if v > 0.25 else "black"
        ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=clr)

fig.colorbar(im, ax=ax, label="BER", shrink=0.8)
fig.tight_layout()
hm_path = os.path.join(OUT_DIR, "heatmap_all_variants_qam16.png")
fig.savefig(hm_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Heatmap -> {hm_path}")


# ═══════════════════════════════════════════════════════════════════
#  CHART 6 -- Winner highlight  (#20)
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
                    markersize=9, markeredgewidth=2, markeredgecolor="black",
                    zorder=10)

ax.semilogy(EVAL_SNRS, bers_af, color="#888888", linewidth=0.8, linestyle=":",
            alpha=0.4, label="AF")
ax.semilogy(EVAL_SNRS, bers_df, color="#333333", linewidth=0.8, linestyle=":",
            alpha=0.4, label="DF")

ax.set_ylim(bottom=min_ber / 10, top=0.55)
ax.grid(True, which="major", linewidth=0.4, alpha=0.3)
ax.set_xlabel("SNR (dB)", fontsize=14)
ax.set_ylabel("Bit Error Rate", fontsize=14)
ax.set_title(f"Winner per SNR -- MLP Variants, {MODULATION.upper()}", fontsize=16)
ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
ax.tick_params(labelsize=12)
fig.tight_layout()
win_path = os.path.join(OUT_DIR, "winner_per_snr_qam16.png")
fig.savefig(win_path, dpi=150)
plt.close(fig)
print(f"  Winner chart -> {win_path}")


# ═══════════════════════════════════════════════════════════════════
#  BER SUMMARY TABLE  (#18)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  BER SUMMARY TABLE (key SNRs)")
print("=" * 60)
key_snrs = [4, 10, 16, 20]
header = f"  {'Variant':30s}" + "".join(f"  {s:6d} dB" for s in key_snrs) + f"  {'AvgBER':>10s}"
print(header)
print("  " + "-" * (len(header) - 2))
for vn in vnames:
    br = ber_results[vn]["ber_mean"]
    row = f"  {vn:30s}"
    for s in key_snrs:
        if s in EVAL_SNRS:
            row += f"  {br[EVAL_SNRS.index(s)]:9.5f}"
    row += f"  {np.mean(br):10.6f}"
    print(row)
for bname, bbers in [("AF", bers_af), ("DF", bers_df)]:
    row = f"  {bname:30s}"
    for s in key_snrs:
        if s in EVAL_SNRS:
            row += f"  {bbers[EVAL_SNRS.index(s)]:9.5f}"
    row += f"  {np.mean(bbers):10.6f}"
    print(row)

# ─── Final summary ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FINAL SUMMARY")
print("=" * 60)
for vn in vnames:
    info = trained[vn]
    br = ber_results[vn]
    print(f"  {vn:30s} | {info['params']:4d}p | best_ep={info['best_ep']:3d} | "
          f"te_acc={info['te_acc']:.4f} | avg_ber={np.mean(br['ber_mean']):.6f}")

print(f"\n  All outputs in: {OUT_DIR}")
