"""
MLP Classification vs Regression — proper training comparison.

• Train/Val/Test split (70/15/15)
• Early stopping (patience=15, monitor val loss)
• Track train/val/test accuracy per epoch
• QAM16 modulation — 4 classes per axis
• BER evaluation via Monte-Carlo at multiple SNRs
• Charts following CHART_GUIDELINES.md
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

from relaynet.utils.activations import (
    generate_training_targets,
    get_num_classes, get_constellation_levels, symbols_to_class_indices,
)
from relaynet.simulation.runner import run_monte_carlo
from relaynet.relays.genai import MinimalGenAIRelay
from relaynet.relays.af import AmplifyAndForwardRelay
from relaynet.relays.df import DecodeAndForwardRelay

OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "classify_vs_regress")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Config ───────────────────────────────────────────────────────
MODULATION      = "qam16"
WINDOW_SIZE     = 5
HIDDEN_SIZE     = 24
NUM_SAMPLES     = 50_000
TRAINING_SNRS   = [5, 10, 15]
MAX_EPOCHS      = 300
PATIENCE        = 15
BATCH_SIZE      = 64
LR              = 1e-3
SEED            = 42
EVAL_SNRS       = list(range(0, 21, 2))
MC_BITS         = 20_000
MC_TRIALS       = 10

np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cpu")

# ─── Data preparation ─────────────────────────────────────────────
print("Generating training data …")
pad = WINDOW_SIZE // 2
samples_per_snr = NUM_SAMPLES // len(TRAINING_SNRS)
X_all, y_all = [], []

for snr in TRAINING_SNRS:
    clean, noisy = generate_training_targets(
        samples_per_snr, snr,
        training_modulation=MODULATION,
        seed=42 + int(snr),
    )
    for i in range(pad, len(noisy) - pad):
        X_all.append(noisy[i - pad : i + pad + 1])
        y_all.append(clean[i])

X = np.array(X_all, dtype=np.float32)
y = np.array(y_all, dtype=np.float32).reshape(-1, 1)

# Class indices for classification
num_classes = get_num_classes(MODULATION)
levels_np   = get_constellation_levels(MODULATION)
y_cls       = symbols_to_class_indices(y.ravel(), MODULATION)

# Train / Val / Test split (70/15/15)
N = len(X)
idx = np.random.permutation(N)
n_train = int(0.70 * N)
n_val   = int(0.15 * N)
train_idx = idx[:n_train]
val_idx   = idx[n_train:n_train + n_val]
test_idx  = idx[n_train + n_val:]

X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
yc_train, yc_val, yc_test = y_cls[train_idx], y_cls[val_idx], y_cls[test_idx]

print(f"  Samples: {N}  → Train {len(train_idx)}, Val {len(val_idx)}, Test {len(test_idx)}")
print(f"  Classes: {num_classes}  Levels: {levels_np}")
print()


# ─── Helper: build model ──────────────────────────────────────────
def build_model(out_dim, act="tanh"):
    layers = [nn.Linear(WINDOW_SIZE, HIDDEN_SIZE), nn.ReLU()]
    if out_dim > 1:
        layers.append(nn.Linear(HIDDEN_SIZE, out_dim))
    else:
        layers.extend([nn.Linear(HIDDEN_SIZE, 1), nn.Tanh()])
    model = nn.Sequential(*layers).to(device)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return model


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ─── Helper: classification accuracy ──────────────────────────────
def accuracy_classify(model, X_t, y_cls_t):
    with torch.no_grad():
        logits = model(X_t)
        preds = logits.argmax(dim=-1)
        return (preds == y_cls_t).float().mean().item()


def accuracy_regress(model, X_t, y_t, levels_t):
    """Regression accuracy = fraction of samples whose nearest-level
    matches the true nearest-level."""
    with torch.no_grad():
        out = model(X_t).flatten()
        pred_idx = torch.argmin(torch.abs(out.unsqueeze(1) - levels_t.unsqueeze(0)), dim=1)
        true_idx = torch.argmin(torch.abs(y_t.flatten().unsqueeze(1) - levels_t.unsqueeze(0)), dim=1)
        return (pred_idx == true_idx).float().mean().item()


# ═══════════════════════════════════════════════════════════════════
#  CLASSIFICATION TRAINING
# ═══════════════════════════════════════════════════════════════════
print("=" * 60)
print("  CLASSIFICATION  (CrossEntropyLoss, 4-class output)")
print("=" * 60)
model_cls = build_model(num_classes)
print(f"  Parameters: {count_params(model_cls)}")
opt_cls   = optim.Adam(model_cls.parameters(), lr=LR)
crit_cls  = nn.CrossEntropyLoss()

Xt_tr  = torch.as_tensor(X_train, device=device)
Xt_val = torch.as_tensor(X_val,   device=device)
Xt_te  = torch.as_tensor(X_test,  device=device)
yc_tr  = torch.as_tensor(yc_train, dtype=torch.long, device=device)
yc_vl  = torch.as_tensor(yc_val,   dtype=torch.long, device=device)
yc_te  = torch.as_tensor(yc_test,  dtype=torch.long, device=device)

hist_cls = {"epoch": [], "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [], "test_acc": []}
best_val_loss_cls = float("inf")
best_epoch_cls    = 0
patience_ctr      = 0
best_state_cls    = None

for ep in range(1, MAX_EPOCHS + 1):
    # --- train ---
    model_cls.train()
    perm = torch.randperm(len(Xt_tr), device=device)
    epoch_loss = 0.0
    n_batch    = 0
    for i in range(0, len(Xt_tr), BATCH_SIZE):
        sl = perm[i:i + BATCH_SIZE]
        logits = model_cls(Xt_tr[sl])
        loss   = crit_cls(logits, yc_tr[sl])
        opt_cls.zero_grad(); loss.backward(); opt_cls.step()
        epoch_loss += loss.item(); n_batch += 1
    epoch_loss /= n_batch

    # --- val loss ---
    model_cls.eval()
    with torch.no_grad():
        val_loss = crit_cls(model_cls(Xt_val), yc_vl).item()

    # --- accuracies ---
    tr_acc  = accuracy_classify(model_cls, Xt_tr, yc_tr)
    vl_acc  = accuracy_classify(model_cls, Xt_val, yc_vl)
    te_acc  = accuracy_classify(model_cls, Xt_te, yc_te)

    hist_cls["epoch"].append(ep)
    hist_cls["train_loss"].append(epoch_loss)
    hist_cls["val_loss"].append(val_loss)
    hist_cls["train_acc"].append(tr_acc)
    hist_cls["val_acc"].append(vl_acc)
    hist_cls["test_acc"].append(te_acc)

    # --- early stopping ---
    if val_loss < best_val_loss_cls:
        best_val_loss_cls = val_loss
        best_epoch_cls    = ep
        patience_ctr      = 0
        best_state_cls    = {k: v.clone() for k, v in model_cls.state_dict().items()}
    else:
        patience_ctr += 1

    if ep % 10 == 0 or ep == 1 or patience_ctr == PATIENCE:
        print(f"  Epoch {ep:3d} | train_loss={epoch_loss:.4f}  val_loss={val_loss:.4f} | "
              f"train_acc={tr_acc:.4f}  val_acc={vl_acc:.4f}  test_acc={te_acc:.4f}")

    if patience_ctr >= PATIENCE:
        print(f"  Early stopping at epoch {ep}  (best epoch {best_epoch_cls})")
        break

model_cls.load_state_dict(best_state_cls)
model_cls.eval()
final_te_cls = accuracy_classify(model_cls, Xt_te, yc_te)
print(f"  Best epoch: {best_epoch_cls}  |  Test accuracy (best model): {final_te_cls:.4f}")
print()


# ═══════════════════════════════════════════════════════════════════
#  REGRESSION TRAINING
# ═══════════════════════════════════════════════════════════════════
print("=" * 60)
print("  REGRESSION  (MSELoss, 1-output neuron + tanh)")
print("=" * 60)
model_reg = build_model(1)
print(f"  Parameters: {count_params(model_reg)}")
opt_reg   = optim.Adam(model_reg.parameters(), lr=LR)
crit_reg  = nn.MSELoss()

yr_tr = torch.as_tensor(y_train, device=device)
yr_vl = torch.as_tensor(y_val,   device=device)
yr_te = torch.as_tensor(y_test,  device=device)
levels_t = torch.as_tensor(levels_np, dtype=torch.float32, device=device)

hist_reg = {"epoch": [], "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [], "test_acc": []}
best_val_loss_reg = float("inf")
best_epoch_reg    = 0
patience_ctr      = 0
best_state_reg    = None

for ep in range(1, MAX_EPOCHS + 1):
    model_reg.train()
    perm = torch.randperm(len(Xt_tr), device=device)
    epoch_loss = 0.0
    n_batch    = 0
    for i in range(0, len(Xt_tr), BATCH_SIZE):
        sl = perm[i:i + BATCH_SIZE]
        out  = model_reg(Xt_tr[sl])
        loss = crit_reg(out, yr_tr[sl])
        opt_reg.zero_grad(); loss.backward(); opt_reg.step()
        epoch_loss += loss.item(); n_batch += 1
    epoch_loss /= n_batch

    model_reg.eval()
    with torch.no_grad():
        val_loss = crit_reg(model_reg(Xt_val), yr_vl).item()

    tr_acc = accuracy_regress(model_reg, Xt_tr, yr_tr, levels_t)
    vl_acc = accuracy_regress(model_reg, Xt_val, yr_vl, levels_t)
    te_acc = accuracy_regress(model_reg, Xt_te, yr_te, levels_t)

    hist_reg["epoch"].append(ep)
    hist_reg["train_loss"].append(epoch_loss)
    hist_reg["val_loss"].append(val_loss)
    hist_reg["train_acc"].append(tr_acc)
    hist_reg["val_acc"].append(vl_acc)
    hist_reg["test_acc"].append(te_acc)

    if val_loss < best_val_loss_reg:
        best_val_loss_reg = val_loss
        best_epoch_reg    = ep
        patience_ctr      = 0
        best_state_reg    = {k: v.clone() for k, v in model_reg.state_dict().items()}
    else:
        patience_ctr += 1

    if ep % 10 == 0 or ep == 1 or patience_ctr == PATIENCE:
        print(f"  Epoch {ep:3d} | train_loss={epoch_loss:.4f}  val_loss={val_loss:.4f} | "
              f"train_acc={tr_acc:.4f}  val_acc={vl_acc:.4f}  test_acc={te_acc:.4f}")

    if patience_ctr >= PATIENCE:
        print(f"  Early stopping at epoch {ep}  (best epoch {best_epoch_reg})")
        break

model_reg.load_state_dict(best_state_reg)
model_reg.eval()
final_te_reg = accuracy_regress(model_reg, Xt_te, yr_te, levels_t)
print(f"  Best epoch: {best_epoch_reg}  |  Test accuracy (best model): {final_te_reg:.4f}")
print()


# ═══════════════════════════════════════════════════════════════════
#  BER  EVALUATION  (Monte-Carlo)
# ═══════════════════════════════════════════════════════════════════
print("=" * 60)
print("  BER EVALUATION  (Monte-Carlo)")
print("=" * 60)

# Build properly-configured relay objects and inject trained weights
relay_cls = MinimalGenAIRelay(
    window_size=WINDOW_SIZE, hidden_size=HIDDEN_SIZE,
    prefer_gpu=False, classify=True, training_modulation=MODULATION,
)
relay_cls._torch_model = model_cls
relay_cls._use_torch = True
relay_cls.is_trained = True

relay_reg = MinimalGenAIRelay(
    window_size=WINDOW_SIZE, hidden_size=HIDDEN_SIZE,
    prefer_gpu=False, classify=False, training_modulation=MODULATION,
)
relay_reg._torch_model = model_reg
relay_reg._use_torch = True
relay_reg.is_trained = True

print(f"\n  {'SNR':>4s}  {'BER_classify':>13s}  {'BER_regress':>13s}  {'Winner':>10s}")
print("  " + "─" * 48)

snrs_cls, bers_cls, _ = run_monte_carlo(
    relay_cls, EVAL_SNRS, MC_BITS, MC_TRIALS, modulation=MODULATION,
)
snrs_reg, bers_reg, _ = run_monte_carlo(
    relay_reg, EVAL_SNRS, MC_BITS, MC_TRIALS, modulation=MODULATION,
)

for snr, bc, br in zip(EVAL_SNRS, bers_cls, bers_reg):
    winner = "CLASSIFY" if bc < br else ("REGRESS" if br < bc else "TIE")
    print(f"  {snr:4d}  {bc:13.6f}  {br:13.6f}  {winner:>10s}")

# ─── AF / DF baselines ───────────────────────────────────────────
print("\n  Computing AF & DF baselines …")
relay_af = AmplifyAndForwardRelay()
relay_df = DecodeAndForwardRelay()
_, bers_af, trials_af = run_monte_carlo(relay_af, EVAL_SNRS, MC_BITS, MC_TRIALS, modulation=MODULATION)
_, bers_df, trials_df = run_monte_carlo(relay_df, EVAL_SNRS, MC_BITS, MC_TRIALS, modulation=MODULATION)

# ─── Per-trial arrays for CI ─────────────────────────────────────
_, _, trials_cls = run_monte_carlo(relay_cls, EVAL_SNRS, MC_BITS, MC_TRIALS, modulation=MODULATION, seed_offset=100)
_, _, trials_reg = run_monte_carlo(relay_reg, EVAL_SNRS, MC_BITS, MC_TRIALS, modulation=MODULATION, seed_offset=100)

def ci95(trials_2d):
    """Return (low, high) arrays for 95% CI."""
    lo = np.percentile(trials_2d, 2.5, axis=1)
    hi = np.percentile(trials_2d, 97.5, axis=1)
    return lo, hi

ci_cls = ci95(trials_cls)
ci_reg = ci95(trials_reg)

print()

# ─── Summary ──────────────────────────────────────────────────────
print("=" * 60)
print("  SUMMARY")
print("=" * 60)
print(f"  Classification: {count_params(model_cls)} params, best epoch {best_epoch_cls}, "
      f"test acc {final_te_cls:.4f}")
print(f"  Regression:     {count_params(model_reg)} params, best epoch {best_epoch_reg}, "
      f"test acc {final_te_reg:.4f}")
avg_ber_cls = np.mean(bers_cls)
avg_ber_reg = np.mean(bers_reg)
print(f"  Avg BER (classify): {avg_ber_cls:.6f}")
print(f"  Avg BER (regress):  {avg_ber_reg:.6f}")
if avg_ber_cls < avg_ber_reg:
    pct = (1 - avg_ber_cls / avg_ber_reg) * 100
    print(f"  → Classification is {pct:.1f}% better (lower avg BER)")
else:
    pct = (1 - avg_ber_reg / avg_ber_cls) * 100
    print(f"  → Regression is {pct:.1f}% better (lower avg BER)")


# ═══════════════════════════════════════════════════════════════════
#  JSON PERSISTENCE  (Guideline #21)
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
    },
    "classification": {
        "params": count_params(model_cls),
        "best_epoch": best_epoch_cls,
        "test_accuracy": float(final_te_cls),
        "snr": EVAL_SNRS,
        "ber_mean": [float(b) for b in bers_cls],
        "ber_ci95_lo": [float(b) for b in ci_cls[0]],
        "ber_ci95_hi": [float(b) for b in ci_cls[1]],
        "ber_per_trial": trials_cls.tolist(),
        "history": hist_cls,
    },
    "regression": {
        "params": count_params(model_reg),
        "best_epoch": best_epoch_reg,
        "test_accuracy": float(final_te_reg),
        "snr": EVAL_SNRS,
        "ber_mean": [float(b) for b in bers_reg],
        "ber_ci95_lo": [float(b) for b in ci_reg[0]],
        "ber_ci95_hi": [float(b) for b in ci_reg[1]],
        "ber_per_trial": trials_reg.tolist(),
        "history": hist_reg,
    },
    "baselines": {
        "af": {"ber_mean": [float(b) for b in bers_af]},
        "df": {"ber_mean": [float(b) for b in bers_df]},
    },
}
json_path = os.path.join(OUT_DIR, "mlp_classify_vs_regress_qam16.json")
with open(json_path, "w") as f:
    json.dump(results_json, f, indent=2)
print(f"\n  Results saved → {json_path}")


# ═══════════════════════════════════════════════════════════════════
#  CHART 1 — BER vs SNR  (Guideline #1-#22)
# ═══════════════════════════════════════════════════════════════════
print("  Generating BER chart …")

# Colorblind-friendly palette (guideline #13)
CLR_CLS  = "#0072B2"   # blue
CLR_REG  = "#D55E00"   # vermillion
CLR_AF   = "#666666"   # grey
CLR_DF   = "#333333"   # dark grey
MRK_CLS  = "o"
MRK_REG  = "s"
MRK_AF   = "^"
MRK_DF   = "D"

fig, ax = plt.subplots(figsize=(8, 5))  # guideline #10

# Baselines (guideline #8)
ax.semilogy(EVAL_SNRS, bers_af, color=CLR_AF, marker=MRK_AF, markersize=5,
            linewidth=1.2, linestyle="--", label=f"AF baseline")
ax.semilogy(EVAL_SNRS, bers_df, color=CLR_DF, marker=MRK_DF, markersize=5,
            linewidth=1.2, linestyle="-.", label=f"DF baseline")

# Classification
ax.semilogy(EVAL_SNRS, bers_cls, color=CLR_CLS, marker=MRK_CLS, markersize=5,
            linewidth=1.4, linestyle="-",
            label=f"MLP classify ({count_params(model_cls)} params)")
ax.fill_between(EVAL_SNRS, ci_cls[0], ci_cls[1], color=CLR_CLS, alpha=0.12)

# Regression
ax.semilogy(EVAL_SNRS, bers_reg, color=CLR_REG, marker=MRK_REG, markersize=5,
            linewidth=1.4, linestyle="-",
            label=f"MLP regress ({count_params(model_reg)} params)")
ax.fill_between(EVAL_SNRS, ci_reg[0], ci_reg[1], color=CLR_REG, alpha=0.12)

# Y-axis focus (guideline #6)
all_bers = np.concatenate([bers_cls, bers_reg, bers_af, bers_df])
min_ber = all_bers[all_bers > 0].min() if np.any(all_bers > 0) else 1e-4
ax.set_ylim(bottom=min_ber / 10, top=0.55)
ax.yaxis.set_minor_locator(LogLocator(subs="all", numticks=20))
ax.yaxis.set_minor_formatter(NullFormatter())

# Gridlines (guideline #12)
ax.grid(True, which="major", linewidth=0.5, alpha=0.4)
ax.grid(True, which="minor", linewidth=0.3, alpha=0.15)

# Labels (guideline #11)
ax.set_xlabel("SNR (dB)", fontsize=14)
ax.set_ylabel("Bit Error Rate", fontsize=14)
ax.set_title(f"MLP Classification vs Regression — {MODULATION.upper()}", fontsize=16)
ax.tick_params(labelsize=12)

# Legend outside (guideline #3)
ax.legend(fontsize=11, loc="upper right", framealpha=0.9)

# Annotation leaders for high-SNR region (guideline #4, #16)
snr_annot = 18
idx_annot = EVAL_SNRS.index(snr_annot)
bc_a, br_a = bers_cls[idx_annot], bers_reg[idx_annot]
if bc_a > 0 and br_a > 0:
    pct_gain = (1 - bc_a / br_a) * 100
    ax.annotate(
        f"classify {pct_gain:.0f}% better\n@ {snr_annot} dB",
        xy=(snr_annot, bc_a), xytext=(snr_annot - 5, bc_a * 3),
        fontsize=10, color=CLR_CLS,
        arrowprops=dict(arrowstyle="->", color=CLR_CLS, lw=1.2),
    )

# Zoomed inset for high-SNR (guideline #2, #15)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
hi_snrs = [s for s in EVAL_SNRS if s >= 12]
hi_idx  = [EVAL_SNRS.index(s) for s in hi_snrs]
if len(hi_snrs) >= 3:
    axins = inset_axes(ax, width="40%", height="35%", loc="center left",
                       bbox_to_anchor=(0.05, 0.05, 1, 1),
                       bbox_transform=ax.transAxes)
    axins.semilogy(hi_snrs, bers_cls[hi_idx[0]:hi_idx[-1]+1], color=CLR_CLS,
                   marker=MRK_CLS, markersize=4, linewidth=1.2)
    axins.semilogy(hi_snrs, bers_reg[hi_idx[0]:hi_idx[-1]+1], color=CLR_REG,
                   marker=MRK_REG, markersize=4, linewidth=1.2)
    axins.semilogy(hi_snrs, bers_df[hi_idx[0]:hi_idx[-1]+1], color=CLR_DF,
                   marker=MRK_DF, markersize=3, linewidth=1.0, linestyle="-.")
    axins.grid(True, which="both", linewidth=0.3, alpha=0.3)
    axins.set_title("High-SNR zoom", fontsize=9)
    axins.tick_params(labelsize=8)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", lw=0.6)

fig.tight_layout()
ber_path = os.path.join(OUT_DIR, "ber_classify_vs_regress_qam16.png")
fig.savefig(ber_path, dpi=150)
plt.close(fig)
print(f"  BER chart saved → {ber_path}")


# ═══════════════════════════════════════════════════════════════════
#  CHART 2 — Accuracy vs Epoch  (train / val / test)
# ═══════════════════════════════════════════════════════════════════
print("  Generating accuracy chart …")

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)  # guideline #10

for ax, hist, title, clr in [
    (axes[0], hist_cls, "Classification (4-class CE)", CLR_CLS),
    (axes[1], hist_reg, "Regression (MSE → nearest-level)", CLR_REG),
]:
    epochs = hist["epoch"]
    ax.plot(epochs, hist["train_acc"], color=clr, linewidth=1.2,
            linestyle="-", label="Train acc")
    ax.plot(epochs, hist["val_acc"], color=clr, linewidth=1.2,
            linestyle="--", label="Val acc")
    ax.plot(epochs, hist["test_acc"], color=clr, linewidth=1.2,
            linestyle=":", label="Test acc")

    # Mark best epoch
    best_ep = hist["epoch"][np.argmax(hist["val_acc"])]
    best_va = max(hist["val_acc"])
    ax.axvline(best_ep, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.annotate(f"best ep={best_ep}", xy=(best_ep, best_va),
                xytext=(best_ep + 3, best_va - 0.03),
                fontsize=9, color="grey",
                arrowprops=dict(arrowstyle="->", color="grey", lw=0.8))

    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11, loc="lower right", framealpha=0.9)
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.tick_params(labelsize=12)

axes[0].set_ylabel("Accuracy", fontsize=14)
fig.suptitle(f"Training Accuracy — MLP on {MODULATION.upper()}", fontsize=16, y=1.01)
fig.tight_layout()
acc_path = os.path.join(OUT_DIR, "accuracy_classify_vs_regress_qam16.png")
fig.savefig(acc_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Accuracy chart saved → {acc_path}")


# ═══════════════════════════════════════════════════════════════════
#  CHART 3 — Loss vs Epoch  (train / val)
# ═══════════════════════════════════════════════════════════════════
print("  Generating loss chart …")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, hist, title, clr, ylabel in [
    (axes[0], hist_cls, "Classification Loss (CE)", CLR_CLS, "Cross-Entropy Loss"),
    (axes[1], hist_reg, "Regression Loss (MSE)", CLR_REG, "MSE Loss"),
]:
    epochs = hist["epoch"]
    ax.plot(epochs, hist["train_loss"], color=clr, linewidth=1.2,
            linestyle="-", label="Train loss")
    ax.plot(epochs, hist["val_loss"], color=clr, linewidth=1.2,
            linestyle="--", label="Val loss")
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11, loc="upper right", framealpha=0.9)
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.tick_params(labelsize=12)

fig.suptitle(f"Training Loss — MLP on {MODULATION.upper()}", fontsize=16, y=1.01)
fig.tight_layout()
loss_path = os.path.join(OUT_DIR, "loss_classify_vs_regress_qam16.png")
fig.savefig(loss_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Loss chart saved → {loss_path}")


# ═══════════════════════════════════════════════════════════════════
#  BER SUMMARY TABLE  (Guideline #18)
# ═══════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("  BER SUMMARY TABLE")
print("=" * 60)
key_snrs = [4, 10, 16]
print(f"  {'SNR':>4s}  {'Classify':>10s}  {'Regress':>10s}  {'AF':>10s}  {'DF':>10s}  {'Δ (cls−reg)':>12s}")
print("  " + "─" * 58)
for s in key_snrs:
    if s in EVAL_SNRS:
        i = EVAL_SNRS.index(s)
        delta = bers_cls[i] - bers_reg[i]
        print(f"  {s:4d}  {bers_cls[i]:10.6f}  {bers_reg[i]:10.6f}  "
              f"{bers_af[i]:10.6f}  {bers_df[i]:10.6f}  {delta:+12.6f}")

# ─── History tables ───────────────────────────────────────────────
print()
print("=" * 60)
print("  TRAINING HISTORY — Classification")
print("=" * 60)
print(f"  {'Epoch':>5s}  {'TrainLoss':>10s}  {'ValLoss':>10s}  {'TrainAcc':>9s}  {'ValAcc':>8s}  {'TestAcc':>8s}")
for i, ep in enumerate(hist_cls["epoch"]):
    if ep % 10 == 0 or ep == 1 or ep == hist_cls["epoch"][-1]:
        print(f"  {ep:5d}  {hist_cls['train_loss'][i]:10.4f}  {hist_cls['val_loss'][i]:10.4f}  "
              f"{hist_cls['train_acc'][i]:9.4f}  {hist_cls['val_acc'][i]:8.4f}  {hist_cls['test_acc'][i]:8.4f}")

print()
print("=" * 60)
print("  TRAINING HISTORY — Regression")
print("=" * 60)
print(f"  {'Epoch':>5s}  {'TrainLoss':>10s}  {'ValLoss':>10s}  {'TrainAcc':>9s}  {'ValAcc':>8s}  {'TestAcc':>8s}")
for i, ep in enumerate(hist_reg["epoch"]):
    if ep % 10 == 0 or ep == 1 or ep == hist_reg["epoch"][-1]:
        print(f"  {ep:5d}  {hist_reg['train_loss'][i]:10.4f}  {hist_reg['val_loss'][i]:10.4f}  "
              f"{hist_reg['train_acc'][i]:9.4f}  {hist_reg['val_acc'][i]:8.4f}  {hist_reg['test_acc'][i]:8.4f}")

print(f"\n  All outputs in: {OUT_DIR}")
