"""
16-Class QAM16 Classification — Full 2D Constellation
=======================================================
Instead of splitting I/Q and doing 4-class per axis, this classifies
each received complex symbol into one of the 16 QAM constellation
points directly.

Variants:
  A) 4-class per-axis baseline (current approach, win=5)
  B) 16-class 2D, win=1  (input = [I, Q] = 2 values)
  C) 16-class 2D, win=5  (input = [5I, 5Q] = 10 values)
  D) 16-class 2D, win=1, h64
  E) 16-class 2D, win=1, 2×64

• Wide training SNR [5, 10, 15, 20, 25]
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

from relaynet.nodes import Source, Destination
from relaynet.channels.awgn import awgn_channel
from relaynet.modulation.bpsk import calculate_ber
from relaynet.simulation.runner import run_monte_carlo
from relaynet.relays.genai import MinimalGenAIRelay
from relaynet.relays.af import AmplifyAndForwardRelay
from relaynet.relays.df import DecodeAndForwardRelay
from relaynet.relays.base import Relay
from relaynet.utils.activations import (
    get_num_classes, get_constellation_levels, symbols_to_class_indices,
    get_clip_range, generate_training_targets,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "classify_16class")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Config ───────────────────────────────────────────────────────
MODULATION     = "qam16"
NUM_SAMPLES    = 50_000
TRAINING_SNRS  = [5, 10, 15, 20, 25]
MAX_EPOCHS     = 300
PATIENCE       = 15
BATCH_SIZE     = 64
LR             = 1e-3
SEED           = 42
EVAL_SNRS      = list(range(0, 21, 2))
MC_BITS        = 10_000
MC_TRIALS      = 10

CLIP = get_clip_range(MODULATION)
NORM = np.sqrt(10.0)

# Per-axis levels (4 values)
_LEVELS_1D = np.array([-3.0, -1.0, 1.0, 3.0]) / NORM

# Full 16-point constellation (complex)
CONSTELLATION_16 = np.array([
    (i_val + 1j * q_val)
    for i_val in _LEVELS_1D
    for q_val in _LEVELS_1D
])  # shape (16,)  – ordering: I varies slow, Q varies fast


def complex_to_class_16(symbols):
    """Map complex symbols to nearest of 16 constellation points -> index 0..15."""
    # symbols: complex array (N,)
    dist = np.abs(symbols[:, None] - CONSTELLATION_16[None, :])  # (N, 16)
    return np.argmin(dist, axis=1)


device = torch.device("cpu")
CONSTELLATION_16_T = torch.as_tensor(
    np.column_stack([CONSTELLATION_16.real, CONSTELLATION_16.imag]),
    dtype=torch.float32, device=device,
)  # (16, 2)


# ═══════════════════════════════════════════════════════════════════
#  DATA: 16-class (complex QAM16 symbols)
# ═══════════════════════════════════════════════════════════════════
def make_data_16class(window_size):
    """Generate windowed complex QAM16 training data for 16-class classification."""
    print(f"    Generating 16-class data (win={window_size}) ...")
    pad = window_size // 2
    samples_per_snr = NUM_SAMPLES // len(TRAINING_SNRS)
    X_all, y_all = [], []

    for snr in TRAINING_SNRS:
        rng = np.random.RandomState(42 + int(snr))
        n_symbols = samples_per_snr
        # Generate random QAM16 symbols
        tx_idx = rng.randint(0, 16, size=n_symbols)
        tx_symbols = CONSTELLATION_16[tx_idx]  # complex

        # Add AWGN
        sig_power = np.mean(np.abs(tx_symbols) ** 2)
        snr_lin = 10 ** (snr / 10)
        noise_std = np.sqrt(sig_power / snr_lin / 2)
        noise = noise_std * (rng.randn(n_symbols) + 1j * rng.randn(n_symbols))
        rx_symbols = tx_symbols + noise

        # Window: concatenate [I_window, Q_window]
        rx_I = np.pad(rx_symbols.real, pad, mode="edge")
        rx_Q = np.pad(rx_symbols.imag, pad, mode="edge")

        for i in range(pad, n_symbols + pad):
            i_win = rx_I[i - pad: i + pad + 1]  # (window_size,)
            q_win = rx_Q[i - pad: i + pad + 1]  # (window_size,)
            X_all.append(np.concatenate([i_win, q_win]))  # (2*window_size,)
            y_all.append(tx_idx[i - pad])  # 16-class label

    X = np.array(X_all, dtype=np.float32)  # (N, 2*window_size)
    y = np.array(y_all, dtype=np.int64)

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
        y_s = torch.as_tensor(y[idx[sl]], dtype=torch.long, device=device)
        splits[name] = (X_s, y_s)

    print(f"    Samples: {N}  -> Train {n_train}, Val {n_val}, Test {N - n_train - n_val}")
    return splits


# ═══════════════════════════════════════════════════════════════════
#  DATA: 4-class per-axis (existing approach, for comparison)
# ═══════════════════════════════════════════════════════════════════
def make_data_4class(window_size):
    """Generate per-axis 4-class data (existing approach)."""
    print(f"    Generating 4-class per-axis data (win={window_size}) ...")
    pad = window_size // 2
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
        y_s = torch.as_tensor(y_cls[idx[sl]], dtype=torch.long, device=device)
        splits[name] = (X_s, y_s)

    print(f"    Samples: {N}  -> Train {n_train}, Val {n_val}, Test {N - n_train - n_val}")
    return splits


# ═══════════════════════════════════════════════════════════════════
#  MODEL BUILDER
# ═══════════════════════════════════════════════════════════════════
def build_model(input_dim, num_classes, hidden_sizes):
    layers = []
    in_dim = input_dim
    for h in hidden_sizes:
        layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
        in_dim = h
    layers.append(nn.Linear(in_dim, num_classes))
    model = nn.Sequential(*layers).to(device)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return model


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ═══════════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════════
def acc_classify(model, X_t, y_t):
    with torch.no_grad():
        return (model(X_t).argmax(dim=-1) == y_t).float().mean().item()


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
#  16-CLASS RELAY  (processes complex signals natively)
# ═══════════════════════════════════════════════════════════════════
class QAM16ClassifyRelay(Relay):
    """Relay that classifies complex QAM16 symbols into 16 constellation points."""

    def __init__(self, model, window_size, target_power=1.0):
        self.model = model
        self.window_size = window_size
        self.target_power = target_power

    def process(self, received_signal):
        """Process complex QAM16 signal through 16-class classifier."""
        # received_signal is complex (from _process_relay which will call us
        # BUT we need to be called directly, not through _process_relay I/Q split)
        # We handle real signals by just returning them (fallback)
        if not np.iscomplexobj(received_signal):
            return received_signal

        pad = self.window_size // 2
        rx_I = np.pad(received_signal.real, pad, mode="edge")
        rx_Q = np.pad(received_signal.imag, pad, mode="edge")

        windows = []
        for i in range(pad, len(received_signal) + pad):
            i_win = rx_I[i - pad: i + pad + 1]
            q_win = rx_Q[i - pad: i + pad + 1]
            windows.append(np.concatenate([i_win, q_win]))

        windows_np = np.array(windows, dtype=np.float32)

        with torch.no_grad():
            inp = torch.as_tensor(windows_np, dtype=torch.float32, device=device)
            logits = self.model(inp)
            indices = logits.argmax(dim=-1).cpu().numpy()

        processed = CONSTELLATION_16[indices]  # complex

        pwr = np.mean(np.abs(processed) ** 2)
        if pwr > 0:
            processed = processed * np.sqrt(self.target_power / pwr)
        return processed


# ═══════════════════════════════════════════════════════════════════
#  CUSTOM SIMULATION (bypasses _process_relay I/Q split)
# ═══════════════════════════════════════════════════════════════════
def simulate_16class(relay, num_bits, snr_db, seed=None):
    """Two-hop transmission with 16-class relay processing complex signals."""
    source = Source(seed=seed, modulation=MODULATION)
    destination = Destination(modulation=MODULATION)

    tx_bits, tx_symbols = source.transmit(num_bits)

    # Hop 1
    np.random.seed(seed * 1000 + 1 if seed else None)
    rx_relay = awgn_channel(tx_symbols, snr_db)

    # Relay: 16-class classification on complex signal directly
    relay_out = relay.process(rx_relay)

    # Hop 2
    np.random.seed(seed * 1000 + 2 if seed else None)
    rx_dest = awgn_channel(relay_out, snr_db)

    if isinstance(rx_dest, tuple):
        rx_dest = rx_dest[0]
    rx_bits = destination.receive(rx_dest)

    return calculate_ber(tx_bits, rx_bits)


def run_mc_16class(relay, snr_range, num_bits, num_trials):
    """Monte-Carlo over many trials."""
    snr_values = np.array(snr_range)
    ber_values = np.zeros(len(snr_values))
    ber_trials = np.zeros((len(snr_values), num_trials))

    for i, snr_db in enumerate(snr_values):
        for trial in range(num_trials):
            ber, _ = simulate_16class(relay, num_bits, snr_db, seed=trial)
            ber_trials[i, trial] = ber
        ber_values[i] = np.mean(ber_trials[i])
    return snr_values, ber_values, ber_trials


# ═══════════════════════════════════════════════════════════════════
#  VARIANT DEFINITIONS
# ═══════════════════════════════════════════════════════════════════
#  (name, num_classes, window_size, hidden_sizes)
VARIANTS = [
    ("4-class per-axis (w5)",      4,  5, [24]),
    ("16-class 2D (w1, h24)",     16,  1, [24]),
    ("16-class 2D (w5, h24)",     16,  5, [24]),
    ("16-class 2D (w1, h64)",     16,  1, [64]),
    ("16-class 2D (w1, 2×64)",    16,  1, [64, 64]),
]


# ═══════════════════════════════════════════════════════════════════
#  TRAIN ALL
# ═══════════════════════════════════════════════════════════════════
trained = {}

for vname, n_cls, win, hsizes in VARIANTS:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if n_cls == 16:
        splits = make_data_16class(win)
        input_dim = 2 * win
    else:
        splits = make_data_4class(win)
        input_dim = win

    model = build_model(input_dim, n_cls, hsizes)
    state, hist, best_ep, te_acc = train_variant(vname, model, splits)
    model.load_state_dict(state)
    trained[vname] = {
        "model": model, "hist": hist, "best_ep": best_ep,
        "te_acc": te_acc, "params": count_params(model),
        "num_classes": n_cls, "window_size": win,
        "hidden_sizes": hsizes,
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
    n_cls = info["num_classes"]
    win = info["window_size"]

    if n_cls == 16:
        # 16-class: use custom simulation (no I/Q split)
        relay = QAM16ClassifyRelay(model, win)
        snrs, bers, trials = run_mc_16class(
            relay, EVAL_SNRS, MC_BITS, MC_TRIALS)
    else:
        # 4-class per-axis: standard pipeline (I/Q split happens in runner)
        relay = MinimalGenAIRelay(
            window_size=win, hidden_size=info["hidden_sizes"][0],
            prefer_gpu=False,
            classify=True, training_modulation=MODULATION,
        )
        relay._torch_model = model
        relay._use_torch = True
        relay.is_trained = True
        snrs, bers, trials = run_monte_carlo(
            relay, EVAL_SNRS, MC_BITS, MC_TRIALS, modulation=MODULATION)

    lo = np.percentile(trials, 2.5, axis=1)
    hi = np.percentile(trials, 97.5, axis=1)
    ber_results[vname] = {
        "ber_mean": bers, "ber_trials": trials,
        "ci95_lo": lo, "ci95_hi": hi,
    }
    print(f"  {vname:30s} -> BER@20dB={bers[-1]:.6f}  avg={np.mean(bers):.6f}")

# Baselines
print("  Computing AF & DF baselines ...")
_, bers_af, _ = run_monte_carlo(
    AmplifyAndForwardRelay(), EVAL_SNRS, MC_BITS, MC_TRIALS, modulation=MODULATION)
_, bers_df, _ = run_monte_carlo(
    DecodeAndForwardRelay(), EVAL_SNRS, MC_BITS, MC_TRIALS, modulation=MODULATION)
print(f"  {'AF':30s} -> BER@20dB={bers_af[-1]:.6f}  avg={np.mean(bers_af):.6f}")
print(f"  {'DF':30s} -> BER@20dB={bers_df[-1]:.6f}  avg={np.mean(bers_df):.6f}")


# ═══════════════════════════════════════════════════════════════════
#  JSON PERSISTENCE
# ═══════════════════════════════════════════════════════════════════
results_json = {
    "timestamp": datetime.datetime.now().isoformat(),
    "modulation": MODULATION,
    "experiment": "16class_vs_4class_qam16",
    "config": {
        "num_samples": NUM_SAMPLES, "training_snrs": TRAINING_SNRS,
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
        "batch_size": BATCH_SIZE, "lr": LR, "seed": SEED,
        "mc_bits": MC_BITS, "mc_trials": MC_TRIALS,
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
        "num_classes": info["num_classes"],
        "window_size": info["window_size"],
        "hidden_sizes": info["hidden_sizes"],
        "best_epoch": info["best_ep"],
        "test_accuracy": float(info["te_acc"]),
        "ber_mean": [float(b) for b in br["ber_mean"]],
        "ber_ci95_lo": [float(b) for b in br["ci95_lo"]],
        "ber_ci95_hi": [float(b) for b in br["ci95_hi"]],
        "history": info["hist"],
    }
json_path = os.path.join(OUT_DIR, "classify_16class_qam16.json")
with open(json_path, "w") as f:
    json.dump(results_json, f, indent=2)
print(f"\n  JSON -> {json_path}")


# ═══════════════════════════════════════════════════════════════════
#  CHART STYLE
# ═══════════════════════════════════════════════════════════════════
COLORS = {
    "4-class per-axis (w5)":      "#D55E00",
    "16-class 2D (w1, h24)":      "#56B4E9",
    "16-class 2D (w5, h24)":      "#CC79A7",
    "16-class 2D (w1, h64)":      "#009E73",
    "16-class 2D (w1, 2×64)":     "#0072B2",
    "AF": "#888888", "DF": "#333333",
}
MARKERS = {
    "4-class per-axis (w5)":      "o",
    "16-class 2D (w1, h24)":      "s",
    "16-class 2D (w5, h24)":      "^",
    "16-class 2D (w1, h64)":      "D",
    "16-class 2D (w1, 2×64)":     "h",
    "AF": "<", "DF": ">",
}
LSTYLES = {
    "4-class per-axis (w5)":      "--",
    "16-class 2D (w1, h24)":      "-",
    "16-class 2D (w5, h24)":      "-",
    "16-class 2D (w1, h64)":      "-",
    "16-class 2D (w1, 2×64)":     "-",
    "AF": ":", "DF": ":",
}
vnames = [v[0] for v in VARIANTS]


# ═══════════════════════════════════════════════════════════════════
#  CHART 1 -- BER vs SNR
# ═══════════════════════════════════════════════════════════════════
print("  Generating BER chart ...")
fig, ax = plt.subplots(figsize=(10, 6))

ax.semilogy(EVAL_SNRS, bers_af, color=COLORS["AF"], marker="<",
            markersize=5, linewidth=1.0, linestyle=":", label="AF baseline")
ax.semilogy(EVAL_SNRS, bers_df, color=COLORS["DF"], marker=">",
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
ax.set_title(f"16-Class vs 4-Class Classification -- {MODULATION.upper()}", fontsize=16)
ax.tick_params(labelsize=12)
ax.legend(fontsize=9, loc="upper right", framealpha=0.9)

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
fig.savefig(os.path.join(OUT_DIR, "ber_16class_vs_4class_qam16.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  BER chart saved.")


# ═══════════════════════════════════════════════════════════════════
#  CHART 2 -- Waterfall (BER@20dB)
# ═══════════════════════════════════════════════════════════════════
print("  Generating waterfall chart ...")
fig, ax = plt.subplots(figsize=(10, 5))
snr_20_idx = EVAL_SNRS.index(20)
ber_at_20 = [ber_results[vn]["ber_mean"][snr_20_idx] for vn in vnames]

bars = ax.bar(range(len(vnames)), ber_at_20,
              color=[COLORS[vn] for vn in vnames], edgecolor="black", linewidth=0.5)
ax.axhline(y=bers_df[snr_20_idx], color=COLORS["DF"], linewidth=1.5, linestyle="--",
           label=f"DF = {bers_df[snr_20_idx]:.5f}")
ax.axhline(y=bers_af[snr_20_idx], color=COLORS["AF"], linewidth=1.0, linestyle=":",
           label=f"AF = {bers_af[snr_20_idx]:.5f}", alpha=0.6)

for i, (b, v) in enumerate(zip(bars, ber_at_20)):
    ax.text(i, v + max(ber_at_20) * 0.02, f"{v:.4f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold")

short_labels = [vn for vn in vnames]
ax.set_xticks(range(len(vnames)))
ax.set_xticklabels(short_labels, fontsize=9, rotation=20, ha="right")
ax.set_ylabel("BER @ 20 dB", fontsize=14)
ax.set_title(f"BER @ 20 dB — 16-Class vs 4-Class, {MODULATION.upper()}", fontsize=15)
ax.legend(fontsize=11, loc="upper right")
ax.grid(True, axis="y", linewidth=0.4, alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "waterfall_16class_qam16.png"), dpi=150)
plt.close(fig)
print(f"  Waterfall chart saved.")


# ═══════════════════════════════════════════════════════════════════
#  CHART 3 -- Accuracy vs Epoch
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
axes[2].legend(fontsize=8, loc="lower right", framealpha=0.9)
fig.suptitle(f"Training Accuracy — 16-Class vs 4-Class, {MODULATION.upper()}",
             fontsize=16, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "accuracy_16class_qam16.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Accuracy chart saved.")


# ═══════════════════════════════════════════════════════════════════
#  CHART 4 -- Heatmap
# ═══════════════════════════════════════════════════════════════════
print("  Generating heatmap ...")
ber_matrix = np.array([ber_results[vn]["ber_mean"] for vn in vnames])
fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(ber_matrix, aspect="auto", cmap="RdYlGn_r")
ax.set_xticks(range(len(EVAL_SNRS)))
ax.set_xticklabels([str(s) for s in EVAL_SNRS], fontsize=11)
ax.set_yticks(range(len(vnames)))
ax.set_yticklabels(vnames, fontsize=9)
ax.set_xlabel("SNR (dB)", fontsize=14)
ax.set_title(f"BER Heatmap — 16-Class vs 4-Class, {MODULATION.upper()}", fontsize=15)
for i in range(len(vnames)):
    for j in range(len(EVAL_SNRS)):
        v = ber_matrix[i, j]
        txt = f"{v:.3f}" if v >= 0.01 else f"{v:.4f}"
        clr = "white" if v > 0.25 else "black"
        ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=clr)
fig.colorbar(im, ax=ax, label="BER", shrink=0.8)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "heatmap_16class_qam16.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Heatmap saved.")


# ═══════════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  BER SUMMARY TABLE")
print("=" * 70)
key_snrs = [4, 10, 16, 20]
header = f"  {'Variant':30s}" + "".join(f"  {s:6d}dB" for s in key_snrs) + f"  {'AvgBER':>10s}"
print(header)
print("  " + "-" * (len(header) - 2))
for vn in vnames:
    br = ber_results[vn]["ber_mean"]
    row = f"  {vn:30s}"
    for s in key_snrs:
        if s in EVAL_SNRS:
            row += f"  {br[EVAL_SNRS.index(s)]:8.5f}"
    row += f"  {np.mean(br):10.6f}"
    print(row)
for bname, bbers in [("AF", bers_af), ("DF", bers_df)]:
    row = f"  {bname:30s}"
    for s in key_snrs:
        if s in EVAL_SNRS:
            row += f"  {bbers[EVAL_SNRS.index(s)]:8.5f}"
    row += f"  {np.mean(bbers):10.6f}"
    print(row)

print("\n  GAP vs DF @ 20 dB:")
df_20 = bers_df[snr_20_idx]
for vn in vnames:
    mlp_20 = ber_results[vn]["ber_mean"][snr_20_idx]
    gap = mlp_20 - df_20
    print(f"    {vn:30s}  BER={mlp_20:.5f}  gap={gap:+.5f}")

print("\n" + "=" * 70)
print("  FINAL SUMMARY")
print("=" * 70)
for vn in vnames:
    info = trained[vn]
    br = ber_results[vn]
    print(f"  {vn:30s} | {info['num_classes']:2d}-cls | {info['params']:5d}p | "
          f"best_ep={info['best_ep']:3d} | te_acc={info['te_acc']:.4f} | "
          f"ber@20={br['ber_mean'][snr_20_idx]:.5f}")
print(f"  {'DF':30s} | {'--':>5s} | {'---':>5s}p | {'---':>8s} | "
      f"{'---':>10s} | ber@20={df_20:.5f}")
print(f"\n  All outputs in: {OUT_DIR}")
