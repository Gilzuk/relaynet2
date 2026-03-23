"""
Checkpoint 24 — End-to-End Autoencoder Transmitter / Receiver

Trains an autoencoder that jointly learns transmitter (constellation mapper)
and receiver (detector) over differentiable fading / AWGN channels.

Key design choices
──────────────────
- nn.Embedding maps message indices → latent space (trainable constellation LUT).
- BatchNorm1d enforces the average power constraint with stable running stats.
- **Curriculum training**: starts at high SNR (easy) then gradually widens to
  the full SNR range so the constellation structure forms first.
- Deeper 3-layer MLPs; receiver uses **residual blocks** with LayerNorm.
- **Label smoothing** (0.1) for better low-SNR generalisation.
- **Exact BER** via Hamming-distance bit comparison (not the SER/log2(M)
  approximation).
- Multi-trial BER evaluation with 95 % confidence intervals.
- Constellation quality metrics: d_min, PAPR, centroid.
- Comparative two-hop relay benchmark against AF / DF / MLP.

Usage
─────
    python checkpoints/checkpoint_24_e2e_transmitter.py
    python checkpoints/checkpoint_24_e2e_transmitter.py --epochs 20000
    python checkpoints/checkpoint_24_e2e_transmitter.py --save-weights trained_weights/e2e.pt
"""

import os
import sys
import math
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure repo root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Force UTF-8 for Windows console
for _stream in (sys.stdout, sys.stderr):
    if _stream and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


# ═══════════════════════════════════════════════════════════════════
# Helper: exact BER via Hamming distance
# ═══════════════════════════════════════════════════════════════════

def _messages_to_bits(indices: torch.Tensor, bits_per_symbol: int) -> torch.Tensor:
    """Convert integer message indices to binary representation.

    Returns a (batch, bits_per_symbol) uint8 tensor.
    """
    # Right-shift and mask each bit position
    shifts = torch.arange(bits_per_symbol - 1, -1, -1, device=indices.device)
    return ((indices.unsqueeze(1) >> shifts) & 1).to(torch.uint8)


def exact_ber(true_indices: torch.Tensor, pred_indices: torch.Tensor,
              bits_per_symbol: int) -> float:
    """Compute exact BER via Hamming distance between bit representations."""
    true_bits = _messages_to_bits(true_indices, bits_per_symbol)
    pred_bits = _messages_to_bits(pred_indices, bits_per_symbol)
    return (true_bits != pred_bits).float().mean().item()


# ═══════════════════════════════════════════════════════════════════
# Model definitions
# ═══════════════════════════════════════════════════════════════════

class E2ETransmitter(nn.Module):
    """Learned constellation mapper.

    Maps each message index (0 … M-1) to a 2-D point (I, Q) that satisfies
    the average-power constraint via BatchNorm1d (tracks running statistics
    for stable inference).

    Parameters
    ----------
    M : int
        Constellation size (e.g. 16 for 16-ary).
    n_complex_uses : int
        Number of complex channel uses per symbol.
    hidden_dim : int
        Width of the hidden layers.
    """

    def __init__(self, M=16, n_complex_uses=1, hidden_dim=64):
        super().__init__()
        self.M = M
        self.n_real_dims = 2 * n_complex_uses

        # Learnable constellation lookup table
        self.embedding = nn.Embedding(M, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_real_dims),
        )

        # Average-power normalisation with running stats for inference
        self.norm = nn.BatchNorm1d(self.n_real_dims, affine=False)

    def forward(self, message_indices):
        """Encode message indices to normalised transmitted symbols.

        Parameters
        ----------
        message_indices : Tensor of shape (batch,)
            Integer message labels in [0, M).

        Returns
        -------
        x : Tensor of shape (batch, n_real_dims)
            Power-normalised transmitted signal.
        """
        z = self.embedding(message_indices)
        raw_x = self.mlp(z)
        x = self.norm(raw_x)
        return x


# ── Differentiable channels ─────────────────────────────────────────

class DifferentiableRayleighChannel(nn.Module):
    """Differentiable Rayleigh flat-fading channel.

    Implements  y = h·x + w  with explicit complex multiplication,
    where h ~ CN(0, 1) and w ~ CN(0, σ²).

    All random tensors are created on the same device as the input signal.

    Parameters
    ----------
    perfect_csi : bool
        If True (default), the channel coefficients *h* are returned to
        the receiver (perfect CSI).  If False, only *y* is returned and
        *h* is replaced with zeros.
    """

    def __init__(self, perfect_csi=True):
        super().__init__()
        self.perfect_csi = perfect_csi

    def forward(self, x, snr_linear):
        """Forward pass through the Rayleigh fading channel.

        Parameters
        ----------
        x : Tensor of shape (batch, 2)
            Transmitted signal [I, Q].
        snr_linear : float
            Linear-scale SNR.

        Returns
        -------
        y : Tensor of shape (batch, 2)
            Received signal.
        h : Tensor of shape (batch, 2)
            Channel coefficients (zeros when perfect_csi is False).
        """
        device = x.device
        batch_size = x.shape[0]

        # Rayleigh fading: h ~ CN(0,1)  →  each component ~ N(0, 0.5)
        h_r = torch.randn(batch_size, device=device) * math.sqrt(0.5)
        h_i = torch.randn(batch_size, device=device) * math.sqrt(0.5)

        x_r, x_i = x[:, 0], x[:, 1]

        # Complex multiplication: y_faded = h * x
        y_r = h_r * x_r - h_i * x_i
        y_i = h_r * x_i + h_i * x_r

        # AWGN noise
        noise_std = math.sqrt(1.0 / snr_linear)
        w_r = torch.randn(batch_size, device=device) * noise_std
        w_i = torch.randn(batch_size, device=device) * noise_std

        y = torch.stack([y_r + w_r, y_i + w_i], dim=1)

        if self.perfect_csi:
            h = torch.stack([h_r, h_i], dim=1)
        else:
            h = torch.zeros(batch_size, 2, device=device)

        return y, h


class DifferentiableAWGNChannel(nn.Module):
    """Differentiable AWGN channel (no fading).

    Implements  y = x + w  where w ~ CN(0, σ²).  Since there is no fading,
    the CSI vector returned is always [1, 0] (identity channel).
    """

    def forward(self, x, snr_linear):
        """Forward pass through the AWGN channel.

        Parameters
        ----------
        x : Tensor of shape (batch, 2)
        snr_linear : float

        Returns
        -------
        y : Tensor of shape (batch, 2)
        h : Tensor of shape (batch, 2)
            Always [1, 0] (no fading).
        """
        device = x.device
        batch_size = x.shape[0]

        noise_std = math.sqrt(1.0 / snr_linear)
        w = torch.randn_like(x) * noise_std
        y = x + w

        # Identity channel: h = 1 + 0j
        h = torch.zeros(batch_size, 2, device=device)
        h[:, 0] = 1.0

        return y, h


# ── Receiver with residual blocks ──────────────────────────────────

class _ResBlock(nn.Module):
    """Pre-norm residual block: LayerNorm → ReLU → Linear → LayerNorm → ReLU → Linear + skip."""

    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.block(x)


class E2EReceiver(nn.Module):
    """Learned detector with residual connections.

    Concatenates received signal *y* with channel state *h* (perfect CSI)
    and outputs logits for each of the M messages.

    Architecture: Linear projection → 2 × ResBlock → Linear head.

    Parameters
    ----------
    M : int
        Constellation size.
    n_complex_uses : int
        Complex channel uses per symbol.
    hidden_dim : int
        Width of the hidden layers.
    """

    def __init__(self, M=16, n_complex_uses=1, hidden_dim=64):
        super().__init__()
        self.M = M
        n_real_dims = 2 * n_complex_uses
        input_dim = n_real_dims * 2  # y + h concatenated

        self.proj = nn.Linear(input_dim, hidden_dim)
        self.res1 = _ResBlock(hidden_dim)
        self.res2 = _ResBlock(hidden_dim)
        self.head = nn.Linear(hidden_dim, M)
        # No Softmax — CrossEntropyLoss expects raw logits

    def forward(self, y, h):
        """Decode received signal to message logits.

        Parameters
        ----------
        y : Tensor of shape (batch, 2)
        h : Tensor of shape (batch, 2)

        Returns
        -------
        logits : Tensor of shape (batch, M)
        """
        u = torch.cat([y, h], dim=1)
        z = torch.relu(self.proj(u))
        z = self.res1(z)
        z = self.res2(z)
        return self.head(z)


# ═══════════════════════════════════════════════════════════════════
# Training (with curriculum)
# ═══════════════════════════════════════════════════════════════════

def train_e2e(transmitter, channel, receiver, *,
              epochs=10_000, batch_size=1024, lr=0.01,
              snr_range_db=(0.0, 20.0), seed=42, device="cpu",
              grad_clip=1.0, label_smoothing=0.1,
              curriculum=True, verbose=True):
    """Train the E2E autoencoder over a range of SNR values.

    When *curriculum* is True the training proceeds in three phases:
      1. **Warm-up** (first 20 % of epochs): train at high SNR only
         (top quarter of the range) so the network finds a clean
         constellation geometry.
      2. **Widening** (20–50 %): linearly expand the SNR range from
         the high-SNR region down to the full range.
      3. **Full range** (50–100 %): train over the entire SNR range.

    Parameters
    ----------
    transmitter : E2ETransmitter
    channel : DifferentiableRayleighChannel or DifferentiableAWGNChannel
    receiver : E2EReceiver
    epochs : int
    batch_size : int
    lr : float
        Initial learning rate.
    snr_range_db : tuple of (float, float)
        Full target SNR range.
    seed : int
    device : str or torch.device
    grad_clip : float
        Max gradient norm.
    label_smoothing : float
        Label smoothing factor for CrossEntropyLoss (0 = off).
    curriculum : bool
        Enable curriculum training schedule.
    verbose : bool

    Returns
    -------
    loss_history : list of float
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    M = transmitter.M
    all_params = list(transmitter.parameters()) + list(receiver.parameters())
    optimizer = optim.Adam(all_params, lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(epochs // 4, 1), T_mult=2, eta_min=1e-5
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    snr_lo, snr_hi = snr_range_db
    loss_history = []

    phase1_end = int(0.20 * epochs) if curriculum else 0
    phase2_end = int(0.50 * epochs) if curriculum else 0

    for epoch in range(epochs):
        optimizer.zero_grad()

        # ── Curriculum SNR schedule ─────────────────────────────
        if curriculum and epoch < phase1_end:
            # Phase 1: high SNR only (top quarter)
            cur_lo = snr_lo + 0.75 * (snr_hi - snr_lo)
            cur_hi = snr_hi
        elif curriculum and epoch < phase2_end:
            # Phase 2: linearly widen toward full range
            progress = (epoch - phase1_end) / max(phase2_end - phase1_end, 1)
            cur_lo = snr_lo + (1.0 - progress) * 0.75 * (snr_hi - snr_lo)
            cur_hi = snr_hi
        else:
            # Phase 3 (or no curriculum): full range
            cur_lo = snr_lo
            cur_hi = snr_hi

        # Random messages
        messages = torch.randint(0, M, (batch_size,), device=device)

        # Random SNR per batch (uniform in dB within current range)
        snr_db = np.random.uniform(cur_lo, cur_hi)
        snr_lin = 10 ** (snr_db / 10.0)

        # Forward pass
        x = transmitter(messages)
        y, h = channel(x, snr_lin)
        logits = receiver(y, h)

        loss = criterion(logits, messages)
        loss.backward()
        nn.utils.clip_grad_norm_(all_params, grad_clip)
        optimizer.step()
        scheduler.step(epoch + epoch / epochs)  # fractional epoch for warm restarts

        loss_val = loss.item()
        loss_history.append(loss_val)

        if verbose and epoch % 500 == 0:
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                acc = (preds == messages).float().mean().item()
            lr_now = scheduler.get_last_lr()[0]
            phase = ("warmup" if epoch < phase1_end else
                     "widen" if epoch < phase2_end else "full")
            print(f"  Epoch {epoch:>5d}/{epochs} | Loss {loss_val:.4f} | "
                  f"Acc {acc:.4f} | SNR {snr_db:5.1f} dB | "
                  f"lr {lr_now:.2e} | [{phase}]")

    return loss_history


# ═══════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_ber(transmitter, channel, receiver, *,
                 snr_range_db=None, num_symbols=50_000,
                 num_trials=5, device="cpu"):
    """Compute BER vs SNR using Monte-Carlo symbol transmission.

    Uses exact Hamming-distance BER, not the SER/log2(M) approximation.
    Runs multiple independent trials and returns per-trial data for
    confidence-interval computation.

    Parameters
    ----------
    snr_range_db : array-like
        SNR values to evaluate. Default: 0-25 dB in 1-dB steps.
    num_symbols : int
        Number of symbols per trial per SNR point.
    num_trials : int
        Independent evaluation trials.

    Returns
    -------
    snr_values : ndarray
    mean_ber : ndarray
    ber_trials : ndarray, shape (n_snr, num_trials)
    """
    if snr_range_db is None:
        snr_range_db = np.arange(0, 26, 1.0)
    snr_range_db = np.asarray(snr_range_db, dtype=float)

    M = transmitter.M
    bits_per_symbol = int(math.log2(M))

    transmitter.eval()
    receiver.eval()

    ber_trials = np.zeros((len(snr_range_db), num_trials))

    for t in range(num_trials):
        for i, snr_db in enumerate(snr_range_db):
            snr_lin = 10 ** (snr_db / 10.0)
            messages = torch.randint(0, M, (num_symbols,), device=device)

            x = transmitter(messages)
            y, h = channel(x, snr_lin)
            logits = receiver(y, h)
            preds = torch.argmax(logits, dim=1)

            ber_trials[i, t] = exact_ber(messages, preds, bits_per_symbol)

    mean_ber = ber_trials.mean(axis=1)

    transmitter.train()
    receiver.train()

    return snr_range_db, mean_ber, ber_trials


# ═══════════════════════════════════════════════════════════════════
# Constellation analysis
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def constellation_metrics(transmitter):
    """Compute quality metrics for the learned constellation.

    Returns
    -------
    metrics : dict
        ``d_min``  - minimum Euclidean distance between any two points.
        ``d_mean`` - mean pairwise distance.
        ``papr``   - Peak-to-Average Power Ratio (dB).
        ``centroid`` - (I, Q) centroid of the constellation.
        ``points`` - (M, 2) numpy array of constellation coordinates.
    """
    transmitter.eval()
    M = transmitter.M
    device = next(transmitter.parameters()).device

    indices = torch.arange(M, device=device)
    points = transmitter(indices).cpu().numpy()

    # Pairwise distances
    from itertools import combinations
    dists = [np.linalg.norm(points[i] - points[j])
             for i, j in combinations(range(M), 2)]
    dists = np.array(dists)

    powers = np.sum(points ** 2, axis=1)
    avg_power = np.mean(powers)
    peak_power = np.max(powers)
    papr_db = 10 * np.log10(peak_power / avg_power) if avg_power > 0 else 0.0

    transmitter.train()

    return {
        "d_min": float(dists.min()),
        "d_mean": float(dists.mean()),
        "papr": float(papr_db),
        "centroid": tuple(points.mean(axis=0)),
        "points": points,
    }


# ═══════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def plot_constellation(transmitter, save_path=None,
                       title="Learned E2E Constellation"):
    """Scatter-plot the learned constellation with decision boundaries."""
    if not _HAS_MATPLOTLIB:
        print("[WARNING] matplotlib not installed — skipping constellation plot.")
        return None

    metrics = constellation_metrics(transmitter)
    points = metrics["points"]
    M = transmitter.M

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # ── Left panel: constellation points ────────────────────────
    ax = axes[0]
    ax.scatter(points[:, 0], points[:, 1], s=140, c="royalblue",
               edgecolors="k", zorder=3, linewidths=1.2)

    for idx in range(M):
        bits = format(idx, f"0{int(math.log2(M))}b")
        ax.annotate(f"{idx}\n{bits}",
                    (points[idx, 0], points[idx, 1]),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=7, ha="left",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7))

    ax.set_xlabel("I (In-phase)", fontsize=11)
    ax.set_ylabel("Q (Quadrature)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)

    # Add metrics annotation
    info = (f"d_min = {metrics['d_min']:.3f}\n"
            f"d_mean = {metrics['d_mean']:.3f}\n"
            f"PAPR = {metrics['papr']:.2f} dB")
    ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9))

    # ── Right panel: decision regions ───────────────────────────
    ax2 = axes[1]
    margin = 0.5
    x_lim = (points[:, 0].min() - margin, points[:, 0].max() + margin)
    y_lim = (points[:, 1].min() - margin, points[:, 1].max() + margin)

    grid_res = 200
    xx, yy = np.meshgrid(np.linspace(*x_lim, grid_res),
                         np.linspace(*y_lim, grid_res))
    grid_pts = np.column_stack([xx.ravel(), yy.ravel()])

    # Nearest constellation point for each grid point
    dists = np.linalg.norm(grid_pts[:, None, :] - points[None, :, :], axis=2)
    regions = np.argmin(dists, axis=1).reshape(xx.shape)

    cmap = plt.colormaps.get_cmap("tab20").resampled(M)
    ax2.contourf(xx, yy, regions, levels=np.arange(-0.5, M, 1),
                 cmap=cmap, alpha=0.35)
    ax2.contour(xx, yy, regions, levels=np.arange(-0.5, M, 1),
                colors="gray", linewidths=0.5, alpha=0.5)
    ax2.scatter(points[:, 0], points[:, 1], s=100, c="k", marker="x",
                zorder=3, linewidths=2)

    ax2.set_xlabel("I (In-phase)", fontsize=11)
    ax2.set_ylabel("Q (Quadrature)", fontsize=11)
    ax2.set_title("Decision Regions (Nearest Neighbour)", fontsize=12,
                  fontweight="bold")
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Constellation plot saved to: {save_path}")

    return fig


def plot_ber_with_ci(snr_values, mean_ber, ber_trials, *,
                     classical_ber=None, save_path=None, title="", M=16):
    """BER curve with 95 % confidence intervals."""
    if not _HAS_MATPLOTLIB:
        return None

    from relaynet.simulation.statistics import compute_confidence_interval

    fig, ax = plt.subplots(figsize=(10, 6))
    snr = np.asarray(snr_values)

    # E2E autoencoder
    lower, upper = compute_confidence_interval(ber_trials)
    ax.semilogy(snr, mean_ber, "o-", color="blue", lw=2, ms=6,
                label=f"E2E Autoencoder ({M}-ary)")
    ax.fill_between(snr, np.maximum(lower, 1e-8), upper,
                    alpha=0.2, color="blue")

    # Classical baseline
    if classical_ber is not None:
        ax.semilogy(snr, classical_ber, "k--", lw=1.5,
                    label=f"Classical {M}-QAM Rayleigh (approx)")

    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.set_xlabel("SNR (dB)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Bit Error Rate (BER)", fontsize=12, fontweight="bold")
    ax.set_title(title or f"E2E Autoencoder BER ({M}-ary, Rayleigh)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=11)
    ax.set_ylim([1e-6, 1])
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"BER plot saved to: {save_path}")

    return fig


# ═══════════════════════════════════════════════════════════════════
# Two-hop relay comparison
# ═══════════════════════════════════════════════════════════════════

def run_relay_comparison(results_dir, snr_range, num_bits=5000,
                         num_trials=5, seed=42):
    """Benchmark E2E relay against AF, DF, and MLP via two-hop simulation.

    Uses the project's existing ``run_monte_carlo`` pipeline so results
    are directly comparable with the master BER chart.
    """
    from relaynet.relays.af import AmplifyAndForwardRelay
    from relaynet.relays.df import DecodeAndForwardRelay
    from relaynet.relays.e2e import E2ERelay
    from relaynet.channels.fading import rayleigh_fading_channel
    from relaynet.simulation.runner import run_monte_carlo
    from relaynet.visualization.plots import plot_ber_curves

    print("\n── Two-Hop Relay Comparison (Rayleigh) ──")

    # Train E2E relay (abbreviated for comparison)
    e2e = E2ERelay(M=16, hidden_dim=64, prefer_gpu=False)
    print("  Training E2E relay (3000 epochs) ...")
    e2e.train(epochs=3000, seed=seed)
    print(f"  E2E relay trained.")

    relays = {
        "AF": AmplifyAndForwardRelay(),
        "DF": DecodeAndForwardRelay(),
        "E2E Autoencoder": e2e,
    }

    ber_dict = {}
    for name, relay in relays.items():
        print(f"  Evaluating {name} ...")
        _, ber_vals, _ = run_monte_carlo(
            relay, snr_range,
            num_bits_per_trial=num_bits,
            num_trials=num_trials,
            channel_fn=rayleigh_fading_channel,
            seed_offset=seed,
        )
        ber_dict[name] = ber_vals

    save_path = os.path.join(results_dir, "e2e_relay_comparison.png")
    plot_ber_curves(
        snr_range, ber_dict,
        title="Two-Hop Relay Comparison (Rayleigh)",
        save_path=save_path,
    )

    return ber_dict


# ═══════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="E2E Autoencoder Transmitter/Receiver (Checkpoint 24)")
    p.add_argument("--M", type=int, default=16, help="Constellation size")
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10_000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--snr-lo", type=float, default=0.0,
                   help="Training SNR lower bound (dB)")
    p.add_argument("--snr-hi", type=float, default=20.0,
                   help="Training SNR upper bound (dB)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-symbols", type=int, default=50_000)
    p.add_argument("--eval-trials", type=int, default=5)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--no-curriculum", action="store_true",
                   help="Disable curriculum training")
    p.add_argument("--save-weights", type=str, default=None)
    p.add_argument("--results-dir", type=str, default="results/e2e")
    p.add_argument("--gpu", action="store_true", default=False)
    p.add_argument("--skip-relay-comparison", action="store_true",
                   help="Skip the two-hop relay benchmark")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    t0 = time.time()

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Constellation size M = {args.M}  ({int(math.log2(args.M))} bits/symbol)")
    print(f"Training SNR range: [{args.snr_lo}, {args.snr_hi}] dB")
    print(f"Epochs: {args.epochs},  Batch: {args.batch_size},  "
          f"Label smoothing: {args.label_smoothing}")
    print(f"Curriculum: {'ON' if not args.no_curriculum else 'OFF'}")
    print()

    # ── Build models ────────────────────────────────────────────────
    transmitter = E2ETransmitter(M=args.M, hidden_dim=args.hidden_dim).to(device)
    channel = DifferentiableRayleighChannel(perfect_csi=True).to(device)
    receiver = E2EReceiver(M=args.M, hidden_dim=args.hidden_dim).to(device)

    tx_params = sum(p.numel() for p in transmitter.parameters())
    rx_params = sum(p.numel() for p in receiver.parameters())
    print(f"Transmitter parameters: {tx_params:,}")
    print(f"Receiver parameters:    {rx_params:,}")
    print(f"Total E2E parameters:   {tx_params + rx_params:,}")
    print()

    # ── Train ───────────────────────────────────────────────────────
    print("── Training ──")
    loss_history = train_e2e(
        transmitter, channel, receiver,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        snr_range_db=(args.snr_lo, args.snr_hi),
        seed=args.seed, device=device,
        label_smoothing=args.label_smoothing,
        curriculum=not args.no_curriculum,
    )

    # ── Constellation quality ───────────────────────────────────────
    metrics = constellation_metrics(transmitter)
    print(f"\n── Constellation Metrics ──")
    print(f"  d_min  = {metrics['d_min']:.4f}  (minimum inter-point distance)")
    print(f"  d_mean = {metrics['d_mean']:.4f}  (mean pairwise distance)")
    print(f"  PAPR   = {metrics['papr']:.2f} dB")
    print(f"  Centroid = ({metrics['centroid'][0]:.4f}, {metrics['centroid'][1]:.4f})")

    # ── Evaluate BER vs SNR ─────────────────────────────────────────
    print(f"\n── BER Evaluation ({args.eval_trials} trials x "
          f"{args.eval_symbols:,} symbols) ──")
    snr_values, mean_ber, ber_trials = evaluate_ber(
        transmitter, channel, receiver,
        snr_range_db=np.arange(0, 26, 1.0),
        num_symbols=args.eval_symbols,
        num_trials=args.eval_trials,
        device=device,
    )

    for snr, ber in zip(snr_values, mean_ber):
        if ber > 0:
            print(f"  SNR {snr:5.1f} dB  ->  BER = {ber:.6f}")

    # ── Save weights ────────────────────────────────────────────────
    if args.save_weights:
        state = {
            "type": "E2EAutoencoder",
            "config": {"M": args.M, "hidden_dim": args.hidden_dim},
            "transmitter": transmitter.state_dict(),
            "receiver": receiver.state_dict(),
            "constellation": metrics["points"].tolist(),
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.save_weights)), exist_ok=True)
        torch.save(state, args.save_weights)
        print(f"\nWeights saved to: {args.save_weights}")

    # ── Plots ───────────────────────────────────────────────────────
    os.makedirs(args.results_dir, exist_ok=True)

    # 1. Constellation scatter + decision regions
    plot_constellation(
        transmitter,
        save_path=os.path.join(args.results_dir, "e2e_constellation.png"),
        title=f"Learned {args.M}-ary Constellation (E2E Autoencoder)",
    )

    # 2. BER curve with confidence intervals
    if _HAS_MATPLOTLIB:
        snr_lin = 10 ** (snr_values / 10.0)
        ber_classical = 3.0 / (2.0 * math.log2(args.M)) * (
            1.0 - np.sqrt(snr_lin / (snr_lin + 1.0))
        )
        plot_ber_with_ci(
            snr_values, mean_ber, ber_trials,
            classical_ber=ber_classical,
            save_path=os.path.join(args.results_dir, "e2e_ber_comparison.png"),
            title=f"E2E Autoencoder vs Classical {args.M}-QAM (Rayleigh)",
            M=args.M,
        )

    # 3. Training loss curve
    if _HAS_MATPLOTLIB:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(loss_history, linewidth=0.5, alpha=0.4, color="steelblue")
        # Smoothed version
        window = min(200, max(len(loss_history) // 10, 2))
        if window > 1:
            smoothed = np.convolve(loss_history,
                                   np.ones(window) / window, mode="valid")
            ax.plot(np.arange(window - 1, len(loss_history)), smoothed,
                    linewidth=2, color="darkblue",
                    label=f"Moving avg (w={window})")

        # Mark curriculum phases
        if not args.no_curriculum:
            p1 = int(0.20 * args.epochs)
            p2 = int(0.50 * args.epochs)
            ax.axvline(p1, color="green", ls="--", alpha=0.5, label="End warm-up")
            ax.axvline(p2, color="orange", ls="--", alpha=0.5, label="End widening")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.set_title("E2E Autoencoder Training Loss")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.results_dir, "e2e_training_loss.png"),
                    dpi=150, bbox_inches="tight")
        print(f"Training loss plot saved.")

    # 4. Two-hop relay comparison
    if not args.skip_relay_comparison:
        run_relay_comparison(
            args.results_dir,
            snr_range=np.arange(0, 21, 2),
            num_bits=5000,
            num_trials=5,
            seed=args.seed,
        )

    elapsed = time.time() - t0
    print(f"\n✓ Checkpoint 24 complete.  ({elapsed:.1f}s)")
