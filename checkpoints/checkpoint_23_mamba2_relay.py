"""
Checkpoint 23: Mamba-2 Relay (Structured State Space Duality)

This module implements a Mamba-2 relay based on the SSD (Structured
State Space Duality) formulation introduced by Dao & Gu (2024).

Key difference from Mamba-1 (S6, checkpoint 20):
    Mamba-1 computes the SSM recurrence *sequentially* — one time
    step at a time (a Python for-loop of length *seq_len*, each
    triggering a separate CUDA kernel). This is the root cause of
    the 4.5× training-time penalty relative to the Transformer
    (see thesis §8.3).

    Mamba-2 exploits the *dual* view: the SSM can be written as a
    structured (semi-separable) matrix multiply

        y = M · (B ⊙ u),   M_{ij} = C_i^T A_bar_{i:j} B_j

    which is executed as a **single batched matmul** over chunks of
    the sequence. This eliminates the sequential loop and allows
    GPU-parallel computation comparable to attention.

Architecture (default ~24K params to match Mamba S6):
    - Input projection  : 1 → d_model
    - SSD blocks × 2    : each with chunk-parallel SSM + SiLU gate
    - Output head        : LN → d_model → d_model//2 → 1 → Tanh

Reference:
    Dao, T. & Gu, A. (2024). "Transformers are SSMs: Generalized
    Models and Efficient Algorithms Through Structured State Space
    Duality." ICML 2024.

Author: Copilot
Date: 2026-03-15
Checkpoint: CP-23
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from checkpoint_01_channel import awgn_channel
from checkpoint_02_modulation import bpsk_modulate, calculate_ber
from checkpoint_03_nodes import Source, Relay, Destination

from relaynet.utils.activations import make_torch_activation, generate_training_targets


# ======================================================================
# Core SSD layer
# ======================================================================


class SSDLayer(nn.Module):
    """Structured State Space Duality (SSD) layer.

    Instead of the sequential recurrence of S6, SSD re-expresses the
    SSM as a structured (semi-separable) matrix multiply.  For a
    chunk of length *L*:

        M[i, j] = C_i^T  diag(exp(Δ_k A))_{k=j+1..i}  B_j   (i >= j)
        y = M · (B ⊙ u)                                       (causal)

    The matrix *M* is lower-triangular and can be built and applied
    in O(L² · N) — quadratic in chunk length but **parallel** on GPU.
    With small chunks (e.g. L ≤ 16) this is faster than L serial
    CUDA-kernel launches.

    For inter-chunk state passing we use a simple running state
    that is updated once per chunk (not per time step).
    """

    def __init__(self, d_model, d_state=16, num_heads=1,
                 dt_min=0.001, dt_max=0.1, chunk_size=8):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.num_heads = num_heads
        self.chunk_size = chunk_size

        # Selective projections (input-dependent Δ, B, C)
        self.delta_proj = nn.Linear(d_model, num_heads)
        self.B_proj = nn.Linear(d_model, d_state)
        self.C_proj = nn.Linear(d_model, d_state)

        # Diagonal state matrix A (S4D-style log-linear init for stable decay)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
        )

        # Direct feed-through D (init ≈ 1 for skip connection)
        self.D = nn.Parameter(torch.ones(d_model))

        # Value projection (input → value used in the matmul)
        self.value_proj = nn.Linear(d_model, d_model)

        self.dt_min = dt_min
        self.dt_max = dt_max

    # ------------------------------------------------------------------
    def _build_ssm_matrix(self, A_bar, B, C, L):
        """Build the lower-triangular SSM matrix M for one chunk.

        Parameters
        ----------
        A_bar : (batch, L, d_state)   – discretised diagonal A per step
        B     : (batch, L, d_state)
        C     : (batch, L, d_state)
        L     : int – chunk length

        Returns
        -------
        M : (batch, L, L)  – lower-triangular causal kernel
        """
        # Cumulative product of A_bar along time (log-space for stability)
        log_A = torch.log(A_bar.clamp(min=1e-8))          # (B, L, N)
        log_A_cum = torch.cumsum(log_A, dim=1)             # (B, L, N)

        # M[i,j] = sum_n  C[i,n] * (prod_{k=j+1}^{i} A_bar[k,n]) * B[j,n]
        #        = sum_n  C[i,n] * exp(log_A_cum[i,n] - log_A_cum[j,n]) * B[j,n]
        # Shape gymnastics: (B, L, 1, N) - (B, 1, L, N) → (B, L, L, N)
        log_decay = log_A_cum.unsqueeze(2) - log_A_cum.unsqueeze(1)  # (B, L, L, N)
        decay = torch.exp(log_decay)                                 # (B, L, L, N)

        # C[i] · decay · B[j]  →  sum over d_state
        # C: (B, L, 1, N),  B: (B, 1, L, N)
        M = torch.sum(
            C.unsqueeze(2) * decay * B.unsqueeze(1),
            dim=-1,
        )  # (B, L, L)

        # Causal mask: M[i,j] = 0 for j > i
        causal = torch.tril(torch.ones(L, L, device=M.device))
        M = M * causal.unsqueeze(0)

        return M

    # ------------------------------------------------------------------
    def forward(self, x):
        """Forward pass — chunk-parallel SSD.

        The SSM kernel M is applied to the *value* stream V via a
        single batched matmul per chunk: Y = M · V  (B, L, D).
        Inter-chunk state (B, N, D) carries across chunks.

        Parameters
        ----------
        x : (batch, seq_len, d_model)

        Returns
        -------
        (batch, seq_len, d_model)
        """
        batch, seq_len, d = x.shape
        C_size = self.chunk_size

        # Input-dependent projections
        delta_raw = self.delta_proj(x)                       # (B, T, H)
        delta = self.dt_min + (self.dt_max - self.dt_min) * torch.sigmoid(delta_raw)
        delta = delta.mean(dim=-1, keepdim=True)             # (B, T, 1)

        B = self.B_proj(x)                                   # (B, T, N)
        C = self.C_proj(x)                                   # (B, T, N)
        V = self.value_proj(x)                               # (B, T, D)

        A = -torch.exp(self.A_log)                           # (N,)
        A_bar = torch.exp(delta * A.unsqueeze(0).unsqueeze(0))  # (B, T, N)
        B_bar = delta * B                                    # (B, T, N)

        # Pad sequence to a multiple of chunk_size
        # A_bar must be padded with 1.0 (identity/no-decay) so that
        # log(A_bar) = 0 in padded positions; zero-padding would
        # create log(0) = -inf → exp(+inf) = NaN in _build_ssm_matrix.
        pad = (C_size - seq_len % C_size) % C_size
        if pad > 0:
            A_bar = torch.nn.functional.pad(A_bar, (0, 0, 0, pad), value=1.0)
            B_bar = torch.nn.functional.pad(B_bar, (0, 0, 0, pad))
            C     = torch.nn.functional.pad(C,     (0, 0, 0, pad))
            V     = torch.nn.functional.pad(V,     (0, 0, 0, pad))
            x_pad = torch.nn.functional.pad(x,     (0, 0, 0, pad))
        else:
            x_pad = x

        T_padded = A_bar.shape[1]
        num_chunks = T_padded // C_size

        # Reshape into chunks: (B, num_chunks, C_size, ...)
        A_bar_c = A_bar.reshape(batch, num_chunks, C_size, -1)
        B_bar_c = B_bar.reshape(batch, num_chunks, C_size, -1)
        C_c     = C.reshape(batch, num_chunks, C_size, -1)
        V_c     = V.reshape(batch, num_chunks, C_size, -1)
        x_c     = x_pad.reshape(batch, num_chunks, C_size, -1)

        # Process each chunk with the parallel SSD matmul
        outputs = []
        # State tracks d_model channels independently: (B, N, D)
        state = torch.zeros(batch, self.d_state, d, device=x.device)

        for ci in range(num_chunks):
            a_chunk = A_bar_c[:, ci]     # (B, L, N)
            b_chunk = B_bar_c[:, ci]     # (B, L, N)
            c_chunk = C_c[:, ci]         # (B, L, N)
            v_chunk = V_c[:, ci]         # (B, L, D)
            x_orig  = x_c[:, ci]        # (B, L, D)

            # ---- Intra-chunk: parallel SSD matrix multiply ----
            M = self._build_ssm_matrix(a_chunk, b_chunk, c_chunk, C_size)
            y_intra = torch.bmm(M, v_chunk)          # (B, L, D)

            # ---- Inter-chunk: contribution from previous state ----
            log_A_chunk = torch.log(a_chunk.clamp(min=1e-8))
            log_A_cum = torch.cumsum(log_A_chunk, dim=1)   # (B, L, N)
            cumA = torch.exp(log_A_cum)                    # (B, L, N)

            # y_inter[t,d] = sum_n C[t,n] * cumA[t,n] * state[n,d]
            y_inter = torch.einsum(
                'bln, bnd -> bld', c_chunk * cumA, state,
            )  # (B, L, D)

            # Combined output + direct feed-through D
            chunk_out = y_intra + y_inter + self.D * x_orig

            # ---- Update running state for next chunk ----
            # h(L-1) = cumA[-1]*h_init + sum_t reverseA(t)*B_bar(t) (x) v(t)
            # reverseA(t) = prod_{k=t+1}^{L-1} A_bar(k)
            reverse_cumA = torch.exp(
                log_A_cum[:, -1:, :] - log_A_cum,
            )  # (B, L, N)
            state_contrib = torch.einsum(
                'bln, bld -> bnd', reverse_cumA * b_chunk, v_chunk,
            )  # (B, N, D)
            state = cumA[:, -1, :].unsqueeze(-1) * state + state_contrib

            outputs.append(chunk_out)

        output = torch.cat(outputs, dim=1)  # (B, T_padded, d_model)
        output = output[:, :seq_len, :]     # remove padding

        return output


# ======================================================================
# Mamba-2 block (SSD + gate + residual)
# ======================================================================


class Mamba2Block(nn.Module):
    """Mamba-2 block: SSD layer with SiLU gate and residual."""

    def __init__(self, d_model, d_state=16, expand_factor=2, chunk_size=8):
        super().__init__()
        d_inner = d_model * expand_factor

        self.in_proj = nn.Linear(d_model, d_inner)
        self.gate_proj = nn.Linear(d_model, d_inner)
        self.ssd = SSDLayer(d_inner, d_state, chunk_size=chunk_size)
        self.activation = nn.SiLU()
        self.out_proj = nn.Linear(d_inner, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.norm(x)

        # Gated branch (Mamba-2 uses a parallel gate instead of a
        # sequential conv-then-SSM pipeline)
        h = self.activation(self.in_proj(x))
        h = self.ssd(h)
        gate = torch.sigmoid(self.gate_proj(x))
        x = self.out_proj(h * gate)

        return x + residual


# ======================================================================
# Full Mamba-2 relay model
# ======================================================================


class Mamba2Relay(nn.Module):
    """Mamba-2 based relay for signal denoising."""

    def __init__(self, window_size=11, d_model=32, d_state=16,
                 num_layers=2, chunk_size=8, output_activation="tanh",
                 use_input_norm=False, clip_range=None, in_channels=1):
        super().__init__()
        self.window_size = window_size
        self.d_model = d_model
        self.use_input_norm = use_input_norm

        self.input_proj = nn.Linear(in_channels, d_model)

        # Optional input LayerNorm — stabilises the distribution entering
        # the Mamba-2 blocks and prevents extreme activations.
        if use_input_norm:
            self.input_norm = nn.LayerNorm(d_model)

        self.blocks = nn.ModuleList([
            Mamba2Block(d_model, d_state, chunk_size=chunk_size)
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
            make_torch_activation(output_activation, clip_range=clip_range),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Parameters
        ----------
        x : (batch, window_size, 1)

        Returns
        -------
        (batch, 1)
        """
        x_raw = x
        x = self.input_proj(x)           # -> (B, W, d_model)
        if self.use_input_norm:
            x = self.input_norm(x)
        for block in self.blocks:
            x = block(x)
        center = self.window_size // 2
        x = x[:, center, :]              # -> (B, d_model)
        return self.output_proj(x) + x_raw[:, center, 0:1]


# ======================================================================
# Relay wrapper (same interface as MambaRelayWrapper)
# ======================================================================


class Mamba2RelayWrapper(Relay):
    """Wrapper that integrates Mamba2Relay into the relay pipeline."""

    def __init__(self, target_power=1.0, window_size=11, d_model=32,
                 d_state=16, num_layers=2, chunk_size=8, prefer_gpu=False,
                 output_activation="tanh", use_input_norm=False, clip_range=None,
                 in_channels=1):
        self.target_power = target_power
        self.window_size = window_size
        self.output_activation = output_activation
        self.use_input_norm = use_input_norm
        self.clip_range = clip_range
        self.in_channels = in_channels

        if prefer_gpu:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device("cpu")

        self.model = Mamba2Relay(
            window_size=window_size,
            d_model=d_model,
            d_state=d_state,
            num_layers=num_layers,
            chunk_size=chunk_size,
            output_activation=output_activation,
            use_input_norm=use_input_norm,
            clip_range=clip_range,
            in_channels=in_channels,
        ).to(self.device)

        self.is_trained = False
        self.num_params = sum(p.numel() for p in self.model.parameters())

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _compute_accuracy(self, X, y_true, batch_size=512):
        """Compute symbol-level accuracy."""
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y_true[i:i+batch_size]
                output = self.model(X_batch)
                correct += (torch.sign(output) == torch.sign(y_batch)).sum().item()
                total += y_batch.numel()
        return correct / total if total > 0 else 0.0

    def train(self, training_snrs=[5, 10, 15], num_samples=50000,
              epochs=100, lr=0.001, seed=None, log_timings=False,
              training_modulation="bpsk", use_rayleigh=False,
              patience=0, min_delta=1e-5, val_split=0.15):
        """Train the Mamba-2 relay.

        Returns
        -------
        history : dict
            Keys: 'train_loss', 'train_acc', 'val_acc'.
        """
        print(f"  Mamba-2 (SSD) Training Configuration:")
        print(f"    Device: {self.device}")
        print(f"    Architecture: {self.window_size}-token Mamba-2 (SSD)")
        print(f"    d_model: {self.model.d_model}, "
              f"d_state: {self.model.blocks[0].ssd.d_state}")
        print(f"    in_channels: {self.model.input_proj.in_features}")
        print(f"    Layers: {len(self.model.blocks)}")
        print(f"    Parameters: {self.num_params:,}")
        print(f"    Training SNRs: {training_snrs} dB")
        print(f"    Samples: {num_samples:,}, Epochs: {epochs}")
        if patience > 0:
            print(f"    Early stopping: patience={patience}, min_delta={min_delta}")
        if self.use_input_norm:
            print(f"    Input LayerNorm: ENABLED")
        if training_modulation != "bpsk":
            print(f"    Training modulation: {training_modulation}")

        # Generate training data
        samples_per_snr = num_samples // len(training_snrs)
        X_all, y_all = [], []

        in_channels = self.model.input_proj.in_features
        return_csi = (in_channels > 1)

        for snr in training_snrs:
            rng_seed = (42 + int(snr)) if seed is None else (seed + int(snr))
            if return_csi:
                clean, noisy, h_csi = generate_training_targets(
                    samples_per_snr, snr,
                    training_modulation=training_modulation,
                    seed=rng_seed,
                    use_rayleigh=use_rayleigh,
                    return_csi=True
                )
                features = np.column_stack([noisy, np.abs(h_csi)])
            else:
                clean, noisy = generate_training_targets(
                    samples_per_snr, snr,
                    training_modulation=training_modulation,
                    seed=rng_seed,
                    use_rayleigh=use_rayleigh,
                    return_csi=False
                )
                features = noisy.reshape(-1, 1)

            half = self.window_size // 2
            for i in range(half, len(noisy) - half):
                window = features[i - half: i + half + 1]
                X_all.append(window)
                y_all.append(clean[i])

        X_full = torch.FloatTensor(np.array(X_all)).to(self.device)
        y_full = torch.FloatTensor(np.array(y_all).reshape(-1, 1)).to(self.device)

        # Train / validation split
        n_total = len(X_full)
        n_val = max(1, int(n_total * val_split))
        perm = torch.randperm(n_total)
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        X_train, y_train = X_full[train_idx], y_full[train_idx]
        X_val,   y_val   = X_full[val_idx],   y_full[val_idx]

        print(f"  Training data: {len(X_train):,} train, {len(X_val):,} val samples")

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        batch_size = 64
        best_loss = float("inf")
        patience_counter = 0
        best_state = None

        history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

        for epoch in range(epochs):
            self.model.train()
            indices = torch.randperm(len(X_train), device=self.device)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            total_loss = 0.0
            num_batches = 0

            for i in range(0, len(X_train), batch_size):
                xb = X_shuffled[i: i + batch_size]
                yb = y_shuffled[i: i + batch_size]

                optimizer.zero_grad()
                out = self.model(xb)
                loss = criterion(out, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0,
                )
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches

            train_acc = self._compute_accuracy(X_train, y_train)
            val_acc   = self._compute_accuracy(X_val,   y_val)
            history['train_loss'].append(avg_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)

            if avg_loss < best_loss - min_delta:
                best_loss = avg_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
                print(f"    Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}, "
                      f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, "
                      f"Best: {best_loss:.6f}")

            if patience > 0 and patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

        if best_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        self.is_trained = True
        print(f"\n  Mamba-2 training complete!  "
              f"Final loss: {avg_loss:.6f}, Best: {best_loss:.6f}  "
              f"Parameters: {self.num_params:,}")

        return history

    # ------------------------------------------------------------------
    # Inference (batched — same approach as Mamba S6 / Transformer)
    # ------------------------------------------------------------------
    def process(self, received_signal):
        """Process signal through Mamba-2 (batched)."""
        if not self.is_trained:
            if isinstance(received_signal, tuple):
                return received_signal[0] * 1.5
            return received_signal * 1.5

        self.model.eval()

        # Unpack CSI if provided
        if isinstance(received_signal, tuple):
            if self.in_channels == 1:
                received_signal = received_signal[0]
            else:
                y, h_csi = received_signal
                n = len(y)
                features = np.column_stack([y, h_csi])
                pad_size = self.window_size // 2
                padded_features = np.pad(features, ((pad_size, pad_size), (0, 0)), mode='edge')
                windows = np.lib.stride_tricks.as_strided(
                    padded_features,
                    shape=(n, self.window_size, 2),
                    strides=(padded_features.strides[0], padded_features.strides[0], padded_features.strides[1]),
                ).copy()

        if not isinstance(received_signal, tuple):
            n = len(received_signal)
            pad_size = self.window_size // 2
            padded = np.pad(received_signal, pad_size, mode="edge")
            windows = np.lib.stride_tricks.as_strided(
                padded,
                shape=(n, self.window_size),
                strides=(padded.strides[0], padded.strides[0]),
            ).copy()
            windows = np.expand_dims(windows, -1)

        with torch.no_grad():
            inp = torch.as_tensor(
                windows, dtype=torch.float32, device=self.device,
            )
            processed = self.model(inp).cpu().numpy().flatten()

        current_power = np.mean(np.abs(processed) ** 2)
        if current_power > 0:
            return processed * np.sqrt(self.target_power / current_power)
        return processed

    # ------------------------------------------------------------------
    # Weight persistence
    # ------------------------------------------------------------------
    def _arch_config(self):
        """Return a dict describing the full architecture (for cache invalidation)."""
        return {
            "window_size": self.window_size,
            "d_model": self.model.d_model,
            "d_state": self.model.blocks[0].ssd.d_state,
            "num_layers": len(self.model.blocks),
            "in_channels": self.in_channels,
            "use_input_norm": self.use_input_norm,
            "output_activation": self.output_activation,
            "clip_range": self.clip_range,
            "num_params": self.num_params,
        }

    def save_weights(self, path):
        torch.save({
            "type": "Mamba2RelayWrapper",
            "model_state_dict": self.model.state_dict(),
            "config": self._arch_config(),
        }, path)

    def load_weights(self, path):
        """Load model weights if architecture matches. Returns True/False."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        saved_cfg = state.get("config", {})
        if saved_cfg != self._arch_config():
            return False
        self.model.load_state_dict(state["model_state_dict"])
        self.is_trained = True
        return True


# ======================================================================
# Standalone test
# ======================================================================


def test_mamba2_relay():
    """Smoke-test for Mamba-2 relay."""
    print("=" * 70)
    print("MAMBA-2 (SSD) RELAY")
    print("=" * 70)

    relay = Mamba2RelayWrapper(
        target_power=1.0, window_size=11,
        d_model=32, d_state=16, num_layers=2,
    )
    print(f"Parameters: {relay.num_params:,}")
    relay.train(training_snrs=[5, 10, 15], num_samples=50000,
                epochs=100, lr=0.001)

    print("\nQuick validation (10 000 bits per SNR):")
    for snr in [0, 2, 4, 6, 8, 10]:
        source = Source(seed=42)
        dest = Destination()
        tx_bits, tx_sym = source.transmit(10000)
        rx = awgn_channel(tx_sym, snr)
        relay_out = relay.process(rx)
        rx2 = awgn_channel(relay_out, snr)
        rx_bits = dest.receive(rx2)
        ber, _ = calculate_ber(tx_bits, rx_bits)
        print(f"  SNR {snr:>2} dB  BER {ber:.6f}")

    print("\n✓ Mamba-2 relay complete!")
    return True


if __name__ == "__main__":
    test_mamba2_relay()
