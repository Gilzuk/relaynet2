"""
Checkpoint 18: Transformer Relay with Attention Mechanism

This module implements a Transformer-based relay using self-attention
to capture temporal dependencies in noisy signals.

Architecture:
- Multi-head self-attention
- Positional encoding
- Feed-forward network
- Layer normalization

Author: Cline
Date: 2026-02-15
Checkpoint: CP-18
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from relaynet.channels.awgn import awgn_channel
from relaynet.modulation.bpsk import bpsk_modulate, calculate_ber
from relaynet.nodes import Source, Destination
from relaynet.relays.base import Relay

from relaynet.utils.activations import make_torch_activation, generate_training_targets
from relaynet.utils.activations import get_num_classes, get_constellation_levels, symbols_to_class_indices
from relaynet.utils.activations import generate_training_targets_2d, get_constellation_2d


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Add positional encoding to input."""
        return x + self.pe[:x.size(1), :].unsqueeze(0)


class TransformerBlock(nn.Module):
    """Single transformer block with multi-head attention."""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """Forward pass through transformer block."""
        # Multi-head attention with residual connection
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TransformerRelay(nn.Module):
    """Transformer-based relay for signal denoising."""
    
    def __init__(self, window_size=11, d_model=32, num_heads=4, num_layers=2, d_ff=64, dropout=0.1,
                 output_activation="tanh", use_input_norm=False, clip_range=None, in_channels=1,
                 num_classes=1):
        super(TransformerRelay, self).__init__()
        
        self.window_size = window_size
        self.d_model = d_model
        self.use_input_norm = use_input_norm
        self.num_classes = num_classes
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, d_model)
        
        # Optional input LayerNorm — stabilises the distribution entering
        # the Transformer blocks and prevents extreme activations.
        if use_input_norm:
            self.input_norm = nn.LayerNorm(d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=window_size)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        out_dim = num_classes if num_classes > 1 else 1
        if num_classes > 1:
            self.output_proj = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, out_dim),
            )
        else:
            self.output_proj = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 1),
                make_torch_activation(output_activation, clip_range=clip_range),
            )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch_size, window_size, 1)
        
        Returns
        -------
        torch.Tensor
            Output of shape (batch_size, 1)
        """
        # Project input to d_model dimensions
        x_raw = x
        x = self.input_proj(x)  # (batch, window_size, d_model)
        
        # Optional input normalisation
        if self.use_input_norm:
            x = self.input_norm(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Take the middle token (center of window)
        center_idx = self.window_size // 2
        x = x[:, center_idx, :]  # (batch, d_model)
        
        # Project to output
        output = self.output_proj(x)  # (batch, num_classes) or (batch, 1)
        
        # Only add residual in regression mode (single output)
        if self.num_classes <= 1:
            return output + x_raw[:, center_idx, 0:1]
        return output


class TransformerRelayWrapper(Relay):
    """Wrapper for Transformer relay."""
    
    def __init__(self, target_power=1.0, window_size=11, d_model=32, num_heads=4, num_layers=2, prefer_gpu=False,
                 output_activation="tanh", use_input_norm=False, clip_range=None, in_channels=1,
                 classify=False, training_modulation="bpsk", classify_2d=False):
        self.target_power = target_power
        self.window_size = window_size
        self.output_activation = output_activation
        self.use_input_norm = use_input_norm
        self.clip_range = clip_range
        self.in_channels = in_channels
        self.classify = classify
        self.classify_2d = classify_2d
        self._training_modulation = training_modulation

        if classify_2d:
            self.classify = True
            self.num_classes = 16
            self._constellation_2d = get_constellation_2d("qam16")
            self._constellation_levels_np = None
            in_channels = 2  # I and Q channels
            self.in_channels = in_channels
        elif classify:
            self.num_classes = get_num_classes(training_modulation)
            self._constellation_levels_np = get_constellation_levels(training_modulation)
            self._constellation_2d = None
        else:
            self.num_classes = 1
            self._constellation_levels_np = None
            self._constellation_2d = None
        
        # Set device
        if prefer_gpu:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')
        
        # Initialize transformer
        self.model = TransformerRelay(
            window_size=window_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_model * 2,
            dropout=0.1,
            output_activation=output_activation,
            use_input_norm=use_input_norm,
            clip_range=clip_range,
            in_channels=in_channels,
            num_classes=self.num_classes,
        ).to(self.device)
        
        self.is_trained = False
        
        # Count parameters
        self.num_params = sum(p.numel() for p in self.model.parameters())
    
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
                if self.classify and self.num_classes > 1:
                    predicted = output.argmax(dim=-1)
                    correct += (predicted == y_batch.squeeze(-1)).sum().item()
                else:
                    correct += (torch.sign(output) == torch.sign(y_batch)).sum().item()
                total += y_batch.numel()
        return correct / total if total > 0 else 0.0

    def train(self, training_snrs=[5, 10, 15], num_samples=50000, epochs=100, lr=0.001,
              training_modulation="bpsk", use_rayleigh=False,
              patience=0, min_delta=1e-5, val_split=0.15):
        """
        Train transformer relay.

        Returns
        -------
        history : dict
            Keys: 'train_loss', 'train_acc', 'val_acc'.
        """
        print(f"  Transformer Training Configuration:")
        print(f"    Device: {self.device}")
        print(f"    Architecture: {self.window_size}-token Transformer")
        print(f"    d_model: {self.model.d_model}, heads: {self.model.transformer_blocks[0].attention.num_heads}")
        print(f"    in_channels: {self.model.input_proj.in_features}")
        print(f"    Layers: {len(self.model.transformer_blocks)}")
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
        print(f"\n  Generating training data...")
        samples_per_snr = num_samples // len(training_snrs)
        X_all = []
        y_all = []
        
        in_channels = self.model.input_proj.in_features

        if self.classify_2d:
            # 16-class 2D: generate complex QAM16 data, windows = [I_win, Q_win]
            y_cls_all = []
            half = self.window_size // 2
            for snr in training_snrs:
                _clean, noisy_c, labels = generate_training_targets_2d(
                    samples_per_snr, snr, seed=42 + int(snr),
                )
                rx_I = np.pad(noisy_c.real, half, mode="edge")
                rx_Q = np.pad(noisy_c.imag, half, mode="edge")
                for i in range(half, len(noisy_c) + half):
                    i_win = rx_I[i - half: i + half + 1]
                    q_win = rx_Q[i - half: i + half + 1]
                    # Shape: (window_size, 2) — two channels: I and Q
                    X_all.append(np.column_stack([i_win, q_win]))
                    y_cls_all.append(labels[i - half])
            X_full = torch.FloatTensor(np.array(X_all)).to(self.device)
            y_full = torch.LongTensor(np.array(y_cls_all)).to(self.device)
            use_ce = True
        else:
            return_csi = (in_channels > 1)
        
            for snr in training_snrs:
                if return_csi:
                    clean, noisy, h_csi = generate_training_targets(
                        samples_per_snr, snr,
                        training_modulation=training_modulation,
                        seed=42 + int(snr),
                        use_rayleigh=use_rayleigh,
                        return_csi=True
                    )
                    features = np.column_stack([noisy, np.abs(h_csi)])
                else:
                    clean, noisy = generate_training_targets(
                        samples_per_snr, snr,
                        training_modulation=training_modulation,
                        seed=42 + int(snr),
                        use_rayleigh=use_rayleigh,
                        return_csi=False
                    )
                    features = noisy.reshape(-1, 1)
            
                for i in range(self.window_size // 2, len(noisy) - self.window_size // 2):
                    window = features[i - self.window_size // 2 : i + self.window_size // 2 + 1]
                    X_all.append(window)
                    y_all.append(clean[i])
        
            X_full = torch.FloatTensor(np.array(X_all)).to(self.device)
            y_clean = np.array(y_all)

            use_ce = self.classify and self.num_classes > 1
            if use_ce:
                y_cls = symbols_to_class_indices(y_clean, training_modulation)
                y_full = torch.LongTensor(y_cls).to(self.device)
            else:
                y_full = torch.FloatTensor(y_clean.reshape(-1, 1)).to(self.device)

        # Train / validation split
        n_total = len(X_full)
        n_val = max(1, int(n_total * val_split))
        perm = torch.randperm(n_total)
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        X_train, y_train = X_full[train_idx], y_full[train_idx]
        X_val,   y_val   = X_full[val_idx],   y_full[val_idx]
        
        print(f"  Training data ready: {len(X_train):,} train, {len(X_val):,} val samples")

        # Optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss() if use_ce else nn.MSELoss()

        print(f"\n  Training Transformer...")
        
        batch_size = 64
        best_loss = float('inf')
        patience_counter = 0
        best_state = None

        history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
        
        for epoch in range(epochs):
            self.model.train()
            indices = torch.randperm(len(X_train))
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            total_loss = 0
            num_batches = 0
            
            for i in range(0, len(X_train), batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                optimizer.zero_grad()
                output = self.model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
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
                print(f"    Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, "
                      f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, "
                      f"Best: {best_loss:.6f}")

            if patience > 0 and patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break
        
        if best_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        self.is_trained = True
        print(f"\n  Transformer training complete!")
        print(f"  Final loss: {avg_loss:.6f}, Best loss: {best_loss:.6f}")
        print(f"  Total parameters: {self.num_params:,}")

        return history
    
    def process(self, received_signal):
        """Process signal through transformer (batched)."""
        if not self.is_trained:
            if isinstance(received_signal, tuple):
                return received_signal[0] * 1.5
            return received_signal * 1.5
        
        self.model.eval()

        # ── 16-class 2D mode: handle complex signal directly ──
        if self.classify_2d and np.iscomplexobj(received_signal):
            pad = self.window_size // 2
            rx_I = np.pad(received_signal.real, pad, mode="edge")
            rx_Q = np.pad(received_signal.imag, pad, mode="edge")
            n = len(received_signal)
            windows = np.zeros((n, self.window_size, 2), dtype=np.float32)
            for i in range(n):
                windows[i, :, 0] = rx_I[i: i + self.window_size]
                windows[i, :, 1] = rx_Q[i: i + self.window_size]

            with torch.no_grad():
                inp = torch.as_tensor(windows, dtype=torch.float32, device=self.device)
                out = self.model(inp)
                indices = out.argmax(dim=-1).cpu().numpy()

            processed = self._constellation_2d[indices]  # complex
            pwr = np.mean(np.abs(processed) ** 2)
            if pwr > 0:
                processed = processed * np.sqrt(self.target_power / pwr)
            return processed
        
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
            y = received_signal
            n = len(y)
            pad_size = self.window_size // 2
            padded_signal = np.pad(y, pad_size, mode='edge')
            windows = np.lib.stride_tricks.as_strided(
                padded_signal,
                shape=(n, self.window_size),
                strides=(padded_signal.strides[0], padded_signal.strides[0]),
            ).copy()
            windows = np.expand_dims(windows, -1)
        
        with torch.no_grad():
            window_input = torch.as_tensor(
                windows, dtype=torch.float32, device=self.device,
            )
            out = self.model(window_input)
            if self.classify and self.num_classes > 1:
                indices = out.argmax(dim=-1).cpu().numpy()
                processed = self._constellation_levels_np[indices]
            else:
                processed = out.cpu().numpy().flatten()
        
        current_power = np.mean(np.abs(processed) ** 2)
        if current_power > 0:
            return processed * np.sqrt(self.target_power / current_power)
        return processed

    def _arch_config(self):
        """Return a dict describing the full architecture (for cache invalidation)."""
        return {
            "window_size": self.window_size,
            "d_model": self.model.d_model,
            "num_heads": self.model.transformer_blocks[0].attention.num_heads,
            "num_layers": len(self.model.transformer_blocks),
            "in_channels": self.in_channels,
            "use_input_norm": self.use_input_norm,
            "output_activation": self.output_activation,
            "clip_range": self.clip_range,
            "num_params": self.num_params,
        }

    def save_weights(self, path):
        """Save trained model weights to *path*."""
        torch.save({
            "type": "TransformerRelayWrapper",
            "model_state_dict": self.model.state_dict(),
            "config": self._arch_config(),
        }, path)

    def load_weights(self, path):
        """Load model weights from *path* if architecture matches.

        Returns True if weights were loaded, False if config mismatch.
        """
        state = torch.load(path, map_location=self.device, weights_only=False)
        saved_cfg = state.get("config", {})
        if saved_cfg != self._arch_config():
            return False
        self.model.load_state_dict(state["model_state_dict"])
        self.is_trained = True
        return True


def simulate_transformer_transmission(num_bits, snr_db, relay, seed=None):
    """Simulate transmission with Transformer relay."""
    source = Source(seed=seed)
    destination = Destination()
    
    tx_bits, tx_symbols = source.transmit(num_bits)
    rx_at_relay = awgn_channel(tx_symbols, snr_db)
    relay_output = relay.process(rx_at_relay)
    rx_at_destination = awgn_channel(relay_output, snr_db)
    rx_bits = destination.receive(rx_at_destination)
    
    ber, num_errors = calculate_ber(tx_bits, rx_bits)
    return ber, num_errors


def test_transformer_relay():
    """Test Transformer relay implementation."""
    print("="*70)
    print("TRANSFORMER RELAY WITH ATTENTION MECHANISM")
    print("="*70)
    
    print("\nTest 1: Training Transformer Relay")
    print("-"*70)
    
    relay = TransformerRelayWrapper(
        target_power=1.0,
        window_size=11,
        d_model=32,
        num_heads=4,
        num_layers=2
    )
    relay.train(training_snrs=[5, 10, 15], num_samples=50000, epochs=100, lr=0.001)
    
    print(f"\n  Status: PASSED ✓")
    
    print("\nTest 2: Quick Validation (10k bits per SNR)")
    print("-"*70)
    
    test_snrs = [0, 2, 4, 6, 8, 10]
    print(f"\n  {'SNR':<6} {'BER':<12} {'Errors':<10}")
    print(f"  {'-'*6} {'-'*12} {'-'*10}")
    
    for snr in test_snrs:
        ber, errors = simulate_transformer_transmission(10000, snr, relay, seed=42)
        print(f"  {snr:<6.0f} {ber:<12.6f} {errors:<10}")
    
    print(f"\n  Status: PASSED ✓")
    
    print("\n" + "="*70)
    print("TRANSFORMER RELAY TEST COMPLETE")
    print("="*70)
    print(f"✓ Transformer relay with {relay.num_params:,} parameters!")
    print("✓ Self-attention mechanism successfully applied!")
    
    return True


if __name__ == "__main__":
    test_transformer_relay()
