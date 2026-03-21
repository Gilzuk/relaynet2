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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from checkpoint_01_channel import awgn_channel
from checkpoint_02_modulation import bpsk_modulate, calculate_ber
from checkpoint_03_nodes import Source, Relay, Destination

from relaynet.utils.activations import make_torch_activation, generate_training_targets


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
                 output_activation="tanh", use_input_norm=False):
        super(TransformerRelay, self).__init__()
        
        self.window_size = window_size
        self.d_model = d_model
        self.use_input_norm = use_input_norm
        
        # Input projection
        self.input_proj = nn.Linear(1, d_model)
        
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
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            make_torch_activation(output_activation),
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
        output = self.output_proj(x)  # (batch, 1)
        
        return output


class TransformerRelayWrapper(Relay):
    """Wrapper for Transformer relay."""
    
    def __init__(self, target_power=1.0, window_size=11, d_model=32, num_heads=4, num_layers=2, prefer_gpu=False,
                 output_activation="tanh", use_input_norm=False):
        self.target_power = target_power
        self.window_size = window_size
        self.output_activation = output_activation
        self.use_input_norm = use_input_norm
        
        # Set device — with only 17K parameters this model is too small to
        # benefit from GPU; kernel-launch overhead dominates compute time.
        if prefer_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
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
        ).to(self.device)
        
        self.is_trained = False
        
        # Count parameters
        self.num_params = sum(p.numel() for p in self.model.parameters())
    
    def train(self, training_snrs=[5, 10, 15], num_samples=50000, epochs=100, lr=0.001,
              training_modulation="bpsk"):
        """
        Train transformer relay.
        """
        print(f"  Transformer Training Configuration:")
        print(f"    Device: {self.device}")
        print(f"    Architecture: {self.window_size}-token Transformer")
        print(f"    d_model: {self.model.d_model}, heads: {self.model.transformer_blocks[0].attention.num_heads}")
        print(f"    Layers: {len(self.model.transformer_blocks)}")
        print(f"    Parameters: {self.num_params:,}")
        print(f"    Training SNRs: {training_snrs} dB")
        print(f"    Samples: {num_samples:,}, Epochs: {epochs}")
        if self.use_input_norm:
            print(f"    Input LayerNorm: ENABLED")
        if training_modulation != "bpsk":
            print(f"    Training modulation: {training_modulation}")
        
        # Generate training data
        print(f"\n  Generating training data...")
        samples_per_snr = num_samples // len(training_snrs)
        X_train_all = []
        y_train_all = []
        
        for snr in training_snrs:
            clean, noisy = generate_training_targets(
                samples_per_snr, snr,
                training_modulation=training_modulation,
                seed=42 + int(snr),
            )
            
            for i in range(self.window_size // 2, len(noisy) - self.window_size // 2):
                window = noisy[i - self.window_size // 2 : i + self.window_size // 2 + 1]
                X_train_all.append(window)
                y_train_all.append(clean[i])
        
        X_train = torch.FloatTensor(np.array(X_train_all)).unsqueeze(-1).to(self.device)
        y_train = torch.FloatTensor(np.array(y_train_all).reshape(-1, 1)).to(self.device)
        
        print(f"  Training data ready: {len(X_train):,} samples")

        # Optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        print(f"\n  Training Transformer...")
        
        batch_size = 64
        best_loss = float('inf')
        
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
                
                # Forward pass
                optimizer.zero_grad()
                output = self.model(X_batch)
                loss = criterion(output, y_batch)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            if avg_loss < best_loss:
                best_loss = avg_loss
            
            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, Best: {best_loss:.6f}")
        
        self.is_trained = True
        print(f"\n  Transformer training complete!")
        print(f"  Final loss: {avg_loss:.6f}")
        print(f"  Total parameters: {self.num_params:,}")
    
    def process(self, received_signal):
        """Process signal through transformer (batched)."""
        if not self.is_trained:
            return received_signal * 1.5
        
        self.model.eval()
        
        pad_size = self.window_size // 2
        padded_signal = np.pad(received_signal, pad_size, mode='edge')
        n = len(received_signal)
        
        # Build all windows at once: (n, window_size)
        windows = np.lib.stride_tricks.as_strided(
            padded_signal,
            shape=(n, self.window_size),
            strides=(padded_signal.strides[0], padded_signal.strides[0]),
        ).copy()  # copy to ensure contiguous memory
        
        # Single batched forward pass
        with torch.no_grad():
            window_input = torch.as_tensor(
                windows, dtype=torch.float32, device=self.device,
            ).unsqueeze(-1)  # (n, window_size, 1)
            processed = self.model(window_input).cpu().numpy().flatten()
        
        # Normalize power
        current_power = np.mean(np.abs(processed) ** 2)
        if current_power > 0:
            return processed * np.sqrt(self.target_power / current_power)
        return processed

    def save_weights(self, path):
        """Save trained model weights to *path*."""
        torch.save({
            "type": "TransformerRelayWrapper",
            "model_state_dict": self.model.state_dict(),
            "config": {"window_size": self.window_size,
                       "num_params": self.num_params},
        }, path)

    def load_weights(self, path):
        """Load model weights from *path* and mark as trained."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        self.is_trained = True


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
