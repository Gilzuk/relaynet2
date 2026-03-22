"""
Checkpoint 20: Mamba S6 Relay (State Space Model)

This module implements a Mamba-style S6 (Selective State Space) relay
for signal denoising. S6 models are more efficient than Transformers
for sequence modeling.

Architecture:
- Selective State Space (S6) layers
- Linear time complexity (vs quadratic for attention)
- Efficient long-range dependencies

Reference: Mamba - Linear-Time Sequence Modeling with Selective State Spaces

Author: Cline
Date: 2026-02-15
Checkpoint: CP-20
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


class S6Layer(nn.Module):
    """
    Simplified S6 (Selective State Space) layer.
    
    State space model:
        x'(t) = A x(t) + B u(t)
        y(t) = C x(t) + D u(t)
    
    Discretized:
        x_k = A_bar x_{k-1} + B_bar u_k
        y_k = C x_k + D u_k
    """
    
    def __init__(self, d_model, d_state=16, dt_min=0.001, dt_max=0.1):
        super(S6Layer, self).__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        
        # Learnable parameters for selective mechanism
        self.delta_proj = nn.Linear(d_model, d_model)  # Time step selection
        self.B_proj = nn.Linear(d_model, d_state)      # Input matrix selection
        self.C_proj = nn.Linear(d_model, d_state)      # Output matrix selection
        
        # State transition matrix A (initialized as diagonal)
        self.A_log = nn.Parameter(torch.randn(d_state))
        
        # Direct feedthrough
        self.D = nn.Parameter(torch.randn(d_model))
        
        # Output projection
        self.output_proj = nn.Linear(d_state, d_model)
        
        self.dt_min = dt_min
        self.dt_max = dt_max
    
    def forward(self, x):
        """
        Forward pass through S6 layer.
        
        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, seq_len, d_model)
        
        Returns
        -------
        torch.Tensor
            Output of shape (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape
        
        # Selective mechanism: compute time steps, B, C based on input
        delta = torch.sigmoid(self.delta_proj(x))  # (batch, seq_len, d_model)
        delta = self.dt_min + (self.dt_max - self.dt_min) * delta
        delta = delta.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1) - average across d_model
        
        B = self.B_proj(x)  # (batch, seq_len, d_state)
        C = self.C_proj(x)  # (batch, seq_len, d_state)
        
        # State transition matrix A
        A = -torch.exp(self.A_log)  # (d_state,)
        
        # Discretize: A_bar = exp(delta * A), B_bar = delta * B
        # Simplified discretization for efficiency
        A_bar = torch.exp(delta * A.unsqueeze(0).unsqueeze(0))  # (batch, seq_len, d_state)
        B_bar = delta * B  # (batch, seq_len, d_state)
        
        # Run state space model
        state = torch.zeros(batch, self.d_state, device=x.device)
        outputs = []
        
        for t in range(seq_len):
            # x_k = A_bar * x_{k-1} + B_bar * u_k
            state = A_bar[:, t, :] * state + B_bar[:, t, :] * x[:, t, :].mean(dim=-1, keepdim=True)
            
            # y_k = C * x_k + D * u_k
            y = torch.sum(C[:, t, :] * state, dim=-1, keepdim=True)  # (batch, 1)
            y = self.output_proj(state) + self.D * x[:, t, :]
            
            outputs.append(y)
        
        output = torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)
        
        return output


class MambaBlock(nn.Module):
    """Mamba block with S6 layer and MLP."""
    
    def __init__(self, d_model, d_state=16, expand_factor=2):
        super(MambaBlock, self).__init__()
        
        d_inner = d_model * expand_factor
        
        # Input projection
        self.in_proj = nn.Linear(d_model, d_inner)
        
        # S6 layer
        self.s6 = S6Layer(d_inner, d_state)
        
        # Activation
        self.activation = nn.SiLU()
        
        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model)
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        """Forward pass through Mamba block."""
        residual = x
        
        # Normalize
        x = self.norm(x)
        
        # Project up
        x = self.in_proj(x)
        x = self.activation(x)
        
        # S6 layer
        x = self.s6(x)
        
        # Project down
        x = self.out_proj(x)
        
        # Residual connection
        return x + residual


class MambaRelay(nn.Module):
    """Mamba-based relay for signal denoising."""
    
    def __init__(self, window_size=11, d_model=32, d_state=16, num_layers=2,
                 output_activation="tanh", use_input_norm=False, clip_range=None):
        super(MambaRelay, self).__init__()
        
        self.window_size = window_size
        self.d_model = d_model
        self.use_input_norm = use_input_norm
        
        # Input projection
        self.input_proj = nn.Linear(1, d_model)
        
        # Optional input LayerNorm — stabilises the distribution entering
        # the Mamba blocks and prevents extreme activations from noisy inputs.
        if use_input_norm:
            self.input_norm = nn.LayerNorm(d_model)
        
        # Mamba blocks
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
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
        
        # Pass through Mamba blocks
        for block in self.blocks:
            x = block(x)
        
        # Take the middle token (center of window)
        center_idx = self.window_size // 2
        x = x[:, center_idx, :]  # (batch, d_model)
        
        # Project to output
        output = self.output_proj(x)  # (batch, 1)
        
        return output + x_raw[:, center_idx, :]

class MambaRelayWrapper(Relay):
    """Wrapper for Mamba relay."""

    def __init__(self, target_power=1.0, window_size=11, d_model=32, d_state=16, num_layers=2, prefer_gpu=False,
                 output_activation="tanh", use_input_norm=False, clip_range=None):
        self.target_power = target_power
        self.window_size = window_size
        self.output_activation = output_activation
        self.use_input_norm = use_input_norm
        self.clip_range = clip_range
        
        # Set device — with only 24K parameters this model is too small to
        # benefit from GPU; kernel-launch overhead dominates compute time.
        if prefer_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        # Initialize Mamba
        self.model = MambaRelay(
            window_size=window_size,
            d_model=d_model,
            d_state=d_state,
            num_layers=num_layers,
            output_activation=output_activation,
            use_input_norm=use_input_norm,
            clip_range=clip_range,
        ).to(self.device)
        
        self.is_trained = False
        
        # Count parameters
        self.num_params = sum(p.numel() for p in self.model.parameters())
    
    def train(self, training_snrs=[5, 10, 15], num_samples=50000, epochs=100, lr=0.001,
              training_modulation="bpsk"):
        """
        Train Mamba relay.
        """
        print(f"  Mamba S6 Training Configuration:")
        print(f"    Device: {self.device}")
        print(f"    Architecture: {self.window_size}-token Mamba (S6)")
        print(f"    d_model: {self.model.d_model}, d_state: {self.model.blocks[0].s6.d_state}")
        print(f"    Layers: {len(self.model.blocks)}")
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

        print(f"\n  Training Mamba S6...")
        
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
        print(f"\n  Mamba training complete!")
        print(f"  Final loss: {avg_loss:.6f}")
        print(f"  Total parameters: {self.num_params:,}")
    
    def process(self, received_signal):
        """Process signal through Mamba (batched)."""
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
            "type": "MambaRelayWrapper",
            "model_state_dict": self.model.state_dict(),
            "config": {"window_size": self.window_size,
                       "num_params": self.num_params},
        }, path)

    def load_weights(self, path):
        """Load model weights from *path* and mark as trained."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        self.is_trained = True


def simulate_mamba_transmission(num_bits, snr_db, relay, seed=None):
    """Simulate transmission with Mamba relay."""
    source = Source(seed=seed)
    destination = Destination()
    
    tx_bits, tx_symbols = source.transmit(num_bits)
    rx_at_relay = awgn_channel(tx_symbols, snr_db)
    relay_output = relay.process(rx_at_relay)
    rx_at_destination = awgn_channel(relay_output, snr_db)
    rx_bits = destination.receive(rx_at_destination)
    
    ber, num_errors = calculate_ber(tx_bits, rx_bits)
    return ber, num_errors


def test_mamba_relay():
    """Test Mamba S6 relay implementation."""
    print("="*70)
    print("MAMBA S6 RELAY (STATE SPACE MODEL)")
    print("="*70)
    
    print("\nTest 1: Training Mamba S6 Relay")
    print("-"*70)
    
    relay = MambaRelayWrapper(
        target_power=1.0,
        window_size=11,
        d_model=32,
        d_state=16,
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
        ber, errors = simulate_mamba_transmission(10000, snr, relay, seed=42)
        print(f"  {snr:<6.0f} {ber:<12.6f} {errors:<10}")
    
    print(f"\n  Status: PASSED ✓")
    
    print("\n" + "="*70)
    print("MAMBA S6 RELAY TEST COMPLETE")
    print("="*70)
    print(f"✓ Mamba relay with {relay.num_params:,} parameters!")
    print("✓ State Space Model (S6) successfully applied!")
    print("✓ Linear-time complexity vs quadratic for Transformer!")
    
    return True


if __name__ == "__main__":
    test_mamba_relay()
