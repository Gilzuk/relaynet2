"""
Checkpoint 12: Maximum GenAI Training - Push the Limits!

This module implements maximum training for GenAI relay:
- 500,000 training samples (25x original)
- 500 epochs (5x original)
- All SNR training (0-20 dB in 2 dB steps)
- Larger architecture (11→64→64→64→32→1)
- Advanced learning rate scheduling

Author: Cline
Date: 2026-02-14
Checkpoint: CP-12
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from checkpoint_01_channel import awgn_channel
from checkpoint_02_modulation import bpsk_modulate, bpsk_demodulate, calculate_ber
from checkpoint_03_nodes import Source, Relay, Destination
from checkpoint_04_simulation import simulate_two_hop_transmission
from checkpoint_06_decode_forward import simulate_df_transmission
from checkpoint_08_genai_relay import SimpleNeuralNetwork, simulate_genai_transmission


class MaximumNeuralNetwork:
    """
    Larger neural network for maximum performance.
    
    Architecture: Input(11) → 64 → 64 → 64 → 32 → Output(1)
    """
    
    def __init__(self, input_size, output_size):
        """Initialize larger network."""
        # Layer 1: input → 64
        self.W1 = np.random.randn(input_size, 64) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(64)
        
        # Layer 2: 64 → 64
        self.W2 = np.random.randn(64, 64) * np.sqrt(2.0 / 64)
        self.b2 = np.zeros(64)
        
        # Layer 3: 64 → 64
        self.W3 = np.random.randn(64, 64) * np.sqrt(2.0 / 64)
        self.b3 = np.zeros(64)
        
        # Layer 4: 64 → 32
        self.W4 = np.random.randn(64, 32) * np.sqrt(2.0 / 64)
        self.b4 = np.zeros(32)
        
        # Layer 5: 32 → output
        self.W5 = np.random.randn(32, output_size) * 0.1
        self.b5 = np.zeros(output_size)
    
    def forward(self, X):
        """Forward pass."""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = np.maximum(0, self.z2)  # ReLU
        
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = np.maximum(0, self.z3)  # ReLU
        
        self.z4 = np.dot(self.a3, self.W4) + self.b4
        self.a4 = np.maximum(0, self.z4)  # ReLU
        
        self.z5 = np.dot(self.a4, self.W5) + self.b5
        output = np.tanh(self.z5)  # Tanh for BPSK
        
        return output
    
    def train_step(self, X, y, learning_rate=0.01):
        """Training step with backpropagation."""
        batch_size = X.shape[0]
        
        # Forward pass
        output = self.forward(X)
        
        # Loss
        loss = np.mean((output - y) ** 2)
        
        # Backward pass
        # Layer 5
        dz5 = 2 * (output - y) / batch_size * (1 - output ** 2)
        dW5 = np.dot(self.a4.T, dz5)
        db5 = np.sum(dz5, axis=0)
        
        # Layer 4
        da4 = np.dot(dz5, self.W5.T)
        dz4 = da4 * (self.z4 > 0)
        dW4 = np.dot(self.a3.T, dz4)
        db4 = np.sum(dz4, axis=0)
        
        # Layer 3
        da3 = np.dot(dz4, self.W4.T)
        dz3 = da3 * (self.z3 > 0)
        dW3 = np.dot(self.a2.T, dz3)
        db3 = np.sum(dz3, axis=0)
        
        # Layer 2
        da2 = np.dot(dz3, self.W3.T)
        dz2 = da2 * (self.z2 > 0)
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0)
        
        # Layer 1
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (self.z1 > 0)
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0)
        
        # Update weights
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3
        self.W4 -= learning_rate * dW4
        self.b4 -= learning_rate * db4
        self.W5 -= learning_rate * dW5
        self.b5 -= learning_rate * db5
        
        return loss


class MaximumGenAIRelay(Relay):
    """Maximum performance GenAI relay."""
    
    def __init__(self, target_power=1.0, window_size=11):
        self.target_power = target_power
        self.window_size = window_size
        self.nn = MaximumNeuralNetwork(input_size=window_size, output_size=1)
        self.is_trained = False
    
    def train_maximum(self, num_samples=500000, epochs=500):
        """
        Maximum training across all SNRs.
        """
        print(f"  MAXIMUM TRAINING CONFIGURATION:")
        print(f"    Architecture: 11→64→64→64→32→1 (5 layers)")
        print(f"    Total samples: {num_samples:,}")
        print(f"    Epochs: {epochs}")
        print(f"    Training SNRs: 0-20 dB (all 11 points)")
        print(f"    Window size: {self.window_size}")
        
        print(f"\n  Generating massive training dataset...")
        
        # Train on ALL SNRs
        training_snrs = np.arange(0, 21, 2)
        samples_per_snr = num_samples // len(training_snrs)
        
        X_train_all = []
        y_train_all = []
        
        for snr in training_snrs:
            np.random.seed(42 + int(snr))
            clean_bits = np.random.randint(0, 2, samples_per_snr)
            clean_symbols = bpsk_modulate(clean_bits)
            noisy_symbols = awgn_channel(clean_symbols, snr)
            
            for i in range(self.window_size // 2, len(noisy_symbols) - self.window_size // 2):
                window = noisy_symbols[i - self.window_size // 2 : i + self.window_size // 2 + 1]
                X_train_all.append(window)
                y_train_all.append(clean_symbols[i])
        
        X_train = np.array(X_train_all)
        y_train = np.array(y_train_all).reshape(-1, 1)
        
        print(f"  Dataset ready: {len(X_train):,} samples")
        print(f"  Network parameters: ~{self._count_parameters():,}")
        print(f"\n  Starting maximum training (this will take several minutes)...")
        
        # Advanced learning rate schedule
        learning_rate = 0.01
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Cosine annealing
            learning_rate = 0.001 + 0.009 * (1 + np.cos(np.pi * epoch / epochs)) / 2
            
            # Shuffle
            indices = np.random.permutation(len(X_train))
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Train
            batch_size = 128
            total_loss = 0
            num_batches = 0
            
            for i in range(0, len(X_train), batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                loss = self.nn.train_step(X_batch, y_batch, learning_rate=learning_rate)
                total_loss += loss
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 50 == 0:
                print(f"    Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, Best: {best_loss:.6f}, LR: {learning_rate:.6f}")
        
        self.is_trained = True
        print(f"\n  MAXIMUM TRAINING COMPLETE!")
        print(f"  Final loss: {avg_loss:.6f}")
        print(f"  Best loss: {best_loss:.6f}")
    
    def _count_parameters(self):
        """Count network parameters."""
        return (11*64 + 64 + 64*64 + 64 + 64*64 + 64 + 64*32 + 32 + 32*1 + 1)
    
    def process(self, received_signal):
        """Process signal."""
        if not self.is_trained:
            return received_signal * 1.5
        
        processed = np.zeros_like(received_signal)
        pad_size = self.window_size // 2
        padded_signal = np.pad(received_signal, pad_size, mode='edge')
        
        for i in range(len(received_signal)):
            window = padded_signal[i:i+self.window_size]
            window_input = window.reshape(1, -1)
            processed[i] = self.nn.forward(window_input)[0, 0]
        
        # Normalize power
        current_power = np.mean(np.abs(processed) ** 2)
        if current_power > 0:
            power_factor = np.sqrt(self.target_power / current_power)
            forwarded_signal = power_factor * processed
        else:
            forwarded_signal = processed
        
        return forwarded_signal


def simulate_maximum_genai(num_bits, snr_db, relay, seed=None):
    """Simulate with maximum GenAI relay."""
    source = Source(seed=seed)
    destination = Destination()
    
    tx_bits, tx_symbols = source.transmit(num_bits)
    rx_at_relay = awgn_channel(tx_symbols, snr_db)
    relay_output = relay.process(rx_at_relay)
    rx_at_destination = awgn_channel(relay_output, snr_db)
    rx_bits = destination.receive(rx_at_destination)
    
    ber, num_errors = calculate_ber(tx_bits, rx_bits)
    return ber, num_errors


def test_maximum_training():
    """Test maximum training."""
    print("=" * 70)
    print("MAXIMUM GENAI TRAINING - PUSHING THE LIMITS!")
    print("=" * 70)
    
    # Train
    print("\nPhase 1: Maximum Training")
    print("-" * 70)
    
    relay = MaximumGenAIRelay(target_power=1.0, window_size=11)
    relay.train_maximum(num_samples=500000, epochs=500)
    
    # Validate
    print("\nPhase 2: Comprehensive Validation (1M bits per SNR)")
    print("-" * 70)
    
    snr_range = np.arange(0, 21, 2)
    results = {'snr': [], 'df_ber': [], 'genai_ber': []}
    
    for snr_db in snr_range:
        print(f"\n  Testing SNR = {snr_db} dB...")
        
        df_errors = 0
        genai_errors = 0
        
        for trial in range(100):
            _, df_err = simulate_df_transmission(10000, snr_db, seed=trial)
            _, genai_err = simulate_maximum_genai(10000, snr_db, relay, seed=trial)
            df_errors += df_err
            genai_errors += genai_err
        
        df_ber = df_errors / 1000000
        genai_ber = genai_errors / 1000000
        
        results['snr'].append(snr_db)
        results['df_ber'].append(df_ber)
        results['genai_ber'].append(genai_ber)
        
        winner = "GenAI ✓" if genai_ber < df_ber else ("DF ✓" if df_ber < genai_ber else "Tie")
        print(f"    DF: {df_ber:.6f}, GenAI: {genai_ber:.6f} - {winner}")
    
    # Analysis
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    genai_wins = sum(1 for g, d in zip(results['genai_ber'], results['df_ber']) if g < d)
    df_wins = sum(1 for g, d in zip(results['genai_ber'], results['df_ber']) if d < g)
    ties = sum(1 for g, d in zip(results['genai_ber'], results['df_ber']) if g == d)
    
    print(f"\nGenAI wins: {genai_wins}/11 ({100*genai_wins/11:.1f}%)")
    print(f"DF wins: {df_wins}/11 ({100*df_wins/11:.1f}%)")
    print(f"Ties: {ties}/11 ({100*ties/11:.1f}%)")
    
    if genai_wins > df_wins:
        print(f"\n🏆 VICTORY! Maximum GenAI BEATS DF!")
    elif genai_wins == df_wins:
        print(f"\n⚖️  PARITY! GenAI matches DF!")
    else:
        print(f"\n📊 GenAI improved but DF still leads")
    
    return True


if __name__ == "__main__":
    test_maximum_training()
