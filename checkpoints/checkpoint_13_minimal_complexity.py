"""
Checkpoint 13: Minimal Complexity Search

This module systematically searches for the minimal complexity
configuration that achieves similar performance to Enhanced Training.

Goal: Find smallest network with 2-3 wins at low SNR (0-4 dB)

Experiments:
1. Tiny networks (200-500 params)
2. Reduced training data (10k-50k samples)
3. Simplified training strategies
4. Smaller window sizes

Author: Cline
Date: 2026-02-14
Checkpoint: CP-13
"""

import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from checkpoint_01_channel import awgn_channel
from checkpoint_02_modulation import bpsk_modulate, calculate_ber
from checkpoint_03_nodes import Source, Relay, Destination
from checkpoint_06_decode_forward import simulate_df_transmission


class TinyNeuralNetwork:
    """
    Minimal neural network for efficiency.
    """
    
    def __init__(self, input_size, hidden1, hidden2, output_size):
        """Initialize tiny network."""
        self.W1 = np.random.randn(input_size, hidden1) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden1)
        
        if hidden2 > 0:
            self.W2 = np.random.randn(hidden1, hidden2) * np.sqrt(2.0 / hidden1)
            self.b2 = np.zeros(hidden2)
            self.W3 = np.random.randn(hidden2, output_size) * 0.1
            self.b3 = np.zeros(output_size)
            self.has_hidden2 = True
        else:
            self.W2 = np.random.randn(hidden1, output_size) * 0.1
            self.b2 = np.zeros(output_size)
            self.has_hidden2 = False
    
    def forward(self, X):
        """Forward pass."""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1)
        
        if self.has_hidden2:
            self.z2 = np.dot(self.a1, self.W2) + self.b2
            self.a2 = np.maximum(0, self.z2)
            self.z3 = np.dot(self.a2, self.W3) + self.b3
            output = np.tanh(self.z3)
        else:
            self.z2 = np.dot(self.a1, self.W2) + self.b2
            output = np.tanh(self.z2)
        
        return output
    
    def train_step(self, X, y, learning_rate=0.01):
        """Training step."""
        batch_size = X.shape[0]
        output = self.forward(X)
        loss = np.mean((output - y) ** 2)
        
        if self.has_hidden2:
            # 3-layer backprop
            dz3 = 2 * (output - y) / batch_size * (1 - output ** 2)
            dW3 = np.dot(self.a2.T, dz3)
            db3 = np.sum(dz3, axis=0)
            
            da2 = np.dot(dz3, self.W3.T)
            dz2 = da2 * (self.z2 > 0)
            dW2 = np.dot(self.a1.T, dz2)
            db2 = np.sum(dz2, axis=0)
            
            da1 = np.dot(dz2, self.W2.T)
            dz1 = da1 * (self.z1 > 0)
            dW1 = np.dot(X.T, dz1)
            db1 = np.sum(dz1, axis=0)
            
            self.W3 -= learning_rate * dW3
            self.b3 -= learning_rate * db3
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1
        else:
            # 2-layer backprop
            dz2 = 2 * (output - y) / batch_size * (1 - output ** 2)
            dW2 = np.dot(self.a1.T, dz2)
            db2 = np.sum(dz2, axis=0)
            
            da1 = np.dot(dz2, self.W2.T)
            dz1 = da1 * (self.z1 > 0)
            dW1 = np.dot(X.T, dz1)
            db1 = np.sum(dz1, axis=0)
            
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1
        
        return loss


class MinimalGenAIRelay(Relay):
    """Minimal complexity GenAI relay."""
    
    def __init__(self, window_size, hidden1, hidden2, target_power=1.0):
        self.window_size = window_size
        self.target_power = target_power
        self.nn = TinyNeuralNetwork(window_size, hidden1, hidden2, 1)
        self.is_trained = False
        self.params = self._count_params(window_size, hidden1, hidden2)
    
    def _count_params(self, inp, h1, h2):
        """Count parameters."""
        if h2 > 0:
            return inp*h1 + h1 + h1*h2 + h2 + h2*1 + 1
        else:
            return inp*h1 + h1 + h1*1 + 1
    
    def train_minimal(self, training_snrs, num_samples, epochs):
        """Minimal training."""
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
        
        for epoch in range(epochs):
            indices = np.random.permutation(len(X_train))
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            batch_size = 32
            for i in range(0, len(X_train), batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                self.nn.train_step(X_batch, y_batch, learning_rate=0.01)
        
        self.is_trained = True
    
    def process(self, received_signal):
        """Process signal."""
        if not self.is_trained:
            return received_signal * 1.5
        
        processed = np.zeros_like(received_signal)
        pad_size = self.window_size // 2
        padded_signal = np.pad(received_signal, pad_size, mode='edge')
        
        for i in range(len(received_signal)):
            window = padded_signal[i:i+self.window_size]
            processed[i] = self.nn.forward(window.reshape(1, -1))[0, 0]
        
        current_power = np.mean(np.abs(processed) ** 2)
        if current_power > 0:
            return processed * np.sqrt(self.target_power / current_power)
        return processed


def simulate_minimal_genai(num_bits, snr_db, relay, seed=None):
    """Simulate with minimal GenAI."""
    source = Source(seed=seed)
    destination = Destination()
    
    tx_bits, tx_symbols = source.transmit(num_bits)
    rx_at_relay = awgn_channel(tx_symbols, snr_db)
    relay_output = relay.process(rx_at_relay)
    rx_at_destination = awgn_channel(relay_output, snr_db)
    rx_bits = destination.receive(rx_at_destination)
    
    ber, num_errors = calculate_ber(tx_bits, rx_bits)
    return ber, num_errors


def test_configuration(config_name, window, h1, h2, samples, epochs, training_snrs):
    """Test a single configuration."""
    print(f"\n{'='*70}")
    print(f"Testing: {config_name}")
    print(f"{'='*70}")
    
    relay = MinimalGenAIRelay(window, h1, h2)
    print(f"  Architecture: {window}→{h1}→{h2}→1" if h2 > 0 else f"  Architecture: {window}→{h1}→1")
    print(f"  Parameters: {relay.params:,}")
    print(f"  Training: {samples:,} samples, {epochs} epochs, SNRs {training_snrs}")
    
    start_time = time.time()
    relay.train_minimal(training_snrs, samples, epochs)
    train_time = time.time() - start_time
    
    print(f"  Training time: {train_time:.1f} seconds")
    
    # Quick validation (100k bits per SNR)
    print(f"\n  Validation Results:")
    snr_range = [0, 2, 4, 6, 8, 10]
    wins = 0
    
    for snr_db in snr_range:
        df_errors = 0
        genai_errors = 0
        
        for trial in range(10):
            _, df_err = simulate_df_transmission(10000, snr_db, seed=trial)
            _, genai_err = simulate_minimal_genai(10000, snr_db, relay, seed=trial)
            df_errors += df_err
            genai_errors += genai_err
        
        df_ber = df_errors / 100000
        genai_ber = genai_errors / 100000
        
        if genai_ber < df_ber:
            winner = "GenAI ✓"
            wins += 1
        elif df_ber < genai_ber:
            winner = "DF ✓"
        else:
            winner = "Tie"
        
        print(f"    {snr_db:2.0f} dB: DF={df_ber:.6f}, GenAI={genai_ber:.6f} - {winner}")
    
    print(f"\n  Summary: {wins}/6 wins, {relay.params:,} params, {train_time:.1f}s training")
    
    return {
        'name': config_name,
        'params': relay.params,
        'wins': wins,
        'train_time': train_time,
        'window': window,
        'h1': h1,
        'h2': h2,
        'samples': samples,
        'epochs': epochs
    }


def test_minimal_complexity():
    """Run minimal complexity experiments."""
    print("="*70)
    print("MINIMAL COMPLEXITY SEARCH")
    print("="*70)
    print("\nGoal: Find smallest network with 2-3 wins at low SNR")
    print("Baseline: Enhanced Training has 3 wins with 3,000 params\n")
    
    results = []
    
    # Experiment 1: Ultra-tiny (200 params)
    results.append(test_configuration(
        "Ultra-Tiny", window=5, h1=16, h2=8, 
        samples=25000, epochs=100, training_snrs=[5, 10, 15]
    ))
    
    # Experiment 2: Tiny (300 params)
    results.append(test_configuration(
        "Tiny", window=5, h1=24, h2=12, 
        samples=25000, epochs=100, training_snrs=[5, 10, 15]
    ))
    
    # Experiment 3: Small (500 params)
    results.append(test_configuration(
        "Small", window=7, h1=24, h2=12, 
        samples=50000, epochs=100, training_snrs=[5, 10, 15]
    ))
    
    # Experiment 4: Medium-Small (800 params)
    results.append(test_configuration(
        "Medium-Small", window=7, h1=32, h2=16, 
        samples=50000, epochs=150, training_snrs=[5, 10, 15]
    ))
    
    # Experiment 5: 2-layer tiny (150 params)
    results.append(test_configuration(
        "2-Layer Tiny", window=5, h1=24, h2=0, 
        samples=25000, epochs=100, training_snrs=[5, 10, 15]
    ))
    
    # Final Analysis
    print("\n" + "="*70)
    print("FINAL ANALYSIS")
    print("="*70)
    
    print(f"\n{'Config':<20} {'Params':<10} {'Wins':<8} {'Time(s)':<10} {'Efficiency':<12}")
    print(f"{'-'*20} {'-'*10} {'-'*8} {'-'*10} {'-'*12}")
    
    for r in results:
        efficiency = r['wins'] / (r['params'] / 100)  # wins per 100 params
        print(f"{r['name']:<20} {r['params']:<10,} {r['wins']:<8} {r['train_time']:<10.1f} {efficiency:<12.2f}")
    
    # Find best
    best = max(results, key=lambda x: (x['wins'], -x['params']))
    
    print(f"\n🏆 BEST CONFIGURATION: {best['name']}")
    print(f"   Parameters: {best['params']:,} ({3000/best['params']:.1f}x smaller than enhanced)")
    print(f"   Wins: {best['wins']}/6")
    print(f"   Training time: {best['train_time']:.1f}s ({150/best['train_time']:.1f}x faster)")
    print(f"   Architecture: {best['window']}→{best['h1']}→{best['h2']}→1" if best['h2'] > 0 
          else f"   Architecture: {best['window']}→{best['h1']}→1")
    
    return True


if __name__ == "__main__":
    test_minimal_complexity()
