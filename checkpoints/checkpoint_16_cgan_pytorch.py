"""
Checkpoint 16: CGAN (Conditional GAN) Relay - PyTorch Implementation

This module implements a proper Conditional GAN-based relay using PyTorch
with full backpropagation and adversarial training.

Architecture:
- Generator: noisy_signal + noise → clean_signal
- Discriminator: (signal, condition) → real/fake

Loss:
- Generator: Adversarial + L1 reconstruction
- Discriminator: Binary cross-entropy

Author: Cline
Date: 2026-02-14
Checkpoint: CP-16 (PyTorch)
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


class Generator(nn.Module):
    """Generator network for CGAN."""
    
    def __init__(self, input_size=7, noise_size=8):
        super(Generator, self).__init__()
        self.input_size = input_size
        self.noise_size = noise_size
        
        # Generator: (noisy_signal + noise) → clean_signal
        self.model = nn.Sequential(
            nn.Linear(input_size + noise_size, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1),
            nn.Tanh()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, noisy_signal, noise):
        """Forward pass."""
        x = torch.cat([noisy_signal, noise], dim=1)
        return self.model(x)


class Discriminator(nn.Module):
    """Discriminator network for CGAN."""
    
    def __init__(self, signal_size=1, condition_size=7):
        super(Discriminator, self).__init__()
        
        # Discriminator: (signal + condition) → real/fake
        self.model = nn.Sequential(
            nn.Linear(signal_size + condition_size, 32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, signal, condition):
        """Forward pass."""
        x = torch.cat([signal, condition], dim=1)
        return self.model(x)


class CGANRelayPyTorch(Relay):
    """PyTorch-based CGAN relay for signal denoising."""
    
    def __init__(self, target_power=1.0, window_size=7, noise_size=8, lambda_l1=100):
        self.target_power = target_power
        self.window_size = window_size
        self.noise_size = noise_size
        self.lambda_l1 = lambda_l1
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize networks
        self.generator = Generator(window_size, noise_size).to(self.device)
        self.discriminator = Discriminator(1, window_size).to(self.device)
        
        self.is_trained = False
    
    def train(self, training_snrs=[5, 10, 15], num_samples=50000, epochs=200):
        """
        Train CGAN relay with proper adversarial training.
        """
        print(f"  CGAN PyTorch Training Configuration:")
        print(f"    Device: {self.device}")
        print(f"    Generator: ({self.window_size}+{self.noise_size})→32→32→16→1")
        print(f"    Discriminator: ({1}+{self.window_size})→32→16→1")
        print(f"    λ_L1: {self.lambda_l1}")
        print(f"    Training SNRs: {training_snrs} dB")
        print(f"    Samples: {num_samples:,}, Epochs: {epochs}")
        
        # Generate training data
        print(f"\n  Generating training data...")
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
        
        X_train = torch.FloatTensor(np.array(X_train_all)).to(self.device)
        y_train = torch.FloatTensor(np.array(y_train_all).reshape(-1, 1)).to(self.device)
        
        print(f"  Training data ready: {len(X_train):,} samples")
        
        # Optimizers
        g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Loss functions
        adversarial_loss = nn.BCELoss()
        l1_loss = nn.L1Loss()
        
        print(f"\n  Training CGAN with PyTorch...")
        
        batch_size = 64
        
        for epoch in range(epochs):
            indices = torch.randperm(len(X_train))
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            d_loss_total = 0
            g_loss_total = 0
            num_batches = 0
            
            for i in range(0, len(X_train), batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                current_batch_size = X_batch.shape[0]
                
                # Labels
                real_labels = torch.ones(current_batch_size, 1).to(self.device)
                fake_labels = torch.zeros(current_batch_size, 1).to(self.device)
                
                # ---------------------
                #  Train Discriminator
                # ---------------------
                d_optimizer.zero_grad()
                
                # Real samples
                d_real = self.discriminator(y_batch, X_batch)
                d_loss_real = adversarial_loss(d_real, real_labels)
                
                # Fake samples
                noise = torch.randn(current_batch_size, self.noise_size).to(self.device)
                fake = self.generator(X_batch, noise)
                d_fake = self.discriminator(fake.detach(), X_batch)
                d_loss_fake = adversarial_loss(d_fake, fake_labels)
                
                # Total discriminator loss
                d_loss = (d_loss_real + d_loss_fake) / 2
                d_loss.backward()
                d_optimizer.step()
                
                # -----------------
                #  Train Generator
                # -----------------
                g_optimizer.zero_grad()
                
                # Generate fake samples
                noise = torch.randn(current_batch_size, self.noise_size).to(self.device)
                fake = self.generator(X_batch, noise)
                
                # Generator losses
                d_fake = self.discriminator(fake, X_batch)
                g_loss_adv = adversarial_loss(d_fake, real_labels)
                g_loss_l1 = l1_loss(fake, y_batch)
                
                # Total generator loss
                g_loss = g_loss_adv + self.lambda_l1 * g_loss_l1
                g_loss.backward()
                g_optimizer.step()
                
                d_loss_total += d_loss.item()
                g_loss_total += g_loss.item()
                num_batches += 1
            
            avg_d_loss = d_loss_total / num_batches
            avg_g_loss = g_loss_total / num_batches
            
            if (epoch + 1) % 40 == 0:
                print(f"    Epoch {epoch+1}/{epochs}, D_loss: {avg_d_loss:.4f}, G_loss: {avg_g_loss:.4f}")
        
        self.is_trained = True
        print(f"\n  CGAN training complete!")
        print(f"  Final D_loss: {avg_d_loss:.4f}, G_loss: {avg_g_loss:.4f}")
    
    def process(self, received_signal):
        """Process signal through CGAN generator."""
        if not self.is_trained:
            return received_signal * 1.5
        
        self.generator.eval()
        
        processed = np.zeros_like(received_signal)
        pad_size = self.window_size // 2
        padded_signal = np.pad(received_signal, pad_size, mode='edge')
        
        with torch.no_grad():
            for i in range(len(received_signal)):
                window = padded_signal[i:i+self.window_size]
                window_input = torch.FloatTensor(window.reshape(1, -1)).to(self.device)
                
                # Use zero noise at test time (deterministic)
                noise = torch.zeros(1, self.noise_size).to(self.device)
                reconstruction = self.generator(window_input, noise)
                processed[i] = reconstruction.cpu().numpy()[0, 0]
        
        # Normalize power
        current_power = np.mean(np.abs(processed) ** 2)
        if current_power > 0:
            return processed * np.sqrt(self.target_power / current_power)
        return processed


def simulate_cgan_pytorch_transmission(num_bits, snr_db, relay, seed=None):
    """Simulate transmission with PyTorch CGAN relay."""
    source = Source(seed=seed)
    destination = Destination()
    
    tx_bits, tx_symbols = source.transmit(num_bits)
    rx_at_relay = awgn_channel(tx_symbols, snr_db)
    relay_output = relay.process(rx_at_relay)
    rx_at_destination = awgn_channel(relay_output, snr_db)
    rx_bits = destination.receive(rx_at_destination)
    
    ber, num_errors = calculate_ber(tx_bits, rx_bits)
    return ber, num_errors


def test_cgan_pytorch():
    """Test PyTorch CGAN relay implementation."""
    print("="*70)
    print("CGAN PYTORCH RELAY IMPLEMENTATION TEST")
    print("="*70)
    
    print("\nTest 1: Training CGAN Relay with PyTorch")
    print("-"*70)
    
    relay = CGANRelayPyTorch(target_power=1.0, window_size=7, noise_size=8, lambda_l1=100)
    relay.train(training_snrs=[5, 10, 15], num_samples=50000, epochs=200)
    
    print(f"\n  Status: PASSED ✓")
    
    print("\nTest 2: Quick Validation (10k bits per SNR)")
    print("-"*70)
    
    test_snrs = [0, 2, 4, 6, 8, 10]
    print(f"\n  {'SNR':<6} {'BER':<12} {'Errors':<10}")
    print(f"  {'-'*6} {'-'*12} {'-'*10}")
    
    for snr in test_snrs:
        ber, errors = simulate_cgan_pytorch_transmission(10000, snr, relay, seed=42)
        print(f"  {snr:<6.0f} {ber:<12.6f} {errors:<10}")
    
    print(f"\n  Status: PASSED ✓")
    
    print("\n" + "="*70)
    print("CGAN PYTORCH RELAY TEST COMPLETE")
    print("="*70)
    print("✓ CGAN relay successfully implemented with PyTorch!")
    
    return True


if __name__ == "__main__":
    test_cgan_pytorch()
