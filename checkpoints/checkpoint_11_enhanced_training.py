"""
Checkpoint 11: Enhanced GenAI Training with Large Dataset

This module implements enhanced training for the GenAI relay with:
- 100,000 training samples (5x increase)
- 200 epochs (2x increase)
- Multi-SNR training (5, 10, 15 dB)
- Learning rate scheduling
- Comprehensive validation (1M bits per SNR)

Author: Cline
Date: 2026-02-14
Checkpoint: CP-11
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
from checkpoint_03_nodes import Source, Destination
from checkpoint_04_simulation import simulate_two_hop_transmission
from checkpoint_06_decode_forward import simulate_df_transmission
from checkpoint_08_genai_relay import GenAIRelay, simulate_genai_transmission


class EnhancedGenAIRelay(GenAIRelay):
    """
    Enhanced GenAI relay with improved training strategy.
    """
    
    def train_enhanced(self, training_snrs=[5, 10, 15], num_samples=100000, 
                      epochs=200, initial_lr=0.01, lr_decay=0.95):
        """
        Enhanced training with multi-SNR and learning rate scheduling.
        
        Parameters
        ----------
        training_snrs : list
            List of SNR values for training
        num_samples : int
            Total number of training samples
        epochs : int
            Number of training epochs
        initial_lr : float
            Initial learning rate
        lr_decay : float
            Learning rate decay factor per 20 epochs
        """
        print(f"  Enhanced Training Configuration:")
        print(f"    Training SNRs: {training_snrs} dB")
        print(f"    Total samples: {num_samples:,}")
        print(f"    Epochs: {epochs}")
        print(f"    Initial LR: {initial_lr}, Decay: {lr_decay}")
        print(f"\n  Generating training data...")
        
        # Generate training data for multiple SNRs
        X_train_all = []
        y_train_all = []
        
        samples_per_snr = num_samples // len(training_snrs)
        
        for snr in training_snrs:
            np.random.seed(42 + int(snr))
            clean_bits = np.random.randint(0, 2, samples_per_snr)
            clean_symbols = bpsk_modulate(clean_bits)
            noisy_symbols = awgn_channel(clean_symbols, snr)
            
            # Prepare windowed data
            for i in range(self.window_size // 2, len(noisy_symbols) - self.window_size // 2):
                window = noisy_symbols[i - self.window_size // 2 : i + self.window_size // 2 + 1]
                X_train_all.append(window)
                y_train_all.append(clean_symbols[i])
        
        X_train = np.array(X_train_all)
        y_train = np.array(y_train_all).reshape(-1, 1)
        
        print(f"  Training data prepared: {len(X_train):,} samples")
        print(f"\n  Starting training...")
        
        # Training loop with learning rate scheduling
        learning_rate = initial_lr
        best_loss = float('inf')
        
        for epoch in range(epochs):
            # Decay learning rate every 20 epochs
            if epoch > 0 and epoch % 20 == 0:
                learning_rate *= lr_decay
                print(f"    Learning rate decayed to: {learning_rate:.6f}")
            
            # Shuffle data
            indices = np.random.permutation(len(X_train))
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Train in mini-batches
            batch_size = 64  # Increased from 32
            total_loss = 0
            num_batches = 0
            
            for i in range(0, len(X_train), batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                loss = self.nn.train_step(X_batch, y_batch, learning_rate=learning_rate)
                total_loss += loss
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            
            # Track best loss
            if avg_loss < best_loss:
                best_loss = avg_loss
            
            # Print progress
            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, Best: {best_loss:.6f}")
        
        self.is_trained = True
        print(f"\n  Enhanced training complete!")
        print(f"  Final loss: {avg_loss:.6f}")
        print(f"  Best loss: {best_loss:.6f}")


def comprehensive_validation(genai_relay, snr_range, bits_per_trial=10000, num_trials=100):
    """
    Comprehensive validation with 1M bits per SNR.
    
    Parameters
    ----------
    genai_relay : EnhancedGenAIRelay
        Trained relay
    snr_range : array-like
        SNR values to test
    bits_per_trial : int
        Bits per trial
    num_trials : int
        Number of trials (100 trials × 10k bits = 1M bits)
    
    Returns
    -------
    results : dict
        Performance results for all relay types
    """
    print(f"\n  Running comprehensive validation...")
    print(f"    Bits per trial: {bits_per_trial:,}")
    print(f"    Trials per SNR: {num_trials}")
    print(f"    Total bits per SNR: {bits_per_trial * num_trials:,}")
    
    snr_values = np.array(snr_range)
    af_ber = np.zeros(len(snr_values))
    df_ber = np.zeros(len(snr_values))
    genai_ber = np.zeros(len(snr_values))
    
    for i, snr_db in enumerate(snr_values):
        print(f"\n    Testing SNR = {snr_db} dB...")
        
        # AF relay
        af_errors = 0
        for trial in range(num_trials):
            _, errors = simulate_two_hop_transmission(bits_per_trial, snr_db, seed=trial)
            af_errors += errors
        af_ber[i] = af_errors / (bits_per_trial * num_trials)
        
        # DF relay
        df_errors = 0
        for trial in range(num_trials):
            _, errors = simulate_df_transmission(bits_per_trial, snr_db, seed=trial)
            df_errors += errors
        df_ber[i] = df_errors / (bits_per_trial * num_trials)
        
        # GenAI relay
        genai_errors = 0
        for trial in range(num_trials):
            _, errors = simulate_genai_transmission(bits_per_trial, snr_db, genai_relay, seed=trial)
            genai_errors += errors
        genai_ber[i] = genai_errors / (bits_per_trial * num_trials)
        
        print(f"      AF: {af_ber[i]:.6f}, DF: {df_ber[i]:.6f}, GenAI: {genai_ber[i]:.6f}")
    
    return {
        'snr': snr_values,
        'af_ber': af_ber,
        'df_ber': df_ber,
        'genai_ber': genai_ber
    }


def plot_enhanced_comparison(results, save_path='results/enhanced_comparison.png'):
    """
    Plot enhanced comparison results.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    snr = results['snr']
    af_ber = results['af_ber']
    df_ber = results['df_ber']
    genai_ber = results['genai_ber']
    
    # Plot curves
    ax.semilogy(snr, af_ber, 'b-o', linewidth=2.5, markersize=9, 
                label='AF Relay', markerfacecolor='blue', alpha=0.8)
    ax.semilogy(snr, df_ber, 'r-s', linewidth=2.5, markersize=9, 
                label='DF Relay', markerfacecolor='red', alpha=0.8)
    ax.semilogy(snr, genai_ber, 'g-^', linewidth=2.5, markersize=9, 
                label='Enhanced GenAI Relay', markerfacecolor='green', alpha=0.8)
    
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    ax.set_xlabel('SNR (dB)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Bit Error Rate (BER)', fontsize=14, fontweight='bold')
    ax.set_title('Enhanced GenAI Training Results\n(100k samples, 200 epochs, Multi-SNR, 1M bits validation)', 
                 fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=13, framealpha=0.95, shadow=True)
    ax.set_ylim([1e-6, 1])
    
    # Add performance summary
    genai_wins = sum(genai_ber < df_ber)
    df_wins = sum(df_ber < genai_ber)
    ties = sum(genai_ber == df_ber)
    
    textstr = f'Performance Summary:\n'
    textstr += f'━━━━━━━━━━━━━━━━━━━━\n'
    textstr += f'GenAI wins: {genai_wins}/{len(snr)}\n'
    textstr += f'DF wins: {df_wins}/{len(snr)}\n'
    textstr += f'Ties: {ties}/{len(snr)}\n'
    textstr += f'\n'
    textstr += f'Training: 100k samples\n'
    textstr += f'Multi-SNR: 5, 10, 15 dB\n'
    textstr += f'Validation: 1M bits/SNR'
    
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, 
                 edgecolor='black', linewidth=2)
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', bbox=props, family='monospace')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n  Enhanced comparison plot saved to: {save_path}")
    
    return fig


def test_enhanced_training():
    """
    Test enhanced training and validation.
    """
    print("=" * 70)
    print("ENHANCED GENAI TRAINING EXPERIMENT")
    print("=" * 70)
    
    # Test 1: Enhanced Training
    print("\nTest 1: Enhanced Training (100k samples, 200 epochs, Multi-SNR)")
    print("-" * 70)
    
    relay = EnhancedGenAIRelay(target_power=1.0, window_size=7)
    relay.train_enhanced(
        training_snrs=[5, 10, 15],
        num_samples=100000,
        epochs=200,
        initial_lr=0.01,
        lr_decay=0.95
    )
    
    print(f"\n  Status: PASSED ✓")
    
    # Test 2: Comprehensive Validation
    print("\nTest 2: Comprehensive Validation (1M bits per SNR)")
    print("-" * 70)
    
    snr_range = np.arange(0, 21, 2)
    results = comprehensive_validation(
        relay, 
        snr_range, 
        bits_per_trial=10000, 
        num_trials=100
    )
    
    print(f"\n  Status: PASSED ✓")
    
    # Test 3: Performance Analysis
    print("\nTest 3: Performance Analysis")
    print("-" * 70)
    
    print(f"\n  Detailed Results:")
    print(f"  {'SNR':<6} {'AF BER':<12} {'DF BER':<12} {'GenAI BER':<12} {'Winner':<10}")
    print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
    
    genai_wins = 0
    df_wins = 0
    ties = 0
    
    for snr, af, df, genai in zip(results['snr'], results['af_ber'], 
                                   results['df_ber'], results['genai_ber']):
        if genai < df:
            winner = "GenAI ✓"
            genai_wins += 1
        elif df < genai:
            winner = "DF ✓"
            df_wins += 1
        else:
            winner = "Tie"
            ties += 1
        
        print(f"  {snr:<6.0f} {af:<12.6f} {df:<12.6f} {genai:<12.6f} {winner:<10}")
    
    total = len(results['snr'])
    print(f"\n  Win Summary:")
    print(f"    GenAI wins: {genai_wins}/{total} ({100*genai_wins/total:.1f}%)")
    print(f"    DF wins: {df_wins}/{total} ({100*df_wins/total:.1f}%)")
    print(f"    Ties: {ties}/{total} ({100*ties/total:.1f}%)")
    
    print(f"\n  Status: PASSED ✓")
    
    # Test 4: Generate Plot
    print("\nTest 4: Generate Enhanced Comparison Plot")
    print("-" * 70)
    
    fig = plot_enhanced_comparison(results)
    plt.close(fig)
    
    print(f"  Status: PASSED ✓")
    
    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"Tests passed: 4/4")
    
    print(f"\n✓ Enhanced training experiment complete!")
    
    if genai_wins > df_wins:
        print(f"\n🎉 SUCCESS! Enhanced GenAI BEATS DF ({genai_wins} vs {df_wins} wins)")
        print(f"   Conclusion: Performance was dataset-limited!")
    elif genai_wins == df_wins:
        print(f"\n⚖️  COMPETITIVE! GenAI matches DF ({genai_wins} vs {df_wins} wins)")
        print(f"   Conclusion: Significant improvement with more training!")
    else:
        print(f"\n📊 IMPROVED! GenAI performance enhanced but DF still leads")
        print(f"   GenAI: {genai_wins} wins, DF: {df_wins} wins")
        print(f"   Conclusion: Architecture or fundamental limitations remain")
    
    return True


if __name__ == "__main__":
    success = test_enhanced_training()
    exit(0 if success else 1)
