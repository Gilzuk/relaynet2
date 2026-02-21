"""
Checkpoint 19: Transformer vs DF Comparison

Compare Transformer with attention mechanism against:
1. DF (Classical baseline)
2. Minimal GenAI (169 params)
3. Transformer (17.7k params with attention)

Can attention beat DF?

Author: Cline
Date: 2026-02-15
Checkpoint: CP-19
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from checkpoint_06_decode_forward import simulate_df_transmission
from checkpoint_13_minimal_complexity import MinimalGenAIRelay, simulate_minimal_genai
from checkpoint_18_transformer_relay import TransformerRelayWrapper, simulate_transformer_transmission


def run_transformer_comparison():
    """
    Run comprehensive Transformer vs DF comparison.
    """
    print("="*70)
    print("TRANSFORMER VS DF COMPARISON")
    print("="*70)
    print("\nQuestion: Can Transformer with Attention beat DF?")
    print("\nComparing:")
    print("  1. DF (Classical, 0 params)")
    print("  2. Minimal GenAI (169 params)")
    print("  3. Transformer (17.7k params, multi-head attention)")
    
    snr_range = np.arange(0, 21, 2)
    
    # Initialize results
    results = {
        'snr': snr_range,
        'df': np.zeros(len(snr_range)),
        'minimal': np.zeros(len(snr_range)),
        'transformer': np.zeros(len(snr_range))
    }
    
    # Train models
    print("\n1. Training Minimal GenAI (169 params)...")
    minimal_relay = MinimalGenAIRelay(window_size=5, hidden1=24, hidden2=0)
    minimal_relay.train_minimal(
        training_snrs=[5, 10, 15],
        num_samples=25000,
        epochs=100
    )
    
    print("\n2. Training Transformer (17.7k params)...")
    transformer_relay = TransformerRelayWrapper(
        target_power=1.0,
        window_size=11,
        d_model=32,
        num_heads=4,
        num_layers=2
    )
    transformer_relay.train(training_snrs=[5, 10, 15], num_samples=50000, epochs=100, lr=0.001)
    
    # Run validation
    print("\n3. Running Validation (100k bits per SNR)...")
    for i, snr_db in enumerate(snr_range):
        print(f"\n  Testing SNR = {snr_db} dB...")
        
        # DF
        df_errors = 0
        for trial in range(10):
            _, errors = simulate_df_transmission(10000, snr_db, seed=trial)
            df_errors += errors
        results['df'][i] = df_errors / 100000
        
        # Minimal
        min_errors = 0
        for trial in range(10):
            _, errors = simulate_minimal_genai(10000, snr_db, minimal_relay, seed=trial)
            min_errors += errors
        results['minimal'][i] = min_errors / 100000
        
        # Transformer
        trans_errors = 0
        for trial in range(10):
            _, errors = simulate_transformer_transmission(10000, snr_db, transformer_relay, seed=trial)
            trans_errors += errors
        results['transformer'][i] = trans_errors / 100000
        
        print(f"    DF:          {results['df'][i]:.6f}")
        print(f"    Minimal:     {results['minimal'][i]:.6f}")
        print(f"    Transformer: {results['transformer'][i]:.6f}")
    
    return results


def create_transformer_plot(results, save_path='results/transformer_comparison.png'):
    """
    Create Transformer comparison plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    snr = results['snr']
    
    # Plot 1: Full BER Comparison
    ax1.semilogy(snr, results['df'], 'k-o', linewidth=3, markersize=10,
                 label='DF (0 params)', markerfacecolor='black', alpha=0.8)
    ax1.semilogy(snr, results['minimal'], 'm-d', linewidth=3, markersize=10,
                 label='Minimal (169 params)', markerfacecolor='magenta', alpha=0.8)
    ax1.semilogy(snr, results['transformer'], 'g-^', linewidth=3, markersize=10,
                 label='Transformer (17.7k params)', markerfacecolor='green', alpha=0.8)
    
    ax1.grid(True, which='both', linestyle='--', alpha=0.6)
    ax1.set_xlabel('SNR (dB)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Bit Error Rate (BER)', fontsize=14, fontweight='bold')
    ax1.set_title('Transformer vs DF: Can Attention Win?', fontsize=15, fontweight='bold')
    ax1.legend(loc='best', fontsize=12, framealpha=0.95)
    ax1.set_ylim([1e-6, 1])
    
    # Count wins
    min_wins = sum(results['minimal'] < results['df'])
    trans_wins = sum(results['transformer'] < results['df'])
    
    textstr = 'Performance vs DF:\n'
    textstr += '━━━━━━━━━━━━━━━━━━━━━\n'
    textstr += f'Minimal: {min_wins}/11 wins\n'
    textstr += f'Transformer: {trans_wins}/11 wins\n'
    textstr += '\n'
    textstr += 'Parameters:\n'
    textstr += 'DF: 0\n'
    textstr += 'Minimal: 169\n'
    textstr += 'Transformer: 17,697\n'
    textstr += '\n'
    textstr += 'Attention Layers: 2\n'
    textstr += 'Attention Heads: 4'
    
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.9, edgecolor='black', linewidth=1.5)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props, family='monospace')
    
    # Plot 2: Low SNR Focus (0-10 dB)
    low_snr_mask = snr <= 10
    ax2.semilogy(snr[low_snr_mask], results['df'][low_snr_mask], 'k-o', 
                 linewidth=3, markersize=12, label='DF', markerfacecolor='black', alpha=0.8)
    ax2.semilogy(snr[low_snr_mask], results['minimal'][low_snr_mask], 'm-d',
                 linewidth=3, markersize=12, label='Minimal', markerfacecolor='magenta', alpha=0.8)
    ax2.semilogy(snr[low_snr_mask], results['transformer'][low_snr_mask], 'g-^',
                 linewidth=3, markersize=12, label='Transformer', markerfacecolor='green', alpha=0.8)
    
    ax2.grid(True, which='both', linestyle='--', alpha=0.6)
    ax2.set_xlabel('SNR (dB)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Bit Error Rate (BER)', fontsize=14, fontweight='bold')
    ax2.set_title('Low SNR: Attention Mechanism Performance', fontsize=15, fontweight='bold')
    ax2.legend(loc='best', fontsize=12, framealpha=0.95)
    
    # Add winner annotations
    for i, s in enumerate(snr[low_snr_mask]):
        bers = [results['df'][i], results['minimal'][i], results['transformer'][i]]
        winner_idx = np.argmin(bers)
        winners = ['DF', 'Min', 'Trans']
        if winner_idx > 0:  # Not DF
            ax2.annotate(winners[winner_idx], xy=(s, bers[winner_idx]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Transformer comparison plot saved to: {save_path}")
    
    return fig


def print_transformer_summary(results):
    """
    Print comprehensive summary.
    """
    print("\n" + "="*70)
    print("TRANSFORMER VS DF: FINAL VERDICT")
    print("="*70)
    
    print(f"\n{'SNR':<6} {'DF':<12} {'Minimal':<12} {'Transformer':<12} {'Winner':<12}")
    print(f"{'-'*6} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    
    for i, snr in enumerate(results['snr']):
        bers = [results['df'][i], results['minimal'][i], results['transformer'][i]]
        winner_idx = np.argmin(bers)
        winners = ['DF', 'Minimal', 'Transformer']
        winner = winners[winner_idx]
        
        print(f"{snr:<6.0f} {results['df'][i]:<12.6f} {results['minimal'][i]:<12.6f} "
              f"{results['transformer'][i]:<12.6f} {winner:<12}")
    
    # Win counts
    print(f"\n{'Model':<15} {'Params':<10} {'Wins vs DF':<12} {'Efficiency':<12}")
    print(f"{'-'*15} {'-'*10} {'-'*12} {'-'*12}")
    
    models = [
        ('Minimal', 169, results['minimal']),
        ('Transformer', 17697, results['transformer'])
    ]
    
    for name, params, ber in models:
        wins = sum(ber < results['df'])
        efficiency = wins / (params / 100) if params > 0 else 0
        print(f"{name:<15} {params:<10,} {wins}/11{'':<7} {efficiency:<12.2f}")
    
    print("\n" + "="*70)
    print("KEY FINDINGS: CAN ATTENTION BEAT DF?")
    print("="*70)
    
    # Determine best at each SNR range
    low_snr_wins = {'Minimal': 0, 'Transformer': 0}
    for i in range(3):  # 0, 2, 4 dB
        bers = [results['minimal'][i], results['transformer'][i]]
        winner_idx = np.argmin(bers)
        winners_list = ['Minimal', 'Transformer']
        low_snr_wins[winners_list[winner_idx]] += 1
    
    print(f"\n✓ Low SNR (0-4 dB) Performance:")
    for model, wins in low_snr_wins.items():
        print(f"  {model}: {wins}/3 wins")
    
    # Overall verdict
    trans_total_wins = sum(results['transformer'] < results['df'])
    min_total_wins = sum(results['minimal'] < results['df'])
    
    print(f"\n✓ Overall Performance vs DF:")
    print(f"  Transformer: {trans_total_wins}/11 wins")
    print(f"  Minimal:     {min_total_wins}/11 wins")
    
    print(f"\n✓ Efficiency (wins per 100 params):")
    print(f"  Minimal:     {min_total_wins / 1.69:.2f}")
    print(f"  Transformer: {trans_total_wins / 176.97:.2f}")
    
    # Final verdict
    print(f"\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    
    if trans_total_wins > min_total_wins:
        print(f"\n🏆 TRANSFORMER WINS!")
        print(f"  Attention mechanism beats simple feedforward!")
        print(f"  {trans_total_wins}/11 wins vs DF")
    elif trans_total_wins == min_total_wins:
        print(f"\n⚖️  TIE!")
        print(f"  Both win {trans_total_wins}/11 times vs DF")
        print(f"  But Minimal is 105x smaller (169 vs 17,697 params)")
    else:
        print(f"\n🎯 MINIMAL WINS!")
        print(f"  Simple network beats complex attention!")
        print(f"  {min_total_wins}/11 wins vs {trans_total_wins}/11")
        print(f"  And 105x smaller (169 vs 17,697 params)")
    
    # Can it beat DF?
    if trans_total_wins >= 6:
        print(f"\n✅ YES! Transformer CAN beat DF")
        print(f"  Wins {trans_total_wins}/11 SNR points")
    else:
        print(f"\n❌ NO. Transformer cannot beat DF overall")
        print(f"  Only wins {trans_total_wins}/11 SNR points")
        print(f"  DF dominates at medium/high SNR")


def test_transformer_comparison():
    """
    Main test function.
    """
    print("\nStarting Transformer vs DF comparison...")
    print("Testing if attention mechanism can beat classical DF...\n")
    
    # Run comparison
    results = run_transformer_comparison()
    
    # Create plot
    print("\n4. Creating Comparison Plot...")
    fig = create_transformer_plot(results)
    plt.close(fig)
    
    # Print summary
    print_transformer_summary(results)
    
    print("\n✓ Transformer comparison complete!")
    return True


if __name__ == "__main__":
    test_transformer_comparison()
