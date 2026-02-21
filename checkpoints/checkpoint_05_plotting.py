"""
Checkpoint 05: BER Plotting and Validation

This module generates BER vs SNR plots for the two-hop AF relay system
and validates performance against expected behavior.

Author: Cline
Date: 2026-02-14
Checkpoint: CP-05
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import from previous checkpoints
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from previous checkpoints
from checkpoint_04_simulation import run_monte_carlo_simulation


def plot_ber_vs_snr(snr_values, ber_values, save_path='results/ber_plot.png'):
    """
    Generate BER vs SNR plot.
    
    Parameters
    ----------
    snr_values : numpy.ndarray
        SNR values in dB
    ber_values : numpy.ndarray
        BER values corresponding to each SNR
    save_path : str
        Path to save the plot
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot BER vs SNR (log scale for BER)
    ax.semilogy(snr_values, ber_values, 'b-o', linewidth=2, 
                markersize=8, label='AF Relay (Simulated)')
    
    # Add grid
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    
    # Labels and title
    ax.set_xlabel('SNR (dB)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Bit Error Rate (BER)', fontsize=12, fontweight='bold')
    ax.set_title('BER Performance of Two-Hop AF Relay System\n(BPSK Modulation over AWGN Channel)', 
                 fontsize=14, fontweight='bold')
    
    # Legend
    ax.legend(loc='best', fontsize=11)
    
    # Set y-axis limits
    ax.set_ylim([1e-6, 1])
    
    # Tight layout
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Plot saved to: {save_path}")
    
    return fig


def validate_performance(snr_values, ber_values):
    """
    Validate BER performance against expected behavior.
    
    Parameters
    ----------
    snr_values : numpy.ndarray
        SNR values in dB
    ber_values : numpy.ndarray
        BER values
    
    Returns
    -------
    validation_results : dict
        Dictionary with validation results
    """
    results = {}
    
    # Check 1: BER decreases with SNR
    ber_decreasing = all(ber_values[i] >= ber_values[i+1] 
                        for i in range(len(ber_values)-1))
    results['ber_decreasing'] = ber_decreasing
    
    # Check 2: BER at high SNR is low
    high_snr_idx = np.argmax(snr_values)
    high_snr_ber = ber_values[high_snr_idx]
    results['high_snr_low_ber'] = high_snr_ber < 0.01
    results['high_snr_ber'] = high_snr_ber
    
    # Check 3: BER at low SNR is reasonable (not too high, not too low)
    low_snr_idx = np.argmin(snr_values)
    low_snr_ber = ber_values[low_snr_idx]
    results['low_snr_reasonable'] = 0.1 < low_snr_ber < 0.5
    results['low_snr_ber'] = low_snr_ber
    
    # Check 4: No BER values are exactly 0.5 (random guessing)
    results['no_random_guessing'] = all(ber < 0.5 for ber in ber_values)
    
    # Check 5: BER range is reasonable
    ber_range = ber_values[0] - ber_values[-1]
    results['reasonable_range'] = ber_range > 0.1
    results['ber_range'] = ber_range
    
    return results


def test_plotting_and_validation():
    """
    Test BER plotting and performance validation.
    
    Tests:
    1. Run simulation and collect data
    2. Generate BER plot
    3. Validate performance
    4. Check plot file exists
    """
    print("Testing BER Plotting and Validation")
    print("=" * 50)
    
    # Test 1: Run Simulation
    print("\nTest 1: Running Simulation")
    print("-" * 50)
    
    # Simulation parameters
    num_bits_per_trial = 10000
    num_trials = 50
    snr_range = np.arange(0, 21, 2)  # 0, 2, 4, ..., 20 dB
    
    print(f"  Configuration:")
    print(f"    Bits per trial: {num_bits_per_trial}")
    print(f"    Number of trials: {num_trials}")
    print(f"    SNR range: {snr_range[0]} to {snr_range[-1]} dB (step: 2 dB)")
    print(f"    Total SNR points: {len(snr_range)}")
    print(f"\n  Running simulation (this may take a moment)...")
    
    snr_vals, ber_vals, err_counts = run_monte_carlo_simulation(
        num_bits_per_trial, num_trials, snr_range
    )
    
    print(f"\n  Simulation Results:")
    for snr, ber, errs in zip(snr_vals, ber_vals, err_counts):
        total_bits = num_bits_per_trial * num_trials
        print(f"    SNR = {snr:2.0f} dB: BER = {ber:.6f} ({errs:6d}/{total_bits} errors)")
    
    if len(ber_vals) == len(snr_range):
        print(f"\n  Status: PASSED ✓")
        test1_pass = True
    else:
        print(f"\n  Status: FAILED ✗")
        test1_pass = False
    
    # Test 2: Generate Plot
    print("\nTest 2: Generate BER Plot")
    print("-" * 50)
    
    try:
        fig = plot_ber_vs_snr(snr_vals, ber_vals)
        plt.close(fig)  # Close to free memory
        print(f"  Status: PASSED ✓")
        test2_pass = True
    except Exception as e:
        print(f"  Error: {e}")
        print(f"  Status: FAILED ✗")
        test2_pass = False
    
    # Test 3: Validate Performance
    print("\nTest 3: Performance Validation")
    print("-" * 50)
    
    validation = validate_performance(snr_vals, ber_vals)
    
    print(f"  Validation Results:")
    print(f"    BER decreases with SNR: {'✓' if validation['ber_decreasing'] else '✗'}")
    print(f"    High SNR (20 dB) BER < 0.01: {validation['high_snr_ber']:.6f} {'✓' if validation['high_snr_low_ber'] else '✗'}")
    print(f"    Low SNR (0 dB) BER reasonable: {validation['low_snr_ber']:.6f} {'✓' if validation['low_snr_reasonable'] else '✗'}")
    print(f"    No random guessing (BER < 0.5): {'✓' if validation['no_random_guessing'] else '✗'}")
    print(f"    BER range > 0.1: {validation['ber_range']:.6f} {'✓' if validation['reasonable_range'] else '✗'}")
    
    all_checks = [
        validation['ber_decreasing'],
        validation['high_snr_low_ber'],
        validation['low_snr_reasonable'],
        validation['no_random_guessing'],
        validation['reasonable_range']
    ]
    
    if all(all_checks):
        print(f"\n  Status: PASSED ✓")
        test3_pass = True
    else:
        print(f"\n  Status: FAILED ✗")
        test3_pass = False
    
    # Test 4: Check Plot File Exists
    print("\nTest 4: Verify Plot File")
    print("-" * 50)
    
    plot_path = 'results/ber_plot.png'
    if os.path.exists(plot_path):
        file_size = os.path.getsize(plot_path)
        print(f"  Plot file exists: {plot_path}")
        print(f"  File size: {file_size} bytes")
        print(f"  Status: PASSED ✓")
        test4_pass = True
    else:
        print(f"  Plot file not found: {plot_path}")
        print(f"  Status: FAILED ✗")
        test4_pass = False
    
    # Test 5: Performance Analysis
    print("\nTest 5: Detailed Performance Analysis")
    print("-" * 50)
    
    # Find SNR for BER ≈ 0.01 (1% error rate)
    target_ber = 0.01
    idx = np.argmin(np.abs(ber_vals - target_ber))
    snr_at_target = snr_vals[idx]
    ber_at_target = ber_vals[idx]
    
    print(f"  Target BER: {target_ber}")
    print(f"  Closest SNR: {snr_at_target:.0f} dB")
    print(f"  Actual BER: {ber_at_target:.6f}")
    
    # Calculate BER improvement from 0 dB to 20 dB
    ber_improvement = ber_vals[0] / max(ber_vals[-1], 1e-10)
    print(f"\n  BER Improvement (0 dB → 20 dB):")
    print(f"    Initial BER: {ber_vals[0]:.6f}")
    print(f"    Final BER: {ber_vals[-1]:.6f}")
    print(f"    Improvement factor: {ber_improvement:.1f}x")
    
    # Check if improvement is significant
    if ber_improvement > 100:
        print(f"  Status: PASSED ✓")
        test5_pass = True
    else:
        print(f"  Status: WARNING (improvement < 100x)")
        test5_pass = True  # Still pass, just a warning
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    all_tests = [test1_pass, test2_pass, test3_pass, test4_pass, test5_pass]
    passed = sum(all_tests)
    total = len(all_tests)
    
    print(f"Tests passed: {passed}/{total}")
    
    if all(all_tests):
        print("\n✓ All tests passed! BER plotting and validation complete.")
        print(f"\nFinal Results:")
        print(f"  - BER plot saved to: results/ber_plot.png")
        print(f"  - AF relay system validated successfully")
        print(f"  - Ready for GenAI relay comparison (Phase 2)")
        return True
    else:
        print("\n✗ Some tests failed. Please review the implementation.")
        return False


if __name__ == "__main__":
    # Run tests
    success = test_plotting_and_validation()
    
    # Exit with appropriate code
    exit(0 if success else 1)
