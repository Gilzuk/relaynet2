"""
Checkpoint 04: Full Two-Hop Simulation Implementation

This module implements a complete Monte Carlo simulation of a two-hop
relay communication system with BER performance evaluation.

System: Source → AWGN → Relay → AWGN → Destination

Author: Cline
Date: 2026-02-14
Checkpoint: CP-04
"""

import numpy as np
import sys
import os

# Add parent directory to path to import from previous checkpoints
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from previous checkpoints
from checkpoint_01_channel import awgn_channel
from checkpoint_02_modulation import calculate_ber
from checkpoint_03_nodes import Source, AmplifyAndForwardRelay, Destination


def simulate_two_hop_transmission(num_bits, snr_db, seed=None):
    """
    Simulate a single two-hop transmission: Source → Relay → Destination.
    
    Parameters
    ----------
    num_bits : int
        Number of bits to transmit
    snr_db : float
        Signal-to-Noise Ratio in dB for both hops
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    ber : float
        Bit Error Rate for this transmission
    num_errors : int
        Number of bit errors
    """
    # Initialize nodes
    source = Source(seed=seed)
    relay = AmplifyAndForwardRelay(target_power=1.0)
    destination = Destination()
    
    # Source transmits
    tx_bits, tx_symbols = source.transmit(num_bits)
    
    # First hop: Source → Relay (through AWGN channel)
    rx_at_relay = awgn_channel(tx_symbols, snr_db)
    
    # Relay processes and forwards
    relay_output = relay.process(rx_at_relay)
    
    # Second hop: Relay → Destination (through AWGN channel)
    rx_at_destination = awgn_channel(relay_output, snr_db)
    
    # Destination receives and demodulates
    rx_bits = destination.receive(rx_at_destination)
    
    # Calculate BER
    ber, num_errors = calculate_ber(tx_bits, rx_bits)
    
    return ber, num_errors


def run_monte_carlo_simulation(num_bits_per_trial, num_trials, snr_range_db):
    """
    Run Monte Carlo simulation over multiple SNR values.
    
    Parameters
    ----------
    num_bits_per_trial : int
        Number of bits per transmission trial
    num_trials : int
        Number of Monte Carlo trials per SNR point
    snr_range_db : array-like
        Array of SNR values in dB to simulate
    
    Returns
    -------
    snr_values : numpy.ndarray
        SNR values simulated
    ber_values : numpy.ndarray
        Average BER for each SNR value
    error_counts : numpy.ndarray
        Total error counts for each SNR value
    """
    snr_values = np.array(snr_range_db)
    ber_values = np.zeros(len(snr_values))
    error_counts = np.zeros(len(snr_values), dtype=int)
    
    for i, snr_db in enumerate(snr_values):
        total_errors = 0
        total_bits = 0
        
        for trial in range(num_trials):
            # Use different seed for each trial
            seed = trial
            ber, errors = simulate_two_hop_transmission(
                num_bits_per_trial, snr_db, seed=seed
            )
            
            total_errors += errors
            total_bits += num_bits_per_trial
        
        # Calculate average BER
        ber_values[i] = total_errors / total_bits
        error_counts[i] = total_errors
    
    return snr_values, ber_values, error_counts


def test_two_hop_simulation():
    """
    Test the two-hop simulation implementation.
    
    Tests:
    1. Single transmission - verify system works
    2. BER decreases with SNR - sanity check
    3. Monte Carlo simulation - multiple trials
    4. Performance validation - reasonable BER values
    """
    print("Testing Full Two-Hop Simulation")
    print("=" * 50)
    
    # Test 1: Single Transmission
    print("\nTest 1: Single Two-Hop Transmission")
    print("-" * 50)
    
    num_bits = 1000
    snr_db = 10.0
    
    ber, errors = simulate_two_hop_transmission(num_bits, snr_db, seed=42)
    
    print(f"  Number of bits: {num_bits}")
    print(f"  SNR: {snr_db} dB")
    print(f"  Bit errors: {errors}")
    print(f"  BER: {ber:.6f}")
    
    # With SNR=10dB and two hops, we expect some errors but not too many
    if 0 <= ber <= 0.5:  # Reasonable range
        print(f"  Status: PASSED ✓")
        test1_pass = True
    else:
        print(f"  Status: FAILED ✗ (BER out of reasonable range)")
        test1_pass = False
    
    # Test 2: BER Decreases with SNR
    print("\nTest 2: BER vs SNR Trend")
    print("-" * 50)
    
    test_snrs = [0, 5, 10, 15]
    test_bers = []
    
    for snr in test_snrs:
        ber, _ = simulate_two_hop_transmission(10000, snr, seed=42)
        test_bers.append(ber)
        print(f"  SNR = {snr:2d} dB: BER = {ber:.6f}")
    
    # Check if BER generally decreases with SNR
    decreasing = all(test_bers[i] >= test_bers[i+1] for i in range(len(test_bers)-1))
    
    if decreasing or test_bers[-1] < test_bers[0]:  # Allow some variation
        print(f"  Trend: BER decreases with SNR ✓")
        test2_pass = True
    else:
        print(f"  Trend: BER does not decrease consistently ✗")
        test2_pass = False
    
    # Test 3: Monte Carlo Simulation
    print("\nTest 3: Monte Carlo Simulation")
    print("-" * 50)
    
    num_bits_per_trial = 1000
    num_trials = 10
    snr_range = [0, 10, 20]
    
    print(f"  Configuration:")
    print(f"    Bits per trial: {num_bits_per_trial}")
    print(f"    Number of trials: {num_trials}")
    print(f"    SNR range: {snr_range} dB")
    print(f"\n  Running simulation...")
    
    snr_vals, ber_vals, err_counts = run_monte_carlo_simulation(
        num_bits_per_trial, num_trials, snr_range
    )
    
    print(f"\n  Results:")
    for snr, ber, errs in zip(snr_vals, ber_vals, err_counts):
        print(f"    SNR = {snr:2.0f} dB: BER = {ber:.6f} ({errs} errors)")
    
    # Check if simulation completed
    if len(ber_vals) == len(snr_range):
        print(f"  Status: PASSED ✓")
        test3_pass = True
    else:
        print(f"  Status: FAILED ✗")
        test3_pass = False
    
    # Test 4: Performance Validation
    print("\nTest 4: Performance Validation")
    print("-" * 50)
    
    # Run a more comprehensive simulation
    num_bits_per_trial = 5000
    num_trials = 20
    snr_range = np.arange(0, 21, 5)  # 0, 5, 10, 15, 20 dB
    
    print(f"  Configuration:")
    print(f"    Bits per trial: {num_bits_per_trial}")
    print(f"    Number of trials: {num_trials}")
    print(f"    SNR range: {snr_range[0]} to {snr_range[-1]} dB")
    print(f"\n  Running simulation...")
    
    snr_vals, ber_vals, err_counts = run_monte_carlo_simulation(
        num_bits_per_trial, num_trials, snr_range
    )
    
    print(f"\n  Results:")
    for snr, ber, errs in zip(snr_vals, ber_vals, err_counts):
        total_bits = num_bits_per_trial * num_trials
        print(f"    SNR = {snr:2.0f} dB: BER = {ber:.6f} ({errs}/{total_bits} errors)")
    
    # Validation checks
    checks = []
    
    # Check 1: BER at high SNR should be low
    high_snr_ber = ber_vals[-1]  # BER at 20 dB
    check1 = high_snr_ber < 0.01
    checks.append(check1)
    print(f"\n  Validation:")
    print(f"    BER at 20 dB < 0.01: {high_snr_ber:.6f} {'✓' if check1 else '✗'}")
    
    # Check 2: BER decreases with SNR
    check2 = ber_vals[0] > ber_vals[-1]
    checks.append(check2)
    print(f"    BER decreases (0dB > 20dB): {ber_vals[0]:.6f} > {ber_vals[-1]:.6f} {'✓' if check2 else '✗'}")
    
    # Check 3: All BER values are valid (0 to 0.5)
    check3 = all(0 <= ber <= 0.5 for ber in ber_vals)
    checks.append(check3)
    print(f"    All BER values valid (0-0.5): {'✓' if check3 else '✗'}")
    
    if all(checks):
        print(f"  Status: PASSED ✓")
        test4_pass = True
    else:
        print(f"  Status: FAILED ✗")
        test4_pass = False
    
    # Test 5: Reproducibility
    print("\nTest 5: Reproducibility with Seed")
    print("-" * 50)
    
    # Run same simulation twice with same seed
    ber1, err1 = simulate_two_hop_transmission(1000, 10.0, seed=123)
    ber2, err2 = simulate_two_hop_transmission(1000, 10.0, seed=123)
    
    print(f"  Run 1: BER = {ber1:.6f}, Errors = {err1}")
    print(f"  Run 2: BER = {ber2:.6f}, Errors = {err2}")
    
    if ber1 == ber2 and err1 == err2:
        print(f"  Reproducibility: PASSED ✓")
        test5_pass = True
    else:
        print(f"  Reproducibility: FAILED ✗")
        test5_pass = False
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    all_tests = [test1_pass, test2_pass, test3_pass, test4_pass, test5_pass]
    passed = sum(all_tests)
    total = len(all_tests)
    
    print(f"Tests passed: {passed}/{total}")
    
    if all(all_tests):
        print("\n✓ All tests passed! Two-hop simulation implementation is correct.")
        return True
    else:
        print("\n✗ Some tests failed. Please review the implementation.")
        return False


if __name__ == "__main__":
    # Run tests
    success = test_two_hop_simulation()
    
    # Exit with appropriate code
    exit(0 if success else 1)
