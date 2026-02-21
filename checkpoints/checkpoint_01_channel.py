"""
Checkpoint 01: AWGN Channel Implementation

This module implements an Additive White Gaussian Noise (AWGN) channel
for digital communication simulation.

Author: Cline
Date: 2026-02-14
Checkpoint: CP-01
"""

import numpy as np


def awgn_channel(signal, snr_db):
    """
    Add Additive White Gaussian Noise (AWGN) to a signal.
    
    The function adds white Gaussian noise to achieve a specified
    Signal-to-Noise Ratio (SNR) in decibels.
    
    Parameters
    ----------
    signal : numpy.ndarray
        Input signal (can be real or complex)
    snr_db : float
        Target Signal-to-Noise Ratio in decibels
    
    Returns
    -------
    noisy_signal : numpy.ndarray
        Signal with added AWGN
    
    Notes
    -----
    SNR is defined as: SNR_dB = 10 * log10(P_signal / P_noise)
    where P_signal is the average signal power and P_noise is the noise power.
    """
    # Calculate signal power
    signal_power = np.mean(np.abs(signal) ** 2)
    
    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_db / 10)
    
    # Calculate required noise power
    noise_power = signal_power / snr_linear
    
    # Generate noise with appropriate variance
    # For complex signals, split noise power between real and imaginary parts
    if np.iscomplexobj(signal):
        noise_std = np.sqrt(noise_power / 2)
        noise = noise_std * (np.random.randn(len(signal)) + 
                            1j * np.random.randn(len(signal)))
    else:
        noise_std = np.sqrt(noise_power)
        noise = noise_std * np.random.randn(len(signal))
    
    # Add noise to signal
    noisy_signal = signal + noise
    
    return noisy_signal


def calculate_snr(signal, noisy_signal):
    """
    Calculate the actual SNR between a clean signal and noisy signal.
    
    Parameters
    ----------
    signal : numpy.ndarray
        Original clean signal
    noisy_signal : numpy.ndarray
        Signal with noise added
    
    Returns
    -------
    snr_db : float
        Measured SNR in decibels
    """
    # Calculate signal power
    signal_power = np.mean(np.abs(signal) ** 2)
    
    # Calculate noise (difference between noisy and clean signal)
    noise = noisy_signal - signal
    noise_power = np.mean(np.abs(noise) ** 2)
    
    # Calculate SNR in dB
    if noise_power > 0:
        snr_db = 10 * np.log10(signal_power / noise_power)
    else:
        snr_db = np.inf  # Perfect signal (no noise)
    
    return snr_db


def test_awgn_channel():
    """
    Test the AWGN channel implementation.
    
    Tests:
    1. SNR verification - check if achieved SNR matches target
    2. Noise addition - verify noise power is correct
    3. Multiple trials - ensure consistency
    """
    print("Testing AWGN Channel Implementation")
    print("=" * 50)
    
    # Test parameters
    num_samples = 100000  # Large number for accurate statistics
    target_snr_db = 10.0
    tolerance_db = 0.5  # Allow 0.5 dB deviation
    
    # Test 1: SNR Verification with real signal
    print("\nTest 1: SNR Verification (Real Signal)")
    print("-" * 50)
    
    # Generate a simple real signal (BPSK-like: +1 or -1)
    np.random.seed(42)  # For reproducibility
    signal = 2 * np.random.randint(0, 2, num_samples) - 1
    signal = signal.astype(float)
    
    # Add noise
    noisy_signal = awgn_channel(signal, target_snr_db)
    
    # Measure actual SNR
    measured_snr_db = calculate_snr(signal, noisy_signal)
    snr_error = abs(measured_snr_db - target_snr_db)
    
    print(f"  Target SNR: {target_snr_db:.2f} dB")
    print(f"  Measured SNR: {measured_snr_db:.2f} dB")
    print(f"  Difference: {snr_error:.2f} dB")
    
    if snr_error < tolerance_db:
        print(f"  Status: PASSED ✓")
        test1_pass = True
    else:
        print(f"  Status: FAILED ✗ (Error > {tolerance_db} dB)")
        test1_pass = False
    
    # Test 2: Noise Power Verification
    print("\nTest 2: Noise Power Verification")
    print("-" * 50)
    
    signal_power = np.mean(np.abs(signal) ** 2)
    noise = noisy_signal - signal
    noise_power = np.mean(np.abs(noise) ** 2)
    expected_noise_power = signal_power / (10 ** (target_snr_db / 10))
    
    print(f"  Signal power: {signal_power:.6f}")
    print(f"  Measured noise power: {noise_power:.6f}")
    print(f"  Expected noise power: {expected_noise_power:.6f}")
    print(f"  Relative error: {abs(noise_power - expected_noise_power) / expected_noise_power * 100:.2f}%")
    
    if abs(noise_power - expected_noise_power) / expected_noise_power < 0.05:  # 5% tolerance
        print(f"  Status: PASSED ✓")
        test2_pass = True
    else:
        print(f"  Status: FAILED ✗")
        test2_pass = False
    
    # Test 3: Complex Signal
    print("\nTest 3: Complex Signal Support")
    print("-" * 50)
    
    # Generate complex signal (like QPSK)
    complex_signal = (2 * np.random.randint(0, 2, num_samples) - 1) + \
                     1j * (2 * np.random.randint(0, 2, num_samples) - 1)
    complex_signal = complex_signal.astype(complex)
    
    # Add noise
    noisy_complex_signal = awgn_channel(complex_signal, target_snr_db)
    
    # Measure SNR
    measured_complex_snr_db = calculate_snr(complex_signal, noisy_complex_signal)
    complex_snr_error = abs(measured_complex_snr_db - target_snr_db)
    
    print(f"  Target SNR: {target_snr_db:.2f} dB")
    print(f"  Measured SNR: {measured_complex_snr_db:.2f} dB")
    print(f"  Difference: {complex_snr_error:.2f} dB")
    
    if complex_snr_error < tolerance_db:
        print(f"  Status: PASSED ✓")
        test3_pass = True
    else:
        print(f"  Status: FAILED ✗")
        test3_pass = False
    
    # Test 4: Different SNR values
    print("\nTest 4: Multiple SNR Values")
    print("-" * 50)
    
    test_snrs = [0, 5, 10, 15, 20]
    test4_pass = True
    
    for test_snr in test_snrs:
        noisy = awgn_channel(signal, test_snr)
        measured = calculate_snr(signal, noisy)
        error = abs(measured - test_snr)
        status = "✓" if error < tolerance_db else "✗"
        print(f"  SNR = {test_snr:2d} dB: Measured = {measured:5.2f} dB, Error = {error:.2f} dB {status}")
        if error >= tolerance_db:
            test4_pass = False
    
    if test4_pass:
        print(f"  Status: PASSED ✓")
    else:
        print(f"  Status: FAILED ✗")
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    all_tests = [test1_pass, test2_pass, test3_pass, test4_pass]
    passed = sum(all_tests)
    total = len(all_tests)
    
    print(f"Tests passed: {passed}/{total}")
    
    if all(all_tests):
        print("\n✓ All tests passed! AWGN channel implementation is correct.")
        return True
    else:
        print("\n✗ Some tests failed. Please review the implementation.")
        return False


if __name__ == "__main__":
    # Run tests
    success = test_awgn_channel()
    
    # Exit with appropriate code
    exit(0 if success else 1)
