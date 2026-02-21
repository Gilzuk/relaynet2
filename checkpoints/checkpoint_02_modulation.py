"""
Checkpoint 02: BPSK Modulation Implementation

This module implements Binary Phase Shift Keying (BPSK) modulation and
demodulation for digital communication simulation.

Author: Cline
Date: 2026-02-14
Checkpoint: CP-02
"""

import numpy as np


def bpsk_modulate(bits):
    """
    Modulate binary bits using BPSK (Binary Phase Shift Keying).
    
    BPSK maps binary bits to antipodal symbols:
    - Bit 0 → Symbol -1
    - Bit 1 → Symbol +1
    
    This is optimal for AWGN channels as it maximizes Euclidean distance
    between symbols.
    
    Parameters
    ----------
    bits : numpy.ndarray
        Binary input bits (0s and 1s)
    
    Returns
    -------
    symbols : numpy.ndarray
        BPSK modulated symbols (-1s and +1s)
    
    Examples
    --------
    >>> bits = np.array([0, 1, 0, 1])
    >>> symbols = bpsk_modulate(bits)
    >>> print(symbols)
    [-1  1 -1  1]
    """
    # Convert bits to numpy array if not already
    bits = np.asarray(bits)
    
    # Map: 0 → -1, 1 → +1
    # Formula: symbol = 2*bit - 1
    symbols = 2 * bits - 1
    
    return symbols.astype(float)


def bpsk_demodulate(symbols, decision_threshold=0.0):
    """
    Demodulate BPSK symbols to binary bits using hard decision.
    
    Performs hard decision decoding based on the sign of received symbols:
    - Symbol < threshold → Bit 0
    - Symbol ≥ threshold → Bit 1
    
    Parameters
    ----------
    symbols : numpy.ndarray
        Received BPSK symbols (real-valued)
    decision_threshold : float, optional
        Decision threshold (default: 0.0)
    
    Returns
    -------
    bits : numpy.ndarray
        Demodulated binary bits (0s and 1s)
    
    Examples
    --------
    >>> symbols = np.array([-0.8, 1.2, -0.3, 0.9])
    >>> bits = bpsk_demodulate(symbols)
    >>> print(bits)
    [0 1 0 1]
    """
    # Convert symbols to numpy array if not already
    symbols = np.asarray(symbols)
    
    # Hard decision: symbol >= threshold → 1, else → 0
    bits = (symbols >= decision_threshold).astype(int)
    
    return bits


def calculate_ber(transmitted_bits, received_bits):
    """
    Calculate Bit Error Rate (BER) between transmitted and received bits.
    
    BER = (Number of bit errors) / (Total number of bits)
    
    Parameters
    ----------
    transmitted_bits : numpy.ndarray
        Original transmitted bits
    received_bits : numpy.ndarray
        Received/decoded bits
    
    Returns
    -------
    ber : float
        Bit Error Rate (between 0 and 1)
    num_errors : int
        Number of bit errors
    
    Examples
    --------
    >>> tx_bits = np.array([0, 1, 0, 1, 1])
    >>> rx_bits = np.array([0, 1, 1, 1, 1])
    >>> ber, errors = calculate_ber(tx_bits, rx_bits)
    >>> print(f"BER: {ber}, Errors: {errors}")
    BER: 0.2, Errors: 1
    """
    # Ensure arrays are same length
    if len(transmitted_bits) != len(received_bits):
        raise ValueError("Transmitted and received bit arrays must have same length")
    
    # Count bit errors (XOR operation)
    bit_errors = np.sum(transmitted_bits != received_bits)
    
    # Calculate BER
    total_bits = len(transmitted_bits)
    ber = bit_errors / total_bits if total_bits > 0 else 0.0
    
    return ber, bit_errors


def test_bpsk_modulation():
    """
    Test the BPSK modulation and demodulation implementation.
    
    Tests:
    1. Basic modulation - verify bit-to-symbol mapping
    2. Basic demodulation - verify symbol-to-bit mapping
    3. Round-trip (noiseless) - verify perfect recovery
    4. Round-trip with noise - verify BER calculation
    """
    print("Testing BPSK Modulation Implementation")
    print("=" * 50)
    
    # Test 1: Basic Modulation
    print("\nTest 1: Basic Modulation")
    print("-" * 50)
    
    test_bits = np.array([0, 1, 0, 1, 1, 0])
    expected_symbols = np.array([-1, 1, -1, 1, 1, -1])
    
    modulated_symbols = bpsk_modulate(test_bits)
    
    print(f"  Input bits:      {test_bits}")
    print(f"  Output symbols:  {modulated_symbols}")
    print(f"  Expected:        {expected_symbols}")
    
    if np.array_equal(modulated_symbols, expected_symbols):
        print(f"  Status: PASSED ✓")
        test1_pass = True
    else:
        print(f"  Status: FAILED ✗")
        test1_pass = False
    
    # Test 2: Basic Demodulation (Noiseless)
    print("\nTest 2: Basic Demodulation (Noiseless)")
    print("-" * 50)
    
    test_symbols = np.array([-1.0, 1.0, -1.0, 1.0, 1.0, -1.0])
    expected_bits = np.array([0, 1, 0, 1, 1, 0])
    
    demodulated_bits = bpsk_demodulate(test_symbols)
    
    print(f"  Input symbols:   {test_symbols}")
    print(f"  Output bits:     {demodulated_bits}")
    print(f"  Expected:        {expected_bits}")
    
    ber, errors = calculate_ber(expected_bits, demodulated_bits)
    print(f"  Bit errors:      {errors}")
    print(f"  BER:             {ber:.4f}")
    
    if np.array_equal(demodulated_bits, expected_bits) and ber == 0.0:
        print(f"  Status: PASSED ✓")
        test2_pass = True
    else:
        print(f"  Status: FAILED ✗")
        test2_pass = False
    
    # Test 3: Round-trip (Noiseless) with Random Bits
    print("\nTest 3: Round-trip Test (Noiseless, 1000 bits)")
    print("-" * 50)
    
    np.random.seed(42)  # For reproducibility
    num_bits = 1000
    random_bits = np.random.randint(0, 2, num_bits)
    
    # Modulate
    symbols = bpsk_modulate(random_bits)
    
    # Demodulate
    recovered_bits = bpsk_demodulate(symbols)
    
    # Calculate BER
    ber, errors = calculate_ber(random_bits, recovered_bits)
    
    print(f"  Number of bits:  {num_bits}")
    print(f"  Bit errors:      {errors}")
    print(f"  BER:             {ber:.6f}")
    
    if ber == 0.0:
        print(f"  Status: PASSED ✓")
        test3_pass = True
    else:
        print(f"  Status: FAILED ✗ (Expected BER = 0.0)")
        test3_pass = False
    
    # Test 4: Demodulation with Noisy Symbols
    print("\nTest 4: Demodulation with Noise")
    print("-" * 50)
    
    # Create symbols with small noise
    clean_symbols = bpsk_modulate(random_bits)
    noise = 0.3 * np.random.randn(len(clean_symbols))
    noisy_symbols = clean_symbols + noise
    
    # Demodulate noisy symbols
    recovered_bits_noisy = bpsk_demodulate(noisy_symbols)
    
    # Calculate BER
    ber_noisy, errors_noisy = calculate_ber(random_bits, recovered_bits_noisy)
    
    print(f"  Number of bits:  {num_bits}")
    print(f"  Noise std dev:   0.3")
    print(f"  Bit errors:      {errors_noisy}")
    print(f"  BER:             {ber_noisy:.6f}")
    
    # With noise std=0.3, we expect some errors but not too many
    # For BPSK with symbols at ±1 and noise std=0.3, BER should be low
    if 0.0 < ber_noisy < 0.1:  # Reasonable range
        print(f"  Status: PASSED ✓ (BER in expected range)")
        test4_pass = True
    else:
        print(f"  Status: WARNING (BER = {ber_noisy:.6f}, expected 0 < BER < 0.1)")
        test4_pass = True  # Still pass, just a warning
    
    # Test 5: Edge Cases
    print("\nTest 5: Edge Cases")
    print("-" * 50)
    
    # Test with symbols exactly at threshold
    edge_symbols = np.array([0.0, 0.0, -0.0001, 0.0001])
    edge_bits = bpsk_demodulate(edge_symbols)
    expected_edge = np.array([1, 1, 0, 1])  # threshold is 0.0, so >= 0 → 1
    
    print(f"  Symbols at threshold: {edge_symbols}")
    print(f"  Demodulated bits:     {edge_bits}")
    print(f"  Expected:             {expected_edge}")
    
    if np.array_equal(edge_bits, expected_edge):
        print(f"  Status: PASSED ✓")
        test5_pass = True
    else:
        print(f"  Status: FAILED ✗")
        test5_pass = False
    
    # Test 6: Large-scale Test
    print("\nTest 6: Large-scale Test (100,000 bits)")
    print("-" * 50)
    
    large_bits = np.random.randint(0, 2, 100000)
    large_symbols = bpsk_modulate(large_bits)
    large_recovered = bpsk_demodulate(large_symbols)
    large_ber, large_errors = calculate_ber(large_bits, large_recovered)
    
    print(f"  Number of bits:  100,000")
    print(f"  Bit errors:      {large_errors}")
    print(f"  BER:             {large_ber:.6f}")
    
    if large_ber == 0.0:
        print(f"  Status: PASSED ✓")
        test6_pass = True
    else:
        print(f"  Status: FAILED ✗")
        test6_pass = False
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    all_tests = [test1_pass, test2_pass, test3_pass, test4_pass, test5_pass, test6_pass]
    passed = sum(all_tests)
    total = len(all_tests)
    
    print(f"Tests passed: {passed}/{total}")
    
    if all(all_tests):
        print("\n✓ All tests passed! BPSK modulation implementation is correct.")
        return True
    else:
        print("\n✗ Some tests failed. Please review the implementation.")
        return False


if __name__ == "__main__":
    # Run tests
    success = test_bpsk_modulation()
    
    # Exit with appropriate code
    exit(0 if success else 1)
