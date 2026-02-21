"""
Checkpoint 03: Communication Nodes Implementation

This module implements the communication nodes for a two-hop relay system:
- Source: Generates and modulates bits
- Relay: Base class for relay strategies
- AmplifyAndForwardRelay: Classical AF relay implementation
- Destination: Receives and demodulates signals

Author: Cline
Date: 2026-02-14
Checkpoint: CP-03
"""

import numpy as np
import sys
import os

# Add parent directory to path to import from checkpoint_02
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import modulation functions from checkpoint 02
from checkpoint_02_modulation import bpsk_modulate, bpsk_demodulate, calculate_ber


class Source:
    """
    Source node that generates and modulates binary data.
    
    The source generates random bits and modulates them using BPSK
    for transmission over the channel.
    """
    
    def __init__(self, seed=None):
        """
        Initialize the Source node.
        
        Parameters
        ----------
        seed : int, optional
            Random seed for reproducible bit generation
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    def generate_bits(self, num_bits):
        """
        Generate random binary bits.
        
        Parameters
        ----------
        num_bits : int
            Number of bits to generate
        
        Returns
        -------
        bits : numpy.ndarray
            Random binary bits (0s and 1s)
        """
        bits = np.random.randint(0, 2, num_bits)
        return bits
    
    def transmit(self, num_bits):
        """
        Generate bits and modulate them for transmission.
        
        Parameters
        ----------
        num_bits : int
            Number of bits to transmit
        
        Returns
        -------
        bits : numpy.ndarray
            Generated bits
        symbols : numpy.ndarray
            BPSK modulated symbols
        """
        # Generate bits
        bits = self.generate_bits(num_bits)
        
        # Modulate using BPSK
        symbols = bpsk_modulate(bits)
        
        return bits, symbols


class Relay:
    """
    Base class for relay strategies.
    
    This abstract class defines the interface that all relay
    implementations must follow.
    """
    
    def process(self, received_signal):
        """
        Process the received signal and forward it.
        
        Parameters
        ----------
        received_signal : numpy.ndarray
            Signal received from the source
        
        Returns
        -------
        forwarded_signal : numpy.ndarray
            Processed signal to forward to destination
        """
        raise NotImplementedError("Subclasses must implement process()")


class AmplifyAndForwardRelay(Relay):
    """
    Amplify-and-Forward (AF) relay implementation.
    
    The AF relay amplifies the received signal and forwards it to the
    destination. Power normalization ensures the transmitted power is
    controlled.
    
    Operation:
    1. Receive noisy signal from source
    2. Amplify signal with gain factor
    3. Normalize to target power
    4. Forward to destination
    """
    
    def __init__(self, target_power=1.0):
        """
        Initialize the AF relay.
        
        Parameters
        ----------
        target_power : float, optional
            Target power for the forwarded signal (default: 1.0)
        """
        self.target_power = target_power
    
    def process(self, received_signal):
        """
        Amplify and forward the received signal with power normalization.
        
        The relay amplifies the signal and normalizes it to maintain
        a constant transmitted power, regardless of the received signal
        power.
        
        Parameters
        ----------
        received_signal : numpy.ndarray
            Noisy signal received from source
        
        Returns
        -------
        forwarded_signal : numpy.ndarray
            Amplified and power-normalized signal
        """
        # Calculate received signal power
        received_power = np.mean(np.abs(received_signal) ** 2)
        
        # Calculate amplification factor for power normalization
        # We want: mean(|amplified_signal|^2) = target_power
        if received_power > 0:
            amplification_factor = np.sqrt(self.target_power / received_power)
        else:
            amplification_factor = 1.0
        
        # Amplify and forward
        forwarded_signal = amplification_factor * received_signal
        
        return forwarded_signal
    
    def get_amplification_factor(self, received_signal):
        """
        Calculate the amplification factor for a given received signal.
        
        Useful for analysis and debugging.
        
        Parameters
        ----------
        received_signal : numpy.ndarray
            Received signal
        
        Returns
        -------
        amplification_factor : float
            Amplification factor applied
        """
        received_power = np.mean(np.abs(received_signal) ** 2)
        if received_power > 0:
            return np.sqrt(self.target_power / received_power)
        else:
            return 1.0


class Destination:
    """
    Destination node that receives and demodulates signals.
    
    The destination receives the signal from the relay and demodulates
    it to recover the transmitted bits.
    """
    
    def __init__(self):
        """Initialize the Destination node."""
        pass
    
    def receive(self, received_signal):
        """
        Receive and demodulate the signal to recover bits.
        
        Parameters
        ----------
        received_signal : numpy.ndarray
            Signal received from relay
        
        Returns
        -------
        bits : numpy.ndarray
            Demodulated bits
        """
        # Demodulate using BPSK
        bits = bpsk_demodulate(received_signal)
        
        return bits


def test_communication_nodes():
    """
    Test the communication nodes implementation.
    
    Tests:
    1. Source node - bit generation and modulation
    2. AF Relay - amplification and power normalization
    3. Destination node - demodulation
    4. End-to-end (no channel) - perfect transmission
    5. Power normalization validation
    """
    print("Testing Communication Nodes Implementation")
    print("=" * 50)
    
    # Test 1: Source Node
    print("\nTest 1: Source Node")
    print("-" * 50)
    
    source = Source(seed=42)
    num_bits = 100
    tx_bits, tx_symbols = source.transmit(num_bits)
    
    print(f"  Generated bits: {num_bits}")
    print(f"  Modulated symbols: {len(tx_symbols)}")
    print(f"  Bits range: [{tx_bits.min()}, {tx_bits.max()}]")
    print(f"  Symbols range: [{tx_symbols.min():.1f}, {tx_symbols.max():.1f}]")
    
    # Verify bits are binary
    if set(tx_bits).issubset({0, 1}) and len(tx_symbols) == num_bits:
        print(f"  Status: PASSED ✓")
        test1_pass = True
    else:
        print(f"  Status: FAILED ✗")
        test1_pass = False
    
    # Test 2: Amplify-and-Forward Relay
    print("\nTest 2: Amplify-and-Forward Relay")
    print("-" * 50)
    
    relay = AmplifyAndForwardRelay(target_power=1.0)
    
    # Create a signal with different power
    test_signal = tx_symbols * 1.5  # Increase power
    input_power = np.mean(np.abs(test_signal) ** 2)
    
    # Process through relay
    forwarded_signal = relay.process(test_signal)
    output_power = np.mean(np.abs(forwarded_signal) ** 2)
    
    amplification_factor = relay.get_amplification_factor(test_signal)
    
    print(f"  Input signal power: {input_power:.6f}")
    print(f"  Amplification factor: {amplification_factor:.6f}")
    print(f"  Output signal power: {output_power:.6f}")
    print(f"  Target power: {relay.target_power:.6f}")
    print(f"  Power error: {abs(output_power - relay.target_power):.6f}")
    
    # Check if output power matches target (within 0.1% tolerance)
    if abs(output_power - relay.target_power) < 0.001:
        print(f"  Status: PASSED ✓")
        test2_pass = True
    else:
        print(f"  Status: FAILED ✗")
        test2_pass = False
    
    # Test 3: Destination Node
    print("\nTest 3: Destination Node")
    print("-" * 50)
    
    destination = Destination()
    
    # Receive clean symbols (no noise)
    rx_bits = destination.receive(tx_symbols)
    
    print(f"  Received symbols: {len(tx_symbols)}")
    print(f"  Demodulated bits: {len(rx_bits)}")
    print(f"  Bits range: [{rx_bits.min()}, {rx_bits.max()}]")
    
    # Verify demodulation
    if set(rx_bits).issubset({0, 1}) and len(rx_bits) == len(tx_symbols):
        print(f"  Status: PASSED ✓")
        test3_pass = True
    else:
        print(f"  Status: FAILED ✗")
        test3_pass = False
    
    # Test 4: End-to-End (No Channel)
    print("\nTest 4: End-to-End Communication (No Channel)")
    print("-" * 50)
    
    # Full chain: Source → Relay → Destination (no noise)
    source = Source(seed=42)
    relay = AmplifyAndForwardRelay(target_power=1.0)
    destination = Destination()
    
    num_bits = 1000
    tx_bits, tx_symbols = source.transmit(num_bits)
    relay_output = relay.process(tx_symbols)
    rx_bits = destination.receive(relay_output)
    
    ber, errors = calculate_ber(tx_bits, rx_bits)
    
    print(f"  Transmitted bits: {num_bits}")
    print(f"  Received bits: {len(rx_bits)}")
    print(f"  Bit errors: {errors}")
    print(f"  BER: {ber:.6f}")
    
    if ber == 0.0:
        print(f"  Status: PASSED ✓")
        test4_pass = True
    else:
        print(f"  Status: FAILED ✗ (Expected BER = 0.0)")
        test4_pass = False
    
    # Test 5: Power Normalization with Various Input Powers
    print("\nTest 5: Power Normalization Validation")
    print("-" * 50)
    
    relay = AmplifyAndForwardRelay(target_power=1.0)
    test5_pass = True
    
    # Test with different input power levels
    power_levels = [0.5, 1.0, 2.0, 5.0]
    
    for input_power_level in power_levels:
        test_signal = tx_symbols * np.sqrt(input_power_level)
        input_power = np.mean(np.abs(test_signal) ** 2)
        
        forwarded = relay.process(test_signal)
        output_power = np.mean(np.abs(forwarded) ** 2)
        
        power_error = abs(output_power - relay.target_power)
        status = "✓" if power_error < 0.001 else "✗"
        
        print(f"  Input power: {input_power:.3f} → Output power: {output_power:.6f} (error: {power_error:.6f}) {status}")
        
        if power_error >= 0.001:
            test5_pass = False
    
    if test5_pass:
        print(f"  Status: PASSED ✓")
    else:
        print(f"  Status: FAILED ✗")
    
    # Test 6: Relay with Different Target Powers
    print("\nTest 6: Different Target Powers")
    print("-" * 50)
    
    test6_pass = True
    target_powers = [0.5, 1.0, 2.0]
    
    for target_power in target_powers:
        relay = AmplifyAndForwardRelay(target_power=target_power)
        forwarded = relay.process(tx_symbols)
        output_power = np.mean(np.abs(forwarded) ** 2)
        
        power_error = abs(output_power - target_power)
        status = "✓" if power_error < 0.001 else "✗"
        
        print(f"  Target: {target_power:.1f} → Actual: {output_power:.6f} (error: {power_error:.6f}) {status}")
        
        if power_error >= 0.001:
            test6_pass = False
    
    if test6_pass:
        print(f"  Status: PASSED ✓")
    else:
        print(f"  Status: FAILED ✗")
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    all_tests = [test1_pass, test2_pass, test3_pass, test4_pass, test5_pass, test6_pass]
    passed = sum(all_tests)
    total = len(all_tests)
    
    print(f"Tests passed: {passed}/{total}")
    
    if all(all_tests):
        print("\n✓ All tests passed! Communication nodes implementation is correct.")
        return True
    else:
        print("\n✗ Some tests failed. Please review the implementation.")
        return False


if __name__ == "__main__":
    # Run tests
    success = test_communication_nodes()
    
    # Exit with appropriate code
    exit(0 if success else 1)
