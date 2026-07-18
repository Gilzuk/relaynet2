#!/usr/bin/env python3
"""Quick verification test of E6 core algorithms (no relaynet dependencies)."""

import numpy as np

# Test 1: ISI channel simulation
print("Test 1: ISI Channel")
H_ISI = np.array([1.0, 0.7, 0.5])
H_ISI = H_ISI / np.linalg.norm(H_ISI)
x = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
snr_db = 10.0
sigma = 10 ** (-snr_db / 20.0)
rng = np.random.default_rng(42)
n = sigma * rng.standard_normal(x.size)
y_isi = np.convolve(x, H_ISI)[:x.size] + n
print(f"  Input shape: {x.shape}, Output shape: {y_isi.shape}, H_ISI norm: {np.linalg.norm(H_ISI):.4f}")
assert y_isi.shape == x.shape, "ISI output shape mismatch"
print("  ✓ ISI channel works")

# Test 2: MLP weight initialization
print("\nTest 2: MLP Weight Initialization")
W = 11
HID = 13
seed = 0
r = np.random.default_rng(seed)
W1 = r.standard_normal((W, HID)) * np.sqrt(2.0 / W)
b1 = np.zeros(HID)
W2 = r.standard_normal((HID, 1)) * np.sqrt(2.0 / HID)
b2 = np.zeros(1)
n_params = W1.size + b1.size + W2.size + b2.size
print(f"  W1 shape: {W1.shape}, W2 shape: {W2.shape}")
print(f"  Total params: {n_params} (expected 170: {11*13 + 13 + 13*1 + 1})")
assert n_params == 170, f"Parameter count mismatch: {n_params} != 170"
print("  ✓ MLP weight initialization correct")

# Test 3: Forward pass
print("\nTest 3: MLP Forward Pass")
X = np.random.randn(10, W)  # 10 samples, window=11
h = np.tanh(X @ W1 + b1)    # hidden layer
o = np.tanh(h @ W2 + b2).ravel()  # output
print(f"  Input shape: {X.shape}, Hidden shape: {h.shape}, Output shape: {o.shape}")
assert o.shape[0] == 10, "Output batch size mismatch"
print("  ✓ Forward pass works")

# Test 4: SNR convention check
print("\nTest 4: SNR Convention")
snr_db_test = 10.0
sigma_e6 = 10 ** (-snr_db_test / 20.0)
snr_linear_e6 = 1.0 / (sigma_e6 ** 2)
snr_linear_relaynet = 10 ** (snr_db_test / 10.0)
print(f"  E6 SNR convention: gamma = 1/sigma^2 = {snr_linear_e6:.2f}")
print(f"  relaynet convention: SNR_linear = 10^(SNR_dB/10) = {snr_linear_relaynet:.2f}")
assert np.isclose(snr_linear_e6, snr_linear_relaynet), "SNR convention mismatch"
print("  ✓ SNR conventions match")

# Test 5: Window extraction
print("\nTest 5: Window Extraction (sliding windows)")
y_signal = np.random.randn(30)
W_size = 11
pad_size = W_size // 2
yp = np.pad(y_signal, (pad_size, pad_size), mode='constant')
windows = np.lib.stride_tricks.sliding_window_view(yp, W_size)
print(f"  Signal shape: {y_signal.shape}, Padded shape: {yp.shape}")
print(f"  Windows shape: {windows.shape} (expected ({y_signal.size}, {W_size}))")
assert windows.shape == (y_signal.size, W_size), "Window shape mismatch"
print("  ✓ Window extraction works")

# Test 6: Rayleigh fading magnitude
print("\nTest 6: Rayleigh Channel (magnitude compensation)")
x_test = np.ones(100)
rng = np.random.default_rng(42)
h_mag = np.abs((rng.standard_normal(100) + 1j * rng.standard_normal(100)) / np.sqrt(2))
y_rayleigh = h_mag * x_test + sigma * rng.standard_normal(100)
print(f"  Input shape: {x_test.shape}, Output shape: {y_rayleigh.shape}")
print(f"  Mean fading magnitude: {h_mag.mean():.3f} (expected ~0.886)")
print("  ✓ Rayleigh channel works")

print("\n" + "="*60)
print("All core algorithm tests passed! ✓")
print("="*60)
