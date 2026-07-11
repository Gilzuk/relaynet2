# System Patterns — relaynet architecture

## Package layout
```
relaynet/
├── relays/
│   ├── base.py        # Relay base class/interface
│   ├── af.py           AmplifyAndForwardRelay
│   ├── df.py            DecodeAndForwardRelay (hard-decision)
│   ├── mlp.py            MLPRelay              (E6 port: general windowed neural relay)
│   ├── viterbi.py         ViterbiMLSERelay      (E6 port: MLSE for ISI channels)
│   ├── e2e.py, vae.py, cgan.py, hybrid.py, rl.py, genai.py   (other thesis chapters' architectures)
├── channels/
│   ├── awgn.py, fading.py, mimo.py
│   └── e6_channels.py   (E6 port: ISIChannel, NonlinearBiasChannel, RayleighChannel,
│                          FlatPhaseChannel, FlatGainChannel, BranchAsymmetryChannel,
│                          PowerAmplifierChannel, CompositeChannel)
├── nodes.py             Source, Destination
├── modulation/           bpsk.py (qpsk/qam16 modules referenced but calculate_ber()
│                          not present for them yet — see techContext.md gotcha)
└── simulation/runner.py
```

## Conventions to preserve when adding relays/channels
- **Relay interface**: a relay exposes `.process(received_signal) -> forwarded_signal`. `MLPRelay` also exposes `.fwd(X)` for raw windowed-batch inference (used when windowing is done manually, e.g. complex-signal cases in `e6_flat_ported.py`).
- **Channel interface**: a channel is callable as `channel(signal, snr_db) -> noisy_signal`, and is a class so it can hold internal RNG/seed state (`ISIChannel(seed=...)`, etc.).
- **SNR convention**: γ = 1/σ² = 10^(SNR_dB/10); noise_power = signal_power / 10^(SNR_dB/10). Verified identical between standalone E6 scripts and `relaynet` — no rescaling anywhere. Any new channel MUST follow this exact convention.
- **Windowed neural relays**: `MLPRelay(input_size, hidden_size, output_size, window_size, seed)`. If `window_size` is set, `.process()` auto-extracts sliding windows via `np.lib.stride_tricks.sliding_window_view` with `pad_size = window_size // 2` zero-padding on both sides. For complex signals, windows are extracted manually (I and Q concatenated) and passed to `.fwd()` directly — see `extract_windows()` in `e6_flat_ported.py`.
- **Viterbi relay**: `ViterbiMLSERelay(channel_taps=...)` for genie CSI, or LS-estimation variant — set `self.L` (channel length) BEFORE calling `_ls_estimate()` (this was a real bug, see `progress.md` Known Issues Fixed).
- **Hard vs soft decision relays** (added for the enhanced multi-architecture comparison): a "hard" DF relay quantizes to the nearest constellation point (`DecodeAndForwardRelay`, BPSK-only, or the modulation-aware `DFHardRelay` in `e6_sim_enhanced_multimod.py` which wraps `get_modulation_functions(modulation)`); a "soft" DF relay does power normalization only, no quantization (`DFSoftRelay`, same signature works for real or complex signals via `np.abs`). Neither hard/soft class is promoted into `relaynet/relays/` yet — kept local to the experiment scripts.
- **Complex-signal channels** (`ComplexISIChannel`, `ComplexAWGNChannel` in `relaynet/channels/e6_channels.py`): same taps/SNR convention as their real-valued counterparts (`ISIChannel`) but add circularly-symmetric complex AWGN (`sigma * (randn + 1j*randn) / sqrt(2)`), required for QPSK/16-QAM where symbols are complex. `ComplexAWGNChannel` auto-detects real vs complex input, so it's safe to use as hop 2 for any modulation.

## File naming pattern for ported experiments
`e6_<name>_ported.py` at repo root — mirrors `experiments-standalone/e6_<name>.py` 1:1 in structure/scenarios but calls into `relaynet` classes instead of hand-rolled NumPy. Results saved to `/tmp/e6_<name>_ported_results.npy` (or similar), figures to `/tmp/*.png` during iteration — NOT yet moved into `results/` (that happens only at final thesis-integration pass, see `progress.md`).

## Testing pattern
`test_e6_core.py` — plain assertion-based checks (no pytest framework requirement observed) covering: ISI channel convolution/normalization, MLP weight init param counts, forward-pass shapes, SNR convention agreement, window extraction correctness, Rayleigh magnitude distribution. Run with `python3 test_e6_core.py`.
