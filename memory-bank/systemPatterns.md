# System Patterns — relaynet architecture

## Package layout
```
relaynet/
├── relays/
│   ├── base.py        # Relay base class/interface
│   ├── af.py           AmplifyAndForwardRelay
│   ├── df.py            DecodeAndForwardRelay (hard-decision)
│   ├── mlp.py            MLPRelay (BPSK regression), MLPQPSKClassifierRelay (QPSK,
│   │                       4-class softmax over the Gray-coded alphabet)
│   ├── viterbi.py         ViterbiMLSERelay (BPSK), ViterbiMLSEQPSKRelay (QPSK, 4-symbol
│   │                       Gray-coded trellis) — both MLSE for ISI channels
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
- **Viterbi relay**: `ViterbiMLSERelay(channel_taps=...)` for genie CSI, or LS-estimation variant — set `self.L` (channel length) BEFORE calling `_ls_estimate()` (this was a real bug, see `progress.md` Known Issues Fixed). `ViterbiMLSEQPSKRelay` follows the identical trellis pattern generalized to base-M states (`itertools.product(range(M), repeat=L-1)` instead of binary digit unpacking) and complex branch metrics (`np.abs(y - exp_y)**2`); `process()` returns decoded complex constellation symbols rather than bits, consistent with `.process()` returning a forwarded signal, not decoded bits. If a 16-QAM version is ever needed, follow the same generalization (M=16, 256 states for L=3) — not built, explicitly deferred by user request.
- **Hard vs soft decision relays** (added for the enhanced multi-architecture comparison): a "hard" DF relay quantizes to the nearest constellation point (`DecodeAndForwardRelay`, BPSK-only, or the modulation-aware `DFHardRelay` in `e6_sim_enhanced_multimod.py` which wraps `get_modulation_functions(modulation)`); a "soft" DF relay does power normalization only, no quantization (`DFSoftRelay`, same signature works for real or complex signals via `np.abs`). Neither hard/soft class is promoted into `relaynet/relays/` yet — kept local to the experiment scripts.
- **Complex-signal channels** (`ComplexISIChannel`, `ComplexAWGNChannel` in `relaynet/channels/e6_channels.py`): same taps/SNR convention as their real-valued counterparts (`ISIChannel`) but add circularly-symmetric complex AWGN (`sigma * (randn + 1j*randn) / sqrt(2)`), required for QPSK/16-QAM where symbols are complex. `ComplexAWGNChannel` auto-detects real vs complex input, so it's safe to use as hop 2 for any modulation.
- **Symmetric-hop channels** (`ISIRayleighChannel`, `ComplexISIRayleighChannel`): combine ISI + coherently-compensated Rayleigh magnitude fading + AWGN in one channel class, meant to be instantiated twice (once per hop, independent RNG/seed) with the *same* taps so hop 1 and hop 2 face an identical impairment model — use this pattern whenever a comparison needs to isolate relay-architecture effects from channel asymmetry (a relay that only equalizes hop 1 will still show a floor from hop 2's uncorrected impairment, and that's expected/correct, not a bug).

## Comparison-methodology gotcha (important, cost real time this session)
When pulling BER numbers from two different scripts to compare relays "fairly," **verify both scripts use the literal same hop-2 (and hop-1) channel objects** before trusting the comparison — even scenario descriptions that sound the same (e.g. both say "unknown ISI") can differ in whether hop 2 has Rayleigh fading or not, and that alone can dwarf any real relay-architecture difference. This actually happened: an MLP-170-vs-Viterbi-QPSK comparison looked dramatic until it turned out one script used `RayleighChannel` and the other `ComplexAWGNChannel` for hop 2. Always rerun the comparison set in one script under one shared channel setup rather than stitching together numbers from separate runs.

## File naming pattern for ported experiments
`e6_<name>_ported.py` at repo root — mirrors `experiments-standalone/e6_<name>.py` 1:1 in structure/scenarios but calls into `relaynet` classes instead of hand-rolled NumPy. Results saved to `/tmp/e6_<name>_ported_results.npy` (or similar), figures to `/tmp/*.png` during iteration — NOT yet moved into `results/` (that happens only at final thesis-integration pass, see `progress.md`).

## Testing pattern
`test_e6_core.py` — plain assertion-based checks (no pytest framework requirement observed) covering: ISI channel convolution/normalization, MLP weight init param counts, forward-pass shapes, SNR convention agreement, window extraction correctness, Rayleigh magnitude distribution. Run with `python3 test_e6_core.py`.
