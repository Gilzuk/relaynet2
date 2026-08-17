# Archived results — do not cite these

Nothing in this directory backs any table or figure in the thesis. It is kept
for provenance only. Two separate reasons put files here.

## 1. Superseded: measured before the SNR-convention fix

`relaynet/channels/awgn.py` carried a 3 dB error until commit `97ca397`
(2026-08-16). The fix matters more widely than it looks, because every learned
relay generates its *training* data through `awgn_channel`
(`relaynet/utils/activations.py:214`) — so the corrected convention moved the
trained weights, and with them results on channels the fix never touched.

- `bpsk_comparison/` — the old canonical relay comparison.
  - `rayleigh.json` carries `created: 2026-03-23`, predating the fix entirely.
    Its file mtime is later only because it was restored from a backup after a
    lean run overwrote it. **Judge freshness by the `created` field inside the
    JSON, never by mtime.**
  - `awgn.json` is post-fix but only five relays, and every value in it is
    reproduced exactly by `results/modulation/bpsk_awgn.json`, which has eight.
    Checked at 8 dB: AF `0.00650`, DF `0.00038` in both. Strict subset.

## 2. Abolished: configurations the thesis no longer evaluates

The thesis pairs a real constellation with a real channel and a complex
constellation with a complex channel, giving exactly three configurations
(`CANONICAL_PAIRS` in `run_experiments.py`, and the configuration table in the
system model). These files are outside that set:

- `qpsk_awgn.*`, `qam16_awgn.*` — complex constellations on the calibration
  channel. AWGN is retained solely as the analytical baseline for BPSK, where
  the closed forms live; no relay is ranked on it.
- `bpsk_comparison/rayleigh.json` — a real constellation on the complex
  channel, which forces the simulator to discard the imaginary part of the
  equalized sample (`relaynet/channels/fading.py:51`). QPSK holds the
  canonical slot instead.
- `3k_mimo_{zf,mmse,sic}.*`, `3k_rician_k3.*` — the MIMO second hop and the
  Rician robustness study, both removed from the thesis and redesignated
  future work (Appendix E, response to comment 4).

## If you need any of these again

Re-run rather than un-archive. The scripts are current and correct; the data
here is not.
