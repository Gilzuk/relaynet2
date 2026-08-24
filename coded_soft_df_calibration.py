"""Why does BCJR soft block-DF get *worse* than hard block-DF at 20 dB?

The soft-decision sweep found soft-DF matching or beating hard block-DF
everywhere up to 16 dB, then losing badly at 20 dB (0.00238 vs 0.00138,
0/20 paired trials). That is the opposite of what the repairability
argument predicts, so it needs an explanation rather than a footnote.

Hypothesis: mis-calibration, not a flaw in soft forwarding. The relay is
given the *nominal* per-real-dimension noise variance 1/(2*snr_linear),
but the channel is Rayleigh with perfect-CSI equalization, so the noise
actually reaching the relay has variance 1/(2*snr_linear*|h|^2) -- much
larger whenever the symbol sat in a deep fade. At high SNR the nominal
variance is tiny, so BCJR becomes extremely overconfident: deep-fade
symbols that are effectively garbage are forwarded as near-full-magnitude
"certain" values, and the destination believes them. Viterbi is immune to
this because its pure squared-distance metric is invariant to any uniform
scaling of the assumed variance -- the mis-calibration cannot hurt it.

Prediction, if true: deliberately inflating the assumed variance above
nominal should *recover* soft-DF's performance at 20 dB, with an optimum
well above 1x nominal. If instead performance is flat or monotonically
worse in the inflation factor, the hypothesis is wrong and soft
forwarding really does break down here.
"""

import json

import numpy as np

from relaynet.coding.convolutional import ConvolutionalEncoder, ViterbiCodeDecoder
from relaynet.relays.coded_df import CodedDecodeAndForwardRelay
from relaynet.relays.soft_coded_df import SoftCodedDecodeAndForwardRelay
from relaynet.channels.fading import rayleigh_fading_channel
from relaynet.modulation.qpsk import qpsk_modulate

from coded_df_experiment import FRAME_INFO_BITS, decode_all_frames

SNRS = [16, 20]
N_TRIALS = 100
N_FRAMES = 500
# Multiples of the nominal per-real-dimension noise variance.
FACTORS = [1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0]


def main():
    encoder = ConvolutionalEncoder()
    decoder = ViterbiCodeDecoder()
    frame_symbols = FRAME_INFO_BITS + decoder.num_tail
    hard = CodedDecodeAndForwardRelay(frame_info_bits=FRAME_INFO_BITS)
    soft = SoftCodedDecodeAndForwardRelay(frame_info_bits=FRAME_INFO_BITS)

    out = {}
    for snr_db in SNRS:
        nominal = 1.0 / (2.0 * 10 ** (snr_db / 10.0))
        print(f"\n=== {snr_db} dB (nominal sigma^2 = {nominal:.3e}), "
              f"{N_TRIALS} trials x {N_FRAMES} frames ===")

        hard_bers, soft_bers = [], {f: [] for f in FACTORS}
        for t in range(N_TRIALS):
            seed = 950000 + 1000 * snr_db + t
            rng = np.random.default_rng(seed)
            info_bits = rng.integers(0, 2, N_FRAMES * FRAME_INFO_BITS)
            coded = np.concatenate([
                encoder.encode(info_bits[f * FRAME_INFO_BITS:(f + 1) * FRAME_INFO_BITS])
                for f in range(N_FRAMES)
            ])
            tx = qpsk_modulate(coded)

            def run(relay):
                np.random.seed(seed % (2 ** 31))
                rx = rayleigh_fading_channel(tx, snr_db)
                out_sig = relay.process(rx)
                rx_d = rayleigh_fading_channel(out_sig, snr_db)
                ih = decode_all_frames(rx_d, decoder, frame_symbols)
                n = min(len(ih), len(info_bits))
                return float(np.mean(ih[:n] != info_bits[:n]))

            hard_bers.append(run(hard))
            for fac in FACTORS:
                soft.decoder.set_noise_var(nominal * fac)
                soft_bers[fac].append(run(soft))

        hb = float(np.mean(hard_bers))
        print(f"  {'hard block-DF (Viterbi)':32s} {hb:.6f}")
        row = {}
        for fac in FACTORS:
            sb = float(np.mean(soft_bers[fac]))
            mark = "  <-- best" if sb == min(np.mean(soft_bers[f]) for f in FACTORS) else ""
            print(f"  soft-DF, sigma^2 = {fac:6.1f}x nominal   {sb:.6f}"
                  f"  ({'beats' if sb < hb else 'loses to'} hard){mark}")
            row[str(fac)] = sb
        out[str(snr_db)] = {"hard": hb, "soft_by_factor": row, "nominal_var": nominal}

    with open("results/coded_soft_df_calibration.json", "w") as fh:
        json.dump({"n_trials": N_TRIALS, "n_frames": N_FRAMES,
                   "factors": FACTORS, "results": out}, fh, indent=2)
    print("\nSaved results/coded_soft_df_calibration.json")


if __name__ == "__main__":
    main()
