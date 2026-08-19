#!/usr/bin/env python3
"""Why does the symbol-level learned relay edge out block-DF at high SNR,
when Viterbi is the optimal decoder for this code?

Hypothesis under test -- an architectural asymmetry, NOT a claim that a
neural network out-decodes the Viterbi algorithm:

  Block-DF decodes the frame at the relay and *re-encodes* it. When that
  decode is wrong, the relay does not emit a corrupted codeword, it emits
  a different, perfectly VALID codeword. The destination's decoder has no
  way to detect or repair that -- the redundancy has been spent and
  regenerated around the error, so a single relay-frame error becomes a
  locked-in burst of information-bit errors.

  The symbol-level learned relay never re-encodes. Its mistakes are
  isolated wrong symbols sitting inside an otherwise-valid codeword, so
  the destination's Viterbi decoder still has the code's redundancy
  available and can repair them.

If true, the prediction is specific and falsifiable:
  (a) block-DF's end-to-end errors should be concentrated in a few
      catastrophic frames (high bit-errors-per-failed-frame),
  (b) the learned relay's relay-output symbol errors should be largely
      *repaired* by the destination decoder (relay-output error rate much
      higher than final BER), whereas block-DF's should pass through
      essentially unrepaired,
  (c) both should converge to the same hop-2-only floor when the relay is
      replaced by an oracle that forwards the clean codeword.

Measured at high SNR, where the effect was observed.
"""

import json

import numpy as np

from relaynet.coding.convolutional import ConvolutionalEncoder, ViterbiCodeDecoder
from relaynet.relays.coded_df import CodedDecodeAndForwardRelay
from relaynet.relays.mlp import MLPQPSKClassifierRelay
from relaynet.channels.fading import rayleigh_fading_channel
from relaynet.modulation.qpsk import qpsk_modulate

from coded_df_experiment import FRAME_INFO_BITS, decode_all_frames
from coded_learned_relay import (
    WINDOW as MLP_WINDOW,
    generate_coded_training_data as gen_mlp_data,
    TRAIN_FRAMES_PER_SNR as MLP_TRAIN_FRAMES,
)

SNRS = [16, 20]
N_TRIALS = 20
N_FRAMES = 500

ALPHABET = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)


class OracleRelay:
    """Control: forwards the true clean codeword. Isolates hop-2-only errors."""

    def __init__(self):
        self.truth = None

    def process(self, received_signal):
        return self.truth


def symbol_errors(a, b):
    return int(np.sum(~np.isclose(a, b, atol=1e-9)))


def main():
    encoder = ConvolutionalEncoder()
    decoder = ViterbiCodeDecoder()
    frame_symbols = FRAME_INFO_BITS + decoder.num_tail

    print("Training MLP-coded ...", flush=True)
    mlp = MLPQPSKClassifierRelay(window_size=MLP_WINDOW, hidden_size=16, seed=0)
    X, T = gen_mlp_data(encoder, mlp, MLP_TRAIN_FRAMES)
    mlp.train_on_data(X, T, epochs=25, batch_size=512, lr=3e-3)

    coded_df = CodedDecodeAndForwardRelay(frame_info_bits=FRAME_INFO_BITS)
    oracle = OracleRelay()
    relays = {"coded_df": coded_df, "mlp_coded": mlp, "oracle": oracle}

    out = {}
    for snr_db in SNRS:
        acc = {k: dict(bit_err=0, bits=0, relay_sym_err=0, syms=0,
                       failed_frames=0, frames=0, bit_err_in_failed=0)
               for k in relays}

        for t in range(N_TRIALS):
            rng = np.random.default_rng(700000 + 1000 * snr_db + t)
            info_bits = rng.integers(0, 2, N_FRAMES * FRAME_INFO_BITS)
            coded = np.concatenate([
                encoder.encode(info_bits[f * FRAME_INFO_BITS:(f + 1) * FRAME_INFO_BITS])
                for f in range(N_FRAMES)
            ])
            tx = qpsk_modulate(coded)
            oracle.truth = tx

            for name, relay in relays.items():
                np.random.seed((700000 + 1000 * snr_db + t) % (2 ** 31))
                rx_relay = rayleigh_fading_channel(tx, snr_db)
                relay_out = relay.process(rx_relay)
                rx_dest = rayleigh_fading_channel(relay_out, snr_db)
                info_hat = decode_all_frames(rx_dest, decoder, frame_symbols)

                n = min(len(info_hat), len(info_bits))
                errs = info_hat[:n] != info_bits[:n]
                a = acc[name]
                a["bit_err"] += int(errs.sum())
                a["bits"] += n
                # How wrong is what the relay actually put on the air?
                a["relay_sym_err"] += symbol_errors(relay_out, tx[:len(relay_out)])
                a["syms"] += len(relay_out)
                # Per-frame failure structure.
                for f in range(N_FRAMES):
                    seg = errs[f * FRAME_INFO_BITS:(f + 1) * FRAME_INFO_BITS]
                    a["frames"] += 1
                    if seg.any():
                        a["failed_frames"] += 1
                        a["bit_err_in_failed"] += int(seg.sum())

        print(f"\n=== {snr_db} dB, {N_TRIALS} trials x {N_FRAMES} frames ===")
        hdr = f"{'relay':14s} {'final BER':>11s} {'relay sym ER':>13s} {'repaired':>10s} {'FER':>9s} {'bit err/failed frame':>21s}"
        print(hdr)
        snr_out = {}
        for name, a in acc.items():
            ber = a["bit_err"] / a["bits"]
            rser = a["relay_sym_err"] / a["syms"]
            fer = a["failed_frames"] / a["frames"]
            per_failed = (a["bit_err_in_failed"] / a["failed_frames"]) if a["failed_frames"] else 0.0
            # Fraction of the relay's own symbol errors that did NOT survive to
            # the output -- i.e. that the destination decoder repaired.
            repaired = (1 - ber / rser) if rser > 0 else float("nan")
            print(f"{name:14s} {ber:11.6f} {rser:13.6f} {repaired:9.1%} {fer:9.4f} {per_failed:21.1f}")
            snr_out[name] = dict(ber=ber, relay_sym_er=rser, repaired_frac=repaired,
                                 fer=fer, bit_err_per_failed_frame=per_failed)
        out[str(snr_db)] = snr_out

    with open("results/coded_error_mechanism.json", "w") as fh:
        json.dump({"n_trials": N_TRIALS, "n_frames": N_FRAMES, "results": out}, fh, indent=2)
    print("\nSaved results/coded_error_mechanism.json")


if __name__ == "__main__":
    main()
