#!/usr/bin/env python3
"""Does block-DF's re-encoding liability survive into the reliable-decoding regime?

Remark~rem:df-terminology defines information-theoretic block-DF as using
"a strong (ideally capacity-achieving) channel code", and the supervisor's
comment that prompted the coded study made the same assumption: with a
strong code the relay recovers the message, so re-encoding costs nothing.

The coded study of Section~sec:coded-block-df does not reach that regime.
Its rate-1/2, K=3 code at a 200-information-bit frame leaves a *frame*
error rate of 0.2023 for coded-DF at 20 dB, and 0.1058 even for an oracle
relay that forwards the clean codeword -- so roughly one frame in ten
fails for reasons that have nothing to do with the relay at all. The
measured reversal (a denoising relay overtaking block-DF at 20 dB) is
therefore established only where decoding is unreliable, which is not
where block-DF's information-theoretic optimality is claimed.

This script closes that gap without changing the code. The mechanism the
coded study identifies -- block-DF re-encodes, so a frame it decodes
wrongly leaves the relay as a valid-but-wrong codeword the destination
cannot repair -- depends on the relay's frame error rate, not on which
family of code produced it. Driving that FER down by extending the same
link up the SNR axis therefore tests the same premise as swapping in a
capacity-approaching code, at a fraction of the implementation cost, and
with the code held fixed so that operating regime is the only variable.

Two learned relays are evaluated to separate the mechanism from a
training artefact:

  mlp_thesis  trained on the thesis recipe (SNRs 5/10/15), so at 28--32 dB
              it is extrapolating far outside its training range;
  mlp_ext     identical architecture and hyperparameters, retrained on
              5..30 dB, which brackets three of the four evaluated points
              (20/24/28) and leaves only 32 dB as a mild, 2 dB
              extrapolation -- so any residual gap there is not a large
              train/test mismatch.

Because mlp_ext is a differently-trained instance, this script's 20 dB
column is NOT required to reproduce Table~tbl:table34's 20 dB entry, and
no such match is claimed; the table stands on its own axis.

Writes results/coded_reliable_regime.json. Touches no existing result
file: every number already reported by the thesis is left untouched.
"""

import json
import numpy as np

from relaynet.coding.convolutional import ConvolutionalEncoder, ViterbiCodeDecoder
from relaynet.relays.coded_df import CodedDecodeAndForwardRelay
from relaynet.relays.mlp import MLPQPSKClassifierRelay
from relaynet.channels.fading import rayleigh_fading_channel
from relaynet.modulation.qpsk import qpsk_modulate

from coded_df_experiment import decode_all_frames, FRAME_INFO_BITS
from coded_learned_relay import WINDOW as MLP_WINDOW

# Evaluation axis: starts at the coded study's top point and climbs until
# the code decodes reliably, so the FER-driven mechanism can be read off
# directly against the BER ordering.
SNRS = [20, 24, 28, 32]
N_TRIALS = 100
N_FRAMES = 1000  # 1000 * 200 = 200,000 info bits per trial

TRAIN_SNRS_THESIS = [5, 10, 15]
TRAIN_SNRS_EXTENDED = [5, 10, 15, 20, 25, 30]
TRAIN_FRAMES_PER_SNR = 2000


class OracleRelay:
    """Control: forwards the true clean codeword. Isolates hop-2-only errors."""

    def __init__(self):
        self.truth = None

    def process(self, received_signal):
        return self.truth


def make_training_data(encoder, relay, train_snrs, n_frames_per_snr, seed=0):
    """Same construction as coded_learned_relay.generate_coded_training_data,
    with the SNR list as a parameter so the two relays differ only in it."""
    rng = np.random.default_rng(seed)
    # rayleigh_fading_channel draws from the global RNG, so seeding only the
    # bit generator above would leave the fading and noise irreproducible.
    # Same convention as coded_error_mechanism.py / coded_soft_decision.py.
    np.random.seed(seed % (2 ** 31))
    frame_symbols = FRAME_INFO_BITS + encoder.num_tail
    X_list, T_list = [], []
    for snr_db in train_snrs:
        info = rng.integers(0, 2, n_frames_per_snr * FRAME_INFO_BITS)
        coded = np.concatenate([
            encoder.encode(info[f * FRAME_INFO_BITS:(f + 1) * FRAME_INFO_BITS])
            for f in range(n_frames_per_snr)
        ])
        tx = qpsk_modulate(coded)
        rx = rayleigh_fading_channel(tx, snr_db)
        windows = relay._extract_windows(rx)
        bit_pairs = coded.reshape(-1, 2)
        X_list.append(windows)
        T_list.append(bit_pairs[:, 0] * 2 + bit_pairs[:, 1])
        assert len(T_list[-1]) == frame_symbols * n_frames_per_snr
    return np.vstack(X_list), np.concatenate(T_list)


def train_mlp(encoder, train_snrs, label):
    relay = MLPQPSKClassifierRelay(window_size=MLP_WINDOW, hidden_size=16, seed=0)
    print(f"  training {label} on SNRs {train_snrs} ...", flush=True)
    X, T = make_training_data(encoder, relay, train_snrs, TRAIN_FRAMES_PER_SNR)
    relay.train_on_data(X, T, epochs=25, batch_size=512, lr=3e-3)
    return relay


def run_trial(relay, snr_db, seed, encoder, decoder, frame_symbols, is_oracle=False):
    rng = np.random.default_rng(seed)
    # Both RNGs must be seeded: the bit generator below, and the global one the
    # channel draws its fading and noise from. Without the second, the seed
    # arithmetic in main() would not make a trial reproducible.
    np.random.seed(seed % (2 ** 31))
    info_bits = rng.integers(0, 2, N_FRAMES * FRAME_INFO_BITS)
    coded = np.concatenate([
        encoder.encode(info_bits[f * FRAME_INFO_BITS:(f + 1) * FRAME_INFO_BITS])
        for f in range(N_FRAMES)
    ])
    tx = qpsk_modulate(coded)
    rx_relay = rayleigh_fading_channel(tx, snr_db)

    if is_oracle:
        relay.truth = tx
    relay_out = relay.process(rx_relay)
    rx_dest = rayleigh_fading_channel(relay_out, snr_db)

    info_hat = decode_all_frames(rx_dest, decoder, frame_symbols)
    n = min(len(info_hat), len(info_bits))
    ber = float(np.mean(info_hat[:n] != info_bits[:n]))

    a = info_bits[:N_FRAMES * FRAME_INFO_BITS].reshape(N_FRAMES, FRAME_INFO_BITS)
    b = info_hat[:N_FRAMES * FRAME_INFO_BITS].reshape(N_FRAMES, FRAME_INFO_BITS)
    fer = float(np.mean(np.any(a != b, axis=1)))
    return ber, fer


def main():
    encoder = ConvolutionalEncoder()
    decoder = ViterbiCodeDecoder()
    frame_symbols = FRAME_INFO_BITS + decoder.num_tail

    print("Preparing relays...")
    relays = {
        "coded_df": (CodedDecodeAndForwardRelay(frame_info_bits=FRAME_INFO_BITS), False),
        "mlp_thesis": (train_mlp(encoder, TRAIN_SNRS_THESIS, "mlp_thesis"), False),
        "mlp_ext": (train_mlp(encoder, TRAIN_SNRS_EXTENDED, "mlp_ext"), False),
        "oracle": (OracleRelay(), True),
    }

    results = {
        "snr_db": list(SNRS),
        "n_trials": N_TRIALS,
        "n_frames": N_FRAMES,
        "frame_info_bits": FRAME_INFO_BITS,
        "train_snrs_thesis": TRAIN_SNRS_THESIS,
        "train_snrs_extended": TRAIN_SNRS_EXTENDED,
    }
    for name in relays:
        results[name] = []
        results[f"{name}_fer"] = []

    hdr = f"{'SNR':>5} " + " ".join(f"{n:>22}" for n in relays)
    print("\n" + hdr)
    print("-" * len(hdr))
    for snr_db in SNRS:
        row = f"{snr_db:>5} "
        for offset, (name, (relay, is_oracle)) in enumerate(relays.items()):
            trials = [run_trial(relay, snr_db, 90000 + 7000 * offset + 100 * int(snr_db) + t,
                                encoder, decoder, frame_symbols, is_oracle)
                      for t in range(N_TRIALS)]
            ber = float(np.mean([x[0] for x in trials]))
            fer = float(np.mean([x[1] for x in trials]))
            results[name].append(ber)
            results[f"{name}_fer"].append(fer)
            row += f" {ber:>11.6f}/{fer:<10.4f}"
        print(row, flush=True)

    with open("results/coded_reliable_regime.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nSaved results/coded_reliable_regime.json")


if __name__ == "__main__":
    main()
