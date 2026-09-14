"""One protocol for every relay on the unknown-ISI channel.

The 14 September 2026 consistency review found that the committed genie and
pilot-LS Viterbi curves do not reproduce under the current channel code. They
do not, and there are two independent reasons, which compound:

1. Hop-2 noise convention. `relaynet/channels/awgn.py` used to put the whole
   of N0 in the single real dimension (Pb = Q(sqrt(gamma))) and now puts N0/2
   (Pb = Q(sqrt(2*gamma))). The stored Viterbi vectors match the old
   convention to three significant figures at 0, 4 and 8 dB, and are 3.01 dB
   pessimistic under the current one. The MLP, AF and DF vectors reproduce
   under the current code, so tbl:tableE6 tabulated two conventions side by
   side.

The tap normalisation that differs between the two scripts is NOT a second
cause, though it looks like one: ISIChannel normalises its taps in __init__,
so both scripts drive the same channel. What it does mean is that the
detector must be given the NORMALISED taps to be genie-informed, because
those are the channel's effective taps. Handing it the raw [1.0, 0.7, 0.5]
mis-specifies it by a factor of 1.319 and makes the pilot-LS estimate beat
"genie" CSI, which is the tell that the configuration is wrong.

This script runs AF, symbol-wise DF, the learned relay, genie-CSI Viterbi
MLSE and pilot-LS Viterbi MLSE through one channel object, one tap vector,
one hop-2 convention and one bit budget, recording per-trial error counts,
seeds and the code revision. Results go to results/ (never /tmp), and the
experiment is registered in provenance_audit.py.

Run: python e6_matched_protocol.py [--bits N] [--trials M]
"""

import argparse
import json
import os
import subprocess
import time

import numpy as np

import e6_sim_ported as e6
from relaynet.relays import AmplifyAndForwardRelay, DecodeAndForwardRelay, ViterbiMLSERelay

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "e6_matched_protocol.json")

# The canonical taps of the unknown-ISI study. ISIChannel normalises whatever
# it is handed, so H_CHANNEL and H_EFFECTIVE drive the same channel -- but the
# detector has to be told the effective taps, which are the normalised ones.
H_CHANNEL = np.array([1.0, 0.7, 0.5])
H_ISI = H_CHANNEL / np.linalg.norm(H_CHANNEL)
SNRS = list(range(0, 21, 2))
N_PILOT = 200
SEEDS = (0, 1, 2)          # the three MLP training seeds


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def run(n_bits, n_trials):
    hop1 = e6.create_channel("isi", seed=1)
    hop2 = e6.create_channel("awgn", seed=2)
    destination = e6.Destination(modulation="bpsk")

    mlps = [e6.train_mlp(hop1, seed=s)[0] for s in SEEDS]
    relays = {
        "AF": lambda: AmplifyAndForwardRelay(target_power=1.0),
        "DF": lambda: DecodeAndForwardRelay(target_power=1.0),
        "VIT-genie": lambda: ViterbiMLSERelay(channel_taps=H_ISI),
    }

    out = {name: {"errors": [], "bits": []} for name in
           list(relays) + ["VIT-est", "MLP"]}
    t0 = time.time()

    for snr in SNRS:
        acc = {k: [0, 0] for k in out}
        for tr in range(n_trials):
            source = e6.Source(seed=10_000 + 97 * snr + tr, modulation="bpsk")
            tx_bits, tx_sym = source.transmit(n_bits)
            y = hop1(tx_sym, snr)

            for name, make in relays.items():
                rx = destination.receive(hop2(make().process(y), snr))
                acc[name][0] += int(np.count_nonzero(tx_bits != rx))
                acc[name][1] += tx_bits.size

            # pilot-LS Viterbi: the estimate comes from the same observation
            est = ViterbiMLSERelay(pilot_symbols=(y[:N_PILOT], tx_sym[:N_PILOT]))
            rx = destination.receive(hop2(est.process(y), snr))
            acc["VIT-est"][0] += int(np.count_nonzero(tx_bits != rx))
            acc["VIT-est"][1] += tx_bits.size

            # the learned relay is averaged over its three training seeds
            for mlp in mlps:
                rx = destination.receive(hop2(mlp.process(y), snr))
                acc["MLP"][0] += int(np.count_nonzero(tx_bits != rx))
                acc["MLP"][1] += tx_bits.size

        for k in out:
            out[k]["errors"].append(acc[k][0])
            out[k]["bits"].append(acc[k][1])
        row = "  ".join(f"{k}={acc[k][0] / acc[k][1]:.3e}" for k in
                        ("AF", "DF", "MLP", "VIT-genie", "VIT-est"))
        print(f"  {snr:2d} dB  {row}")

    payload = {
        "description": "Matched-protocol unknown-ISI comparison: one tap "
                       "vector, one hop-2 noise convention, one bit budget.",
        "taps_as_configured": H_CHANNEL.tolist(),
        "taps_effective_given_to_detector": H_ISI.tolist(),
        "note_taps": "ISIChannel normalises internally, so the configured and "
                     "effective taps describe the same channel; the detector "
                     "is given the effective ones.",
        "hop2": "awgn, sigma^2 = N0/2 per real dimension (current convention)",
        "snrs": SNRS,
        "bits_per_trial": n_bits,
        "trials": n_trials,
        "mlp_training_seeds": list(SEEDS),
        "n_pilot": N_PILOT,
        "git_rev": git_rev(),
        "seconds": round(time.time() - t0, 1),
        "results": {k: {"errors": v["errors"], "bits": v["bits"],
                        "ber": [e / b for e, b in zip(v["errors"], v["bits"])]}
                    for k, v in out.items()},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bits", type=int, default=100_000)
    ap.add_argument("--trials", type=int, default=10)
    a = ap.parse_args()
    run(a.bits, a.trials)
