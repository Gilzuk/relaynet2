"""Re-evaluate only composite AF after correcting complex hop-2 noise.

Original seeds, SNRs, 10 independent trials and 100,000 source bits per trial.
The original driver repeated the same AF seeds for each MLP training instance;
those copies are not additional independent trials. Never overwrite that file.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from e6_composite_ported import H_ISI, N_BITS, N_TRIALS, SNRS, run_ber_trial
from relaynet.channels import AdaptiveRayleighChannel, CompositeChannel
from relaynet.relays import AmplifyAndForwardRelay


def main():
    output = ROOT / "e6_unknown_channel_results/codex_composite_af_validation.json"
    if output.exists():
        raise FileExistsError(f"Preserving existing validation: {output}")
    hop1 = CompositeChannel(isi_taps=H_ISI, pa_sat=1.2, include_phase=True)
    hop2 = AdaptiveRayleighChannel()
    relay = AmplifyAndForwardRelay(target_power=1.0)
    points = {}
    for si, snr in enumerate(SNRS):
        seeds = [7000 * si + tr for tr in range(N_TRIALS)]
        rates = np.array([run_ber_trial("AF", relay, hop1, hop2, N_BITS,
                                       int(snr), seed) for seed in seeds])
        bits = N_BITS - 1  # initial differential reference is not scored
        points[str(snr)] = {
            "seeds": seeds, "bits_per_trial": bits,
            "errors_per_trial": np.rint(rates * bits).astype(int).tolist(),
            "ber_per_trial": rates.tolist(), "mean": float(rates.mean()),
            "ci95_halfwidth": float(t.ppf(.975, N_TRIALS - 1)
                                     * rates.std(ddof=1) / np.sqrt(N_TRIALS)),
        }
        print(snr, points[str(snr)]["mean"], flush=True)
    sources = ["scripts/codex_composite_af_validation.py", "e6_composite_ported.py",
               "relaynet/channels/e6_channels.py", "relaynet/relays/af.py"]
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
              for p in sources}
    result = {"measured_at": "destination, excluding differential reference bit",
              "noise_convention": "Es=1; real-axis variance=N0/2; complex total=N0",
              "protocol": "original composite AF seeds and bit budget; no retraining",
              "interval": "Student-t over 10 independent trial BERs; not repeated seed copies",
              "source_sha256": hashes, "points": points}
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
