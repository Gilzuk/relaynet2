"""Re-run the E6 AWGN Viterbi reference under the current channel code.

The historical ``e6_viterbi_awgn.npy`` is retained for reproducibility.  This
script writes a new, explicitly prefixed artifact so a corrected noise
convention cannot silently replace the published checkpoint.
"""

import json
import os
import subprocess

import numpy as np

import e6_viterbi_ported as baseline


def main() -> None:
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "e6_unknown_channel_results")
    os.makedirs(out_dir, exist_ok=True)

    corrected = {
        "VIT-genie": baseline.run_experiment("awgn", use_genie_csi=True),
        "VIT-est": baseline.run_experiment("awgn", use_genie_csi=False),
    }
    npy_path = os.path.join(out_dir, "codex_viterbi_consistency_awgn.npy")
    np.save(npy_path, corrected, allow_pickle=True)

    repo_dir = os.path.dirname(os.path.abspath(__file__))
    commit = subprocess.check_output(
        ["git", "-c", f"safe.directory={repo_dir}", "rev-parse", "HEAD"],
        text=True,
        cwd=repo_dir,
    ).strip()
    manifest = {
        "artifact": os.path.basename(npy_path),
        "source_script": "e6_viterbi_ported.py",
        "source_commit": commit,
        "hop2": "awgn",
        "snrs_db": baseline.SNRS.tolist(),
        "n_trials": baseline.N_TRIALS,
        "n_bits_per_trial": baseline.N_BITS,
        "pilot_symbols": baseline.N_PILOT,
        "channel_taps": baseline.H_ISI.tolist(),
        "noise_convention": "current relaynet channel implementation",
        "historical_artifact_retained": "e6_viterbi_awgn.npy",
    }
    manifest_path = os.path.join(out_dir, "codex_viterbi_consistency_awgn.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    print(f"Saved corrected Viterbi results to {npy_path}")
    print(f"Saved provenance manifest to {manifest_path}")
    for key, values in corrected.items():
        print(f"{key}: " + " ".join(f"{value:.6g}" for value in values))


if __name__ == "__main__":
    main()
