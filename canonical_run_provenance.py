"""Which runs produced the canonical Tables 5.2 and 5.3, and with what settings.

The canonical result files are in the repository's history, not before it:

* results/modulation/qpsk_rayleigh.json (Table 5.2) was written at
  2026-08-17T19:10:21 by `run_experiments.py --exp 7.10` with fresh training
  (`--retrain`), the cGAN skipped, on a four-core CPU without CUDA, from the
  code of commit b4059e6, and committed four minutes later as 69a470c ("Re-run
  the three canonical configurations", 2,892 s for the stage);
* results/normalized_3k/3k_rayleigh.json (Table 5.3) was written at
  2026-08-17T21:25:14 by `--exp 7.8 --retrain` from the code of 4cf9753 and
  committed as d27fcf7 (746.6 s, retrained rather than cache-loaded).

No driver or relay code that these runs execute has changed since, apart from
the Mamba wrappers gaining an optional `readout` argument whose default is the
centre readout the runs used. This script makes those statements checkable:

1. it re-evaluates the parameter-free AF and DF relays with today's code and
   the runs' budget and seeds, and checks that both tables' AF and DF per-trial
   BERs reproduce bit for bit (they share every random draw with the learned
   relays, so this pins the channel, the seeds and the evaluation budget);
2. it instantiates every learned relay of both tables from today's code and
   records its architecture and parameter count;
3. it derives each relay's training-set size exactly as its `train` method
   forms it (samples per SNR, edge windows dropped, the sequence models' 15%
   hold-out), so the sample counts quoted in Section 4.3.3 are computed rather
   than typed;
4. it confirms from git that the run-time code and today's code agree on the
   files the runs execute.

The learned relays themselves cannot be reproduced exactly: PyTorch's RNG was
not seeded and the canonical checkpoints are absent from the archived record.
The VAE shuffles batches with the seeded NumPy generator but draws its latent
training noise with PyTorch. That limit is recorded rather than hidden.

Output: results/canonical_run_provenance.json
Usage:  python canonical_run_provenance.py
"""
import json
import subprocess

import numpy as np

import run_experiments as R
from checkpoints.checkpoint_22_normalized_3k import build_all_3k
from relaynet.relays import AmplifyAndForwardRelay, DecodeAndForwardRelay

OUT = "results/canonical_run_provenance.json"
SNRS = np.arange(0, 21, 2)
RUNS = {
    "table_5_2": {"file": "results/modulation/qpsk_rayleigh.json", "stage": "7.10",
                  "code_commit": "b4059e6", "data_commit": "69a470c",
                  "command": "python3 run_experiments.py --exp 7.10 --num-trials 10 --bits-per-trial 10000 "
                             "--snr-min 0 --snr-max 20 --snr-step 2 --skip-relays cgan --retrain",
                  "stage_seconds": 2892, "hardware": "4-core CPU, no CUDA"},
    "table_5_3": {"file": "results/normalized_3k/3k_rayleigh.json", "stage": "7.8",
                  "code_commit": "4cf9753", "data_commit": "d27fcf7",
                  "command": "python3 run_experiments.py --exp 7.8 --num-trials 10 --bits-per-trial 10000 "
                             "--snr-min 0 --snr-max 20 --snr-step 2 --skip-relays cgan --retrain",
                  "stage_seconds": 746.6, "hardware": "4-core CPU, no CUDA"},
}
# Files the two stages execute; their content at the run commits is compared with HEAD.
EXECUTED = ["relaynet/relays/genai.py", "relaynet/relays/hybrid.py", "relaynet/relays/vae.py",
            "relaynet/relays/af.py", "relaynet/relays/df.py", "relaynet/simulation/runner.py",
            "relaynet/channels/fading.py", "relaynet/utils/activations.py",
            "checkpoints/checkpoint_18_transformer_relay.py", "checkpoints/checkpoint_20_mamba_s6_relay.py",
            "checkpoints/checkpoint_22_normalized_3k.py", "checkpoints/checkpoint_23_mamba2_relay.py"]


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout


def unchanged_since(commit):
    """Executed files whose content differs between `commit` and HEAD."""
    return [f for f in EXECUTED if git("diff", "--stat", commit, "HEAD", "--", f).strip()]


def windows(samples, n_snr, window):
    """Training windows exactly as the train methods form them (interior windows only)."""
    per = samples // n_snr
    return n_snr * (per - (window - 1))


def torch_params(r):
    for attr in ("_torch_model", "model"):
        m = getattr(r, attr, None)
        if m is not None and hasattr(m, "parameters"):
            return int(sum(p.numel() for p in m.parameters()))
    return int(r.num_params)


def relay_record(name, r, samples, snrs, lr, batch, loss, keep, epochs=100):
    w = getattr(r, "window_size", None) or r.mlp_relay.window_size
    n = windows(samples, len(snrs), w)
    seq = keep.startswith("lowest")
    rec = {"params": torch_params(r), "window": int(w), "training_snrs_db": snrs,
           "training_data": "BPSK +-1 over real AWGN", "nominal_samples": samples,
           "windows": int(n), "gradient_windows": int(n - int(n * 0.15)) if seq else int(n),
           "epochs": epochs, "optimizer": f"Adam lr={lr} batch={batch}", "loss": loss,
           "checkpoint": keep}
    for k in ("hidden_size", "latent_size", "hidden_sizes", "beta", "snr_threshold", "use_input_norm", "readout"):
        if hasattr(r, k):
            v = getattr(r, k)
            rec[k] = list(v) if isinstance(v, tuple) else v
    return rec


def main():
    out = {"description": __doc__.split("\n\n")[0], "git_rev": git("rev-parse", "HEAD").strip(),
           "runs": {}, "relays": {}}
    for key, run in RUNS.items():
        pub = json.load(open(run["file"]))
        res = R.evaluate_relays({"AF": AmplifyAndForwardRelay(), "DF": DecodeAndForwardRelay()}, SNRS,
                                channel_fn=R._CHANNELS["rayleigh"][1], modulation="qpsk",
                                bits_per_trial=10_000, num_trials=10)
        repro = {k: bool(np.array_equal(np.asarray(res[k]["ber_trials"]),
                                        np.asarray(pub["results"][k]["ber_trials"]))) for k in res}
        out["runs"][key] = dict(run, created=pub["created"], relays=list(pub["results"]),
                                af_df_reproduce_bit_for_bit=repro,
                                executed_files_changed_since_code_commit=unchanged_since(run["code_commit"]),
                                change_note=("the Mamba wrappers' only change is an optional readout argument "
                                             "(default 'center', the configuration these runs used)"),
                                trials=10, bits_per_trial=10_000, snr_db=SNRS.tolist(),
                                trial_seeds="0-9, the same at every SNR and for every relay")
        print(f"{key}: AF/DF reproduce {repro}; executed files changed since {run['code_commit']}: "
              f"{out['runs'][key]['executed_files_changed_since_code_commit']}")
    base = R.build_base_relays(gpu=False)
    canon = {
        "MLP (169p)": (25_000, [5, 10, 15], 0.01, 32, "MSE", "final epoch"),
        "Hybrid": (25_000, [2, 4, 6], 0.01, 32, "MSE", "final epoch"),
        "VAE": (50_000, [5, 10, 15], 0.001, 64, "beta-VAE (MSE + 0.1 KL)", "final epoch"),
        "Transformer": (50_000, [5, 10, 15], 0.001, 64, "MSE", "lowest mean training loss: kept when below the best kept loss by more than 1e-5, early stopping off (15% held out)"),
        "Mamba S6": (50_000, [5, 10, 15], 0.001, 64, "MSE", "lowest mean training loss: kept when below the best kept loss by more than 1e-5, early stopping off (15% held out)"),
        "Mamba2 (SSD)": (50_000, [5, 10, 15], 0.001, 64, "MSE", "lowest mean training loss: kept when below the best kept loss by more than 1e-5, early stopping off (15% held out)"),
    }
    out["relays"]["table_5_2"] = {n: relay_record(n, base[n], *v) for n, v in canon.items()}
    k3 = build_all_3k(prefer_gpu=False)
    seq3 = "lowest mean training loss: kept when below the best kept loss by more than 1e-5, early stopping off (15% held out)"
    three = {
        "MLP-3K": (25_000, [5, 10, 15], 0.01, 32, "MSE", "final epoch"),
        "Hybrid-3K": (25_000, [5, 10, 15], 0.01, 32, "MSE", "final epoch"),
        "VAE-3K": (25_000, [5, 10, 15], 0.001, 64, "beta-VAE (MSE + 0.1 KL)", "final epoch"),
        "Transformer-3K": (25_000, [5, 10, 15], 0.001, 64, "MSE", seq3),
        "Mamba-3K": (25_000, [5, 10, 15], 0.001, 64, "MSE", seq3),
        "Mamba2-3K": (25_000, [5, 10, 15], 0.001, 64, "MSE", seq3),
    }
    out["relays"]["table_5_3"] = {n: relay_record(n, k3[n], *v) for n, v in three.items()}
    out["not_reproducible"] = ("PyTorch's RNG was not seeded (weight initialization, sequence splits/dropout, "
                               "MLP/sequence batch order and VAE latent training noise). VAE batch order uses "
                               "seeded NumPy; evaluation decodes its latent mean. The canonical checkpoints "
                               "are absent from the archived record, so learned columns cannot be regenerated "
                               "exactly; training data are seeded (seed 42 + training SNR).")
    for t, rel in out["relays"].items():
        for n, r in rel.items():
            print(f"  {t} {n:15s} {r['params']:6d} params  window {r['window']:2d}  {r['windows']:6d} windows "
                  f"({r['gradient_windows']} for gradients)  {r['optimizer']}  {r['checkpoint']}")
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
