"""
run_sic_only.py
===============
Quick script to run ONLY the 2×2 MIMO SIC comparison (skipping AWGN,
Rayleigh, Rician, MIMO-ZF, and MIMO-MMSE).  Trains the same relay
models as run_full_comparison.py, then evaluates them on the SIC channel only.

Usage::

    python scripts/run_sic_only.py
    python scripts/run_sic_only.py --include-sequence-models --gpu --log-timings
"""

import argparse
import os
import sys
from time import perf_counter

# Ensure UTF-8 output on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from relaynet.relays.af import AmplifyAndForwardRelay
from relaynet.relays.df import DecodeAndForwardRelay
from relaynet.relays.genai import MinimalGenAIRelay
from relaynet.relays.hybrid import HybridRelay
from relaynet.relays.vae import VAERelay
from relaynet.relays.cgan import CGANRelay
from relaynet.simulation.runner import run_monte_carlo
from relaynet.simulation.statistics import (
    compute_confidence_interval,
    significance_table,
)
from relaynet.visualization.plots import plot_ber_curves, plot_ber_with_ci
from relaynet.channels.mimo import (
    mimo_2x2_channel,
    mimo_2x2_mmse_channel,
    mimo_2x2_sic_channel,
)

try:
    from checkpoints.checkpoint_18_transformer_relay import TransformerRelayWrapper
    from checkpoints.checkpoint_20_mamba_s6_relay import MambaRelayWrapper
    _HAS_SEQUENCE_MODELS = True
except Exception:
    TransformerRelayWrapper = None
    MambaRelayWrapper = None
    _HAS_SEQUENCE_MODELS = False


def parse_args():
    p = argparse.ArgumentParser(description="SIC-only relay comparison")
    p.add_argument("--include-sequence-models", action="store_true")
    p.add_argument("--snr-min", type=float, default=0)
    p.add_argument("--snr-max", type=float, default=20)
    p.add_argument("--snr-step", type=float, default=2)
    p.add_argument("--bits-per-trial", type=int, default=10_000)
    p.add_argument("--num-trials", type=int, default=10)
    p.add_argument("--genai-samples", type=int, default=25_000)
    p.add_argument("--genai-epochs", type=int, default=100)
    p.add_argument("--hybrid-samples", type=int, default=25_000)
    p.add_argument("--hybrid-epochs", type=int, default=100)
    p.add_argument("--vae-samples", type=int, default=50_000)
    p.add_argument("--vae-epochs", type=int, default=100)
    p.add_argument("--cgan-samples", type=int, default=50_000)
    p.add_argument("--cgan-epochs", type=int, default=200)
    p.add_argument("--transformer-samples", type=int, default=50_000)
    p.add_argument("--transformer-epochs", type=int, default=100)
    p.add_argument("--mamba-samples", type=int, default=50_000)
    p.add_argument("--mamba-epochs", type=int, default=100)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--log-timings", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _relay_device(relay):
    device = getattr(relay, "device", None)
    return "cpu" if device is None else str(device)


def _timed(label, fn, *args, log_timings=False, **kwargs):
    start = perf_counter()
    result = fn(*args, **kwargs)
    elapsed = perf_counter() - start
    if log_timings:
        print(f"    [time] {label}: {elapsed:.2f}s")
    return result, elapsed


def _make_progress(name, interval=20):
    """Return an epoch_callback that prints every *interval* epochs."""
    def _cb(epoch, total):
        ep = epoch + 1
        if ep % interval == 0 or ep == total:
            print(f"    {name}: epoch {ep}/{total}", flush=True)
    return _cb


def train_models(args):
    """Train relay models (same as full comparison)."""
    print("\n=== Training AI relay models ===")
    timing = {}

    genai = MinimalGenAIRelay(window_size=5, hidden_size=24, prefer_gpu=False)
    print(f"  Training GenAI (169p) … device: {_relay_device(genai)}")
    _, elapsed = _timed("train GenAI", genai.train,
                        training_snrs=[5, 10, 15],
                        num_samples=args.genai_samples, epochs=args.genai_epochs,
                        seed=args.seed, log_timings=args.log_timings,
                        epoch_callback=_make_progress("GenAI"))
    timing["GenAI"] = (_relay_device(genai), elapsed)

    hybrid = HybridRelay(snr_threshold=5.0, prefer_gpu=False)
    print(f"  Training Hybrid … device: {_relay_device(hybrid)}")
    _, elapsed = _timed("train Hybrid", hybrid.train,
                        training_snrs=[2, 4, 6],
                        num_samples=args.hybrid_samples, epochs=args.hybrid_epochs,
                        seed=args.seed, log_timings=args.log_timings,
                        epoch_callback=_make_progress("Hybrid"))
    timing["Hybrid"] = (_relay_device(hybrid), elapsed)

    vae = VAERelay(window_size=5, prefer_gpu=args.gpu)
    print(f"  Training VAE … device: {_relay_device(vae)}")
    _, elapsed = _timed("train VAE", vae.train,
                        training_snrs=[5, 10, 15],
                        num_samples=args.vae_samples, epochs=args.vae_epochs,
                        seed=args.seed, log_timings=args.log_timings,
                        epoch_callback=_make_progress("VAE"))
    timing["VAE"] = (_relay_device(vae), elapsed)

    cgan = CGANRelay(window_size=5, prefer_gpu=args.gpu)
    print(f"  Training CGAN … device: {_relay_device(cgan)}")
    _, elapsed = _timed("train CGAN", cgan.train,
                        training_snrs=[5, 10, 15],
                        num_samples=args.cgan_samples, epochs=args.cgan_epochs,
                        seed=args.seed, log_timings=args.log_timings,
                        epoch_callback=_make_progress("CGAN"))
    timing["CGAN"] = (_relay_device(cgan), elapsed)

    relays = {
        "AF": AmplifyAndForwardRelay(),
        "DF": DecodeAndForwardRelay(),
        "GenAI (169p)": genai,
        "Hybrid": hybrid,
        "VAE": vae,
        "CGAN (WGAN-GP)": cgan,
    }

    if args.include_sequence_models:
        if not _HAS_SEQUENCE_MODELS:
            print("  [WARNING] Sequence models not available, skipping.")
        else:
            transformer = TransformerRelayWrapper(prefer_gpu=args.gpu)
            print(f"  Training Transformer … device: {_relay_device(transformer)}")
            _, elapsed = _timed("train Transformer", transformer.train,
                                training_snrs=[5, 10, 15],
                                num_samples=args.transformer_samples,
                                epochs=args.transformer_epochs, lr=0.001,
                                log_timings=args.log_timings)
            timing["Transformer"] = (_relay_device(transformer), elapsed)

            mamba = MambaRelayWrapper(prefer_gpu=args.gpu)
            print(f"  Training Mamba … device: {_relay_device(mamba)}")
            _, elapsed = _timed("train Mamba", mamba.train,
                                training_snrs=[5, 10, 15],
                                num_samples=args.mamba_samples,
                                epochs=args.mamba_epochs, lr=0.001,
                                log_timings=args.log_timings)
            timing["Mamba"] = (_relay_device(mamba), elapsed)

            relays["Transformer"] = transformer
            relays["Mamba S6"] = mamba

    if args.log_timings:
        print("\n=== Training Time Summary ===")
        for name, (device, elapsed) in timing.items():
            print(f"  {name:<20} {device:<6} {elapsed:>8.2f}s")

    return relays


def run_comparison(relays, snr_range, bits_per_trial, num_trials, channel_fn, title):
    """Run one channel comparison."""
    print(f"\n=== {title} ===")
    n_relays = len(relays)
    all_ber, all_trials = {}, {}
    for r_idx, (name, relay) in enumerate(relays.items(), 1):
        print(f"  [{r_idx}/{n_relays}] {name} …", flush=True)
        start = perf_counter()
        snr_values = np.array(snr_range)
        n_snr = len(snr_values)
        ber_values = np.zeros(n_snr)
        ber_trials = np.zeros((n_snr, num_trials))

        for si, snr_db in enumerate(snr_values):
            from relaynet.simulation.runner import simulate_transmission
            trial_bers = []
            for trial in range(num_trials):
                b, _ = simulate_transmission(
                    relay, bits_per_trial, snr_db,
                    seed=trial, channel_fn=channel_fn,
                )
                trial_bers.append(b)
            ber_trials[si] = trial_bers
            ber_values[si] = np.mean(trial_bers)
            # Progress: print every 3 SNR points or at the last one
            if (si + 1) % 3 == 0 or si + 1 == n_snr:
                print(f"    SNR {si+1}/{n_snr} ({snr_db:.0f} dB) "
                      f"BER={ber_values[si]:.2e}", flush=True)

        all_ber[name] = ber_values
        all_trials[name] = ber_trials
        elapsed = perf_counter() - start
        print(f"    done (device={_relay_device(relay)}, time={elapsed:.2f}s, "
              f"BER [{ber_values.min():.2e}, {ber_values.max():.2e}])")
    return all_ber, all_trials


def main():
    args = parse_args()
    np.random.seed(args.seed)
    total_start = perf_counter()
    os.makedirs("results", exist_ok=True)

    snr_range = np.arange(args.snr_min, args.snr_max + 0.01, args.snr_step)
    relays = train_models(args)

    # ── Run all three MIMO equalizers for comparison ────────────────
    configs = [
        (mimo_2x2_channel,      "2×2 MIMO – Rayleigh – ZF Equalization",
         "results/mimo_2x2_comparison_ci.png"),
        (mimo_2x2_mmse_channel, "2×2 MIMO – Rayleigh – MMSE Equalization",
         "results/mimo_2x2_mmse_comparison_ci.png"),
        (mimo_2x2_sic_channel,  "2×2 MIMO – Rayleigh – SIC Equalization",
         "results/mimo_2x2_sic_comparison_ci.png"),
    ]

    for ch_fn, title, save_path in configs:
        ber, trials = run_comparison(
            relays, snr_range, args.bits_per_trial, args.num_trials, ch_fn, title,
        )

        # Significance
        baseline = "DF"
        methods = [k for k in ber if k != baseline]
        print(f"\n=== Statistical Significance vs {baseline} ===")
        significance_table(
            snr_range, methods,
            {k: trials[k] for k in trials},
            baseline=baseline,
        )

        if not args.no_plots:
            ci_dict = {}
            for name, t in trials.items():
                lo, hi = compute_confidence_interval(t)
                ci_dict[name] = (lo, hi)
            plot_ber_with_ci(
                snr_range, ber, ci_dict,
                title=f"{title} – All Relay Methods (95% CI)",
                save_path=save_path,
            )

    print(f"\nDone. Total elapsed: {perf_counter() - total_start:.2f}s")
    print("Results saved to results/")


if __name__ == "__main__":
    main()
