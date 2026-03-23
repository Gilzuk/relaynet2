"""
run_csi_experiment.py
============================
Experiment: Evaluate whether introducing Channel State Information (CSI) 
as an explicit secondary input feature helps Sequence Realy Models 
exceed Amplify-and-Forward (AF) performance for 16-QAM in Rayleigh.

Usage::
    python scripts/run_csi_experiment.py --quick
"""

import argparse
import json
import os
import sys
from time import perf_counter
from datetime import datetime

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from relaynet.relays.af import AmplifyAndForwardRelay
from relaynet.relays.df import DecodeAndForwardRelay
from relaynet.simulation.runner import run_monte_carlo
from relaynet.channels.fading import rayleigh_fading_channel
from relaynet.utils.activations import get_clip_range

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except Exception:
    _HAS_PLT = False

from checkpoints.checkpoint_20_mamba_s6_relay import MambaRelayWrapper
from checkpoints.checkpoint_18_transformer_relay import TransformerRelayWrapper
from checkpoints.checkpoint_23_mamba2_relay import Mamba2RelayWrapper


from scipy import stats

def plot_training_history(all_histories, out_dir):
    """Plot train loss, train accuracy and val accuracy for each model."""
    if not _HAS_PLT:
        return
    os.makedirs(out_dir, exist_ok=True)
    for name, hist in all_histories.items():
        epochs_range = range(1, len(hist['train_loss']) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(name, fontsize=13)

        # Loss
        ax1.plot(epochs_range, hist['train_loss'], 'b-', label='Train Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy
        ax2.plot(epochs_range, hist['train_acc'], 'g-', label='Train Accuracy')
        ax2.plot(epochs_range, hist['val_acc'], 'r--', label='Val Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy')
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '')
        path = os.path.join(out_dir, f"training_{safe_name}.png")
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved training chart: {path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--snr-min", type=float, default=0)
    p.add_argument("--snr-max", type=float, default=20)
    p.add_argument("--snr-step", type=float, default=2)
    p.add_argument("--bits-per-trial", type=int, default=10000)
    p.add_argument("--num-trials", type=int, default=10,
                   help="(deprecated, use --monte-carlo)")
    p.add_argument("--monte-carlo", "--mc", type=int, default=None,
                   help="Number of Monte-Carlo trials (default: 10)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=0,
                   help="Early-stopping patience (0=disabled)")
    p.add_argument("--min-delta", type=float, default=1e-5,
                   help="Min loss improvement for early stopping")
    p.add_argument("--retrain", action="store_true",
                   help="Force retraining even if cached weights exist")
    p.add_argument("--only", nargs="+", default=None,
                   choices=["baseline", "ln", "csi", "csi+ln"],
                   help="Run only selected configs (e.g. --only baseline csi+ln)")
    p.add_argument("--activations", nargs="+", default=None,
                   choices=["hardtanh", "scaled_tanh", "tanh", "sigmoid"],
                   help="Run only selected activations (e.g. --activations hardtanh tanh)")
    p.add_argument("--models", nargs="+", default=None,
                   choices=["mamba", "transformer", "mamba2"],
                   help="Run only selected model types (e.g. --models mamba transformer)")
    p.add_argument("--epochs", type=int, default=None,
                   help="Number of training epochs (default: 25, or 5 with --quick)")
    p.add_argument("--constellation", default="qam16",
                   choices=["bpsk", "qpsk", "qam16", "psk16"],
                   help="Modulation constellation (default: qam16)")
    args = p.parse_args()
    # --monte-carlo / --mc takes precedence over legacy --num-trials
    if args.monte_carlo is not None:
        args.num_trials = args.monte_carlo
    return args

def evaluate(relays, snr_range, args, modulation="qam16"):
    results = {}
    channel_fn = lambda s, snr: rayleigh_fading_channel(s, snr, return_channel=True)
    for name, relay in relays.items():
        print(f"    {name} ...", end=" ", flush=True)
        t0 = perf_counter()
        _, ber_mean, ber_trials = run_monte_carlo(
            relay, snr_range,
            num_bits_per_trial=args.bits_per_trial,
            num_trials=args.num_trials,
            channel_fn=channel_fn,
            seed_offset=args.seed,
            modulation=modulation,
        )
        print(f"done ({perf_counter()-t0:.1f}s)")
        results[name] = (ber_mean, ber_trials)
    return results

def plot_csi_results(results, snr_range, out_path):
    if not _HAS_PLT: return
    fig, ax = plt.subplots(figsize=(10, 7))
    # Distinct styles per variant
    colors = ["#999999", "#555555",
              "#0072b2", "#d55e00", "#009e73", "#cc79a7",
              "#e69f00", "#56b4e9", "#f0e442", "#000000",
              "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
              "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
              "#bcbd22", "#17becf"]
    markers = ["x", "+", "^", "s", "*", "D", "v", "p", "h", "o",
               "<", ">", "1", "2", "3", "4", "8", "P", "X", "d"]
    linestyles = ["-", "--", "-.", ":"]
    for idx, (name, (ber_mean, ber_trials)) in enumerate(results.items()):
        c = colors[idx % len(colors)]
        m = markers[idx % len(markers)]
        ls = linestyles[idx % len(linestyles)]
        ax.semilogy(snr_range, ber_mean, marker=m, color=c,
                    linestyle=ls, label=name, markersize=7)
        # 95% CI band
        n_trials = ber_trials.shape[1]
        if n_trials >= 2:
            se = np.std(ber_trials, axis=1, ddof=1) / np.sqrt(n_trials)
            t_crit = stats.t.ppf(0.975, df=n_trials - 1)
            ci_lo = np.clip(ber_mean - t_crit * se, 1e-10, None)
            ci_hi = ber_mean + t_crit * se
            ax.fill_between(snr_range, ci_lo, ci_hi, color=c, alpha=0.12)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Bit Error Rate (BER)")
    ax.set_title("Rayleigh: CSI / LayerNorm / Activation Comparison (95% CI)")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(fontsize=8, loc="lower left")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def save_results_json(results, snr_range, args, out_path):
    """Save all BER results to JSON for later retrieval."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "constellation": args.constellation,
        "snr_range": snr_range.tolist(),
        "num_trials": args.num_trials,
        "bits_per_trial": args.bits_per_trial,
        "patience": args.patience,
        "seed": args.seed,
        "quick": args.quick,
        "results": {}
    }
    for name, (ber_mean, ber_trials) in results.items():
        n_trials = ber_trials.shape[1]
        entry = {
            "ber_mean": ber_mean.tolist(),
            "ber_trials": ber_trials.tolist(),
        }
        if n_trials >= 2:
            se = np.std(ber_trials, axis=1, ddof=1) / np.sqrt(n_trials)
            t_crit = stats.t.ppf(0.975, df=n_trials - 1)
            entry["ci_95_lower"] = np.clip(ber_mean - t_crit * se, 0, None).tolist()
            entry["ci_95_upper"] = (ber_mean + t_crit * se).tolist()
        data["results"][name] = entry
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved results JSON to {out_path}")

def plot_top3(results, snr_range, constellation, out_dir):
    """Plot the 3 best neural architectures vs AF and DF baselines."""
    if not _HAS_PLT:
        return
    # Separate classical from neural
    classical = {}
    neural = {}
    for name, (ber_mean, ber_trials) in results.items():
        if name in ("AF", "DF"):
            classical[name] = (ber_mean, ber_trials)
        else:
            neural[name] = (ber_mean, ber_trials)
    if not neural:
        return
    # Rank neural by average BER across the upper half of SNR range
    mid = len(snr_range) // 2
    ranking = sorted(neural.items(), key=lambda kv: np.mean(kv[1][0][mid:]))
    top3 = ranking[:3]

    fig, ax = plt.subplots(figsize=(10, 7))
    # Classical baselines in grey/black
    baseline_styles = {"AF": ("#999999", "x", "--"), "DF": ("#555555", "+", "-.")}
    for bname, (ber_mean, ber_trials) in classical.items():
        c, m, ls = baseline_styles[bname]
        ax.semilogy(snr_range, ber_mean, marker=m, color=c,
                    linestyle=ls, label=bname, markersize=7, linewidth=1.0)
        n_trials = ber_trials.shape[1]
        if n_trials >= 2:
            se = np.std(ber_trials, axis=1, ddof=1) / np.sqrt(n_trials)
            t_crit = stats.t.ppf(0.975, df=n_trials - 1)
            ci_lo = np.clip(ber_mean - t_crit * se, 1e-10, None)
            ci_hi = ber_mean + t_crit * se
            ax.fill_between(snr_range, ci_lo, ci_hi, color=c, alpha=0.12)

    # Top-3 neural in distinct bold colors
    top_colors = ["#0072b2", "#d55e00", "#009e73"]
    top_markers = ["^" , "s", "D"]
    for idx, (name, (ber_mean, ber_trials)) in enumerate(top3):
        c = top_colors[idx]
        m = top_markers[idx]
        rank_label = f"#{idx+1} {name}"
        ax.semilogy(snr_range, ber_mean, marker=m, color=c,
                    linestyle="-", label=rank_label, markersize=8, linewidth=1.5)
        n_trials = ber_trials.shape[1]
        if n_trials >= 2:
            se = np.std(ber_trials, axis=1, ddof=1) / np.sqrt(n_trials)
            t_crit = stats.t.ppf(0.975, df=n_trials - 1)
            ci_lo = np.clip(ber_mean - t_crit * se, 1e-10, None)
            ci_hi = ber_mean + t_crit * se
            ax.fill_between(snr_range, ci_lo, ci_hi, color=c, alpha=0.15)

    # Y-axis: one decade below min nonzero BER
    all_ber = np.concatenate([v[0] for v in results.values()])
    min_nonzero = all_ber[all_ber > 0].min() if np.any(all_ber > 0) else 1e-5
    ax.set_ylim(bottom=min_nonzero / 10)

    ax.set_xlabel("SNR (dB)", fontsize=14)
    ax.set_ylabel("Bit Error Rate (BER)", fontsize=14)
    ax.set_title(f"Top-3 Neural Relays vs Classical — {constellation.upper()} Rayleigh (95% CI)",
                 fontsize=16)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(fontsize=11, loc="best", framealpha=0.9)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"top3_{constellation}_rayleigh.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved top-3 chart to {out_path}")
    return [name for name, _ in top3]

def main():
    args = parse_args()
    snr_range = np.arange(args.snr_min, args.snr_max + args.snr_step, args.snr_step)
    
    samples = 1000 if args.quick else 20000
    epochs = args.epochs if args.epochs is not None else (5 if args.quick else 25)
    
    constellation = args.constellation
    print(f"--- CSI Injection Experiment ({constellation.upper()} / Rayleigh) ---")
    clip = get_clip_range(constellation)
    
    relays = {
        "AF": AmplifyAndForwardRelay(target_power=1.0),
        "DF": DecodeAndForwardRelay(target_power=1.0)
    }
    all_histories = {}
    
    base_kw = dict(target_power=1.0, window_size=11, d_model=32, num_layers=2, clip_range=clip, prefer_gpu=args.gpu)
    train_kw = dict(training_snrs=[5, 10, 15], num_samples=samples, epochs=epochs,
                    training_modulation=constellation, use_rayleigh=True,
                    patience=args.patience, min_delta=args.min_delta)
    
    model_list = args.models or ["mamba", "transformer", "mamba2"]
    
    # --- Build all variants (model_type x CSI x LN x activation) ---
    act_list = args.activations or ["hardtanh", "scaled_tanh", "tanh", "sigmoid"]
    configs = []
    for model_type in model_list:
        for act in act_list:
            for csi in [False, True]:
                for ln in [False, True]:
                    cfg_key = "+".join(filter(None, ["csi" if csi else "", "ln" if ln else ""])) or "baseline"
                    if args.only and cfg_key not in args.only:
                        continue
                    ch = 2 if csi else 1
                    parts = []
                    if csi:
                        parts.append("CSI")
                    if ln:
                        parts.append("LN")
                    if parts:
                        label = "+" + "+".join(parts) + " " + act
                    else:
                        label = "Baseline " + act
                    configs.append((model_type, label, ch, ln, act))

    weights_dir = os.path.join("results", "csi", "weights")
    os.makedirs(weights_dir, exist_ok=True)

    for model_type, label, ch, ln, act in configs:
        tag = model_type.capitalize()
        if model_type == "mamba":
            tag = "Mamba"
        elif model_type == "transformer":
            tag = "Transformer"
        elif model_type == "mamba2":
            tag = "Mamba2"
        name = f"{tag} ({label})"
        safe = (model_type + "_" + label).replace(' ', '_').replace('+', '').replace('(', '').replace(')', '')
        wpath = os.path.join(weights_dir, f"{safe}.pt")

        if model_type == "mamba":
            r = MambaRelayWrapper(**base_kw, d_state=16, in_channels=ch, use_input_norm=ln, output_activation=act)
        elif model_type == "transformer":
            r = TransformerRelayWrapper(**base_kw, num_heads=4, in_channels=ch, use_input_norm=ln, output_activation=act)
        elif model_type == "mamba2":
            r = Mamba2RelayWrapper(**base_kw, d_state=16, chunk_size=8, in_channels=ch, use_input_norm=ln, output_activation=act)

        loaded = False
        if not args.retrain and os.path.exists(wpath):
            loaded = r.load_weights(wpath)
            if loaded:
                print(f"\n{name}: loaded cached weights from {wpath}")
            else:
                print(f"\n{name}: architecture changed, retraining...")

        if not loaded:
            print(f"\nBuilding {name}...")
            all_histories[name] = r.train(**train_kw)
            r.save_weights(wpath)
            print(f"  Saved weights to {wpath}")

        relays[name] = r
    
    print("\nEvaluating...")
    res = evaluate(relays, snr_range, args, constellation)
    
    # print BER table with 95% CI
    print("\n--- BER Summary (mean \u00b1 95% CI) ---")
    print(f"{'SNR':>5} |", end="")
    for name in relays.keys():
        print(f" {name[:24]:>24} |", end="")
    print()
    for i, snr in enumerate(snr_range):
        print(f"{snr:5.1f} |", end="")
        for name in relays.keys():
            ber_mean, ber_trials = res[name]
            mean_val = ber_mean[i]
            n_trials = ber_trials.shape[1]
            if n_trials >= 2:
                se = np.std(ber_trials[i], ddof=1) / np.sqrt(n_trials)
                t_crit = stats.t.ppf(0.975, df=n_trials - 1)
                ci = t_crit * se
                print(f" {mean_val:.4e}\u00b1{ci:.1e} |", end="")
            else:
                print(f" {mean_val:24.4e} |", end="")
        print()
    
    # Save results to JSON
    json_path = f"results/csi/csi_experiment_{constellation}_rayleigh.json"
    save_results_json(res, snr_range, args, json_path)

    out_path = f"results/csi/csi_experiment_{constellation}_rayleigh.png"
    plot_csi_results(res, snr_range, out_path)
    print(f"\nSaved plot to {out_path}")

    # Top-3 chart
    top3_names = plot_top3(res, snr_range, constellation, "results/csi")
    if top3_names:
        print(f"\nTop-3 architectures ({constellation.upper()}):")
        for i, n in enumerate(top3_names, 1):
            print(f"  {i}. {n}")

    # Training history charts
    plot_training_history(all_histories, "results/csi")
    print("DONE.")

if __name__ == "__main__":
    main()