"""
Regenerate CI figures with fixed zoom inset (upper right, 30%x28%).
Reads existing JSON files and regenerates only the figures used in the thesis.
No torch required.
"""
import json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

PALETTE = [
    "#e6194b","#3cb44b","#4363d8","#f58231","#911eb4",
    "#42d4f4","#f032e6","#bfef45","#469990","#dcbeff",
    "#9a6324","#800000","#aaffc3","#808000","#000075",
    "#e6beff","#ffe119","#fabebe","#7f7f7f","#a9a9a9",
    "#008080","#e41a1c","#377eb8","#ff7f00","#984ea3",
    "#00ced1","#ff1493","#228b22","#8b4513","#4b0082",
]
MARKERS = ["o","s","^","D","v","P","X","<",">","h","p","*","H","d","8","+","x"]
BASELINE_STYLE = {
    "AF": {"color":"grey",  "marker":"o","ls":"-"},
    "DF": {"color":"black", "marker":"s","ls":"-"},
}
RELAY_STYLE = {
    "MLP (169p)":     {"color":PALETTE[0], "marker":"^"},
    "Hybrid":         {"color":PALETTE[1], "marker":"D"},
    "VAE":            {"color":PALETTE[2], "marker":"v"},
    "CGAN (WGAN-GP)": {"color":PALETTE[3], "marker":"P"},
    "Transformer":    {"color":PALETTE[4], "marker":"X"},
    "Mamba S6":       {"color":PALETTE[5], "marker":"<"},
    "Mamba2 (SSD)":   {"color":PALETTE[6], "marker":">"},
}

def _style_for(name, idx=0):
    if name in BASELINE_STYLE:
        return BASELINE_STYLE[name]
    if name in RELAY_STYLE:
        return {**RELAY_STYLE[name], "ls":"-"}
    return {"color":PALETTE[idx%len(PALETTE)], "marker":MARKERS[idx%len(MARKERS)], "ls":"-"}

def _apply_jitter(ber_arrays):
    names = list(ber_arrays.keys())
    jittered = {n: np.array(b, dtype=float) for n,b in ber_arrays.items()}
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a,b = jittered[names[i]], jittered[names[j]]
            overlap = (a>0)&(b>0)&(np.abs(a-b)/np.maximum(a,1e-15)<0.005)
            if np.any(overlap):
                jittered[names[i]][overlap] *= 1.03
                jittered[names[j]][overlap] *= 0.97
    return jittered

def _add_congestion_inset(ax, snr, ber_dict, style_infos):
    """Fixed zoom inset: upper right, 30%x28%, clipped to axes."""
    ber_arrays = [np.asarray(b, dtype=float) for b in ber_dict.values()]
    if len(ber_arrays) < 3:
        return
    snr = np.asarray(snr)
    n = len(snr)
    lo_idx = max(0, n - max(3, int(n*0.4)))
    hi_idx = n - 1
    snr_lo, snr_hi = snr[lo_idx], snr[hi_idx]
    vals = []
    for b in ber_arrays:
        for si in range(lo_idx, hi_idx+1):
            if b[si] > 0:
                vals.append(b[si])
    if not vals:
        return
    ber_lo = min(vals)*0.3
    ber_hi = max(vals)*3.0

    # Fixed: upper right, smaller, clipped
    axins = inset_axes(ax, width="30%", height="28%", loc="upper right",
                       borderpad=1.2)
    for name, sinfo in style_infos.items():
        ber = np.asarray(ber_dict[name], dtype=float)
        ber_plot = np.where(ber>0, ber, 1e-10)
        axins.semilogy(snr, ber_plot, marker=sinfo["marker"],
                       color=sinfo["color"], linewidth=sinfo.get("lw",1.3),
                       markersize=4, linestyle=sinfo.get("ls","-"),
                       alpha=sinfo.get("alpha",0.9))
    axins.set_xlim(snr_lo-0.5, snr_hi+0.5)
    axins.set_ylim(ber_lo, ber_hi)
    axins.grid(True, which="both", linestyle="--", alpha=0.2, linewidth=0.3)
    axins.tick_params(labelsize=8)
    # Clip inset to axes bounding box to prevent overflow
    axins.set_clip_on(True)
    try:
        ax.indicate_inset_zoom(axins, edgecolor="grey", alpha=0.5)
    except Exception:
        pass

def plot_ber_chart(snr, ber_dict, ci_dict=None, title="", save_path=None,
                   extra_styles=None, show_inset=True):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    snr = np.asarray(snr)
    jittered = _apply_jitter(ber_dict)
    style_map = extra_styles or {}
    style_infos = {}
    for idx, (name, ber) in enumerate(ber_dict.items()):
        ber_j = jittered[name]
        st = style_map.get(name, _style_for(name, idx))
        color = st.get("color", PALETTE[idx%len(PALETTE)])
        marker = st.get("marker", MARKERS[idx%len(MARKERS)])
        ls = st.get("ls", "-")
        lw = st.get("lw", 1.3)
        alpha = st.get("alpha", 0.9)
        style_infos[name] = {"color":color,"marker":marker,"ls":ls,"lw":lw,"alpha":alpha}
        ber_plot = np.where(ber_j>0, ber_j, 1e-10)
        ax.semilogy(snr, ber_plot, marker=marker, color=color,
                    linewidth=lw, markersize=6, label=name,
                    linestyle=ls, alpha=alpha)
        if ci_dict and name in ci_dict:
            lo, hi = ci_dict[name]
            lo = np.maximum(np.asarray(lo, dtype=float), 1e-10)
            hi = np.asarray(hi, dtype=float)
            ax.fill_between(snr, lo, hi, alpha=0.15, color=color)
    all_ber = np.concatenate([np.asarray(b, dtype=float) for b in ber_dict.values()])
    min_ber = all_ber[all_ber>0].min() if np.any(all_ber>0) else 1e-6
    bottom = 10**(np.floor(np.log10(min_ber))-1)
    ax.set_ylim(bottom=max(bottom,1e-8), top=1)
    ax.grid(True, which="both", linestyle="--", alpha=0.3, linewidth=0.5)
    ax.set_xlabel("SNR (dB)", fontsize=14)
    ax.set_ylabel("Bit Error Rate (BER)", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=12, loc="upper left", bbox_to_anchor=(1.02,1),
              borderaxespad=0, framealpha=0.9)
    ax.tick_params(labelsize=12)
    if show_inset and len(ber_dict)>=3:
        try:
            _add_congestion_inset(ax, snr, ber_dict, style_infos)
        except Exception as e:
            print(f"    [inset warning] {e}")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)

def load_json(path):
    with open(path) as f:
        return json.load(f)

def regen_from_json(json_path, save_path, title):
    if not os.path.exists(json_path):
        print(f"  SKIP (no JSON): {json_path}")
        return
    data = load_json(json_path)
    snr = np.array(data["snr_range"])
    results = data["results"]
    ber_dict = {n: np.array(r["ber_mean"]) for n,r in results.items()}
    ci_dict = {}
    for n,r in results.items():
        if "ci_lower" in r and "ci_upper" in r:
            ci_dict[n] = (np.array(r["ci_lower"]), np.array(r["ci_upper"]))
    plot_ber_chart(snr, ber_dict, ci_dict if ci_dict else None,
                   title=title, save_path=save_path)

# ── Figures used in thesis ch04_experiments.tex ──────────────────
REGEN_TASKS = [
    # (json_path, save_path, title)
    # BPSK comparison
    ("results/bpsk_comparison/awgn.json",
     "results/awgn_comparison_ci.png",
     "AWGN — BPSK Relay Comparison"),
    ("results/bpsk_comparison/rayleigh.json",
     "results/fading_comparison.png",
     "Rayleigh Fading — BPSK Relay Comparison"),
    ("results/bpsk_comparison/rician_k3.json",
     "results/rician_comparison_ci.png",
     "Rician K=3 — BPSK Relay Comparison"),
    ("results/bpsk_comparison/mimo_zf.json",
     "results/mimo_2x2_comparison_ci.png",
     "2×2 MIMO ZF — BPSK Relay Comparison"),
    ("results/bpsk_comparison/mimo_mmse.json",
     "results/mimo_2x2_mmse_comparison_ci.png",
     "2×2 MIMO MMSE — BPSK Relay Comparison"),
    ("results/bpsk_comparison/mimo_sic.json",
     "results/mimo_2x2_sic_comparison_ci.png",
     "2×2 MIMO SIC — BPSK Relay Comparison"),
    # Modulation comparison
    ("results/modulation/bpsk_awgn.json",
     "results/modulation/bpsk_awgn_ci.png",
     "BPSK — AWGN Relay Comparison"),
    ("results/modulation/bpsk_rayleigh.json",
     "results/modulation/bpsk_rayleigh_ci.png",
     "BPSK — Rayleigh Relay Comparison"),
    ("results/modulation/qpsk_awgn.json",
     "results/modulation/qpsk_awgn_ci.png",
     "QPSK — AWGN Relay Comparison"),
    ("results/modulation/qpsk_rayleigh.json",
     "results/modulation/qpsk_rayleigh_ci.png",
     "QPSK — Rayleigh Relay Comparison"),
    ("results/modulation/qam16_awgn.json",
     "results/modulation/qam16__awgn_ci.png",
     "16-QAM — AWGN Relay Comparison"),
    ("results/modulation/qam16_rayleigh.json",
     "results/modulation/qam16__rayleigh_ci.png",
     "16-QAM — Rayleigh Relay Comparison"),
    # QAM16 activation
    ("results/qam16_activation/qam16_activation_awgn.json",
     "results/qam16_activation/qam16_activation_awgn.png",
     "QAM16 Activation — AWGN"),
    ("results/qam16_activation/qam16_activation_rayleigh.json",
     "results/qam16_activation/qam16_activation_rayleigh.png",
     "QAM16 Activation — Rayleigh"),
    # Activation comparison
    ("results/activation_comparison/activation_qpsk_awgn.json",
     "results/activation_comparison/bpsk_activation_awgn.png",
     "Activation Comparison — QPSK AWGN"),
    ("results/activation_comparison/activation_qpsk_rayleigh.json",
     "results/activation_comparison/bpsk_activation_rayleigh.png",
     "Activation Comparison — QPSK Rayleigh"),
    ("results/activation_comparison/activation_qam16_awgn.json",
     "results/activation_comparison/qam16_activation_awgn.png",
     "Activation Comparison — QAM16 AWGN"),
    ("results/activation_comparison/activation_qam16_rayleigh.json",
     "results/activation_comparison/qam16_activation_rayleigh.png",
     "Activation Comparison — QAM16 Rayleigh"),
    # CSI
    ("results/csi/csi_experiment_qam16_rayleigh.json",
     "results/csi/csi_experiment_qam16_rayleigh.png",
     "CSI Experiment — QAM16 Rayleigh"),
    ("results/csi/csi_experiment_psk16_rayleigh.json",
     "results/csi/csi_experiment_psk16_rayleigh.png",
     "CSI Experiment — PSK16 Rayleigh"),
    # All relays 16class
    ("results/all_relays_16class/all_relays_16class.json",
     "results/all_relays_16class/ber_all_relays_16class.png",
     "16-Class 2D vs 4-Class — QAM16"),
    # E2E
    ("results/e2e/e2e_relay_comparison.json",
     "results/e2e/e2e_relay_comparison.png",
     "E2E vs AF/DF — 16-QAM Rayleigh"),
]

if __name__ == "__main__":
    print("Regenerating CI figures with fixed zoom inset (upper right, 30%x28%)...")
    count = 0
    skipped = 0
    for json_path, save_path, title in REGEN_TASKS:
        if os.path.exists(json_path):
            regen_from_json(json_path, save_path, title)
            count += 1
        else:
            print(f"  SKIP (no JSON): {json_path}")
            skipped += 1
    print(f"\nDone: {count} figures regenerated, {skipped} skipped.")