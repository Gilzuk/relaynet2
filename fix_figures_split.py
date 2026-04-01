"""
fix_figures_split.py
--------------------
Splits dense figures 4.27-4.38 into 3 separate figures each,
with zoom insets on the low-SNR region.

Figures handled:
  4.27  combined_modulation_awgn.png  → split by modulation (BPSK/QPSK/16-QAM)
  4.28  qam16_activation_awgn.png     → split by activation (tanh/linear/hardtanh)
  4.29  qam16_activation_rayleigh.png → split by activation
  4.30  bpsk_activation_awgn.png      → split by activation (sigmoid/hardtanh/scaled_tanh)
  4.31  bpsk_activation_rayleigh.png  → split by activation
  4.32  qpsk_activation_awgn.png      → split by activation
  4.33  qpsk_activation_rayleigh.png  → split by activation
  4.34  qam16_activation_awgn.png     → split by activation (activation_comparison)
  4.36  various_activation_functions  → split by function group
  4.38  top3_qam16_rayleigh.png       → enlarge + zoom inset
"""

import json, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'lines.linewidth': 1.8,
    'lines.markersize': 5,
})

RELAY_COLORS = {
    'AF':              '#1f77b4',
    'DF':              '#d62728',
    'GenAI (169p)':    '#2ca02c',
    'Hybrid':          '#ff7f0e',
    'VAE':             '#9467bd',
    'CGAN (WGAN-GP)':  '#8c564b',
    'Transformer':     '#e377c2',
    'Mamba S6':        '#7f7f7f',
    'Mamba2 (SSD)':    '#17becf',
}
RELAY_MARKERS = {
    'AF': 'o', 'DF': 's', 'GenAI (169p)': '^', 'Hybrid': 'D',
    'VAE': 'v', 'CGAN (WGAN-GP)': 'P', 'Transformer': '*',
    'Mamba S6': 'X', 'Mamba2 (SSD)': 'h',
}
ACT_COLORS = {
    'tanh': '#1f77b4', 'linear': '#d62728', 'hardtanh': '#2ca02c',
    'sigmoid': '#1f77b4', 'scaled_tanh': '#ff7f0e',
}
ACT_LABELS = {
    'tanh': 'tanh', 'linear': 'linear', 'hardtanh': 'clipped tanh',
    'sigmoid': 'sigmoid', 'scaled_tanh': 'scaled tanh',
}

RELAY_GROUPS = [
    ('Classical',   ['AF', 'DF']),
    ('Supervised',  ['GenAI (169p)', 'Hybrid', 'VAE', 'CGAN (WGAN-GP)']),
    ('Sequence',    ['Transformer', 'Mamba S6', 'Mamba2 (SSD)']),
]


def add_zoom_inset(ax, snr, data_dict, x1=0, x2=8, loc=1):
    """Add a zoom inset for SNR range [x1, x2]."""
    try:
        axins = zoomed_inset_axes(ax, zoom=2.2, loc=loc,
                                  bbox_to_anchor=(0.98, 0.98),
                                  bbox_transform=ax.transAxes)
        mask = [i for i, s in enumerate(snr) if x1 <= s <= x2]
        snr_zoom = [snr[i] for i in mask]
        for label, ber in data_dict.items():
            ber_zoom = [ber[i] for i in mask]
            color = RELAY_COLORS.get(label, '#333333')
            marker = RELAY_MARKERS.get(label, 'o')
            axins.semilogy(snr_zoom, ber_zoom, color=color, marker=marker,
                           markersize=3, linewidth=1.2)
        axins.set_xlim(x1, x2)
        axins.set_ylim(1e-3, 0.5)
        axins.set_xticks([0, 4, 8])
        axins.tick_params(labelsize=7)
        axins.grid(True, alpha=0.3)
        mark_inset(ax, axins, loc1=2, loc2=4, fc='none', ec='0.5', lw=0.8)
    except Exception as e:
        print(f"  Zoom inset failed: {e}")


def plot_ber_group(snr, relay_data, title, ylabel='BER', add_zoom=True,
                   zoom_x1=0, zoom_x2=8):
    """Plot BER curves for a group of relays."""
    fig, ax = plt.subplots(figsize=(7, 5))
    zoom_data = {}
    for relay, ber in relay_data.items():
        color = RELAY_COLORS.get(relay, '#333333')
        marker = RELAY_MARKERS.get(relay, 'o')
        ax.semilogy(snr, ber, color=color, marker=marker,
                    label=relay, markevery=2)
        zoom_data[relay] = ber
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='upper right', ncol=1, fontsize=8)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(min(snr), max(snr))
    ax.set_ylim(1e-5, 1.0)
    if add_zoom and len(zoom_data) > 0:
        add_zoom_inset(ax, snr, zoom_data, x1=zoom_x1, x2=zoom_x2)
    plt.tight_layout()
    return fig


def plot_activation_group(snr, act_relay_data, title, activations, add_zoom=True):
    """Plot BER curves for relays with specific activation functions."""
    fig, ax = plt.subplots(figsize=(7, 5))
    zoom_data = {}
    for act in activations:
        for relay, ber in act_relay_data.get(act, {}).items():
            label = f"{relay} ({ACT_LABELS.get(act, act)})"
            color = RELAY_COLORS.get(relay, '#333333')
            marker = RELAY_MARKERS.get(relay, 'o')
            ls = '-' if act == activations[0] else ('--' if act == activations[1] else ':')
            ax.semilogy(snr, ber, color=color, marker=marker, linestyle=ls,
                        label=label, markevery=2)
            zoom_data[label] = ber
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('BER')
    ax.set_title(title)
    ax.legend(loc='upper right', ncol=1, fontsize=7)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(min(snr), max(snr))
    ax.set_ylim(1e-5, 1.0)
    if add_zoom and len(zoom_data) > 0:
        add_zoom_inset(ax, snr, zoom_data, x1=0, x2=8)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4.27: combined_modulation_awgn → split by modulation
# ═══════════════════════════════════════════════════════════════════════════════
print("[4.27] Splitting combined_modulation_awgn by modulation ...")
mod_files = {
    'BPSK':   'results/modulation/bpsk_awgn.json',
    'QPSK':   'results/modulation/qpsk_awgn.json',
    '16-QAM': 'results/modulation/qam16_awgn.json',
}
for suffix, (mod_name, jpath) in enumerate(mod_files.items(), start=1):
    with open(jpath) as f:
        d = json.load(f)
    snr = d['snr_range']
    relay_data = {k: v['ber_mean'] for k, v in d['results'].items()}
    fig = plot_ber_group(snr, relay_data,
                         f'Combined Modulation (AWGN) — {mod_name}',
                         zoom_x1=0, zoom_x2=8)
    out = f'results/modulation/combined_modulation_awgn_split_{suffix}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figures 4.28-4.29: qam16_activation (tanh/linear/hardtanh)
# ═══════════════════════════════════════════════════════════════════════════════
for channel, jpath in [
    ('AWGN',    'results/qam16_activation/qam16_activation_awgn.json'),
    ('Rayleigh','results/qam16_activation/qam16_activation_rayleigh.json'),
]:
    fig_num = '4.28' if channel == 'AWGN' else '4.29'
    print(f"[{fig_num}] Splitting qam16_activation_{channel.lower()} by activation ...")
    with open(jpath) as f:
        d = json.load(f)
    snr = d['snr_range']
    activations = ['tanh', 'linear', 'hardtanh']
    # Build per-activation dict: {act: {relay: ber}}
    act_data = {a: {} for a in activations}
    for key, val in d['results'].items():
        for act in activations:
            if key.endswith(f'({act})'):
                relay = key[:-(len(act)+3)].strip()
                act_data[act][relay] = val['ber_mean']
    for i, act in enumerate(activations, start=1):
        relay_data = act_data[act]
        fig = plot_ber_group(snr, relay_data,
                             f'16-QAM Activation ({channel}) — {ACT_LABELS[act]}',
                             zoom_x1=0, zoom_x2=8)
        tag = 'awgn' if channel == 'AWGN' else 'rayleigh'
        out = f'results/qam16_activation/qam16_activation_{tag}_split_{i}.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figures 4.30-4.34: activation_comparison (sigmoid/hardtanh/scaled_tanh)
# ═══════════════════════════════════════════════════════════════════════════════
act_comp_files = [
    ('4.30', 'bpsk',  'AWGN',     None),   # no JSON for BPSK
    ('4.31', 'bpsk',  'Rayleigh', None),
    ('4.32', 'qpsk',  'AWGN',     'results/activation_comparison/activation_qpsk_awgn.json'),
    ('4.33', 'qpsk',  'Rayleigh', 'results/activation_comparison/activation_qpsk_rayleigh.json'),
    ('4.34', 'qam16', 'AWGN',     'results/activation_comparison/activation_qam16_awgn.json'),
]
activations_comp = ['sigmoid', 'hardtanh', 'scaled_tanh']

for fig_num, mod, channel, jpath in act_comp_files:
    if jpath is None:
        print(f"[{fig_num}] No JSON for {mod} {channel} — skipping regeneration")
        continue
    print(f"[{fig_num}] Splitting {mod}_activation_{channel.lower()} by activation ...")
    with open(jpath) as f:
        d = json.load(f)
    snr = d['snr_range']
    act_data = {a: {} for a in activations_comp}
    for key, val in d['results'].items():
        for act in activations_comp:
            if key.endswith(f'({act})'):
                relay = key[:-(len(act)+3)].strip()
                act_data[act][relay] = val['ber_mean']
    for i, act in enumerate(activations_comp, start=1):
        relay_data = act_data[act]
        if not relay_data:
            continue
        fig = plot_ber_group(snr, relay_data,
                             f'{mod.upper()} Activation ({channel}) — {ACT_LABELS[act]}',
                             zoom_x1=0, zoom_x2=8)
        tag = channel.lower()
        out = f'results/activation_comparison/{mod}_activation_{tag}_split_{i}.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4.36: various_activation_functions → split by function group
# ═══════════════════════════════════════════════════════════════════════════════
print("[4.36] Generating split activation function shape plots ...")

x = np.linspace(-3, 3, 400)
A_max = 0.9487  # for 16-QAM

def tanh_fn(x):      return np.tanh(x)
def hardtanh_fn(x):  return np.clip(x, -1, 1)
def scaled_tanh_fn(x): return A_max * np.tanh(x / A_max)
def sigmoid_fn(x):   return 1 / (1 + np.exp(-x))
def scaled_sigmoid_fn(x): return A_max * (2 / (1 + np.exp(-x)) - 1)

def dtanh(x):        return 1 - np.tanh(x)**2
def dhardtanh(x):    return np.where(np.abs(x) <= 1, 1.0, 0.0)
def dscaled_tanh(x): return (1 - np.tanh(x/A_max)**2)
def dsigmoid(x):     s = sigmoid_fn(x); return s * (1 - s)
def dscaled_sigmoid(x): s = sigmoid_fn(x); return 2 * A_max * s * (1 - s)

groups = [
    ('tanh & Clipped tanh',
     [('tanh', tanh_fn, dtanh, '#1f77b4'),
      ('clipped tanh', hardtanh_fn, dhardtanh, '#d62728')]),
    ('Scaled tanh',
     [('scaled tanh', scaled_tanh_fn, dscaled_tanh, '#2ca02c')]),
    ('Sigmoid variants',
     [('sigmoid', sigmoid_fn, dsigmoid, '#9467bd'),
      ('scaled sigmoid', scaled_sigmoid_fn, dscaled_sigmoid, '#ff7f0e')]),
]

for i, (group_name, fns) in enumerate(groups, start=1):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for name, fn, dfn, color in fns:
        ax1.plot(x, fn(x), color=color, label=name, linewidth=2)
        ax2.plot(x, dfn(x), color=color, label=name, linewidth=2)
    ax1.set_title(f'Activation: {group_name}')
    ax1.set_xlabel('x'); ax1.set_ylabel('f(x)')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='k', lw=0.5); ax1.axvline(0, color='k', lw=0.5)
    ax2.set_title(f"Derivative: {group_name}")
    ax2.set_xlabel('x'); ax2.set_ylabel("f'(x)")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='k', lw=0.5); ax2.axvline(0, color='k', lw=0.5)
    plt.tight_layout()
    out = f'results/activation_comparison/various_activation_functions_split_{i}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4.38: top3_qam16_rayleigh → 3 panels (all relays / top3 only / zoom)
# ═══════════════════════════════════════════════════════════════════════════════
print("[4.38] Splitting top3_qam16_rayleigh ...")
with open('results/csi/csi_experiment_qam16_rayleigh.json') as f:
    d = json.load(f)
snr = d['snr_range']

# Identify top-3 neural relays (lowest BER at SNR=20)
neural_keys = [k for k in d['results'] if k not in ('AF', 'DF')]
snr20_idx = snr.index(20) if 20 in snr else -1
neural_sorted = sorted(neural_keys,
                       key=lambda k: d['results'][k]['ber_mean'][snr20_idx])
top3 = neural_sorted[:3]
print(f"  Top-3: {top3}")

# Panel 1: AF + DF + top3
panel1 = {k: d['results'][k]['ber_mean'] for k in ['AF', 'DF'] + top3}
fig = plot_ber_group(snr, panel1,
                     '16-QAM Rayleigh — Classical vs Top-3 Neural',
                     zoom_x1=0, zoom_x2=8)
fig.savefig('results/csi/top3_qam16_rayleigh_split_1.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved top3_qam16_rayleigh_split_1.png")

# Panel 2: Top-3 only with CI
fig, ax = plt.subplots(figsize=(7, 5))
zoom_data = {}
for relay in top3:
    r = d['results'][relay]
    color = RELAY_COLORS.get(relay, '#333333')
    marker = RELAY_MARKERS.get(relay, 'o')
    ax.semilogy(snr, r['ber_mean'], color=color, marker=marker,
                label=relay, markevery=2)
    if 'ci_lower' in r and 'ci_upper' in r:
        ax.fill_between(snr, r['ci_lower'], r['ci_upper'],
                        alpha=0.15, color=color)
    zoom_data[relay] = r['ber_mean']
ax.set_xlabel('SNR (dB)'); ax.set_ylabel('BER')
ax.set_title('16-QAM Rayleigh — Top-3 Neural Relays with 95% CI')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, which='both', alpha=0.3)
ax.set_xlim(min(snr), max(snr)); ax.set_ylim(1e-5, 1.0)
add_zoom_inset(ax, snr, zoom_data, x1=0, x2=8)
plt.tight_layout()
fig.savefig('results/csi/top3_qam16_rayleigh_split_2.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved top3_qam16_rayleigh_split_2.png")

# Panel 3: All neural relays (not just top3)
panel3 = {k: d['results'][k]['ber_mean'] for k in neural_keys[:6]}
fig = plot_ber_group(snr, panel3,
                     '16-QAM Rayleigh — Neural Relay Variants',
                     zoom_x1=0, zoom_x2=8)
fig.savefig('results/csi/top3_qam16_rayleigh_split_3.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved top3_qam16_rayleigh_split_3.png")

print("\nDone. All split figures generated.")