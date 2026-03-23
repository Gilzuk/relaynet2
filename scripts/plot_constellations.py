"""Generate constellation diagrams for all four modulation schemes."""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ── BPSK ──
bpsk_symbols = np.array([-1.0, 1.0])
bpsk_labels = ['1', '0']

# ── QPSK ──
qpsk_bits = [(0,0), (0,1), (1,0), (1,1)]
qpsk_symbols = np.array([(1-2*b0 + 1j*(1-2*b1))/np.sqrt(2) for b0,b1 in qpsk_bits])
qpsk_labels = ['00', '01', '10', '11']

# ── 16-QAM ──
def gray_pam4(b0, b1):
    table = {(0,0): 3, (0,1): 1, (1,1): -1, (1,0): -3}
    return table[(b0,b1)]

qam16_symbols = []
qam16_labels = []
for b0 in range(2):
    for b1 in range(2):
        for b2 in range(2):
            for b3 in range(2):
                I = gray_pam4(b0, b1) / np.sqrt(10)
                Q = gray_pam4(b2, b3) / np.sqrt(10)
                qam16_symbols.append(I + 1j*Q)
                qam16_labels.append(f'{b0}{b1}{b2}{b3}')
qam16_symbols = np.array(qam16_symbols)

# ── 16-PSK ──
_GRAY_CODE = np.array([0, 1, 3, 2, 6, 7, 5, 4, 12, 13, 15, 14, 10, 11, 9, 8])
_GRAY_INV = np.empty(16, dtype=int)
_GRAY_INV[_GRAY_CODE] = np.arange(16)
psk16_angles = 2.0 * np.pi * np.arange(16) / 16.0
psk16_symbols = np.exp(1j * psk16_angles)
psk16_labels = [f'{(_GRAY_INV[k]>>3)&1}{(_GRAY_INV[k]>>2)&1}{(_GRAY_INV[k]>>1)&1}{_GRAY_INV[k]&1}' for k in range(16)]


def plot_constellation(ax, symbols, labels, title, color='#1f77b4', show_unit_circle=False):
    """Plot a single constellation diagram."""
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(0, color='grey', linewidth=0.5)
    ax.axvline(0, color='grey', linewidth=0.5)

    if show_unit_circle:
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.2, linewidth=0.8)

    ax.scatter(symbols.real, symbols.imag, s=80, c=color, zorder=5, edgecolors='black', linewidths=0.5)

    for sym, label in zip(symbols, labels):
        offset_x = 0.06 if sym.real >= 0 else -0.06
        offset_y = 0.06
        ha = 'left' if sym.real >= 0 else 'right'
        ax.annotate(label, (sym.real, sym.imag),
                   textcoords='offset points',
                   xytext=(8 if sym.real >= 0 else -8, 8),
                   fontsize=7, ha=ha, fontfamily='monospace',
                   color='#333333')

    ax.set_xlabel('In-Phase (I)', fontsize=9)
    ax.set_ylabel('Quadrature (Q)', fontsize=9)
    ax.set_title(title, fontsize=11, fontweight='bold')


# ── Combined 2×2 figure ──
fig, axes = plt.subplots(2, 2, figsize=(12, 11))

# BPSK (top-left) — real-valued, plot on I axis
ax = axes[0, 0]
ax.set_aspect('equal')
ax.grid(True, alpha=0.3, linestyle='--')
ax.axhline(0, color='grey', linewidth=0.5)
ax.axvline(0, color='grey', linewidth=0.5)
ax.scatter(bpsk_symbols, [0, 0], s=120, c='#d62728', zorder=5, edgecolors='black', linewidths=0.5, marker='D')
for sym, label in zip(bpsk_symbols, bpsk_labels):
    ax.annotate(label, (sym, 0), textcoords='offset points', xytext=(0, 14),
               fontsize=10, ha='center', fontfamily='monospace', fontweight='bold', color='#333')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel('In-Phase (I)', fontsize=9)
ax.set_ylabel('Quadrature (Q)', fontsize=9)
ax.set_title('BPSK (1 bit/symbol)', fontsize=11, fontweight='bold')
ax.text(0, -1.2, '$x = 1 - 2b,\\; b \\in \\{0, 1\\}$', ha='center', fontsize=9, style='italic', color='#555')

# QPSK (top-right)
plot_constellation(axes[0, 1], qpsk_symbols, qpsk_labels,
                  'QPSK (2 bits/symbol)', color='#2ca02c', show_unit_circle=True)
axes[0, 1].set_xlim(-1.3, 1.3)
axes[0, 1].set_ylim(-1.3, 1.3)

# 16-QAM (bottom-left)
plot_constellation(axes[1, 0], qam16_symbols, qam16_labels,
                  '16-QAM (4 bits/symbol)', color='#1f77b4')
lim = 1.3
axes[1, 0].set_xlim(-lim, lim)
axes[1, 0].set_ylim(-lim, lim)
# Draw decision boundaries
for b in [-2/np.sqrt(10), 0, 2/np.sqrt(10)]:
    axes[1, 0].axhline(b, color='#aaa', linewidth=0.4, linestyle=':')
    axes[1, 0].axvline(b, color='#aaa', linewidth=0.4, linestyle=':')

# 16-PSK (bottom-right)
plot_constellation(axes[1, 1], psk16_symbols, psk16_labels,
                  '16-PSK (4 bits/symbol)', color='#ff7f0e', show_unit_circle=True)
axes[1, 1].set_xlim(-1.4, 1.4)
axes[1, 1].set_ylim(-1.4, 1.4)

plt.suptitle('Constellation Diagrams — All Modulation Schemes', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])

out_dir = os.path.join('results', 'modulation')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'constellation_diagrams.png')
plt.savefig(out_path, dpi=200, bbox_inches='tight')
plt.close()
print(f'Saved constellation diagrams to {out_path}')
