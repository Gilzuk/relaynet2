"""Plotting utilities for BER curves."""

import os

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


def plot_ber_curves(snr_values, ber_dict, title="BER Comparison",
                    save_path=None, show=False):
    """Plot BER vs SNR curves for multiple methods.

    Parameters
    ----------
    snr_values : array-like
        SNR values in dB.
    ber_dict : dict
        Mapping ``label → ber_array``.
    title : str
    save_path : str, optional
        File path to save the figure.
    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    fig : matplotlib Figure or None
    """
    if not _HAS_MATPLOTLIB:
        print("[WARNING] matplotlib not installed — skipping plot.")
        return None

    markers = ["o", "s", "^", "d", "v", "x", "P", "*"]
    colors = ["k", "m", "b", "r", "g", "c", "orange", "brown"]
    fig, ax = plt.subplots(figsize=(10, 6))
    snr = np.asarray(snr_values)

    for idx, (label, ber) in enumerate(ber_dict.items()):
        ber = np.asarray(ber)
        ax.semilogy(snr, ber,
                    marker=markers[idx % len(markers)],
                    color=colors[idx % len(colors)],
                    linewidth=2, markersize=8, label=label)

    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.set_xlabel("SNR (dB)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Bit Error Rate (BER)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=11)
    ax.set_ylim([1e-6, 1])
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    if show:
        plt.show()

    return fig


def plot_ber_with_ci(snr_values, ber_dict, ci_dict=None,
                     title="BER with Confidence Intervals",
                     save_path=None, show=False):
    """Plot BER curves with optional 95% confidence interval shading.

    Parameters
    ----------
    snr_values : array-like
    ber_dict : dict
        Mapping ``label → mean_ber_array``.
    ci_dict : dict, optional
        Mapping ``label → (lower_array, upper_array)``.
    title : str
    save_path : str, optional
    show : bool

    Returns
    -------
    fig : matplotlib Figure or None
    """
    if not _HAS_MATPLOTLIB:
        print("[WARNING] matplotlib not installed — skipping plot.")
        return None

    markers = ["o", "s", "^", "d", "v", "x", "P", "*"]
    colors = ["k", "m", "b", "r", "g", "c", "orange", "brown"]
    fig, ax = plt.subplots(figsize=(10, 6))
    snr = np.asarray(snr_values)

    for idx, (label, ber) in enumerate(ber_dict.items()):
        ber = np.asarray(ber)
        color = colors[idx % len(colors)]
        ax.semilogy(snr, ber,
                    marker=markers[idx % len(markers)],
                    color=color, linewidth=2, markersize=8, label=label)
        if ci_dict and label in ci_dict:
            lower, upper = ci_dict[label]
            lower = np.maximum(np.asarray(lower), 1e-8)
            upper = np.asarray(upper)
            ax.fill_between(snr, lower, upper, alpha=0.2, color=color)

    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.set_xlabel("SNR (dB)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Bit Error Rate (BER)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=11)
    ax.set_ylim([1e-6, 1])
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    if show:
        plt.show()

    return fig
