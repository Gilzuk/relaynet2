"""
Statistical significance testing utilities for BER comparisons.

Provides:
  - 95% confidence intervals on BER curves
  - Wilcoxon signed-rank test / paired t-test for BER differences
  - Summary table of statistically significant wins
"""

import math
import numpy as np


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------

def compute_confidence_interval(ber_trials, confidence=0.95):
    """Compute a confidence interval for BER estimates.

    Uses the normal approximation to the binomial proportion CI when
    ``scipy`` is not available, or the Student-t CI otherwise.

    Parameters
    ----------
    ber_trials : array-like, shape (n_snr, n_trials)
        Per-trial BER values.
    confidence : float
        Confidence level (default 0.95).

    Returns
    -------
    lower : numpy.ndarray, shape (n_snr,)
    upper : numpy.ndarray, shape (n_snr,)
    """
    ber_trials = np.asarray(ber_trials)
    n_trials = ber_trials.shape[1]
    mean = ber_trials.mean(axis=1)
    sem = ber_trials.std(axis=1, ddof=1) / math.sqrt(n_trials)

    # Use t-distribution critical value
    z = _t_ppf((1 + confidence) / 2, df=n_trials - 1)

    lower = np.maximum(mean - z * sem, 0)
    upper = mean + z * sem
    return lower, upper


def _t_ppf(p, df):
    """Approximate percent-point function of Student's t distribution.

    Falls back to scipy if available, otherwise uses a numeric approximation.
    """
    try:
        from scipy.stats import t as t_dist
        return t_dist.ppf(p, df)
    except ImportError:
        pass

    # Cornish-Fisher approximation (good for df >= 5)
    z = _norm_ppf(p)
    correction = (
        (z ** 3 + z) / (4 * df)
        + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)
    )
    return z + correction


def _norm_ppf(p):
    """Rational approximation to the standard normal PPF (Abramowitz & Stegun)."""
    if p < 0.5:
        return -_norm_ppf(1 - p)
    t = math.sqrt(-2 * math.log(1 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    num = c0 + c1 * t + c2 * t ** 2
    den = 1 + d1 * t + d2 * t ** 2 + d3 * t ** 3
    return t - num / den


# ---------------------------------------------------------------------------
# Hypothesis tests
# ---------------------------------------------------------------------------

def wilcoxon_test(ber_trials_a, ber_trials_b, alpha=0.05):
    """Perform a Wilcoxon signed-rank test comparing two BER curves.

    For each SNR point, tests whether method A has a *lower* BER than
    method B.

    Parameters
    ----------
    ber_trials_a : array-like, shape (n_snr, n_trials)
        BER trials for method A.
    ber_trials_b : array-like, shape (n_snr, n_trials)
        BER trials for method B.
    alpha : float
        Significance level (default 0.05).

    Returns
    -------
    p_values : numpy.ndarray, shape (n_snr,)
        p-value at each SNR point.
    significant : numpy.ndarray of bool, shape (n_snr,)
        True where the difference is statistically significant (p < alpha).
    a_wins : numpy.ndarray of bool, shape (n_snr,)
        True where method A has a lower mean BER than method B.
    """
    a = np.asarray(ber_trials_a)
    b = np.asarray(ber_trials_b)
    n_snr = a.shape[0]
    p_values = np.ones(n_snr)
    significant = np.zeros(n_snr, dtype=bool)
    a_wins = np.zeros(n_snr, dtype=bool)

    for i in range(n_snr):
        diff = a[i] - b[i]
        a_wins[i] = np.mean(a[i]) < np.mean(b[i])
        try:
            from scipy.stats import wilcoxon
            # alternative='less': test whether A < B
            stat, p = wilcoxon(diff, alternative="less" if a_wins[i] else "greater")
            p_values[i] = p
        except ImportError:
            # Fallback: paired t-test approximation
            p_values[i] = _paired_t_pvalue(diff)
        significant[i] = p_values[i] < alpha

    return p_values, significant, a_wins


def _paired_t_pvalue(diff):
    """One-sided paired t-test p-value (H1: mean(diff) < 0)."""
    n = len(diff)
    if n < 2:
        return 1.0
    mean_d = np.mean(diff)
    std_d = np.std(diff, ddof=1)
    if std_d == 0:
        return 0.0 if mean_d < 0 else 1.0
    t_stat = mean_d / (std_d / math.sqrt(n))
    # Approximate CDF using error function
    x = t_stat / math.sqrt(2)
    p_one_sided = 0.5 * (1 + math.erf(x))
    return p_one_sided


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def significance_table(snr_values, methods, ber_trials_dict, baseline="df",
                        alpha=0.05):
    """Print a summary table of statistically significant BER wins.

    Parameters
    ----------
    snr_values : array-like
        SNR values in dB.
    methods : list of str
        Names of the methods to compare against the baseline.
    ber_trials_dict : dict
        Mapping ``method_name → ber_trials array (n_snr, n_trials)``.
    baseline : str
        Key in ``ber_trials_dict`` used as the reference method.
    alpha : float
        Significance level.

    Returns
    -------
    table : dict
        Nested dict ``{method: {snr: {'p': float, 'sig': bool, 'wins': bool}}}``.
    """
    snr_values = list(snr_values)
    table = {}
    baseline_trials = np.asarray(ber_trials_dict[baseline])

    header = f"{'SNR':>5} | " + " | ".join(f"{m:>12}" for m in methods)
    print(header)
    print("-" * len(header))

    for i, snr in enumerate(snr_values):
        row = f"{snr:>5.0f} | "
        for method in methods:
            if method == baseline:
                row += f"{'(baseline)':>12} | "
                continue
            method_trials = np.asarray(ber_trials_dict[method])
            p_vals, sig, a_wins = wilcoxon_test(
                method_trials, baseline_trials, alpha=alpha
            )
            p = p_vals[i]
            s = sig[i]
            w = a_wins[i]
            symbol = "Y*" if (s and w) else ("Y" if w else ("N*" if s else "N"))
            row += f"{symbol:>10}({p:.2f}) | "
            table.setdefault(method, {})[snr] = {"p": p, "sig": s, "wins": w}
        print(row)

    print()
    print("Legend: ✓=wins vs baseline  * = p<{:.2f} (statistically significant)".format(alpha))
    return table
