"""Fixed-exposure summaries with independent blocks as sampling units.

Overlapping windows and shared normalization can correlate errors inside a
block. A pooled count still estimates BER, but its binomial interval need
not cover. All blocks passed here must be independent and equally sized.
"""
import math
from numbers import Integral

import numpy as np
from scipy.stats import t


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def summarize_blocks(error_counts, bits_per_block):
    bits = positive_int(bits_per_block, "bits_per_block")
    counts = list(error_counts)
    if not counts or any(isinstance(e, bool) or not isinstance(e, Integral)
                         or not 0 <= e <= bits for e in counts):
        raise ValueError("expected nonempty integer error counts within each block")
    rates = np.asarray(counts, dtype=float) / bits
    n, errors = len(counts), int(sum(counts))
    ci = None
    if n > 1 and rates.std(ddof=1) > 0:
        half = float(t.ppf(0.975, n - 1) * rates.std(ddof=1) / np.sqrt(n))
        ci = [max(0.0, float(rates.mean()) - half), min(1.0, float(rates.mean()) + half)]
    # BER <= P(any error in a block). At zero block errors this exact upper
    # bound remains conservative for BER even with arbitrary within-block
    # dependence. It is intentionally NOT the bit-wise rule of three.
    zero_upper = -math.expm1(math.log(0.05) / n) if errors == 0 else None
    return {"errors": errors, "bits": n * bits, "blocks": n,
            "ber": errors / (n * bits), "block_t_ci95_approx": ci,
            "zero_error_block_upper95": zero_upper,
            "uncertainty_unit": "independent equal-size block",
            "scope": "conditional on this trained checkpoint; t interval is approximate"}


def summarize_seed_means(values):
    """Training seeds, not their pooled blocks, are the replication unit."""
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or not len(a) or not np.all(np.isfinite(a)):
        raise ValueError("expected finite seed means")
    half = None
    if len(a) > 1 and a.std(ddof=1) > 0:
        half = float(t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a)))
    return {"seed_means": a.tolist(), "mean": float(a.mean()),
            "seed_t_halfwidth95_approx": half, "training_seeds": len(a)}
