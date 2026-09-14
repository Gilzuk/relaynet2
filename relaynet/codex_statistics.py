"""Statistical helpers for new, non-destructive validation experiments.
The functions in this module deliberately use a predeclared exposure. They
must not be used with a sample size chosen from the observed first error:
that optional-stopping rule does not produce an ordinary fixed-budget BER
estimate. New experiment artefacts that use these helpers should have a
codex_ prefix so historical result files remain untouched.
"""
from __future__ import annotations
from dataclasses import dataclass
from statistics import NormalDist
from typing import Callable
@dataclass(frozen=True)
class FixedBudgetBER:
    """A BER estimate and Wilson confidence interval from fixed exposure."""
    errors: int
    bits: int
    estimate: float
    lower_95: float
    upper_95: float
def wilson_interval(errors: int, bits: int, confidence: float = 0.95) -> tuple[float, float]:
    """Return a two-sided Wilson interval for errors / bits.
    Wilson intervals remain well behaved when no errors are observed, unlike
    a normal approximation. bits must be fixed independently of the
    observed errors for the usual frequentist coverage interpretation.
    """
    if bits <= 0:
        raise ValueError("bits must be positive")
    if not 0 <= errors <= bits:
        raise ValueError("errors must lie in [0, bits]")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    p = errors / bits
    denom = 1.0 + z * z / bits
    center = (p + z * z / (2.0 * bits)) / denom
    radius = z * ((p * (1.0 - p) / bits + z * z / (4.0 * bits * bits)) ** 0.5) / denom
    return max(0.0, center - radius), min(1.0, center + radius)
def fixed_budget_ber(total_bits: int, block_size: int,
                     run_block: Callable[[int], int]) -> FixedBudgetBER:
    """Estimate BER from a fixed number of bits without overwriting results.
    run_block(n) must simulate exactly n payload bits and return its
    error count. The final partial block is included, so the exposure is
    always exactly total_bits regardless of the observed error process.
    """
    if total_bits <= 0:
        raise ValueError("total_bits must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    errors = 0
    bits_done = 0
    while bits_done < total_bits:
        n = min(block_size, total_bits - bits_done)
        block_errors = run_block(n)
        if not 0 <= block_errors <= n:
            raise ValueError("run_block returned an invalid error count")
        errors += block_errors
        bits_done += n
    lower, upper = wilson_interval(errors, bits_done)
    return FixedBudgetBER(errors, bits_done, errors / bits_done, lower, upper)
