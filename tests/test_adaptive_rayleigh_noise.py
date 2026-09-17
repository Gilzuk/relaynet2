"""A relay's real/complex dtype must not change its per-axis noise budget."""
import numpy as np
import pytest

from relaynet.channels import AdaptiveRayleighChannel


@pytest.mark.parametrize("snr_db", [0, 10, 20])
def test_noise_power_per_dimension(snr_db):
    n = 200_000
    variance = 1 / (2 * 10 ** (snr_db / 10))
    real = AdaptiveRayleighChannel(seed=17)(np.zeros(n), snr_db)
    complex_ = AdaptiveRayleighChannel(seed=17)(np.zeros(n, complex), snr_db)
    assert np.var(real) == pytest.approx(variance, rel=0.015)
    assert np.var(complex_.real) == pytest.approx(variance, rel=0.015)
    assert np.var(complex_.imag) == pytest.approx(variance, rel=0.015)
    assert np.mean(np.abs(complex_) ** 2) == pytest.approx(2 * variance, rel=0.015)
    # Same RNG sequence: the real part must exactly match the real-only path.
    np.testing.assert_array_equal(real, complex_.real)
