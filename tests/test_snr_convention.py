"""Pin every BPSK channel to its textbook closed form.

These tests exist because a 3 dB convention error went undetected for a long
time. The old check verified only that ``sigma = 10**(-snr/20)`` and
``noise_power = P/10**(snr/10)`` are the same expression, which they are; what
it did not check is where that power goes. Putting the whole of N0 into a
single real dimension, rather than N0/2, makes a channel 3 dB pessimistic
while leaving the expression algebraically unchanged.

The consequence was that Chapter 7's "control: canonical Rayleigh" row did not
reproduce Chapter 5's canonical Rayleigh numbers (DF 0.1208 against 0.068 at
8 dB) even though the thesis presents them as the same channel.

The convention fixed here: **snr_db is Eb/N0**, noise variance is N0/2 per real
dimension, and BPSK therefore obeys

    AWGN      Pb = Q(sqrt(2*Eb/N0))
    Rayleigh  Pb = 0.5*(1 - sqrt(g/(1+g))),  g = Eb/N0

which is what Proakis gives and what the literature plots against.
"""
import math

import numpy as np
import pytest

from relaynet.channels.e6_channels import ISIRayleighChannel, RayleighChannel
from relaynet.channels.fading import rayleigh_fading_channel

N_BITS = 400_000
TOL = 0.004  # Monte Carlo slack at this sample count


def qfunc(x):
    return 0.5 * math.erfc(x / math.sqrt(2))


def rayleigh_theory(snr_db):
    g = 10 ** (snr_db / 10.0)
    return 0.5 * (1.0 - math.sqrt(g / (1.0 + g)))


def _ber(y, bits):
    return float(np.mean((np.real(y) < 0).astype(int) != bits))


@pytest.mark.parametrize("snr_db", [0, 4, 8])
def test_fading_rayleigh_matches_textbook(snr_db):
    """Chapters 5-6 Rayleigh channel is on the Eb/N0 axis."""
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, N_BITS)
    np.random.seed(1)
    y = rayleigh_fading_channel(1.0 - 2.0 * bits, snr_db)
    assert _ber(y, bits) == pytest.approx(rayleigh_theory(snr_db), abs=TOL)


@pytest.mark.parametrize("snr_db", [0, 4, 8])
def test_e6_rayleigh_matches_textbook(snr_db):
    """Chapter 7 Rayleigh channel must agree with the same closed form.

    This is the assertion that fails under the old sigma^2 = 1/gamma
    convention, which yields 0.5*(1 - sqrt(g/(2+g))) instead.
    """
    rng = np.random.default_rng(2)
    bits = rng.integers(0, 2, N_BITS)
    y = RayleighChannel(seed=7)(1.0 - 2.0 * bits, snr_db)
    assert _ber(y, bits) == pytest.approx(rayleigh_theory(snr_db), abs=TOL)


@pytest.mark.parametrize("snr_db", [0, 4, 8])
def test_both_rayleigh_channels_agree(snr_db):
    """The two implementations must be interchangeable, since the thesis
    presents Chapter 7's control as the Chapter 5 canonical channel."""
    rng = np.random.default_rng(3)
    bits = rng.integers(0, 2, N_BITS)
    x = 1.0 - 2.0 * bits
    a = _ber(RayleighChannel(seed=11)(x, snr_db), bits)
    np.random.seed(11)
    b = _ber(rayleigh_fading_channel(x, snr_db), bits)
    assert a == pytest.approx(b, abs=TOL)


def test_isi_rayleigh_reduces_to_rayleigh_with_trivial_taps():
    """With a single unit tap the ISI channel is the plain Rayleigh channel,
    so it must sit on the same axis rather than 3 dB away from it."""
    rng = np.random.default_rng(4)
    bits = rng.integers(0, 2, N_BITS)
    y = ISIRayleighChannel(np.array([1.0]), seed=13)(1.0 - 2.0 * bits, 8)
    assert _ber(y, bits) == pytest.approx(rayleigh_theory(8), abs=TOL)
