#!/usr/bin/env python3
"""Self-test: bounded-traceback MLSE must converge to the full-block decoder.

TruncatedViterbiQPSKRelay is the only equalizer in the framework with an
explicit decision delay, so the latency claims made from it rest on this
check: as the traceback depth grows it has to reproduce
ViterbiMLSEQPSKRelay symbol for symbol, and at depth 0 it must degrade to
a symbol-by-symbol decision rather than silently misalign its output.
"""

import numpy as np
from relaynet.relays import ViterbiMLSEQPSKRelay, TruncatedViterbiQPSKRelay
from relaynet.channels.e6_channels import ComplexISIChannel


import numpy as np
import pytest

from relaynet.relays import ViterbiMLSEQPSKRelay, TruncatedViterbiQPSKRelay
from relaynet.channels.e6_channels import ComplexISIChannel

TAPS = np.array([1.0, 0.7, 0.5]) / np.linalg.norm(np.array([1.0, 0.7, 0.5]))
DEPTHS = (0, 1, 3, 5, 10, 15, 30)


def _signals():
    rng = np.random.default_rng(0)
    x = ViterbiMLSEQPSKRelay.ALPHABET[rng.integers(0, 4, 4000)]
    return x, ComplexISIChannel(TAPS, seed=1)(x, 12)


@pytest.fixture(scope="module")
def reference():
    x, y = _signals()
    out = ViterbiMLSEQPSKRelay(channel_taps=TAPS).process(y)
    assert np.mean(out != x) < 1e-9, "full-block MLSE should be error-free at 12 dB"
    return x, y, out


@pytest.mark.parametrize("depth", DEPTHS)
def test_output_length_is_preserved(reference, depth):
    _, y, _ = reference
    out = TruncatedViterbiQPSKRelay(channel_taps=TAPS, traceback=depth).process(y)
    assert len(out) == len(y)


def test_converges_to_full_block_decoder(reference):
    _, y, ref = reference
    agree = {d: float(np.mean(
        TruncatedViterbiQPSKRelay(channel_taps=TAPS, traceback=d).process(y) == ref))
        for d in DEPTHS}
    assert agree[0] < 1.0, "depth 0 should not already match the full decoder"
    for d in (5, 10, 15, 30):
        assert agree[d] == 1.0, f"depth {d} must match the full-block decoder exactly"
    assert all(agree[a] <= agree[b] for a, b in zip((0, 1, 3, 5), (1, 3, 5, 10))), \
        "agreement must be non-decreasing in traceback depth"


def test_negative_traceback_is_rejected():
    with pytest.raises(ValueError):
        TruncatedViterbiQPSKRelay(channel_taps=TAPS, traceback=-1)


def test_set_channel_rebuilds_the_predecessor_table():
    """A re-estimated relay must not decode against the old trellis inversion."""
    taps5 = np.array([0.7 ** k for k in range(5)])
    taps5 /= np.linalg.norm(taps5)
    relay = TruncatedViterbiQPSKRelay(channel_taps=TAPS, traceback=5)
    relay.set_channel(channel_taps=taps5)
    assert relay.num_states == 4 ** 4
    assert relay.pred_state.shape == (relay.num_states, relay.M)

    rng = np.random.default_rng(3)
    x = ViterbiMLSEQPSKRelay.ALPHABET[rng.integers(0, 4, 3000)]
    y = ComplexISIChannel(taps5, seed=1)(x, 12)
    assert np.mean(relay.process(y) != x) < 0.01
