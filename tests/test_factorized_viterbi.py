"""FactorizedViterbiQPSKRelay must return the joint QPSK MLSE's decisions exactly.

With real taps and circular noise the QPSK sequence metric separates into I
and Q terms, and the joint state is an (I state, Q state) pair, so the best
joint survivor is the pair of best per-axis survivors. The equivalence is
exact -- for full-sequence MLSE and for the sliding traceback -- so any
disagreement at all is a defect.
"""

import numpy as np
import pytest

from relaynet.channels.e6_channels import ComplexISIChannel
from relaynet.modulation.qpsk import qpsk_modulate
from relaynet.relays import (FactorizedViterbiQPSKRelay, TruncatedViterbiQPSKRelay,
                             ViterbiMLSEQPSKRelay)


def _block(L, snr_db, n=600, seed=0):
    rng = np.random.default_rng(seed)
    h = np.array([0.7 ** k for k in range(L)])
    h /= np.linalg.norm(h)
    x = qpsk_modulate(rng.integers(0, 2, 2 * n))
    return h, x, ComplexISIChannel(h, seed=seed + 1)(x, snr_db)


@pytest.mark.parametrize("L", [2, 3, 4])
@pytest.mark.parametrize("D", [0, 1, 3, 10])
@pytest.mark.parametrize("snr_db", [0, 8, 16])
def test_matches_truncated_joint_trellis(L, D, snr_db):
    h, _, y = _block(L, snr_db)
    joint = TruncatedViterbiQPSKRelay(channel_taps=h, traceback=D).process(y)
    fact = FactorizedViterbiQPSKRelay(h, traceback=D).process(y)
    assert np.array_equal(joint, fact)


@pytest.mark.parametrize("L", [2, 3])
def test_matches_full_sequence_joint_mlse(L):
    h, _, y = _block(L, 6, n=400, seed=7)
    joint = ViterbiMLSEQPSKRelay(channel_taps=h).process(y)
    fact = FactorizedViterbiQPSKRelay(h).process(y)
    assert np.array_equal(joint, fact)


def test_decodes_noiseless_input_exactly():
    h, x, _ = _block(3, 0)
    y = np.convolve(x, h)[:len(x)]
    assert np.array_equal(FactorizedViterbiQPSKRelay(h, traceback=15).process(y), x)


def test_rejects_complex_taps():
    with pytest.raises(ValueError):
        FactorizedViterbiQPSKRelay(np.array([1.0, 0.5j]))
