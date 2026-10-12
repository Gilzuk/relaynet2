"""Decoder-metric switch shared by the coded experiments.

The coded scripts decode the zero-forced observation y/h = x + n/h. By default
they use the published unweighted Euclidean metric. With

    CODED_METRIC=matched

they weight every squared residual by the symbol's |h|^2 at the relay decoder
and at the destination decoder (the matched metric; see Sec. 5.5.2 and
coded_matched_metrics.py), and write their results to ``*_matched.json`` so the
published files are left untouched. ``CODED_TRIALS=N`` overrides a script's
trial count. With the variable unset every script behaves exactly as before:
``hop`` draws the same random numbers as ``rayleigh_fading_channel``.
"""

import inspect
import os

import numpy as np

from relaynet.channels.fading import rayleigh_fading_channel

MATCHED = os.environ.get("CODED_METRIC", "unweighted") == "matched"


def trials(default):
    """Trial count: CODED_TRIALS if set, else the script's own default."""
    return int(os.environ.get("CODED_TRIALS", default))


def path(p):
    """Output (or input) path for the active metric."""
    return p.replace(".json", "_matched.json") if MATCHED else p


def hop(signal, snr_db):
    """One zero-forced Rayleigh hop, plus |h|^2 weights when the metric is matched.

    Uses rayleigh_fading_channel(..., return_channel=True), which draws exactly
    what the plain call draws, so seeded trials are unchanged.
    """
    y, h = rayleigh_fading_channel(signal, snr_db, return_channel=True)
    return y, (np.abs(h) ** 2 if MATCHED else None)


def relay_process(relay, y, w):
    """Hand the weights to relays whose decoders take them (block DF)."""
    if w is not None and "weights" in inspect.signature(relay.process).parameters:
        return relay.process(y, weights=w)
    return relay.process(y)


def bit_weights(w, bits_per_symbol, n_bits):
    """Per-symbol weights repeated onto that symbol's coded bits (BICM)."""
    return None if w is None else np.repeat(w, bits_per_symbol)[:n_bits]
