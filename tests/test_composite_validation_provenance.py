"""Corrected AF records must remain tied to the actual simulation sources."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'e6_unknown_channel_results/codex_composite_af_validation.json'


def test_af_validation_source_fingerprints():
    data = json.loads(DATA.read_text(encoding='utf-8'))
    assert 'relaynet/relays/af.py' in data['source_sha256']
    for filename, expected in data['source_sha256'].items():
        raw = (ROOT / filename).read_bytes()
        # Normalize line endings: Git may convert them, and historical files
        # can contain mixed CRLF/LF. Any content change beyond that is rejected.
        canonical = raw.replace(b'\r\n', b'\n')
        assert expected == hashlib.sha256(canonical).hexdigest(), filename


def test_af_validation_counts_seeds_and_intervals():
    data = json.loads(DATA.read_text(encoding='utf-8'))
    assert list(map(int, data['points'])) == list(range(0, 21, 2))
    for i, record in enumerate(data['points'].values()):
        assert record['seeds'] == list(range(7000*i, 7000*i + 10))
        assert record['bits_per_trial'] == 99999
        count_rates = np.array(record['errors_per_trial']) / record['bits_per_trial']
        rates = np.array(record['ber_per_trial'])
        np.testing.assert_allclose(count_rates, rates, rtol=0, atol=1e-15)
        assert np.isclose(rates.mean(), record['mean'], rtol=0, atol=1e-15)
        assert np.isclose(t.ppf(.975, 9)*rates.std(ddof=1)/np.sqrt(10),
                          record['ci95_halfwidth'], rtol=1e-10, atol=1e-15)
