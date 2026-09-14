import pytest
from relaynet.codex_statistics import fixed_budget_ber, wilson_interval
from relaynet.relays.codex_hybrid import CodexNominalSNRHybridRelay
def test_fixed_budget_uses_the_requested_exposure():
    calls = []
    def run_block(n):
        calls.append(n)
        return 0
    result = fixed_budget_ber(23, 10, run_block)
    assert calls == [10, 10, 3]
    assert result.bits == 23
    assert result.errors == 0
    assert result.estimate == 0.0
    assert result.upper_95 > 0.0
def test_wilson_validates_counts():
    with pytest.raises(ValueError):
        wilson_interval(2, 1)
def test_codex_hybrid_requires_an_explicit_operating_snr():
    relay = CodexNominalSNRHybridRelay(prefer_gpu=False)
    with pytest.raises(RuntimeError):
        relay.process([1.0, -1.0])
