"""An explicit-SNR Hybrid relay for validation experiments.
Unlike HybridRelay, this class never estimates SNR from a
zero-forced Rayleigh observation. That observation has heavy-tailed noise,
so a receive-power estimate is not a reliable proxy for nominal link SNR.
The caller must explicitly supply an operating SNR for each block.
"""
from __future__ import annotations
from .hybrid import HybridRelay
class CodexNominalSNRHybridRelay(HybridRelay):
    """Hybrid relay switched by caller-supplied nominal SNR.
    The explicit setter makes the CSI/control-plane assumption testable. It
    intentionally does not replace HybridRelay or any historical result.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._nominal_snr_db: float | None = None
    def set_nominal_snr_db(self, snr_db: float) -> None:
        self._nominal_snr_db = float(snr_db)
    def process(self, received_signal):
        if self._nominal_snr_db is None:
            raise RuntimeError(
                "Set nominal SNR with set_nominal_snr_db() before processing."
            )
        if self._nominal_snr_db < self.snr_threshold:
            return self.mlp_relay.process(received_signal)
        return self.df_relay.process(received_signal)
