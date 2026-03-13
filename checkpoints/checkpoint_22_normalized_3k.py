"""
Checkpoint 22 – Normalized ~3K-parameter relay models
=====================================================

Provides factory functions that instantiate **all six** AI relay
architectures with approximately **3,000 learnable parameters** each,
enabling a fair apples-to-apples BER comparison.

Target parameter budget: ~3,000 ± 40

+---------------+----------------------------------------------+--------+
| Model         | Configuration                                | Params |
+---------------+----------------------------------------------+--------+
| GenAI-3K      | window=11, hidden=231                        | 3,004  |
| Hybrid-3K     | wraps GenAI-3K (same sub-network)            | 3,004  |
| VAE-3K        | window=11, hidden=(44,20), latent=10         | 3,037  |
| CGAN-3K       | window=11, g_hidden=(30,30,16), c=(32,16)    | 3,004  |
| Transformer-3K| d_model=18, heads=2, layers=1, window=11     | 3,007  |
| Mamba S6-3K   | d_model=16, d_state=6, layers=1, window=11   | 3,027  |
+---------------+----------------------------------------------+--------+

Usage
-----
::

    from checkpoints.checkpoint_22_normalized_3k import build_all_3k

    relays = build_all_3k(prefer_gpu=False)
    for name, relay in relays.items():
        relay.train(training_snrs=[5, 10, 15], num_samples=25000, epochs=100)

Author: GitHub Copilot
"""

import os
import sys

# Allow running from the repository root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from relaynet.relays.genai import MinimalGenAIRelay
from relaynet.relays.hybrid import HybridRelay
from relaynet.relays.vae import VAERelay
from relaynet.relays.cgan import CGANRelay

try:
    from checkpoints.checkpoint_18_transformer_relay import TransformerRelayWrapper
    from checkpoints.checkpoint_20_mamba_s6_relay import MambaRelayWrapper
    _HAS_SEQ = True
except Exception:
    _HAS_SEQ = False

# ── Normalized configurations ───────────────────────────────────────

# GenAI: params = window*hidden + hidden + hidden*1 + 1
#       = 11*231 + 231 + 231 + 1 = 3004
GENAI_3K = dict(window_size=11, hidden_size=231)

# Hybrid: wraps GenAI sub-relay with the same architecture
HYBRID_3K = dict(genai_window_size=11, genai_hidden_size=231)

# VAE: enc 11→44→20→(mu,logvar)10, dec 10→20→44→1
#      = 528+900+210+210 + 220+924+45 = 3037
VAE_3K = dict(window_size=11, latent_size=10, hidden_sizes=(44, 20))

# CGAN: G(19→30→30→16→1)=2043, C(12→32→16→1)=961, total=3004
CGAN_3K = dict(
    window_size=11,
    noise_size=8,
    g_hidden_sizes=(30, 30, 16),
    c_hidden_sizes=(32, 16),
)

# Transformer: 1→18, 1 block (MHA(18,2)+FF(18,36)), out 18→9→1
#              = 36 + 2790 + 181 = 3007
TRANSFORMER_3K = dict(
    window_size=11,
    d_model=18,
    num_heads=2,
    num_layers=1,
)

# Mamba S6: 1→16, 1 block (inner=32, S6(32,6)), out LN+16→8→1
#           = 32 + 2818 + 177 = 3027
MAMBA_3K = dict(
    window_size=11,
    d_model=16,
    d_state=6,
    num_layers=1,
)


# ── Factory functions ───────────────────────────────────────────────

def make_genai_3k(prefer_gpu=False, **kw):
    """Return a ~3004-param GenAI relay."""
    return MinimalGenAIRelay(**GENAI_3K, prefer_gpu=prefer_gpu, **kw)


def make_hybrid_3k(prefer_gpu=False, **kw):
    """Return a Hybrid relay wrapping a ~3004-param GenAI sub-network."""
    return HybridRelay(snr_threshold=5.0, **HYBRID_3K, prefer_gpu=prefer_gpu, **kw)


def make_vae_3k(prefer_gpu=False, **kw):
    """Return a ~3037-param VAE relay."""
    return VAERelay(beta=0.1, **VAE_3K, prefer_gpu=prefer_gpu, **kw)


def make_cgan_3k(prefer_gpu=False, **kw):
    """Return a ~3004-param CGAN (WGAN-GP) relay."""
    return CGANRelay(
        lambda_gp=10, lambda_l1=20, n_critic=5,
        **CGAN_3K, prefer_gpu=prefer_gpu, **kw,
    )


def make_transformer_3k(prefer_gpu=False, **kw):
    """Return a ~3007-param Transformer relay."""
    if not _HAS_SEQ:
        raise ImportError("Transformer checkpoint not available")
    return TransformerRelayWrapper(target_power=1.0, **TRANSFORMER_3K, **kw)


def make_mamba_3k(prefer_gpu=False, **kw):
    """Return a ~3027-param Mamba S6 relay."""
    if not _HAS_SEQ:
        raise ImportError("Mamba S6 checkpoint not available")
    return MambaRelayWrapper(target_power=1.0, **MAMBA_3K, **kw)


def build_all_3k(prefer_gpu=False, include_sequence_models=True):
    """Build all six normalized relay models.

    Returns
    -------
    relays : dict
        Mapping ``{display_name: relay_instance}``.
    """
    relays = {
        "GenAI-3K": make_genai_3k(prefer_gpu=prefer_gpu),
        "Hybrid-3K": make_hybrid_3k(prefer_gpu=prefer_gpu),
        "VAE-3K": make_vae_3k(prefer_gpu=prefer_gpu),
        "CGAN-3K": make_cgan_3k(prefer_gpu=prefer_gpu),
    }
    if include_sequence_models and _HAS_SEQ:
        relays["Transformer-3K"] = make_transformer_3k(prefer_gpu=prefer_gpu)
        relays["Mamba-3K"] = make_mamba_3k(prefer_gpu=prefer_gpu)
    return relays


# ── Self-test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch

    print("=== Normalized 3K-param relay verification ===\n")

    relays = build_all_3k(prefer_gpu=False, include_sequence_models=_HAS_SEQ)

    for name, relay in relays.items():
        # Count parameters
        if hasattr(relay, "num_params"):
            n = relay.num_params
        elif hasattr(relay, "model"):
            n = sum(p.numel() for p in relay.model.parameters())
        elif hasattr(relay, "genai_relay"):
            n = relay.genai_relay.num_params
        elif hasattr(relay, "_torch_model"):
            # CGAN / VAE — count via torch internals
            n = "?"
        else:
            n = "?"
        print(f"  {name:<18}  params = {n}")

    print("\nDone.")
