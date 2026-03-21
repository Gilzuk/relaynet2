"""Quick smoke test for InputLN on all three sequence models."""
import sys, os, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from checkpoints.checkpoint_18_transformer_relay import TransformerRelayWrapper
from checkpoints.checkpoint_20_mamba_s6_relay import MambaRelayWrapper
from checkpoints.checkpoint_23_mamba2_relay import Mamba2RelayWrapper

for name, cls, kw in [
    ("Transformer", TransformerRelayWrapper,
     dict(target_power=1.0, window_size=11, d_model=32, num_heads=4, num_layers=2)),
    ("Mamba S6", MambaRelayWrapper,
     dict(target_power=1.0, window_size=11, d_model=32, d_state=16, num_layers=2)),
    ("Mamba2", Mamba2RelayWrapper,
     dict(target_power=1.0, window_size=11, d_model=32, d_state=16, num_layers=2)),
]:
    for use_ln in [False, True]:
        tag = "+InputLN" if use_ln else "Baseline"
        print(f"\n--- {name} ({tag}) ---")
        try:
            r = cls(**kw, output_activation="tanh", use_input_norm=use_ln)
            r.train(training_snrs=[10], num_samples=2000, epochs=3, lr=0.001)
            import numpy as np
            out = r.process(np.random.randn(100))
            print(f"  OK — params={r.num_params}, output shape={out.shape}")
        except Exception:
            traceback.print_exc()

print("\nAll smoke tests done.")
