"""Checkpoint manager for saving and loading trained relay weights.

Directory layout::

    {base_dir}/
      seed_{seed}/
        metadata.json
        genai_169p.pt
        hybrid.pt
        vae.pt
        cgan_wgan_gp.pt
        transformer.pt
        mamba_s6.pt

Usage
-----
::

    from relaynet.utils.checkpoint_manager import CheckpointManager

    mgr = CheckpointManager("trained_weights")

    # After training:
    mgr.save_all(relays, seed=42, training_config={...})

    # Later, for inference-only:
    mgr.load_all(relays, seed=42)
"""

import json
import os
from datetime import datetime


def _safe_filename(name):
    """Convert a relay display name to a safe filename stem."""
    return (
        name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
    )


class CheckpointManager:
    """Manages saving/loading of trained relay weights associated with a seed.

    Parameters
    ----------
    base_dir : str
        Root directory for all checkpoints (default ``trained_weights``).
    """

    def __init__(self, base_dir="trained_weights"):
        self.base_dir = base_dir

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def checkpoint_dir(self, seed):
        """Return the directory path for a given seed."""
        return os.path.join(self.base_dir, f"seed_{seed}")

    def relay_path(self, seed, name):
        """Return the file path for a relay's weights."""
        return os.path.join(self.checkpoint_dir(seed), _safe_filename(name) + ".pt")

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def has_checkpoint(self, seed):
        """Return *True* if a metadata file exists for *seed*."""
        return os.path.isfile(
            os.path.join(self.checkpoint_dir(seed), "metadata.json")
        )

    def has_relay_checkpoint(self, seed, name):
        """Return *True* if weights exist for relay *name* under *seed*."""
        return os.path.isfile(self.relay_path(seed, name))

    def list_checkpoints(self):
        """Return a sorted list of all seeds that have saved checkpoints."""
        if not os.path.isdir(self.base_dir):
            return []
        seeds = []
        for d in os.listdir(self.base_dir):
            if d.startswith("seed_"):
                try:
                    seeds.append(int(d[5:]))
                except ValueError:
                    pass
        return sorted(seeds)

    def get_metadata(self, seed):
        """Load and return the metadata dict for *seed*, or *None*."""
        meta_path = os.path.join(self.checkpoint_dir(seed), "metadata.json")
        if not os.path.isfile(meta_path):
            return None
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Single relay save / load
    # ------------------------------------------------------------------

    def save_relay(self, relay, name, seed):
        """Save a single relay's trained weights.  Returns *True* on success."""
        if not hasattr(relay, "save_weights"):
            return False
        path = self.relay_path(seed, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        relay.save_weights(path)
        return True

    def load_relay(self, relay, name, seed):
        """Load weights into a relay.  Returns *True* on success."""
        if not hasattr(relay, "load_weights"):
            return False
        path = self.relay_path(seed, name)
        if not os.path.isfile(path):
            return False
        relay.load_weights(path)
        return True

    # ------------------------------------------------------------------
    # Batch save / load
    # ------------------------------------------------------------------

    @staticmethod
    def _param_count(relay):
        if hasattr(relay, "num_params"):
            return int(relay.num_params)
        if hasattr(relay, "model"):
            try:
                return int(sum(p.numel() for p in relay.model.parameters()))
            except Exception:
                pass
        return 0

    def save_all(self, relays, seed, training_config=None):
        """Save weights for every relay that supports it + write metadata.

        Returns
        -------
        saved : dict
            ``{name: info_dict}`` for each relay that was saved.
        """
        ckpt_dir = self.checkpoint_dir(seed)
        os.makedirs(ckpt_dir, exist_ok=True)

        # Merge with any existing metadata (e.g. normalized models added later)
        existing = self.get_metadata(seed) or {
            "seed": seed,
            "date": None,
            "training_config": {},
            "relays": {},
        }

        saved = {}
        for name, relay in relays.items():
            if self.save_relay(relay, name, seed):
                info = {
                    "file": _safe_filename(name) + ".pt",
                    "type": type(relay).__name__,
                    "num_params": self._param_count(relay),
                }
                saved[name] = info

        existing["relays"].update(saved)
        existing["date"] = datetime.now().isoformat()
        if training_config:
            existing["training_config"].update(training_config)

        meta_path = os.path.join(ckpt_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)

        return saved

    def load_all(self, relays, seed):
        """Load weights for every relay that has a saved checkpoint.

        Returns
        -------
        loaded : list of str
            Names of relays whose weights were loaded.
        skipped : list of str
            Names of relays that were skipped (no file or not loadable).
        """
        loaded, skipped = [], []
        for name, relay in relays.items():
            if self.load_relay(relay, name, seed):
                loaded.append(name)
            else:
                skipped.append(name)
        return loaded, skipped
