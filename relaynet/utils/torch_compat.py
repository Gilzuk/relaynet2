"""Lightweight Torch compatibility helpers."""

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


def get_torch_module():
    """Return the imported torch module, or None if unavailable."""
    return torch


def get_preferred_device(prefer_gpu=True):
    """Return a Torch device when Torch is installed, else None."""
    if torch is None:
        return None
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def can_use_gpu(device):
    """Return True when the provided Torch device points to CUDA."""
    return torch is not None and device is not None and getattr(device, "type", None) == "cuda"


def to_numpy(value, dtype=None):
    """Convert a tensor-like value to a NumPy array without using Tensor.numpy()."""
    if torch is not None and isinstance(value, torch.Tensor):
        return np.asarray(value.detach().cpu().tolist(), dtype=dtype)
    return np.asarray(value, dtype=dtype)


def save_state(state, path):
    """Save an arbitrary state dict.  Uses ``torch.save`` when available, else numpy."""
    if torch is not None:
        torch.save(state, path)
    else:
        np.save(path, state, allow_pickle=True)


def load_state(path):
    """Load a state dict saved by :func:`save_state`."""
    if torch is not None:
        return torch.load(path, map_location="cpu", weights_only=False)
    data = np.load(path, allow_pickle=True)
    return data.item() if data.ndim == 0 else data