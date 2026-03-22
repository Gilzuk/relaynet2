import torch
print("PyTorch version:", torch.__version__)
print("CUDA compiled:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
# Check for DirectML
try:
    import torch_directml
    print("DirectML available: True")
    print("DirectML device:", torch_directml.device())
except ImportError:
    print("DirectML available: False")
# Check xpu (Intel)
if hasattr(torch, 'xpu') and torch.xpu.is_available():
    print("XPU available: True")
else:
    print("XPU available: False")
