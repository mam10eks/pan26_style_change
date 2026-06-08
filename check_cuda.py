#!/usr/bin/env python3
"""Verify that PyTorch was installed with CUDA support and can see the GPU.

Run after `pip install -r requirements.txt`:

    python check_cuda.py

If CUDA is not available even though you have a GPU, the most likely cause
is that pip installed the CPU-only torch wheel. To force a CUDA wheel:

    pip uninstall -y torch
    pip install torch --index-url https://download.pytorch.org/whl/cu126
    # ...or cu128 for Blackwell, cu118 for older drivers
"""

import sys


def main() -> int:
	try:
		import torch
	except ImportError:
		print("ERROR: PyTorch is not installed. Run: pip install -r requirements.txt")
		return 1

	print(f"PyTorch version: {torch.__version__}")
	cuda_compiled = torch.version.cuda
	if cuda_compiled is None:
		print("This PyTorch wheel was built WITHOUT CUDA support (CPU-only).")
		print("\nTo install the CUDA wheel:")
		print("  pip uninstall -y torch")
		print("  pip install torch --index-url https://download.pytorch.org/whl/cu126")
		return 1

	print(f"Compiled against CUDA: {cuda_compiled}")
	print(f"CUDA available at runtime: {torch.cuda.is_available()}")

	if not torch.cuda.is_available():
		print("\nPyTorch was compiled with CUDA support but cannot detect a GPU.")
		print("Possible causes:")
		print("  - No NVIDIA GPU in this machine.")
		print("  - NVIDIA driver missing or too old (need 525+ for CUDA 12.x).")
		print("  - Driver/runtime CUDA version mismatch.")
		print("Verify with: nvidia-smi")
		return 1

	print(f"Device count: {torch.cuda.device_count()}")
	for i in range(torch.cuda.device_count()):
		props = torch.cuda.get_device_properties(i)
		print(
			f"  Device {i}: {props.name}  "
			f"(sm_{props.major}{props.minor}, "
			f"{props.total_memory / 1024**3:.1f} GB)"
		)

	# Quick functional test: tensor on GPU + simple op.
	try:
		x = torch.randn(64, 64, device="cuda")
		y = (x @ x.T).sum().item()
		print(f"\nGPU test op succeeded (sample sum = {y:.2f})")
	except Exception as e:
		print(f"\nGPU is visible but tensor op failed: {e}")
		return 1

	print("\nCUDA is working correctly.")
	return 0


if __name__ == "__main__":
	sys.exit(main())
