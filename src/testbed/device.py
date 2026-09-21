"""Choosing where models run: the Apple GPU ("mps") or the CPU.

On an Apple silicon Mac, PyTorch can use the built-in GPU through its "mps" backend (Metal
Performance Shaders). There is no NVIDIA GPU, so "cuda" is never used.
"""

import gc

from testbed.errors import TestbedError


def pick_device(requested: str = "auto") -> str:
    """Return "mps" or "cpu".

    - "auto": use the Apple GPU if PyTorch can use it, otherwise the CPU.
    - "mps" / "cpu": use exactly that (error if "mps" is not available).
    """
    import torch  # imported here because it takes a few seconds to load

    gpu_available = torch.backends.mps.is_available()

    if requested == "auto":
        return "mps" if gpu_available else "cpu"
    if requested == "cpu":
        return "cpu"
    if requested == "mps":
        if not gpu_available:
            raise TestbedError(
                "The Apple GPU (mps) is not available on this computer.",
                hint="Use --device cpu (slower) or --device auto.",
            )
        return "mps"
    raise TestbedError(f"Unknown device '{requested}'.", hint="Use auto, mps, or cpu.")


def free_memory(device: str) -> None:
    """Give memory back after a model is no longer needed.

    Python frees objects on its own eventually; calling this between evaluators makes sure
    the next evaluator starts with as much free memory as possible.
    """
    gc.collect()
    if device == "mps":
        import torch

        torch.mps.empty_cache()
