"""Choosing where models run: an NVIDIA GPU ("cuda"), the Apple GPU ("mps"), or the CPU.

PyTorch can run models on different kinds of hardware:

- "cuda": NVIDIA graphics cards (typically a Linux or Windows PC)
- "mps":  the GPU built into Apple silicon Macs (Metal Performance Shaders)
- "cpu":  the processor; works everywhere, but is much slower

By default ("auto") the testbed asks PyTorch what this computer has and picks the fastest.
"""

import gc

from testbed.errors import TestbedError

# Human-friendly names for error messages.
GPU_NAMES = {"cuda": "An NVIDIA GPU (cuda)", "mps": "The Apple GPU (mps)"}


def available_gpus() -> dict[str, bool]:
    """Ask PyTorch which kinds of GPU this computer can use, e.g. {"cuda": False, "mps": True}."""
    import torch  # imported here because it takes a few seconds to load

    return {
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
    }


def pick_device(requested: str = "auto") -> str:
    """Return "cuda", "mps", or "cpu".

    - "auto": the best device this computer has: an NVIDIA GPU, else the Apple GPU, else CPU.
      (A computer never has both kinds of GPU, so the order only matters in theory.)
    - "cuda" / "mps" / "cpu": use exactly that (error if that GPU is not available).
    """
    gpus = available_gpus()

    if requested == "auto":
        if gpus["cuda"]:
            return "cuda"
        if gpus["mps"]:
            return "mps"
        return "cpu"
    if requested == "cpu":
        return "cpu"
    if requested in gpus:
        if not gpus[requested]:
            raise TestbedError(
                f"{GPU_NAMES[requested]} is not available on this computer.",
                hint="Use --device auto to pick what is available, or --device cpu (slower).",
            )
        return requested
    raise TestbedError(f"Unknown device '{requested}'.", hint="Use auto, cuda, mps, or cpu.")


def free_memory(device: str) -> None:
    """Give memory back after a model is no longer needed.

    Python frees objects on its own eventually; calling this between evaluators makes sure
    the next evaluator starts with as much free (GPU) memory as possible.
    """
    gc.collect()
    if device == "cuda":
        import torch

        torch.cuda.empty_cache()
    elif device == "mps":
        import torch

        torch.mps.empty_cache()
