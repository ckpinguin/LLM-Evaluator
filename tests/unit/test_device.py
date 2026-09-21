"""Tests for choosing where models run (testbed/device.py).

PyTorch is asked which GPUs exist; here we replace those answers with fakes, so the tests
behave the same on a Mac, on a PC with an NVIDIA card, and on a machine without any GPU.
"""

import pytest
import torch

from testbed.device import pick_device
from testbed.errors import TestbedError


@pytest.fixture
def gpus(monkeypatch):
    """Call ``gpus(cuda=..., mps=...)`` to pretend which GPUs this computer has."""

    def pretend(cuda: bool, mps: bool) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)

    return pretend


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [
        (True, False, "cuda"),  # PC with an NVIDIA GPU
        (False, True, "mps"),  # Apple silicon Mac
        (False, False, "cpu"),  # no usable GPU
        (True, True, "cuda"),  # not possible on real hardware, but CUDA would win
    ],
)
def test_auto_picks_the_best_available_device(gpus, cuda, mps, expected):
    gpus(cuda=cuda, mps=mps)
    assert pick_device("auto") == expected


def test_requested_device_is_used_when_available(gpus):
    gpus(cuda=True, mps=False)
    assert pick_device("cuda") == "cuda"
    assert pick_device("cpu") == "cpu"


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_requesting_a_missing_gpu_explains_the_problem(gpus, device):
    gpus(cuda=False, mps=False)
    with pytest.raises(TestbedError) as error:
        pick_device(device)
    assert "not available" in error.value.message
    assert "--device auto" in error.value.hint
