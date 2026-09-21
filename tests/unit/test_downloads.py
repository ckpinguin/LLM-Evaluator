"""Tests for disk-space checks, file selection, cleanup and the run lock (testbed/downloads.py)."""

import shutil
from collections import namedtuple

import pytest

from testbed import settings
from testbed.downloads import check_disk_space, cleanup_downloads, run_lock, select_files
from testbed.errors import TestbedError

GB = 1024**3
DiskUsage = namedtuple("DiskUsage", "total used free")


def test_check_disk_space_refuses_when_too_little_space(workspace, monkeypatch):
    # Needed: 2 GB model x 1.1 + 1 GB for datasets = 3.2 GB, but only 3 GB are free.
    monkeypatch.setattr(shutil, "disk_usage", lambda path: DiskUsage(100 * GB, 97 * GB, 3 * GB))
    with pytest.raises(TestbedError, match="3.2 GB"):
        check_disk_space(2 * GB)


def test_check_disk_space_accepts_enough_space(workspace, monkeypatch):
    monkeypatch.setattr(shutil, "disk_usage", lambda path: DiskUsage(100 * GB, 90 * GB, 10 * GB))
    check_disk_space(2 * GB)  # does not raise


def test_select_files_keeps_only_what_is_needed():
    files = [
        ("model.safetensors", 100),
        ("config.json", 1),
        ("tokenizer.json", 2),
        ("tokenizer_config.json", 1),
        ("vocab.txt", 1),
        ("spiece.model", 1),
        ("1_Pooling/config.json", 1),
        ("pytorch_model.bin", 100),
        ("tf_model.h5", 100),
        ("flax_model.msgpack", 100),
        ("onnx/model.onnx", 100),
        ("onnx/config.json", 1),
        ("openvino/openvino_model.xml", 1),
        ("README.md", 1),
    ]
    kept = [name for name, _ in select_files(files)]
    assert kept == [
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "spiece.model",
        "1_Pooling/config.json",
    ]


def test_cleanup_empties_downloads_and_leaves_other_files_alone(workspace):
    model_file = settings.MODEL_DIR / "model.safetensors"
    model_file.parent.mkdir(parents=True)
    model_file.write_bytes(b"x" * 1000)
    (settings.DOWNLOADS_DIR / "leftover.txt").write_bytes(b"y" * 24)
    outside = workspace / "keep-me.txt"
    outside.write_text("important")

    freed = cleanup_downloads()

    assert freed == 1024
    assert list(settings.DOWNLOADS_DIR.iterdir()) == []
    assert outside.read_text() == "important"


def test_only_one_run_at_a_time(workspace):
    with run_lock():
        with pytest.raises(TestbedError, match="Another testbed run"):
            with run_lock():
                pass
    # After the first run has finished, the lock can be taken again.
    with run_lock():
        pass
