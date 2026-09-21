"""Shared test setup (pytest loads this file automatically before the tests)."""

import os
import tempfile

# Point the testbed at a throw-away workspace BEFORE importing it. Importing the testbed sets
# HF_HOME to "<workspace>/downloads/hf-home", and we never want tests to use the real workspace.
os.environ["TESTBED_WORKSPACE"] = tempfile.mkdtemp(prefix="testbed-tests-")

import pytest  # noqa: E402
import yaml  # noqa: E402

from testbed import settings  # noqa: E402

# One valid catalog entry per model type. Tests copy and change these.
SAMPLE_MODELS = [
    {
        "id": "owner/tiny-generator",
        "type": "generative",
        "description": "A tiny text generator used in tests.",
        "size_mb": 10,
        "license": "apache-2.0",
    },
    {
        "id": "owner/tiny-embedder",
        "type": "embedding",
        "description": "A tiny embedding model used in tests.",
        "size_mb": 5,
        "license": "mit",
    },
    {
        "id": "owner/tiny-sentiment",
        "type": "classification",
        "task": "sentiment",
        "label_map": {"NEGATIVE": "negative", "POSITIVE": "positive"},
        "description": "A tiny sentiment classifier used in tests.",
        "size_mb": 8,
        "license": "unknown",
    },
]


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Give each test its own empty workspace folder (downloads, results, lock file)."""
    root = tmp_path / "workspace"
    downloads = root / "downloads"
    downloads.mkdir(parents=True)
    monkeypatch.setattr(settings, "WORKSPACE", root)
    monkeypatch.setattr(settings, "DOWNLOADS_DIR", downloads)
    monkeypatch.setattr(settings, "MODEL_DIR", downloads / "model")
    monkeypatch.setattr(settings, "HF_HOME_DIR", downloads / "hf-home")
    monkeypatch.setattr(settings, "RESULTS_FILE", root / "results.jsonl")
    monkeypatch.setattr(settings, "LOCK_FILE", root / "testbed.lock")
    return root


def write_catalog(path, models):
    """Write a catalog file with the given list of model dictionaries and return its path."""
    data = {"generated_at": "2026-09-21", "max_size_gb": 2.0, "models": models}
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def catalog_file(tmp_path):
    """A valid catalog.yaml with one model of each type."""
    return write_catalog(tmp_path / "catalog.yaml", SAMPLE_MODELS)
