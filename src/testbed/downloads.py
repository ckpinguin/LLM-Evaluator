"""Downloading models only when needed, and deleting them again afterwards.

Rules from the spec (FR-007 to FR-012):

- A model is downloaded only when a run starts, into ``workspace/downloads/model``.
- Before downloading, we check that the disk has enough free space.
- After every run (successful, failed, or cancelled) everything in ``workspace/downloads/`` is
  deleted. Datasets and caches also live there, because ``settings.configure_environment``
  points Hugging Face's ``HF_HOME`` at ``workspace/downloads/hf-home``.
- Only one run may happen at a time (a lock file), so cleaning up never deletes the files of
  a run that is still going on in another terminal.
"""

import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from fnmatch import fnmatch
from pathlib import Path

from filelock import FileLock, Timeout
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

from testbed import settings
from testbed.errors import TestbedError

# Which files of a model repository we download. Weights only in the safe "safetensors"
# format (pickled *.bin files can run code when loaded), plus configuration and tokenizer
# files. Copies of the model for other frameworks (ONNX, OpenVINO, TensorFlow, Flax) are
# skipped: we don't need them and they can be larger than the model itself.
ALLOW_PATTERNS = ["*.safetensors", "*.json", "tokenizer*", "*.txt", "*.model"]
IGNORE_PATTERNS = ["onnx/*", "openvino/*", "original/*", "*.bin", "*.h5", "*.msgpack", "*.ot"]

GB = 1024**3


def select_files(files: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """From (file name, size) pairs, keep the files we would download.

    Uses the same pattern rules as ``snapshot_download`` (where ``*`` also matches "/").
    """
    selected = []
    for name, size in files:
        wanted = any(fnmatch(name, pattern) for pattern in ALLOW_PATTERNS)
        unwanted = any(fnmatch(name, pattern) for pattern in IGNORE_PATTERNS)
        if wanted and not unwanted:
            selected.append((name, size))
    return selected


def planned_files(repo_id: str) -> list[tuple[str, int]]:
    """Ask Hugging Face which files we would download for this model, and how big they are.

    Also refuses models we can't or shouldn't use (spec edge cases), before downloading.
    """
    hint = "Check the id in catalog.yaml, or regenerate it with: uv run testbed catalog build"
    try:
        info = HfApi().model_info(repo_id, files_metadata=True)
    except GatedRepoError as error:
        raise TestbedError(f"'{repo_id}' requires signing in on Hugging Face.", hint) from error
    except RepositoryNotFoundError as error:
        raise TestbedError(f"'{repo_id}' was not found on Hugging Face.", hint) from error
    except (HfHubHTTPError, OSError) as error:
        raise TestbedError(
            f"Could not reach Hugging Face to look up '{repo_id}': {error}",
            hint="Check your internet connection and try again.",
        ) from error

    if info.gated:
        raise TestbedError(f"'{repo_id}' requires signing in on Hugging Face.", hint)
    if "custom_code" in (info.tags or []):
        raise TestbedError(
            f"'{repo_id}' needs to run its own Python code, which the testbed does not allow.",
            hint,
        )

    files = select_files([(s.rfilename, s.size or 0) for s in info.siblings or []])
    if not any(name.endswith(".safetensors") for name, _ in files):
        raise TestbedError(f"'{repo_id}' has no weights in the safe 'safetensors' format.", hint)
    return files


def check_disk_space(model_bytes: int) -> None:
    """Make sure the model (plus 10% and 1 GB for evaluation datasets) fits on the disk."""
    needed = model_bytes * 1.1 + 1 * GB
    free = shutil.disk_usage(settings.WORKSPACE).free
    if free < needed:
        raise TestbedError(
            f"Not enough free disk space: this run needs {needed / GB:.1f} GB, "
            f"but only {free / GB:.1f} GB are free.",
            hint="Free up some space or choose a smaller model.",
        )


def download_model(repo_id: str) -> Path:
    """Download the model's files into ``workspace/downloads/model`` and return that folder.

    Hugging Face shows its own progress bar while downloading (FR-009).
    """
    try:
        snapshot_download(
            repo_id,
            local_dir=settings.MODEL_DIR,
            allow_patterns=ALLOW_PATTERNS,
            ignore_patterns=IGNORE_PATTERNS,
        )
    except (HfHubHTTPError, OSError) as error:
        raise TestbedError(
            f"Downloading '{repo_id}' failed: {error}",
            hint="Check your internet connection and try again.",
        ) from error
    return settings.MODEL_DIR


def folder_size(folder: Path) -> int:
    """Total size in bytes of all files inside a folder (0 if it does not exist)."""
    if not folder.exists():
        return 0
    return sum(file.stat().st_size for file in folder.rglob("*") if file.is_file())


def cleanup_downloads() -> int:
    """Delete everything inside ``workspace/downloads/`` and return how many bytes were freed.

    Only this one folder is touched, never anything outside it (FR-012).
    """
    downloads = settings.DOWNLOADS_DIR
    freed = folder_size(downloads)
    if downloads.exists():
        for item in downloads.iterdir():
            if item.is_dir() and not item.is_symlink():
                shutil.rmtree(item)
            else:
                item.unlink()
    downloads.mkdir(parents=True, exist_ok=True)
    return freed


@contextmanager
def run_lock() -> Iterator[None]:
    """Hold the workspace lock for the duration of a ``with`` block.

    Raises ``TestbedError`` right away if another run already holds it.
    """
    settings.WORKSPACE.mkdir(parents=True, exist_ok=True)
    lock = FileLock(settings.LOCK_FILE, timeout=0)  # timeout=0: don't wait, fail at once
    try:
        lock.acquire()
    except Timeout:
        raise TestbedError(
            "Another testbed run is in progress.",
            hint="Wait for it to finish (only one run at a time protects its downloads).",
        ) from None
    try:
        yield
    finally:
        lock.release()
