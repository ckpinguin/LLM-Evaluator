"""Central place for the testbed's default values and folder locations.

Everything the testbed writes to disk lives in one "workspace" folder (default: ``workspace/``
next to this project). Inside it:

- ``downloads/``     temporary files; emptied after every run (models, datasets, caches)
- ``results.jsonl``  the history of all evaluation results (kept)
- ``testbed.lock``   lock file that makes sure only one run happens at a time

The workspace can be moved by setting the environment variable ``TESTBED_WORKSPACE``.

Other modules read these values as ``settings.NAME`` *when they need them* (not with
``from settings import NAME``). That way tests can point them at a temporary folder.
"""

import os
from pathlib import Path

# --- Default values (can be changed with command-line options) ---------------------------------

DEFAULT_LIMIT = 200  # number of examples each evaluator uses (spec FR-017)
DEFAULT_MAX_SIZE_GB = 2.0  # largest model download the catalog builder accepts
DEFAULT_PER_TYPE = 5  # how many models per type the catalog builder keeps
SEED = 42  # fixed random seed, so repeated runs pick the same examples
MAX_LENGTH = 2048  # longest text (in tokens) a generative model reads at once

# --- Folder and file locations ------------------------------------------------------------------

# This file is src/testbed/settings.py, so the project root is two folders up from its folder.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = PROJECT_ROOT / "catalog.yaml"

WORKSPACE = Path(os.environ.get("TESTBED_WORKSPACE", PROJECT_ROOT / "workspace"))
DOWNLOADS_DIR = WORKSPACE / "downloads"
MODEL_DIR = DOWNLOADS_DIR / "model"
HF_HOME_DIR = DOWNLOADS_DIR / "hf-home"
RESULTS_FILE = WORKSPACE / "results.jsonl"
LOCK_FILE = WORKSPACE / "testbed.lock"


def configure_environment() -> None:
    """Create the workspace folders and set environment variables for the ML libraries.

    This MUST run before PyTorch or any Hugging Face library is imported, because those
    libraries read these variables only once, when they are first imported.
    """
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

    # Send every Hugging Face download (models, datasets, caches) into our own downloads
    # folder. This keeps the user's normal ~/.cache/huggingface untouched and lets us delete
    # everything after a run by emptying a single folder.
    os.environ["HF_HOME"] = str(HF_HOME_DIR)

    # Some PyTorch operations are not implemented on the Apple GPU ("mps") yet. With this
    # setting PyTorch quietly runs them on the CPU instead of crashing.
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # Transformers 5 loads model weights with several threads at once. When loading straight
    # onto the Apple GPU this can crash the whole program (segmentation fault), so we ask it
    # to load the weights one after the other. For our small models this costs well under a
    # second.
    os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"

    # Keep the terminal readable: no progress bars while datasets are prepared, and only
    # real errors from Transformers. (The download progress bar stays visible.)
    os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"

    # Avoid a noisy warning from the fast tokenizers library about parallelism.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
